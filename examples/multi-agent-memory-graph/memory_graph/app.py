# SPDX-License-Identifier: Apache-2.0

"""LangGraph workflow wiring the four memory agents together.

Importing this module requires the langchain / langgraph / chromadb stack and a
reachable Ollama server; the pure logic it delegates to lives in ``parsing``,
``graph_store`` and ``vault`` so it can be tested without either.
"""

from __future__ import annotations

import json
import operator
import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

try:  # langchain-chroma is the maintained home; community's copy is deprecated.
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - depends on the installed extras
    from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import ChatOllama

from .graph_store import load_graph, merge_triples, render_graph, save_graph
from .parsing import TripleParseError, parse_save_verdict, parse_triples
from .vault import append_entry, format_triples

# ---------------------------------------------------------------------------
# 1. State definition
# ---------------------------------------------------------------------------


class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    facts_extracted: list[str]
    triples_extracted: list[dict]
    vault_updated: bool
    # operator.add so an error raised in extraction is not overwritten
    # by the graph agent's (empty) error list later in the same run.
    memory_errors: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# 2. Model and storage initialization
# ---------------------------------------------------------------------------

OLLAMA_MODEL = os.environ.get("MEMORY_GRAPH_MODEL", "llama3.1:8b")
VAULT_DIR = os.environ.get("MEMORY_GRAPH_VAULT", "./vault")
CHROMA_DIR = os.environ.get("MEMORY_GRAPH_CHROMA", "./chroma_vault")

VAULT_LEDGER = os.path.join(VAULT_DIR, "Memory_Ledger.md")
GRAPH_HTML_PATH = os.path.join(VAULT_DIR, "memory_graph.html")
GRAPH_GML_PATH = os.path.join(VAULT_DIR, "knowledge_graph.gml")
os.makedirs(VAULT_DIR, exist_ok=True)

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)
embeddings = FastEmbedEmbeddings()

vector_store = Chroma(
    collection_name="autonomous_agent_memory",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)


def _last_human_message(messages: list[BaseMessage]) -> str | None:
    """Return the most recent human turn, or None if the state holds none."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return None


# ---------------------------------------------------------------------------
# 3. Agent definitions
# ---------------------------------------------------------------------------


def dialogue_agent(state: MultiAgentState) -> dict:
    """Agent A: core dialogue, grounded in a RAG lookup over prior memory."""
    query = _last_human_message(state["messages"])

    # Indexing on [-1] of a filtered list raises IndexError whenever the graph
    # is invoked with no human turn (a system-only bootstrap, a replayed
    # checkpoint). Skip retrieval instead of crashing the whole run.
    context_str = ""
    if query:
        relevant_context = vector_store.similarity_search(query, k=2)
        context_str = "\n".join(doc.page_content for doc in relevant_context)

    system_prompt = (
        "You are an intelligent assistant. Use the following context if relevant:\n"
        f"{context_str}\n"
        "Provide a direct, concise response."
    )

    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}


def evaluator_agent_condition(
    state: MultiAgentState,
) -> Literal["extraction_agent", "__end__"]:
    """Agent B: decide whether the exchange is worth committing to memory."""
    recent_messages = state["messages"][-2:]

    eval_prompt = [
        SystemMessage(
            content=(
                "Evaluate if the recent exchange contains permanent facts, user "
                "preferences, configuration changes, or system state that should "
                "be stored long-term.\n"
                "Reply with exactly one word and nothing else: TRIGGER_SAVE or IGNORE."
            )
        ),
        *recent_messages,
    ]

    verdict = llm.invoke(eval_prompt).content
    return "extraction_agent" if parse_save_verdict(str(verdict)) else END


def extraction_and_vault_agent(state: MultiAgentState) -> dict:
    """Agent C: distil standalone facts and persist them to both stores."""
    recent_dialogue = "\n".join(
        f"{message.type}: {message.content}" for message in state["messages"][-2:]
    )

    fact_prompt = [
        SystemMessage(
            content=(
                "Extract distinct, standalone facts from the conversation as bullet "
                "points. Do not include conversational filler."
            )
        ),
        HumanMessage(content=recent_dialogue),
    ]
    facts_response = str(llm.invoke(fact_prompt).content).strip()

    if not facts_response:
        return {
            "facts_extracted": [],
            "vault_updated": False,
            "memory_errors": ["fact extraction returned an empty response"],
        }

    vector_store.add_texts(
        texts=[facts_response],
        metadatas=[{"source": "autonomous_trigger", "type": "episodic_fact"}],
    )

    return {"facts_extracted": [facts_response], "vault_updated": True}


def graph_rag_memory_agent(state: MultiAgentState) -> dict:
    """Agent D: extract entity-relation triples, merge, persist, visualize."""
    facts = "\n".join(state.get("facts_extracted", []))
    if not facts:
        return {"triples_extracted": []}

    triple_prompt = [
        SystemMessage(
            content=(
                "Extract knowledge graph triples from the text. Respond with a valid "
                "JSON list of objects with 'source', 'relation' and 'target' keys, "
                "and nothing else.\n"
                'Example: [{"source": "User", "relation": "manages", "target": "TaskTech"}]'
            )
        ),
        HumanMessage(content=facts),
    ]
    triple_raw = str(llm.invoke(triple_prompt).content).strip()

    errors: list[str] = []
    try:
        triples = parse_triples(triple_raw)
    except TripleParseError as exc:
        # Recorded in state rather than swallowed: a broken extractor and a
        # conversation that held no relations must not look identical.
        triples = []
        errors.append(f"triple extraction failed: {exc}")

    if triples:
        graph = load_graph(GRAPH_GML_PATH)
        added = merge_triples(graph, triples)
        save_graph(graph, GRAPH_GML_PATH)
        render_graph(graph, GRAPH_HTML_PATH)

        # Feed the triples back into episodic memory so the graph and the
        # vector store describe the same world instead of drifting apart.
        vector_store.add_texts(
            texts=[format_triples(triples)],
            metadatas=[{"source": "graph_rag", "type": "semantic_triple"}],
        )
        note = f"{added} new edge(s) merged into the knowledge graph."
    else:
        note = errors[0] if errors else "No graph triples found in these facts."

    append_entry(VAULT_LEDGER, facts, triples, note)

    payload = [
        {"source": t.source, "relation": t.relation, "target": t.target} for t in triples
    ]
    return {"triples_extracted": payload, "memory_errors": errors}


# ---------------------------------------------------------------------------
# 4. Workflow compilation
# ---------------------------------------------------------------------------


def build_app():
    """Compile the memory workflow."""
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("dialogue_agent", dialogue_agent)
    workflow.add_node("extraction_agent", extraction_and_vault_agent)
    workflow.add_node("graph_rag_agent", graph_rag_memory_agent)

    workflow.add_edge(START, "dialogue_agent")
    workflow.add_conditional_edges(
        "dialogue_agent",
        evaluator_agent_condition,
        {"extraction_agent": "extraction_agent", END: END},
    )
    workflow.add_edge("extraction_agent", "graph_rag_agent")
    workflow.add_edge("graph_rag_agent", END)

    return workflow.compile()


app = build_app()


# ---------------------------------------------------------------------------
# 5. Execution test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_input: MultiAgentState = {
        "messages": [
            HumanMessage(
                content=(
                    "Note down that our core deployment model is Gemma 4 on the "
                    "ThinkPad P16s workstation."
                )
            )
        ],
        "facts_extracted": [],
        "triples_extracted": [],
        "vault_updated": False,
        "memory_errors": [],
    }

    result = app.invoke(test_input)

    print("\n--- Response ---")
    print(result["messages"][-1].content)
    print("\n--- Extracted Triples ---")
    print(json.dumps(result.get("triples_extracted", []), indent=2))
    for error in result.get("memory_errors", []):
        print(f"\n[warning] {error}")
    print(f"\nVisual graph generated at: {GRAPH_HTML_PATH}")
