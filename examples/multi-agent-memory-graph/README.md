<!-- SPDX-License-Identifier: Apache-2.0 -->

# Multi-Agent Memory Graph

A LangGraph workflow that gives a local Ollama assistant autonomous long-term
memory. Four agents cooperate on every turn:

| Agent | Role |
| --- | --- |
| **A — Dialogue** | Answers the user, grounded in a RAG lookup over prior memory. |
| **B — Evaluator gate** | Decides whether the exchange is worth remembering. |
| **C — Extraction & vault** | Distils standalone facts into ChromaDB and a Markdown ledger. |
| **D — Graph RAG** | Extracts entity-relation triples, merges them into a persistent knowledge graph, and renders it. |

```
START → dialogue_agent ─(evaluator gate)─→ extraction_agent → graph_rag_agent → END
                        └──── IGNORE ─────────────────────────────────────────→ END
```

## Layout

```
memory_graph/
  parsing.py      Pure LLM-output parsing: verdicts, triples, entity normalization
  graph_store.py  networkx merge + atomic GML persistence + pyvis rendering
  vault.py        Timestamped append-only Markdown ledger
  app.py          Agents, LangGraph wiring, CLI entry point
tests/            Runs without an LLM, ChromaDB or a network connection
```

The split exists so the fragile parts — string parsing and graph merging — can
be tested without a running Ollama server. `app.py` is the only module that
imports the LLM stack.

## Running

```bash
pip install -r requirements.txt
ollama serve && ollama pull llama3.1:8b
python -m memory_graph.app
```

Configuration is read from the environment:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEMORY_GRAPH_MODEL` | `llama3.1:8b` | Ollama model tag |
| `MEMORY_GRAPH_VAULT` | `./vault` | Ledger, GML and HTML output directory |
| `MEMORY_GRAPH_CHROMA` | `./chroma_vault` | ChromaDB persistence directory |

Outputs land in the vault directory: `Memory_Ledger.md` (Obsidian compatible),
`knowledge_graph.gml`, and a self-contained `memory_graph.html`.

## Tests

```bash
pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

## Fixes applied to the original prototype

**Relations were being destroyed.** The graph was a `nx.DiGraph`, which holds
one edge per node pair, so `add_edge` overwrote any existing relation. Learning
`User -founded-> TaskTech` after `User -manages-> TaskTech` silently discarded
the first. It is now a `MultiDiGraph` keyed on the relation, which both
preserves parallel relations and makes re-learning a known fact idempotent.

**Triple parsing dropped valid extractions.** `re.search(r"\[.*\]", raw, DOTALL)`
is greedy: it spans from the first `[` to the *last* `]` in the response, so one
stray bracket in trailing prose produced invalid JSON. A balanced-bracket scan
that respects string literals replaces it, with markdown-fence handling. The
bare `except: triples = []` is gone — parse failures now raise
`TripleParseError`, which the agent records in `memory_errors` so a broken
extractor is distinguishable from a conversation that held no relations.

**The save gate false-positived.** `"TRIGGER_SAVE" in verdict` matched any
response that merely named the token, including `IGNORE — this is not
TRIGGER_SAVE material`, writing junk into permanent memory. `parse_save_verdict`
resolves the first standalone verdict line and defaults to not saving when the
response is ambiguous.

**`dialogue_agent` crashed on system-only state.** `[m for m in messages if
isinstance(m, HumanMessage)][-1]` raised `IndexError` when invoked with no human
turn. Retrieval is now skipped instead.

**Entity fragmentation.** `User` and `user`, and `"TaskTech".` and `TaskTech`,
became separate nodes. Nodes are keyed on a case-folded canonical form and keep
the first-seen spelling as `display`. This is deliberately conservative — it
merges formatting noise only. It will *not* merge `ThinkPad P16s` with
`ThinkPad P16s workstation`; fuzzy merging risks collapsing genuinely distinct
entities, so that belongs in an explicit alias map, not in normalization.

**Non-atomic graph writes.** `nx.write_gml` wrote straight over the store, so a
crash or full disk mid-write truncated the file and lost all accumulated memory.
Writes now go to a temporary file in the same directory and are moved into place
with `os.replace`, keeping one `.bak` generation.

**Graph and vector memory drifted apart.** Extracted triples were never fed back
into ChromaDB, so semantic retrieval could not see anything the graph knew. They
are now indexed alongside the episodic facts.

**The ledger was unauditable.** Every entry used an identical
`### Autonomous Log` heading. Entries now carry a UTC timestamp, the triples
merged, and an outcome note.

**Deprecated imports and pyvis asset spill.** `Chroma` now prefers
`langchain_chroma` with a fallback to the deprecated community copy, and the
pyvis renderer inlines vis-network rather than writing a sibling `lib/`
directory into the working directory.

## Known limitations

- The evaluator adds a second full LLM round-trip per turn; on an 8B local model
  that roughly doubles latency.
- Nothing prunes or ages out memory. The graph and ledger grow without bound.
- There is no contradiction handling: if a fact is superseded, both the old and
  the new edge persist side by side.
- Triple quality is entirely dependent on the local model's JSON discipline.
