# SPDX-License-Identifier: Apache-2.0

"""Pure parsing helpers for LLM output.

Nothing in this module imports langchain, chromadb or an LLM client, so the
fragile string-handling can be unit tested without a running Ollama server.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Iterable, NamedTuple

__all__ = [
    "TripleParseError",
    "Triple",
    "canonical_entity",
    "entity_key",
    "parse_save_verdict",
    "parse_triples",
]

_SAVE = "TRIGGER_SAVE"
_IGNORE = "IGNORE"

# Wrapping decoration the models like to add around a one-word verdict.
_VERDICT_DECORATION = " \t`*_\"'.,:;!?()[]{}"

# Narrower for entities: brackets and parentheses occur inside real names
# ("Gemma 4 [preview]", "Jetson (Orin)"), so stripping them corrupts the node.
_ENTITY_DECORATION = " \t`*_\"'.,:;"

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class TripleParseError(ValueError):
    """Raised when a model response contains no recoverable triple list."""


class Triple(NamedTuple):
    """A knowledge-graph edge with both display and identity forms resolved."""

    source: str
    relation: str
    target: str
    source_key: str
    target_key: str
    relation_key: str


def canonical_entity(name: str) -> str:
    """Return the display form of an entity: trimmed, unwrapped, single-spaced.

    This is deliberately conservative. It merges the differences that are pure
    formatting noise ('  User ', '"User"', 'User.') but never guesses that two
    distinct strings mean the same thing -- see ``docs`` in the README on why
    fuzzy merging is left to an explicit alias map.
    """
    text = unicodedata.normalize("NFKC", str(name))
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(_ENTITY_DECORATION)
    return re.sub(r"\s+", " ", text).strip()


def entity_key(name: str) -> str:
    """Return the identity form of an entity: case-folded canonical form.

    Two entities share a node when their keys match, so 'ThinkPad P16s' and
    'thinkpad p16s' collapse into one node while keeping the first-seen
    spelling as the label.
    """
    return canonical_entity(name).casefold()


def parse_save_verdict(raw: str) -> bool:
    """Decide whether the evaluator asked for a save.

    A substring test for 'TRIGGER_SAVE' false-positives on any response that
    restates the instruction ('IGNORE -- this is not TRIGGER_SAVE'), which
    writes junk into long-term memory. Resolution order:

    1. The first line that is *exactly* one of the two verdicts wins.
    2. Otherwise, a response mentioning exactly one verdict token uses it.
    3. Anything else is ambiguous and defaults to not saving.
    """
    if not raw:
        return False

    upper = raw.upper()

    for line in upper.splitlines():
        stripped = line.strip().strip(_VERDICT_DECORATION).strip()
        if stripped == _SAVE:
            return True
        if stripped == _IGNORE:
            return False

    has_save = re.search(rf"\b{_SAVE}\b", upper) is not None
    has_ignore = re.search(rf"\b{_IGNORE}\b", upper) is not None
    if has_save and not has_ignore:
        return True
    return False


def _iter_balanced_arrays(text: str) -> Iterable[str]:
    """Yield every balanced top-level ``[...]`` span, outermost first.

    ``re.search(r"\\[.*\\]", text, re.DOTALL)`` is greedy: it spans from the
    first '[' to the *last* ']' in the response, so a single stray bracket in
    trailing prose swallows the JSON and the parse fails. Scanning for balance
    (while respecting string literals and escapes) does not have that failure.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "[":
            if depth == 0:
                start = index
            depth += 1
        elif char == "]":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : index + 1]
                start = -1


def _coerce_triples(payload: object) -> list[Triple]:
    if not isinstance(payload, list):
        raise TripleParseError(f"expected a JSON list, got {type(payload).__name__}")

    triples: list[Triple] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        source = canonical_entity(item.get("source", ""))
        relation = canonical_entity(item.get("relation", ""))
        target = canonical_entity(item.get("target", ""))
        if not source or not target or not relation:
            continue
        triples.append(
            Triple(
                source=source,
                relation=relation,
                target=target,
                source_key=entity_key(source),
                target_key=entity_key(target),
                relation_key=relation.casefold(),
            )
        )
    return triples


def parse_triples(raw: str) -> list[Triple]:
    """Extract knowledge-graph triples from a model response.

    Raises ``TripleParseError`` rather than returning an empty list on failure:
    a bare ``except: triples = []`` makes a broken extractor indistinguishable
    from a conversation that genuinely held no facts.
    """
    if not raw or not raw.strip():
        raise TripleParseError("empty response")

    candidates: list[str] = []
    stripped = raw.strip()
    candidates.append(stripped)
    candidates.extend(match.strip() for match in _FENCE_RE.findall(raw))
    candidates.extend(_iter_balanced_arrays(raw))

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        try:
            return _coerce_triples(payload)
        except TripleParseError as exc:
            last_error = exc

    raise TripleParseError(f"no JSON triple list found in response: {last_error}")
