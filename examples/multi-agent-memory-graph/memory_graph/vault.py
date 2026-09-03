# SPDX-License-Identifier: Apache-2.0

"""Append-only Markdown ledger (Obsidian compatible)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .parsing import Triple

__all__ = ["append_entry", "format_triples"]


def format_triples(triples: Sequence[Triple]) -> str:
    """Render triples as Markdown bullets."""
    return "\n".join(
        f"- `{t.source}` --{t.relation}--> `{t.target}`" for t in triples
    )


def append_entry(
    path: str | os.PathLike[str],
    facts: str,
    triples: Sequence[Triple] = (),
    note: str | None = None,
) -> None:
    """Append one timestamped entry to the ledger.

    Every entry carries a UTC timestamp: an append-only log of identical
    '### Autonomous Log' headings cannot be audited, diffed against the graph,
    or trimmed by age.
    """
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [f"\n\n### Autonomous Log — {stamp}\n", facts.strip()]

    if triples:
        lines.append("\n\n**Graph triples**\n")
        lines.append(format_triples(triples))
    if note:
        lines.append(f"\n\n> {note}")

    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
