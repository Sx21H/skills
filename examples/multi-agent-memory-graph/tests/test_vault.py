# SPDX-License-Identifier: Apache-2.0

"""Tests for the Markdown ledger."""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_graph.vault import append_entry, format_triples  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_graph_store import triple  # noqa: E402

STAMP_RE = re.compile(r"### Autonomous Log — \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}Z")


class TestLedger(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "vault" / "Memory_Ledger.md"

    def test_entry_is_timestamped(self) -> None:
        append_entry(self.path, "- Deployment model is Gemma 4.")
        self.assertRegex(self.path.read_text(encoding="utf-8"), STAMP_RE)

    def test_entries_append_rather_than_overwrite(self) -> None:
        append_entry(self.path, "- First fact.")
        append_entry(self.path, "- Second fact.")
        content = self.path.read_text(encoding="utf-8")
        self.assertIn("First fact.", content)
        self.assertIn("Second fact.", content)
        self.assertEqual(len(STAMP_RE.findall(content)), 2)

    def test_triples_and_note_are_recorded(self) -> None:
        append_entry(
            self.path,
            "- User manages TaskTech.",
            [triple("User", "manages", "TaskTech")],
            note="1 new edge(s) merged into the knowledge graph.",
        )
        content = self.path.read_text(encoding="utf-8")
        self.assertIn("`User` --manages--> `TaskTech`", content)
        self.assertIn("> 1 new edge(s) merged", content)

    def test_parent_directory_is_created(self) -> None:
        self.assertFalse(self.path.parent.exists())
        append_entry(self.path, "- A fact.")
        self.assertTrue(self.path.exists())

    def test_format_triples_is_empty_for_no_triples(self) -> None:
        self.assertEqual(format_triples([]), "")


if __name__ == "__main__":
    unittest.main()
