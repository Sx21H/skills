# SPDX-License-Identifier: Apache-2.0

"""Tests for the LLM-output parsing helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_graph.parsing import (  # noqa: E402
    TripleParseError,
    canonical_entity,
    entity_key,
    parse_save_verdict,
    parse_triples,
)


class TestSaveVerdict(unittest.TestCase):
    def test_plain_verdicts(self) -> None:
        self.assertTrue(parse_save_verdict("TRIGGER_SAVE"))
        self.assertFalse(parse_save_verdict("IGNORE"))

    def test_decorated_verdicts(self) -> None:
        self.assertTrue(parse_save_verdict("  **TRIGGER_SAVE**  "))
        self.assertTrue(parse_save_verdict("`trigger_save`"))
        self.assertFalse(parse_save_verdict('"IGNORE."'))

    def test_restated_instruction_does_not_false_positive(self) -> None:
        # The substring check `"TRIGGER_SAVE" in verdict` saved junk to memory
        # on every response that merely named the token.
        self.assertFalse(
            parse_save_verdict("IGNORE - this exchange is not TRIGGER_SAVE material")
        )
        self.assertFalse(
            parse_save_verdict("Output strictly: 'TRIGGER_SAVE' or 'IGNORE'. IGNORE")
        )

    def test_verdict_on_first_line_wins(self) -> None:
        self.assertTrue(parse_save_verdict("TRIGGER_SAVE\nBecause it names a model."))

    def test_ambiguous_and_empty_default_to_no_save(self) -> None:
        self.assertFalse(parse_save_verdict(""))
        self.assertFalse(parse_save_verdict("I am not sure about this one."))


class TestEntityNormalization(unittest.TestCase):
    def test_formatting_noise_is_stripped(self) -> None:
        self.assertEqual(canonical_entity('  "User".  '), "User")
        self.assertEqual(canonical_entity("ThinkPad   P16s"), "ThinkPad P16s")

    def test_identity_is_case_insensitive(self) -> None:
        self.assertEqual(entity_key("ThinkPad P16s"), entity_key("thinkpad p16s"))

    def test_distinct_entities_stay_distinct(self) -> None:
        self.assertNotEqual(entity_key("Gemma 4"), entity_key("Gemma 3"))


class TestTripleParsing(unittest.TestCase):
    def test_bare_json_list(self) -> None:
        triples = parse_triples(
            '[{"source":"User","relation":"manages","target":"TaskTech"}]'
        )
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0].source, "User")
        self.assertEqual(triples[0].relation, "manages")
        self.assertEqual(triples[0].target, "TaskTech")

    def test_fenced_json(self) -> None:
        raw = 'Here you go:\n```json\n[{"source":"A","relation":"r","target":"B"}]\n```'
        self.assertEqual(len(parse_triples(raw)), 1)

    def test_trailing_bracket_in_prose(self) -> None:
        # The greedy `\[.*\]` span ran from the first '[' to this last ']',
        # producing invalid JSON and silently dropping the extraction.
        raw = (
            '[{"source":"A","relation":"r","target":"B"}]\n'
            "Note: derived from the transcript [see above]"
        )
        triples = parse_triples(raw)
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0].source, "A")

    def test_bracket_inside_string_literal(self) -> None:
        raw = '[{"source":"A [beta]","relation":"r","target":"B"}]'
        triples = parse_triples(raw)
        self.assertEqual(triples[0].source, "A [beta]")

    def test_incomplete_items_are_dropped(self) -> None:
        raw = (
            '[{"source":"A","relation":"r","target":"B"},'
            '{"source":"","relation":"r","target":"C"},'
            '{"source":"D","target":"E"}]'
        )
        self.assertEqual(len(parse_triples(raw)), 1)

    def test_failure_raises_instead_of_returning_empty(self) -> None:
        for raw in ("", "   ", "I could not find any triples.", "{not json"):
            with self.assertRaises(TripleParseError):
                parse_triples(raw)

    def test_valid_but_empty_list_is_not_an_error(self) -> None:
        self.assertEqual(parse_triples("[]"), [])


if __name__ == "__main__":
    unittest.main()
