# SPDX-License-Identifier: Apache-2.0

"""Tests for knowledge-graph merge, persistence and rendering."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402

from memory_graph.graph_store import (  # noqa: E402
    load_graph,
    merge_triples,
    render_graph,
    save_graph,
)
from memory_graph.parsing import Triple, canonical_entity, entity_key  # noqa: E402


def triple(source: str, relation: str, target: str) -> Triple:
    source, relation, target = (
        canonical_entity(source),
        canonical_entity(relation),
        canonical_entity(target),
    )
    return Triple(
        source=source,
        relation=relation,
        target=target,
        source_key=entity_key(source),
        target_key=entity_key(target),
        relation_key=relation.casefold(),
    )


class GraphStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "knowledge_graph.gml"


class TestMerge(GraphStoreTestCase):
    def test_parallel_relations_are_both_kept(self) -> None:
        # On a DiGraph the second add_edge overwrote the first, so 'manages'
        # vanished the moment 'founded' was learned.
        graph = nx.MultiDiGraph()
        merge_triples(
            graph,
            [triple("User", "manages", "TaskTech"), triple("User", "founded", "TaskTech")],
        )
        relations = {data["relation"] for *_, data in graph.edges(data=True)}
        self.assertEqual(relations, {"manages", "founded"})

    def test_case_variants_collapse_to_one_node(self) -> None:
        graph = nx.MultiDiGraph()
        merge_triples(
            graph,
            [
                triple("ThinkPad P16s", "runs", "Gemma 4"),
                triple("thinkpad p16s", "hosts", "Ollama"),
            ],
        )
        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.nodes[entity_key("ThinkPad P16s")]["display"], "ThinkPad P16s")

    def test_repeated_merge_is_idempotent(self) -> None:
        graph = nx.MultiDiGraph()
        facts = [triple("User", "manages", "TaskTech")]
        self.assertEqual(merge_triples(graph, facts), 1)
        self.assertEqual(merge_triples(graph, facts), 0)
        self.assertEqual(graph.number_of_edges(), 1)

    def test_added_count_reflects_only_new_edges(self) -> None:
        graph = nx.MultiDiGraph()
        merge_triples(graph, [triple("A", "r", "B")])
        added = merge_triples(graph, [triple("A", "r", "B"), triple("A", "s", "B")])
        self.assertEqual(added, 1)


class TestPersistence(GraphStoreTestCase):
    def test_round_trip_preserves_type_nodes_and_relations(self) -> None:
        graph = nx.MultiDiGraph()
        merge_triples(
            graph,
            [triple("User", "manages", "TaskTech"), triple("User", "founded", "TaskTech")],
        )
        save_graph(graph, self.path)

        restored = load_graph(self.path)
        self.assertIsInstance(restored, nx.MultiDiGraph)
        self.assertEqual(restored.number_of_nodes(), 2)
        self.assertEqual(restored.number_of_edges(), 2)
        self.assertEqual(restored.nodes[entity_key("User")]["display"], "User")

    def test_merge_accumulates_across_sessions(self) -> None:
        first = load_graph(self.path)
        merge_triples(first, [triple("User", "manages", "TaskTech")])
        save_graph(first, self.path)

        second = load_graph(self.path)
        merge_triples(second, [triple("TaskTech", "deploys", "Gemma 4")])
        save_graph(second, self.path)

        final = load_graph(self.path)
        self.assertEqual(final.number_of_nodes(), 3)
        self.assertEqual(final.number_of_edges(), 2)

    def test_missing_file_yields_empty_graph(self) -> None:
        graph = load_graph(self.path / "nope.gml")
        self.assertEqual(graph.number_of_nodes(), 0)

    def test_save_keeps_one_backup_generation(self) -> None:
        graph = nx.MultiDiGraph()
        merge_triples(graph, [triple("A", "r", "B")])
        save_graph(graph, self.path)
        self.assertFalse(self.path.with_suffix(".gml.bak").exists())

        merge_triples(graph, [triple("B", "r", "C")])
        save_graph(graph, self.path)
        backup = self.path.with_suffix(".gml.bak")
        self.assertTrue(backup.exists())
        self.assertEqual(load_graph(backup).number_of_edges(), 1)

    def test_save_leaves_no_temp_files_behind(self) -> None:
        graph = nx.MultiDiGraph()
        merge_triples(graph, [triple("A", "r", "B")])
        save_graph(graph, self.path)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])


class TestRender(GraphStoreTestCase):
    def test_render_writes_html_with_display_labels(self) -> None:
        graph = nx.MultiDiGraph()
        merge_triples(graph, [triple("ThinkPad P16s", "runs", "Gemma 4")])
        html_path = self.path.parent / "memory_graph.html"
        render_graph(graph, html_path)

        content = html_path.read_text(encoding="utf-8")
        self.assertIn("ThinkPad P16s", content)
        self.assertIn("runs", content)

    def test_render_of_empty_graph_does_not_raise(self) -> None:
        html_path = self.path.parent / "empty.html"
        render_graph(nx.MultiDiGraph(), html_path)
        self.assertTrue(html_path.exists())


if __name__ == "__main__":
    unittest.main()
