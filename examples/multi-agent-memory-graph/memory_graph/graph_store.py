# SPDX-License-Identifier: Apache-2.0

"""Persistent knowledge graph backing the memory agents.

Depends only on networkx and pyvis, so the merge and persistence behaviour is
testable without an LLM.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import networkx as nx
from pyvis.network import Network

from .parsing import Triple

__all__ = ["load_graph", "merge_triples", "save_graph", "render_graph"]

_PHYSICS_OPTIONS = """
var options = {
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "centralGravity": 0.01,
      "springLength": 100,
      "springConstant": 0.08
    },
    "maxVelocity": 50,
    "solver": "forceAtlas2Based",
    "timestep": 0.35
  }
}
"""


def load_graph(path: str | os.PathLike[str]) -> nx.MultiDiGraph:
    """Read the graph from ``path``, returning an empty graph if absent.

    A ``MultiDiGraph`` -- not a ``DiGraph`` -- because a DiGraph holds a single
    edge per node pair, so recording 'User -> founded -> TaskTech' after
    'User -> manages -> TaskTech' silently destroys the first relation.
    """
    graph_path = Path(path)
    if not graph_path.exists():
        return nx.MultiDiGraph()

    graph = nx.read_gml(graph_path)
    if not isinstance(graph, nx.MultiDiGraph):
        graph = nx.MultiDiGraph(graph)
    return graph


def merge_triples(graph: nx.MultiDiGraph, triples: Iterable[Triple]) -> int:
    """Merge ``triples`` into ``graph`` in place; return the new-edge count.

    Nodes are keyed by their case-folded identity form and carry the first-seen
    spelling as ``display``. The relation is the edge key, which makes re-adding
    an already-known fact idempotent while still allowing several distinct
    relations between the same pair of entities.
    """
    added = 0
    for triple in triples:
        for key, display in (
            (triple.source_key, triple.source),
            (triple.target_key, triple.target),
        ):
            if key not in graph:
                graph.add_node(key, display=display)
            elif not graph.nodes[key].get("display"):
                graph.nodes[key]["display"] = display

        if not graph.has_edge(triple.source_key, triple.target_key, key=triple.relation_key):
            added += 1
        graph.add_edge(
            triple.source_key,
            triple.target_key,
            key=triple.relation_key,
            relation=triple.relation,
        )
    return added


def save_graph(graph: nx.MultiDiGraph, path: str | os.PathLike[str]) -> None:
    """Write ``graph`` to ``path`` atomically, keeping one ``.bak`` generation.

    Writing in place means a crash or a full disk mid-write leaves a truncated
    GML file and the accumulated memory is gone. Write to a temporary file in
    the same directory, then ``os.replace`` it over the target.
    """
    graph_path = Path(path)
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    handle, tmp_name = tempfile.mkstemp(
        dir=graph_path.parent, prefix=graph_path.name, suffix=".tmp"
    )
    os.close(handle)
    tmp_path = Path(tmp_name)
    try:
        nx.write_gml(graph, tmp_path)
        if graph_path.exists():
            shutil.copy2(graph_path, graph_path.with_suffix(graph_path.suffix + ".bak"))
        os.replace(tmp_path, graph_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def render_graph(graph: nx.MultiDiGraph, path: str | os.PathLike[str]) -> None:
    """Render an interactive pyvis view of ``graph`` to ``path``."""
    html_path = Path(path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    # pyvis reads 'label' and 'title'; the stored graph keeps 'display' so the
    # GML writer never has to reconcile a node attribute against the node id.
    view = nx.MultiDiGraph()
    for node, data in graph.nodes(data=True):
        display = data.get("display", node)
        view.add_node(node, label=display, title=display)
    for source, target, data in graph.edges(data=True):
        relation = data.get("relation", "")
        view.add_edge(source, target, label=relation, title=relation)

    # cdn_resources="in_line" embeds vis-network instead of writing a sibling
    # lib/ directory next to (or into the cwd alongside) the output, so the
    # vault gets one self-contained file that opens offline.
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,
        cdn_resources="in_line",
    )
    net.from_nx(view)
    net.set_options(_PHYSICS_OPTIONS)
    net.save_graph(str(html_path))
