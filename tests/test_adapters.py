"""
Unit tests for NetReach graph data loaders and sub-graph extractors.
"""

import json
import tempfile
import pytest
import networkx as nx
import torch

from netreach.adapters.graph_loader import load_graph, parse_dot_string
from netreach.adapters.extractor import extract_subgraph, prepare_graph_batch


def test_load_graph_from_json_list():
    """Verify loading from list of edge dictionaries."""
    data = [
        {"source": "CompA", "target": "User1", "relation": "HasSession"},
        {"source": "User1", "target": "AdminGroup", "relation": "MemberOf"},
    ]
    G = load_graph(data)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert nx.has_path(G, "CompA", "AdminGroup")


def test_load_graph_from_json_file():
    """Verify loading from JSON file on disk."""
    data = [
        {"source": "WS01", "target": "SRV01"},
        {"source": "SRV01", "target": "DC01"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    G = load_graph(path)
    assert G.number_of_nodes() == 3
    assert nx.has_path(G, "WS01", "DC01")


def test_load_graph_from_dot():
    """Verify loading from Graphviz DOT format."""
    dot_content = """
    digraph AttackPath {
        "HOST-A" -> "HOST-B" [label="RDP"];
        "HOST-B" -> "HOST-C" [label="SSH"];
    }
    """
    G = parse_dot_string(dot_content)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert nx.has_path(G, "HOST-A", "HOST-C")


def test_load_graph_from_graphml():
    """Verify loading from GraphML format."""
    G_orig = nx.DiGraph()
    G_orig.add_edge("N1", "N2")
    G_orig.add_edge("N2", "N3")

    with tempfile.NamedTemporaryFile(suffix=".graphml", delete=False) as f:
        path = f.name
    nx.write_graphml(G_orig, path)

    G_loaded = load_graph(path)
    assert G_loaded.number_of_nodes() == 3
    assert nx.has_path(G_loaded, "N1", "N3")


def test_extract_subgraph():
    """Verify K-hop bidirectional corridor pruning."""
    G = nx.DiGraph()
    # Path: A -> B -> C -> D
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    # Distractor branch: B -> X -> Y
    G.add_edges_from([("B", "X"), ("X", "Y")])

    subG, meta = extract_subgraph(G, source="A", target="D", max_nodes=10)
    assert meta["has_ground_truth_path"] is True
    assert meta["shortest_path"] == ["A", "B", "C", "D"]
    assert meta["shortest_path_length"] == 3
    assert set(meta["shortest_path"]).issubset(set(subG.nodes()))


def test_extract_subgraph_missing_nodes():
    """Verify error raised when query nodes do not exist."""
    G = nx.DiGraph()
    G.add_edge("A", "B")
    with pytest.raises(KeyError):
        extract_subgraph(G, "A", "NON_EXISTENT")


def test_prepare_graph_batch():
    """Verify tensor shape and device placement from prepare_graph_batch."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "B"), ("B", "C")])
    batch = prepare_graph_batch(G, "A", "C", device=torch.device("cpu"), target_nodes=48, target_edges=96)

    assert batch["node_ids"].shape == (1, 48)
    assert batch["indicators"].shape == (1, 48, 3)
    assert batch["adj"].shape == (1, 48, 48)
    assert batch["edge_src"].shape == (1, 96)
    assert batch["edge_dst"].shape == (1, 96)
    assert batch["src_idx"].item() == 0
    assert batch["dst_idx"].item() == 1
