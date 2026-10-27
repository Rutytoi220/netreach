"""
Unit tests for NetReachEngine core inference and model components.
"""

import pytest
import torch
import networkx as nx

from netreach.engine import NetReachEngine, AuditResult, compute_pairwise_cosine_similarity
from netreach.model import ChallengerLatentNet


def test_engine_init_cpu():
    """Verify NetReachEngine initializes with default weights on pure CPU."""
    engine = NetReachEngine(device="cpu")
    assert engine.device == torch.device("cpu")
    assert 400_000 <= engine.param_count <= 600_000
    assert engine.param_count == 501_793


def test_model_parameter_count():
    """Verify ChallengerLatentNet matches exact parameter budget."""
    model = ChallengerLatentNet(num_nodes=50)
    count = model.count_parameters()
    assert count == 501_793


def test_cosine_similarity_computation():
    """Verify pairwise cosine similarity calculation."""
    B, N, d = 2, 48, 160
    # Case 1: Random orthogonal vectors -> sim close to 0
    H_rand = torch.randn(B, N, d)
    sim_rand = compute_pairwise_cosine_similarity(H_rand)
    assert -0.2 <= sim_rand <= 0.2

    # Case 2: Identical vectors -> sim close to 1.0
    H_ident = torch.ones(B, N, d)
    sim_ident = compute_pairwise_cosine_similarity(H_ident)
    assert sim_ident > 0.99


def test_engine_audit_path():
    """Verify end-to-end audit on a synthetic linear path."""
    engine = NetReachEngine(device="cpu")

    G = nx.DiGraph()
    path = [f"asset_{i}" for i in range(8)]
    for i in range(len(path) - 1):
        G.add_edge(path[i], path[i + 1])

    result = engine.audit(G, source="asset_0", target="asset_7", depths=(5, 10, 15))
    assert isinstance(result, AuditResult)
    assert result.source == "asset_0"
    assert result.target == "asset_7"
    assert result.shortest_hop_count == 7
    assert len(result.ground_truth_path) == 8
    assert set(result.probabilities.keys()) == {5, 10, 15}
    for p in result.probabilities.values():
        assert 0.0 <= p <= 1.0
    assert result.risk_level in ("CRITICAL", "HIGH", "MEDIUM", "LOW / UNREACHABLE")
    assert result.inference_latency_ms > 0.0


def test_engine_audit_unreachable():
    """Verify audit on completely disconnected nodes."""
    engine = NetReachEngine(device="cpu")

    G = nx.DiGraph()
    G.add_edge("node_a", "node_b")
    G.add_edge("node_c", "node_d")

    result = engine.audit(G, source="node_a", target="node_d", depths=(5, 10))
    assert result.shortest_hop_count is None
    assert len(result.ground_truth_path) == 0
    assert result.risk_level in ("LOW / UNREACHABLE", "MEDIUM")


def test_engine_benchmark_cpu():
    """Verify automated benchmark runner executes without errors."""
    engine = NetReachEngine(device="cpu")
    bench = engine.benchmark(num_samples=4, hops=(3, 5), depths=(5, 10))
    assert bench["tested_samples"] == 4
    assert "mean_latency_ms" in bench
    assert "accuracy_summary" in bench
    assert bench["parameter_count"] == 501_793
