"""
Unit tests for ChallengerLatentNetV4: Momentum Wavefront Recurrence,
Collision Peak Halting, and Neural Path Tracing.
"""

import pytest
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from netreach.model_v4 import ChallengerLatentNetV4, RELATION_TYPES, RELATION_TO_ID
from data.generator import ReachabilityDataset, collate_graph_batch
from train import compute_multitask_loss_v4


def test_v4_parameter_budget():
    """Verify ChallengerLatentNetV4 strictly complies with 35.0M - 45.0M parameters."""
    model = ChallengerLatentNetV4(
        num_nodes=256,
        node_embed_dim=256,
        d_model=640,
        hidden_dim=1280,
        nhead=8,
        rel_dim=128,
        num_relations=8,
        max_steps=22,
    )
    count = model.count_parameters()
    assert 35_000_000 <= count <= 45_000_000, f"Expected params in [35M, 45M], got {count}"
    assert count == 38_041_604


def test_v4_relation_types():
    """Verify the 6 defined directed relation types."""
    expected_relations = [
        "AdminTo",
        "MemberOf",
        "HasSession",
        "AllowedToDelegate",
        "OpenPort",
        "SubnetRoute",
    ]
    assert RELATION_TYPES == expected_relations
    assert len(RELATION_TO_ID) == 6


def test_v4_forward_collision_peak_halting():
    """Verify forward pass with Collision Peak Halting on CPU."""
    dataset = ReachabilityDataset(
        hop_lengths=[6, 12], num_samples=4, num_nodes=256, target_edges=768, seed=42
    )
    batch = collate_graph_batch([dataset[0], dataset[1]])

    model = ChallengerLatentNetV4(num_nodes=256, max_steps=12)
    model.eval()

    with torch.no_grad():
        out = model(batch, steps=None, return_latents=True)

    assert "logit" in out
    assert "hop" in out
    assert "pointer_logits" in out
    assert "steps_taken" in out
    assert "t_hit" in out
    assert "ponder_cost" in out
    assert "collision_map" in out

    B = 2
    assert out["logit"].shape == (B,)
    assert out["hop"].shape == (B,)
    assert out["pointer_logits"].shape == (B, 256, 256)
    assert out["collision_map"].shape == (B, 256)
    assert out["steps_taken"].shape == (B,)

    assert (out["steps_taken"] >= 1.0).all()
    assert (out["steps_taken"] <= 12.0).all()
    assert out["ponder_cost"] >= 0.0


def test_v4_fixed_steps():
    """Verify forward pass with fixed step unrolling."""
    dataset = ReachabilityDataset(
        hop_lengths=[6, 12], num_samples=4, num_nodes=256, target_edges=768, seed=42
    )
    batch = collate_graph_batch([dataset[0], dataset[1]])

    model = ChallengerLatentNetV4(num_nodes=256, max_steps=22)
    model.eval()

    with torch.no_grad():
        out_6 = model(batch, steps=6, return_latents=True)
        out_14 = model(batch, steps=14, return_latents=True)

    assert out_6["logit"].shape == (2,)
    assert out_6["hop"].shape == (2,)
    assert out_6["pointer_logits"].shape == (2, 256, 256)
    assert float(out_6["steps_taken"].mean()) == 6.0

    assert out_14["logit"].shape == (2,)
    assert out_14["hop"].shape == (2,)
    assert out_14["pointer_logits"].shape == (2, 256, 256)
    assert float(out_14["steps_taken"].mean()) == 14.0


def test_v4_multitask_gradients():
    """Verify multi-task loss computation and gradient backpropagation."""
    dataset = ReachabilityDataset(
        hop_lengths=[6, 12], num_samples=4, num_nodes=256, target_edges=768, seed=42
    )
    batch = collate_graph_batch([dataset[0], dataset[1]])

    model = ChallengerLatentNetV4(num_nodes=256, max_steps=6)
    model.train()

    criterion_bce = nn.BCEWithLogitsLoss()
    out = model(batch, steps=3)
    loss, metrics = compute_multitask_loss_v4(out, batch, criterion_bce)

    assert loss > 0.0
    assert "loss_bce" in metrics
    assert "loss_hop" in metrics
    assert "loss_pointer" in metrics
    assert "ponder_cost" in metrics

    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and "halt_head" not in name:
            assert param.grad is not None, f"Parameter {name} has None gradient"


def test_v4_neural_path_tracing():
    """Verify trace_attack_path returns valid attack path dictionary and NetworkX subgraph."""
    dataset = ReachabilityDataset(
        hop_lengths=[6], num_samples=4, num_nodes=256, target_edges=768, seed=42
    )
    # Find reachable sample
    reachable_idx = None
    for idx in range(len(dataset)):
        if dataset[idx]["label"] == 1.0:
            reachable_idx = idx
            break

    assert reachable_idx is not None, "Need at least 1 reachable sample"
    batch = collate_graph_batch([dataset[reachable_idx]])

    model = ChallengerLatentNetV4(num_nodes=256, max_steps=10)
    model.eval()

    names = [f"host-{i:03d}.corp.internal" for i in range(256)]
    res = model.trace_attack_path(batch, sample_idx=0, node_names=names, max_steps=10)

    assert "reachable" in res
    assert "bridge_node" in res
    assert "bridge_node_name" in res
    assert "path" in res
    assert "path_names" in res
    assert "path_length" in res
    assert "is_valid_walk" in res
    assert "subgraph" in res
    assert isinstance(res["subgraph"], nx.DiGraph)
    assert isinstance(res["path"], list)
