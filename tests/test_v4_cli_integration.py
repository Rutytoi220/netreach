"""
Integration tests for NetReach v0.5.0 CLI and Engine with ChallengerLatentNetV4.
"""

import sys
import json
import subprocess
import pytest
import torch

from netreach.engine import NetReachEngine, AuditResult
from netreach.model_v4 import ChallengerLatentNetV4
from netreach.adapters.bloodhound import load_bloodhound_graph
from netreach.adapters.extractor import extract_subgraph, prepare_graph_batch


def test_engine_initialization_v4():
    engine = NetReachEngine(model_version="v4")
    assert isinstance(engine.model, ChallengerLatentNetV4)
    assert 35_000_000 <= engine.param_count <= 45_000_000


def test_engine_int8_quantization():
    engine = NetReachEngine(model_version="v4", use_int8=True)
    assert engine.use_int8 is True
    assert engine.device.type == "cpu"


def test_engine_audit_bloodhound():
    engine = NetReachEngine(model_version="v4")
    result = engine.audit(
        graph_source="examples/bloodhound_enterprise.json",
        source="WORKSTATION-01.CORP.LOCAL",
        target="DC01.CORP.LOCAL",
        depths=(4, 8),
    )
    assert isinstance(result, AuditResult)
    assert result.source == "WORKSTATION-01.CORP.LOCAL"
    assert result.target == "DC01.CORP.LOCAL"
    assert 4 in result.probabilities
    assert 8 in result.probabilities
    assert result.shortest_hop_count == 9
    assert len(result.ground_truth_path) == 10
    assert result.risk_level in ("HIGH", "CRITICAL")
    assert result.is_valid_walk is True
    assert len(result.path_relations) >= 8

    # Telemetry and bridge checks
    assert result.bridge_node is not None
    assert not result.bridge_node.startswith("__distractor_")
    assert result.act_steps >= 4.0
    assert result.peak_collision > 0.0


def test_cli_audit_json_output():
    cmd = [
        sys.executable,
        "-m",
        "netreach.cli",
        "audit",
        "--graph",
        "examples/bloodhound_enterprise.json",
        "--src",
        "WORKSTATION-01.CORP.LOCAL",
        "--dst",
        "DC01.CORP.LOCAL",
        "--depth",
        "4",
        "8",
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = json.loads(proc.stdout)
    assert out["source"] == "WORKSTATION-01.CORP.LOCAL"
    assert out["target"] == "DC01.CORP.LOCAL"
    assert "probabilities" in out
    assert "shortest_hop_count" in out
    assert "path_relations" in out
    assert "act_steps" in out
    assert "peak_collision" in out
    assert out["shortest_hop_count"] == 9

    # Bridge node must never be a distractor
    assert out["bridge_node"] is not None
    assert not str(out["bridge_node"]).startswith("__distractor_")
    assert out["act_steps"] >= 4.0
    assert out["peak_collision"] > 0.0


def test_cli_audit_int8_cpu_flag():
    cmd = [
        sys.executable,
        "-m",
        "netreach.cli",
        "audit",
        "--graph",
        "examples/bloodhound_enterprise.json",
        "--src",
        "WORKSTATION-01.CORP.LOCAL",
        "--dst",
        "DC01.CORP.LOCAL",
        "--depth",
        "4",
        "--int8",
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = json.loads(proc.stdout)
    assert out["source"] == "WORKSTATION-01.CORP.LOCAL"
    assert "4" in out["probabilities"]
    assert out["is_valid_walk"] is True
    assert out["bridge_node"] is not None
    assert not str(out["bridge_node"]).startswith("__distractor_")


def test_regression_bridge_node_strictly_valid_node():
    """Verify that collision bridge selection strictly excludes padding/distractors."""
    engine = NetReachEngine(model_version="v4")
    result = engine.audit(
        graph_source="examples/bloodhound_enterprise.json",
        source="WORKSTATION-01.CORP.LOCAL",
        target="DC01.CORP.LOCAL",
    )
    assert result.bridge_node is not None
    assert not result.bridge_node.startswith("__distractor_")

    valid_domain_assets = {
        "WORKSTATION-01.CORP.LOCAL",
        "BOB.DEV@CORP.LOCAL",
        "HELPDESK-SUPPORT@CORP.LOCAL",
        "SRV-FILES.CORP.LOCAL",
        "SVC-BACKUP@CORP.LOCAL",
        "TIER1-ADMINS@CORP.LOCAL",
        "JUMPBOX-01.CORP.LOCAL",
        "ALICE.ADMIN@CORP.LOCAL",
        "DOMAIN-ADMINS@CORP.LOCAL",
        "DC01.CORP.LOCAL",
    }
    assert result.bridge_node in valid_domain_assets


def test_regression_act_minimum_step_floor():
    """Verify that ACT never halts on t=1 when initial collision affinity is zero."""
    engine = NetReachEngine(model_version="v4")
    G = load_bloodhound_graph("examples/bloodhound_enterprise.json")
    subG, meta = extract_subgraph(G, "WORKSTATION-01.CORP.LOCAL", "DC01.CORP.LOCAL")
    batch = prepare_graph_batch(
        subG,
        "WORKSTATION-01.CORP.LOCAL",
        "DC01.CORP.LOCAL",
        target_nodes=256,
        target_edges=768,
        device=engine.device,
    )
    out = engine.model(batch, steps=None)
    steps_taken = out["steps_taken"].item()

    # Must take at least 4 recurrent passes unless early explicit hit
    assert steps_taken >= 4.0
    # Peak collision affinity must be positive / non-zero
    assert out["max_collision"].item() > 0.0
