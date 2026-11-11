"""
Unit and Integration Tests for Hybrid Cloud & Multi-Tenant IAM Ingestion Adapter.
"""

import os
import pytest
import networkx as nx

from netreach.adapters.cloud_iam import (
    normalize_cloud_relation,
    parse_cloud_iam_data,
    load_cloud_iam_graph,
    CLOUD_RELATION_MAP,
)
from netreach.adapters.graph_loader import load_graph
from netreach.engine import NetReachEngine
from netreach.remediation import generate_remediation_commands


def test_normalize_cloud_relations():
    """Verify projection of cloud privilege escalation primitives to canonical categories."""
    # AWS IAM
    assert normalize_cloud_relation("AssumeRole") == "AllowedToDelegate"
    assert normalize_cloud_relation("PassRole") == "AllowedToDelegate"
    assert normalize_cloud_relation("AttachedPolicy") == "AdminTo"
    assert normalize_cloud_relation("FederateTo") == "AllowedToDelegate"
    assert normalize_cloud_relation("AdministratorAccess") == "AdminTo"
    assert normalize_cloud_relation("HasInstanceProfile") == "HasSession"

    # Azure / Microsoft Entra ID
    assert normalize_cloud_relation("AZRoleAssignment") == "AdminTo"
    assert normalize_cloud_relation("AZAppAdmin") == "AdminTo"
    assert normalize_cloud_relation("AZOwns") == "AdminTo"
    assert normalize_cloud_relation("AZAddMembers") == "MemberOf"
    assert normalize_cloud_relation("AZGlobalAdmin") == "AdminTo"
    assert normalize_cloud_relation("EntraSync") == "AllowedToDelegate"

    # Network / VPC
    assert normalize_cloud_relation("VpcPeering") == "SubnetRoute"
    assert normalize_cloud_relation("SecurityGroupIngress") == "OpenPort"


def test_parse_cloud_iam_data_edge_list():
    """Verify parsing list of cloud IAM edges."""
    data = [
        {
            "source": "arn:aws:iam::123:user/bob",
            "target": "arn:aws:iam::123:role/devops",
            "relation": "AssumeRole",
            "cloud": "aws",
        },
        {
            "source": "arn:aws:iam::123:role/devops",
            "target": "arn:aws:iam::aws:policy/AdministratorAccess",
            "relation": "AttachedPolicy",
            "cloud": "aws",
        },
    ]

    G = parse_cloud_iam_data(data)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2

    edge1 = G["arn:aws:iam::123:user/bob"]["arn:aws:iam::123:role/devops"]
    assert edge1["relation"] == "AllowedToDelegate"
    assert edge1["cloud_relation"] == "AssumeRole"
    assert edge1["cloud"] == "aws"


def test_load_hybrid_cloud_iam_graph_file():
    """Verify loading examples/hybrid_cloud_iam.json via load_graph."""
    graph_path = "examples/hybrid_cloud_iam.json"
    assert os.path.exists(graph_path)

    G = load_graph(graph_path)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 10
    assert G.number_of_edges() == 9

    # Verify edge presence
    assert G.has_edge("WORKSTATION-01.CORP.LOCAL", "DEV-USER@CORP.LOCAL")
    assert G.has_edge("Entra-SSO-AWS-AppRegistration", "arn:aws:iam::123456789012:saml-provider/EntraID")
    assert G.has_edge("arn:aws:iam::123456789012:role/AWS-Production-Cluster-Admin", "arn:aws:iam::aws:policy/AdministratorAccess")


def test_cloud_remediation_commands():
    """Verify mitigation scripts generated for AWS IAM and Azure Entra ID relations."""
    # AWS AssumeRole
    res_assume = generate_remediation_commands(
        "arn:aws:iam::123:user/attacker",
        "arn:aws:iam::123:role/target-role",
        "AllowedToDelegate",
        "AssumeRole",
    )
    assert "aws iam update-assume-role-policy" in res_assume["powershell"]
    assert "AssumeRole" in res_assume["cypher"]

    # AWS AttachedPolicy
    res_pol = generate_remediation_commands(
        "arn:aws:iam::123:role/target-role",
        "arn:aws:iam::aws:policy/AdministratorAccess",
        "AdminTo",
        "AttachedPolicy",
    )
    assert "aws iam detach-role-policy" in res_pol["powershell"]

    # Azure AZRoleAssignment
    res_az = generate_remediation_commands(
        "alice@corp.onmicrosoft.com",
        "/subscriptions/sub1/resourceGroups/rg1",
        "AdminTo",
        "AZRoleAssignment",
    )
    assert "az role assignment delete" in res_az["powershell"]


def test_engine_audit_hybrid_cloud():
    """Verify end-to-end neural audit on hybrid cloud IAM path."""
    engine = NetReachEngine()
    graph_path = "examples/hybrid_cloud_iam.json"

    res = engine.audit(
        graph_source=graph_path,
        source="WORKSTATION-01.CORP.LOCAL",
        target="arn:aws:iam::aws:policy/AdministratorAccess",
        depths=(4, 8),
    )

    assert res.source == "WORKSTATION-01.CORP.LOCAL"
    assert res.target == "arn:aws:iam::aws:policy/AdministratorAccess"
    assert res.probabilities[8] > 0.70
    assert len(res.ground_truth_path) == 10
