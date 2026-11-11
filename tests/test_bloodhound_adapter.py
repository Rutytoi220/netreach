"""
Unit tests for BloodHound / SharpHound CE data ingestion adapter.
"""

import os
import json
import pytest
import networkx as nx

from netreach.adapters.bloodhound import (
    load_bloodhound_graph,
    parse_bloodhound_data,
    normalize_ad_relation,
    AD_RELATION_MAP,
)


def test_normalize_ad_relation():
    # AdminTo group
    assert normalize_ad_relation("AdminTo") == "AdminTo"
    assert normalize_ad_relation("CanRDP") == "AdminTo"
    assert normalize_ad_relation("ExecuteDCOM") == "AdminTo"
    assert normalize_ad_relation("GenericAll") == "AdminTo"
    assert normalize_ad_relation("WriteDacl") == "AdminTo"
    assert normalize_ad_relation("DCSync") == "AdminTo"

    # MemberOf group
    assert normalize_ad_relation("MemberOf") == "MemberOf"
    assert normalize_ad_relation("PrimaryGroup") == "MemberOf"

    # HasSession group
    assert normalize_ad_relation("HasSession") == "HasSession"
    assert normalize_ad_relation("Session") == "HasSession"

    # AllowedToDelegate group
    assert normalize_ad_relation("AllowedToDelegate") == "AllowedToDelegate"
    assert normalize_ad_relation("AllowedToAct") == "AllowedToDelegate"
    assert normalize_ad_relation("RBCD") == "AllowedToDelegate"

    # OpenPort group
    assert normalize_ad_relation("OpenPort") == "OpenPort"
    assert normalize_ad_relation("ConnectsTo") == "OpenPort"

    # SubnetRoute group
    assert normalize_ad_relation("SubnetRoute") == "SubnetRoute"
    assert normalize_ad_relation("Network") == "SubnetRoute"


def test_parse_bloodhound_edge_list():
    raw_edges = [
        {"source": "PC-01.CORP", "target": "USER-BOB", "relation": "HasSession"},
        {"source": "USER-BOB", "target": "IT-ADMINS", "relation": "MemberOf"},
        {"source": "IT-ADMINS", "target": "DC-01.CORP", "relation": "GenericAll"},
    ]
    G = parse_bloodhound_data(raw_edges)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3
    assert G["IT-ADMINS"]["DC-01.CORP"]["relation"] == "AdminTo"
    assert G["IT-ADMINS"]["DC-01.CORP"]["ad_relation"] == "GenericAll"


def test_parse_bloodhound_unified_graph():
    unified_doc = {
        "nodes": [
            {"id": "USER-A", "type": "User"},
            {"id": "GROUP-A", "type": "Group"},
            {"id": "SRV-01", "type": "Computer"},
        ],
        "edges": [
            {"source": "USER-A", "target": "GROUP-A", "relation": "MemberOf"},
            {"source": "GROUP-A", "target": "SRV-01", "relation": "CanRDP"},
        ],
    }
    G = parse_bloodhound_data(unified_doc)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G["GROUP-A"]["SRV-01"]["relation"] == "AdminTo"
    assert G["GROUP-A"]["SRV-01"]["ad_relation"] == "CanRDP"


def test_parse_bloodhound_native_ce_data():
    ce_dump = {
        "data": [
            {
                "Properties": {"name": "WORKSTATION-01.CORP"},
                "Sessions": [{"ComputerName": "WORKSTATION-01.CORP", "UserName": "ALICE.USER"}],
            },
            {
                "Properties": {"name": "DOMAIN-ADMINS@CORP"},
                "Members": [{"name": "ALICE.USER"}],
            },
            {
                "Properties": {"name": "DC01.CORP"},
                "Aces": [
                    {"PrincipalName": "DOMAIN-ADMINS@CORP", "RightName": "DCSync"}
                ],
            },
        ]
    }
    G = parse_bloodhound_data(ce_dump)
    assert "WORKSTATION-01.CORP" in G
    assert "ALICE.USER" in G
    assert "DOMAIN-ADMINS@CORP" in G
    assert "DC01.CORP" in G

    assert nx.has_path(G, "WORKSTATION-01.CORP", "DC01.CORP")
    path = nx.shortest_path(G, "WORKSTATION-01.CORP", "DC01.CORP")
    assert path == ["WORKSTATION-01.CORP", "ALICE.USER", "DOMAIN-ADMINS@CORP", "DC01.CORP"]
    assert G["DOMAIN-ADMINS@CORP"]["DC01.CORP"]["relation"] == "AdminTo"
    assert G["DOMAIN-ADMINS@CORP"]["DC01.CORP"]["ad_relation"] == "DCSync"


def test_load_bloodhound_enterprise_file():
    example_path = "examples/bloodhound_enterprise.json"
    assert os.path.exists(example_path), f"Missing example file: {example_path}"

    G = load_bloodhound_graph(example_path)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() >= 12
    assert G.number_of_edges() >= 10

    # Path from workstation to domain controller
    src = "WORKSTATION-01.CORP.LOCAL"
    dst = "DC01.CORP.LOCAL"
    assert nx.has_path(G, src, dst)
    path = nx.shortest_path(G, src, dst)
    assert len(path) == 10  # 9 hops
    assert path[0] == src
    assert path[-1] == dst
