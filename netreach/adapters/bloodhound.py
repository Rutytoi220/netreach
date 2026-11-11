"""
BloodHound and SharpHound CE Ingestion Adapter for NetReach.

Parses BloodHound CE JSON export files (computers, users, groups, domains,
or unified multi-entity JSON graphs) and maps Active Directory ACE/ACL relationships
into NetReach's 6 relational edge categories.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import networkx as nx

from netreach.model_v4 import RELATION_TYPES, RELATION_TO_ID

# Canonical mapping of Active Directory ACE / ACL / Group relations to NetReach categories
AD_RELATION_MAP: Dict[str, str] = {
    # 1. MemberOf
    "memberof": "MemberOf",
    "primarygroup": "MemberOf",
    "contains": "MemberOf",
    "member": "MemberOf",

    # 2. AdminTo (Privilege execution, ACL takeover, DCSync, RDP, DCOM, Full Control)
    "adminto": "AdminTo",
    "canrdp": "AdminTo",
    "canpsremote": "AdminTo",
    "executedcom": "AdminTo",
    "genericall": "AdminTo",
    "genericwrite": "AdminTo",
    "writedacl": "AdminTo",
    "writeowner": "AdminTo",
    "owns": "AdminTo",
    "allassociatedrights": "AdminTo",
    "dcsync": "AdminTo",
    "getchanges": "AdminTo",
    "getchangesall": "AdminTo",
    "addmember": "AdminTo",
    "addself": "AdminTo",
    "forcechangepassword": "AdminTo",
    "allauthorities": "AdminTo",
    "allextendedrights": "AdminTo",
    "writespn": "AdminTo",
    "addkeycredentiallink": "AdminTo",
    "sync": "AdminTo",

    # 3. HasSession
    "hassession": "HasSession",
    "session": "HasSession",
    "logon": "HasSession",

    # 4. AllowedToDelegate
    "allowedtodelegate": "AllowedToDelegate",
    "allowedtoact": "AllowedToDelegate",
    "constraineddelegation": "AllowedToDelegate",
    "unconstraineddelegation": "AllowedToDelegate",
    "rbcd": "AllowedToDelegate",

    # 5. OpenPort
    "openport": "OpenPort",
    "port": "OpenPort",
    "connectsto": "OpenPort",
    "service": "OpenPort",

    # 6. SubnetRoute
    "subnetroute": "SubnetRoute",
    "routes": "SubnetRoute",
    "route": "SubnetRoute",
    "subnethop": "SubnetRoute",
    "network": "SubnetRoute",
    "gateway": "SubnetRoute",
}


def normalize_ad_relation(rel_name: str) -> str:
    """
    Map an Active Directory / BloodHound relationship string to one of the 6 canonical NetReach categories.
    """
    clean = re.sub(r"[^a-zA-Z0-9]", "", rel_name).lower()
    if clean in AD_RELATION_MAP:
        return AD_RELATION_MAP[clean]

    # Substring heuristics
    if any(k in clean for k in ["admin", "write", "own", "sync", "exec", "rdp", "control", "dcom", "pass"]):
        return "AdminTo"
    if any(k in clean for k in ["member", "group", "contain"]):
        return "MemberOf"
    if any(k in clean for k in ["session", "logon"]):
        return "HasSession"
    if any(k in clean for k in ["delegate", "act", "rbcd"]):
        return "AllowedToDelegate"
    if any(k in clean for k in ["route", "subnet", "net", "vlan", "gateway"]):
        return "SubnetRoute"
    if any(k in clean for k in ["port", "service", "socket"]):
        return "OpenPort"

    # Default fallback
    return "AdminTo"


def parse_bloodhound_data(data: Union[Dict[str, Any], List[Any]]) -> nx.DiGraph:
    """
    Parse a BloodHound JSON dictionary or list into a NetworkX DiGraph with canonical relations.
    """
    G = nx.DiGraph()

    # Case 1: Simple edge list [{'source': ..., 'target': ..., 'relation': ...}]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                src = item.get("source") or item.get("src") or item.get("from")
                dst = item.get("target") or item.get("dst") or item.get("to")
                raw_rel = item.get("relation") or item.get("rel") or item.get("type") or "AdminTo"
                if src and dst:
                    canon_rel = normalize_ad_relation(raw_rel)
                    G.add_edge(str(src), str(dst), relation=canon_rel, ad_relation=str(raw_rel))
        return G

    # Case 2: Unified graph with 'nodes' and 'edges'
    if "edges" in data or "links" in data:
        edges_list = data.get("edges") or data.get("links") or []
        for item in edges_list:
            src = item.get("source") or item.get("src") or item.get("from")
            dst = item.get("target") or item.get("dst") or item.get("to")
            raw_rel = item.get("relation") or item.get("rel") or item.get("type") or "AdminTo"
            if src and dst:
                canon_rel = normalize_ad_relation(raw_rel)
                G.add_edge(str(src), str(dst), relation=canon_rel, ad_relation=str(raw_rel))
        if "nodes" in data:
            for node_item in data["nodes"]:
                nid = node_item.get("id") or node_item.get("name")
                if nid:
                    props = {k: v for k, v in node_item.items() if k not in ("id", "name")}
                    if str(nid) in G:
                        G.nodes[str(nid)].update(props)
                    else:
                        G.add_node(str(nid), **props)
        return G

    # Case 3: BloodHound CE native JSON export with 'data' array of entity objects
    entities = data.get("data")
    if entities and isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            props = entity.get("Properties", {})
            name = props.get("name") or entity.get("name") or props.get("distinguishedname")
            if not name:
                continue
            name = str(name).upper()
            G.add_node(name, **props)

            # Ingest Members (MemberOf relation: Member -> Group)
            members = entity.get("Members", [])
            for m in members:
                m_name = m.get("name") or m.get("ObjectIdentifier")
                if m_name:
                    m_name = str(m_name).upper()
                    G.add_edge(m_name, name, relation="MemberOf", ad_relation="MemberOf")

            # Ingest PrimaryGroup (PrimaryGroup relation: User -> Group)
            pri_group = props.get("primarygroupid") or entity.get("PrimaryGroupSID")
            if pri_group:
                pri_name = str(pri_group).upper()
                G.add_edge(name, pri_name, relation="MemberOf", ad_relation="PrimaryGroup")

            # Ingest Sessions (Computer -> User / User has session on Computer)
            sessions = entity.get("Sessions", []) or entity.get("Session", [])
            for sess in sessions:
                user_id = sess.get("UserName") or sess.get("UserId")
                comp_id = sess.get("ComputerName") or sess.get("ComputerId")
                if user_id and comp_id:
                    u_str = str(user_id).upper()
                    c_str = str(comp_id).upper()
                    G.add_edge(c_str, u_str, relation="HasSession", ad_relation="HasSession")

            # Ingest AllowedToDelegate (Computer/User -> Target)
            delegates = entity.get("AllowedToDelegate", [])
            for target in delegates:
                tgt_name = target.get("name") or target.get("ObjectIdentifier")
                if tgt_name:
                    tgt_str = str(tgt_name).upper()
                    G.add_edge(name, tgt_str, relation="AllowedToDelegate", ad_relation="AllowedToDelegate")

            # Ingest AllowedToAct
            act_delegates = entity.get("AllowedToAct", [])
            for target in act_delegates:
                tgt_name = target.get("name") or target.get("ObjectIdentifier")
                if tgt_name:
                    tgt_str = str(tgt_name).upper()
                    G.add_edge(tgt_str, name, relation="AllowedToDelegate", ad_relation="AllowedToAct")

            # Ingest Aces: Principal -> This Object
            aces = entity.get("Aces", [])
            for ace in aces:
                principal = ace.get("PrincipalName") or ace.get("PrincipalObjectIdentifier")
                right = ace.get("RightName") or "AdminTo"
                if principal:
                    p_str = str(principal).upper()
                    canon = normalize_ad_relation(right)
                    G.add_edge(p_str, name, relation=canon, ad_relation=str(right))

    # Case 4: Multi-type dictionary {"users": [...], "computers": [...], "groups": [...]}
    for section_key in ("users", "computers", "groups", "domains", "ous", "gpos"):
        section = data.get(section_key)
        if isinstance(section, list):
            subG = parse_bloodhound_data({"data": section})
            G = nx.compose(G, subG)

    return G


def load_bloodhound_graph(path_or_obj: Union[str, Dict[str, Any], List[Any]]) -> nx.DiGraph:
    """
    Load a BloodHound export from a JSON file path or Python object into a NetworkX DiGraph.
    """
    if isinstance(path_or_obj, str):
        if not os.path.exists(path_or_obj):
            raise FileNotFoundError(f"BloodHound JSON file not found: {path_or_obj}")
        with open(path_or_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
        return parse_bloodhound_data(data)

    return parse_bloodhound_data(path_or_obj)
