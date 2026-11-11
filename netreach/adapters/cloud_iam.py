"""
Hybrid Cloud & Multi-Tenant IAM Ingestion Adapter for NetReach v0.7.0.

Parses AWS IAM, Microsoft Entra ID (Azure AD), Google Cloud IAM, and hybrid identity
graphs, projecting cloud privilege escalation primitives into NetReach's 6 canonical
relational categories:
- AdminTo
- MemberOf
- HasSession
- AllowedToDelegate
- OpenPort
- SubnetRoute
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import networkx as nx

from netreach.model_v4 import RELATION_TYPES, RELATION_TO_ID

# Mapping of AWS IAM, Azure / Entra ID, and GCP IAM relation types to canonical NetReach categories
CLOUD_RELATION_MAP: Dict[str, str] = {
    # 1. AdminTo: Direct administrative control, policy attachments, role assignments, app ownership
    "attachedpolicy": "AdminTo",
    "attachuserpolicy": "AdminTo",
    "attachrolepolicy": "AdminTo",
    "attachgrouppolicy": "AdminTo",
    "putuserpolicy": "AdminTo",
    "putrolepolicy": "AdminTo",
    "administratoraccess": "AdminTo",
    "iamfullaccess": "AdminTo",
    "azroleassignment": "AdminTo",
    "azappadmin": "AdminTo",
    "azowns": "AdminTo",
    "azglobaladmin": "AdminTo",
    "azprivilegedroleadmin": "AdminTo",
    "azkeyvaultaccess": "AdminTo",
    "azresetpassword": "AdminTo",
    "owner": "AdminTo",
    "contributor": "AdminTo",
    "useraccessadministrator": "AdminTo",
    "gcpowner": "AdminTo",
    "gcpeditor": "AdminTo",

    # 2. MemberOf: Group memberships, IAM group attachments, Entra role member additions
    "azaddmembers": "MemberOf",
    "azmemberof": "MemberOf",
    "memberof": "MemberOf",
    "groupmember": "MemberOf",
    "groupmembership": "MemberOf",
    "addmember": "MemberOf",

    # 3. HasSession: Cached tokens, Instance Profiles, Cloud Shell sessions, STS credentials
    "hassession": "HasSession",
    "instanceprofile": "HasSession",
    "hasinstanceprofile": "HasSession",
    "assumedrolesession": "HasSession",
    "tokensession": "HasSession",
    "ec2credential": "HasSession",

    # 4. AllowedToDelegate: Role assumption (STS AssumeRole), PassRole, SAML/OIDC Federation, Entra Sync
    "assumerole": "AllowedToDelegate",
    "assumerolewithsaml": "AllowedToDelegate",
    "assumerolewithwebidentity": "AllowedToDelegate",
    "passrole": "AllowedToDelegate",
    "federateto": "AllowedToDelegate",
    "federatedto": "AllowedToDelegate",
    "samlfederation": "AllowedToDelegate",
    "oidcfederation": "AllowedToDelegate",
    "dirsync": "AllowedToDelegate",
    "entrasync": "AllowedToDelegate",
    "allowedtodelegate": "AllowedToDelegate",

    # 5. OpenPort: Cloud Security Groups, Network Security Group (NSG) rules, Firewall Allow
    "openport": "OpenPort",
    "securitygroupingress": "OpenPort",
    "nsginbound": "OpenPort",
    "firewallallow": "OpenPort",
    "publicip": "OpenPort",

    # 6. SubnetRoute: VPC Peering, Transit Gateway, Azure VNet Peering, DirectConnect, ExpressRoute
    "subnetroute": "SubnetRoute",
    "vpcpeering": "SubnetRoute",
    "vnetpeering": "SubnetRoute",
    "transitgateway": "SubnetRoute",
    "privatelink": "SubnetRoute",
    "expressroute": "SubnetRoute",
    "directconnect": "SubnetRoute",
}


def normalize_cloud_relation(raw_relation: str) -> str:
    """
    Normalize any Cloud IAM / Hybrid identity relation into one of the 6 canonical NetReach categories.
    Defaults to 'AdminTo' if unrecognized.
    """
    clean_rel = re.sub(r"[^a-zA-Z0-9]", "", raw_relation).lower()
    return CLOUD_RELATION_MAP.get(clean_rel, "AdminTo")


def parse_cloud_iam_data(data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> nx.DiGraph:
    """
    Parse cloud IAM relationships from JSON into a NetworkX DiGraph.
    
    Supports:
    - Unified multi-cloud graphs: {"nodes": [...], "edges": [...]} or {"data": [...]}
    - AWS IAM credential and policy dump structures
    - Azure / Microsoft Entra ID relationship dumps
    - Explicit edge lists: [{"source": "u", "target": "v", "relation": "AssumeRole", "cloud": "aws"}]
    """
    G = nx.DiGraph()

    # Case 1: List of edge dictionaries
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            src = str(item.get("source") or item.get("src") or item.get("from") or item.get("Principal", "")).strip()
            dst = str(item.get("target") or item.get("dst") or item.get("to") or item.get("Resource", "")).strip()
            raw_rel = str(item.get("relation") or item.get("type") or item.get("Action", "AdminTo")).strip()

            if not src or not dst:
                continue

            canon_rel = normalize_cloud_relation(raw_rel)
            cloud_provider = item.get("cloud") or item.get("provider", "hybrid")

            G.add_edge(
                src,
                dst,
                relation=canon_rel,
                cloud_relation=raw_rel,
                cloud=cloud_provider,
                relation_id=RELATION_TO_ID.get(canon_rel, 0),
            )
        return G

    # Case 2: Dict structure
    if isinstance(data, dict):
        # Format 2A: {"nodes": [...], "edges": [...]}
        if "edges" in data and isinstance(data["edges"], list):
            # Parse nodes first if present
            if "nodes" in data and isinstance(data["nodes"], list):
                for node_obj in data["nodes"]:
                    if isinstance(node_obj, dict):
                        n_id = str(node_obj.get("id") or node_obj.get("name", "")).strip()
                        if n_id:
                            G.add_node(n_id, **{k: v for k, v in node_obj.items() if k not in ("id", "name")})

            for edge_obj in data["edges"]:
                if isinstance(edge_obj, dict):
                    src = str(edge_obj.get("source") or edge_obj.get("src") or edge_obj.get("from", "")).strip()
                    dst = str(edge_obj.get("target") or edge_obj.get("dst") or edge_obj.get("to", "")).strip()
                    raw_rel = str(edge_obj.get("relation") or edge_obj.get("type", "AdminTo")).strip()
                    if src and dst:
                        canon_rel = normalize_cloud_relation(raw_rel)
                        G.add_edge(
                            src,
                            dst,
                            relation=canon_rel,
                            cloud_relation=raw_rel,
                            cloud=edge_obj.get("cloud", "cloud"),
                            relation_id=RELATION_TO_ID.get(canon_rel, 0),
                        )
            return G

        # Format 2B: AWS IAM Export {"Roles": [...], "Users": [...], "Policies": [...]}
        if any(k in data for k in ["Roles", "Users", "Policies", "RoleAssignments"]):
            # Process AWS IAM Roles and Trust Relationships (AssumeRole)
            for role in data.get("Roles", []):
                role_arn = role.get("Arn") or role.get("RoleName", "")
                if not role_arn:
                    continue
                # AssumeRolePolicyDocument (Trust Policy)
                trust_doc = role.get("AssumeRolePolicyDocument", {})
                for stmt in trust_doc.get("Statement", []):
                    effect = stmt.get("Effect", "")
                    action = stmt.get("Action", "")
                    actions = [action] if isinstance(action, str) else action
                    if effect == "Allow" and any("assumerole" in a.lower() for a in actions):
                        principal = stmt.get("Principal", {})
                        trusted_principals = []
                        if isinstance(principal, str):
                            trusted_principals.append(principal)
                        elif isinstance(principal, dict):
                            for p_type, p_val in principal.items():
                                if isinstance(p_val, list):
                                    trusted_principals.extend(p_val)
                                else:
                                    trusted_principals.append(p_val)

                        for tp in trusted_principals:
                            tp_str = str(tp).strip()
                            if tp_str:
                                G.add_edge(
                                    tp_str,
                                    role_arn,
                                    relation="AllowedToDelegate",
                                    cloud_relation="AssumeRole",
                                    cloud="aws",
                                    relation_id=RELATION_TO_ID["AllowedToDelegate"],
                                )

                # Attached Policies
                for pol in role.get("AttachedManagedPolicies", []):
                    pol_arn = pol.get("PolicyArn") or pol.get("PolicyName", "")
                    if pol_arn:
                        G.add_edge(
                            role_arn,
                            pol_arn,
                            relation="AdminTo",
                            cloud_relation="AttachedPolicy",
                            cloud="aws",
                            relation_id=RELATION_TO_ID["AdminTo"],
                        )

            # Process Azure / Entra Role Assignments
            for ra in data.get("RoleAssignments", []):
                principal = ra.get("PrincipalName") or ra.get("PrincipalId", "")
                role_def = ra.get("RoleDefinitionName") or ra.get("RoleName", "Contributor")
                scope = ra.get("Scope") or ra.get("ResourceName", "")
                if principal and scope:
                    G.add_edge(
                        principal,
                        scope,
                        relation="AdminTo",
                        cloud_relation=f"AZRoleAssignment:{role_def}",
                        cloud="azure",
                        relation_id=RELATION_TO_ID["AdminTo"],
                    )

            if G.number_of_nodes() > 0 or G.number_of_edges() > 0:
                return G

        # Format 2C: Wrapped {"data": [...]}
        if "data" in data and isinstance(data["data"], list):
            return parse_cloud_iam_data(data["data"])

    return G


def load_cloud_iam_graph(file_path_or_dict: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> nx.DiGraph:
    """
    Load a cloud / hybrid IAM graph from a file path or parsed JSON data.
    """
    if isinstance(file_path_or_dict, (dict, list)):
        return parse_cloud_iam_data(file_path_or_dict)

    if not os.path.exists(file_path_or_dict):
        raise FileNotFoundError(f"Cloud IAM file not found: {file_path_or_dict}")

    with open(file_path_or_dict, "r", encoding="utf-8") as f:
        data = json.load(f)

    return parse_cloud_iam_data(data)
