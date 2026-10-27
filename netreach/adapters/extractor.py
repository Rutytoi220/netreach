"""
Automated Sub-Graph Extractor and Tensor Converter for NetReach.

Extracts the relevant K-hop lateral movement corridor between query nodes (source, target)
from massive enterprise security graphs (e.g., Active Directory / BloodHound) and formats
it into aligned tensor batches for ChallengerLatentNet with distribution density matching.
"""

from typing import Tuple, Dict, Any, Optional, Set
import random
import networkx as nx
import torch


def extract_subgraph(
    G: nx.DiGraph,
    source: str,
    target: str,
    max_nodes: int = 48,
    max_edges: int = 96,
    k_hops: int = 20,
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Extract the relevant attack corridor between source and target nodes.

    Args:
        G: Large directed network graph.
        source: Starting entity (e.g. compromised workstation/user).
        target: Destination entity (e.g. Domain Controller, high-value asset).
        max_nodes: Target node budget (default: 48).
        max_edges: Target edge budget (default: 96).
        k_hops: Max search depth for bidirectional expansion.

    Returns:
        subG: Extracted NetworkX DiGraph.
        meta: Diagnostic dictionary (e.g., ground truth shortest path, distance).
    """
    if source not in G:
        raise KeyError(f"Source node '{source}' not found in graph.")
    if target not in G:
        raise KeyError(f"Target node '{target}' not found in graph.")

    # 1. Bidirectional BFS expansion
    fwd_dist = nx.single_source_shortest_path_length(G, source, cutoff=k_hops)
    rev_G = G.reverse(copy=False)
    bwd_dist = nx.single_source_shortest_path_length(rev_G, target, cutoff=k_hops)

    # 2. Attack corridor: intersection of forward reachable and backward reaching nodes
    corridor_nodes: Set[str] = set(fwd_dist.keys()) & set(bwd_dist.keys())

    has_true_path = len(corridor_nodes) > 0 and target in fwd_dist
    shortest_path_list = []
    shortest_path_len = None

    if has_true_path:
        try:
            shortest_path_list = nx.shortest_path(G, source, target)
            shortest_path_len = len(shortest_path_list) - 1
        except nx.NetworkXNoPath:
            has_true_path = False

    if not corridor_nodes:
        # No path exists: sample local neighborhoods around source and target
        corridor_nodes = {source, target}
        for u in sorted(fwd_dist.keys(), key=lambda n: fwd_dist[n]):
            if len(corridor_nodes) >= max_nodes // 2:
                break
            corridor_nodes.add(u)
        for v in sorted(bwd_dist.keys(), key=lambda n: bwd_dist[n]):
            if len(corridor_nodes) >= max_nodes:
                break
            corridor_nodes.add(v)

    # 3. Cap at max_nodes if corridor is large
    if len(corridor_nodes) > max_nodes:
        sorted_nodes = sorted(
            corridor_nodes,
            key=lambda n: (fwd_dist.get(n, 999) + bwd_dist.get(n, 999), fwd_dist.get(n, 999))
        )
        selected_nodes = set(sorted_nodes[:max_nodes])
        selected_nodes.add(source)
        selected_nodes.add(target)
        if len(selected_nodes) > max_nodes:
            for n in reversed(sorted_nodes):
                if n not in (source, target) and len(selected_nodes) > max_nodes:
                    selected_nodes.remove(n)
        corridor_nodes = selected_nodes

    # 4. Extract induced subgraph
    subG = G.subgraph(corridor_nodes).copy()

    meta = {
        "source": source,
        "target": target,
        "has_ground_truth_path": has_true_path,
        "shortest_path": shortest_path_list,
        "shortest_path_length": shortest_path_len,
        "extracted_nodes": subG.number_of_nodes(),
        "extracted_edges": subG.number_of_edges(),
    }
    return subG, meta


def prepare_graph_batch(
    subG: nx.DiGraph,
    source: str,
    target: str,
    device: torch.device,
    target_nodes: int = 48,
    target_edges: int = 96,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Format an extracted NetworkX sub-graph into batched tensors aligned with ChallengerLatentNet.
    Uses density-matched distractor augmentation to preserve invariant message passing.
    """
    sub_nodes = list(subG.nodes())
    if source not in sub_nodes or target not in sub_nodes:
        raise ValueError("Source or target missing from extracted subgraph.")

    # Source at index 0, target at index 1
    ordered_nodes = [source, target] + [n for n in sub_nodes if n not in (source, target)]
    distractor_nodes = [f"__distractor_{j}__" for j in range(target_nodes - len(ordered_nodes))]
    all_nodes = ordered_nodes + distractor_nodes

    fullG = nx.DiGraph()
    fullG.add_nodes_from(all_nodes)
    for u, v in subG.edges():
        fullG.add_edge(u, v)

    # Detect current shortest path length if any
    try:
        H_curr = nx.shortest_path_length(fullG, source, target)
    except nx.NetworkXNoPath:
        H_curr = None

    # Density matching: augment with candidate edges without creating illegal shortcuts
    candidates = [
        (u, v)
        for u in all_nodes
        for v in all_nodes
        if u != v and not fullG.has_edge(u, v)
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)

    for u, v in candidates:
        if fullG.number_of_edges() >= target_edges:
            break
        fullG.add_edge(u, v)
        if H_curr is not None:
            try:
                spl = nx.shortest_path_length(fullG, source, target)
                if spl < H_curr:
                    fullG.remove_edge(u, v)
            except nx.NetworkXNoPath:
                pass
        else:
            if nx.has_path(fullG, source, target):
                fullG.remove_edge(u, v)

    node_map = {n: i for i, n in enumerate(all_nodes)}
    N = target_nodes
    M = target_edges
    B = 1

    node_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0)
    indicators = torch.zeros(B, N, 3, dtype=torch.float32, device=device)
    indicators[0, :, 2] = 1.0
    indicators[0, 0] = torch.tensor([1.0, 0.0, 0.0], device=device)  # source
    indicators[0, 1] = torch.tensor([0.0, 1.0, 0.0], device=device)  # target

    adj = torch.zeros(B, N, N, dtype=torch.float32, device=device)
    edge_src = torch.zeros(B, M, dtype=torch.long, device=device)
    edge_dst = torch.zeros(B, M, dtype=torch.long, device=device)

    for k, (u, v) in enumerate(fullG.edges()):
        if k < M:
            adj[0, node_map[u], node_map[v]] = 1.0
            edge_src[0, k] = node_map[u]
            edge_dst[0, k] = node_map[v]

    # If still below target_edges, pad remaining with self-loops on distractor nodes
    for k in range(fullG.number_of_edges(), M):
        d_idx = len(ordered_nodes) + (k % len(distractor_nodes)) if distractor_nodes else (N - 1)
        adj[0, d_idx, d_idx] = 1.0
        edge_src[0, k] = d_idx
        edge_dst[0, k] = d_idx

    return {
        "node_ids": node_ids,
        "indicators": indicators,
        "adj": adj,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "src_idx": torch.tensor([0], dtype=torch.long, device=device),
        "dst_idx": torch.tensor([1], dtype=torch.long, device=device),
    }
