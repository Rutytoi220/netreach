"""
Built-in Synthetic Benchmark Graph Generator for NetReach.

Generates directed graphs with exact shortest-path controls for latency and accuracy benchmarks.
"""

from typing import List, Tuple, Dict, Optional
import random
import networkx as nx
import torch


def generate_benchmark_sample(
    H: int,
    reachable: bool,
    num_nodes: int = 48,
    target_edges: int = 96,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Tuple[int, int]], int, int, int, int]:
    """Generate a single benchmark graph with controlled shortest path H."""
    if rng is None:
        rng = random.Random()

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    if reachable:
        all_nodes = list(range(num_nodes))
        rng.shuffle(all_nodes)
        path = all_nodes[: H + 1]
        src, dst = path[0], path[-1]
        for i in range(H):
            G.add_edge(path[i], path[i + 1])

        candidates = [
            (u, v)
            for u in range(num_nodes)
            for v in range(num_nodes)
            if u != v and not G.has_edge(u, v)
        ]
        rng.shuffle(candidates)
        for u, v in candidates:
            if G.number_of_edges() >= target_edges:
                break
            G.add_edge(u, v)
            try:
                spl = nx.shortest_path_length(G, src, dst)
                if spl < H:
                    G.remove_edge(u, v)
            except nx.NetworkXNoPath:
                pass
        label = 1
    else:
        all_nodes = list(range(num_nodes))
        rng.shuffle(all_nodes)
        half = num_nodes // 2
        S1, S2 = all_nodes[:half], all_nodes[half:]

        h_s1 = min(H, len(S1) - 1)
        path1 = S1[: h_s1 + 1]
        src = path1[0]
        for i in range(h_s1):
            G.add_edge(path1[i], path1[i + 1])

        h_s2 = min(H, len(S2) - 1)
        path2 = S2[: h_s2 + 1]
        dst = path2[-1]
        for i in range(h_s2):
            G.add_edge(path2[i], path2[i + 1])

        candidates = []
        for u in S1:
            for v in S1:
                if u != v and not G.has_edge(u, v):
                    candidates.append((u, v))
        for u in S2:
            for v in S2:
                if u != v and not G.has_edge(u, v):
                    candidates.append((u, v))
        for u in S2:
            for v in S1:
                if not G.has_edge(u, v):
                    candidates.append((u, v))

        rng.shuffle(candidates)
        for u, v in candidates:
            if G.number_of_edges() >= target_edges:
                break
            G.add_edge(u, v)
        label = 0

    perm = list(range(num_nodes))
    rng.shuffle(perm)
    perm_map = {old: new for old, new in enumerate(perm)}
    new_src = perm_map[src]
    new_dst = perm_map[dst]
    edges = [(perm_map[u], perm_map[v]) for u, v in G.edges()]
    rng.shuffle(edges)

    return edges, new_src, new_dst, label, H


def collate_benchmark_batch(
    edges: List[Tuple[int, int]],
    src: int,
    dst: int,
    label: int,
    num_nodes: int = 48,
    target_edges: int = 96,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Collate single sample into tensor batch for inference."""
    B = 1
    node_ids = torch.arange(num_nodes, dtype=torch.long, device=device).unsqueeze(0)
    indicators = torch.zeros(B, num_nodes, 3, dtype=torch.float32, device=device)
    indicators[0, :, 2] = 1.0
    indicators[0, src] = torch.tensor([1.0, 0.0, 0.0], device=device)
    indicators[0, dst] = torch.tensor([0.0, 1.0, 0.0], device=device)

    adj = torch.zeros(B, num_nodes, num_nodes, dtype=torch.float32, device=device)
    edge_src = torch.zeros(B, target_edges, dtype=torch.long, device=device)
    edge_dst = torch.zeros(B, target_edges, dtype=torch.long, device=device)

    for k, (u, v) in enumerate(edges[:target_edges]):
        adj[0, u, v] = 1.0
        edge_src[0, k] = u
        edge_dst[0, k] = v

    return {
        "node_ids": node_ids,
        "indicators": indicators,
        "adj": adj,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "src_idx": torch.tensor([src], dtype=torch.long, device=device),
        "dst_idx": torch.tensor([dst], dtype=torch.long, device=device),
        "label": torch.tensor([float(label)], dtype=torch.float32, device=device),
    }
