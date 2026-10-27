"""
Data generator for Directed Graph Reachability with Harder Benchmark Distribution.

Features:
- Exact shortest-path controls using NetworkX.
- Supports expanded graph sizes: default N = 48 nodes, target_edges = 96.
- Training distribution: Path lengths H in [3, 5].
- Harder evaluation distribution: H in [3, 5, 8, 12, 16, 20].
- Perfectly balanced classes (50% reachable, 50% unreachable).
- Unreachable pairs matched in node count (N = 48) and edge density (target_edges = 96)
  to prevent structural cheating.
- Node ID permutation and edge order shuffling to prevent positional shortcut learning.
"""

from typing import List, Tuple, Dict, Optional
import random
import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader


def generate_graph_sample(
    H: int,
    reachable: bool,
    num_nodes: int = 48,
    target_edges: int = 96,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Tuple[int, int]], int, int, int, int]:
    """
    Generate a single directed graph reachability sample.

    Args:
        H: Target shortest path length between src and dst.
        reachable: Whether a directed path from src to dst should exist.
        num_nodes: Number of nodes in the graph (default: 48).
        target_edges: Exact target edge count (default: 96).
        rng: Optional random.Random instance for reproducibility.

    Returns:
        edges: List of directed (u, v) edge pairs.
        src: Source node ID.
        dst: Destination node ID.
        label: 1 if reachable, 0 if unreachable.
        H: Target shortest path length.
    """
    if rng is None:
        rng = random.Random()

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    if reachable:
        # 1. Pick H + 1 distinct nodes for the backbone path
        all_nodes = list(range(num_nodes))
        rng.shuffle(all_nodes)
        path = all_nodes[: H + 1]
        src, dst = path[0], path[-1]
        for i in range(H):
            G.add_edge(path[i], path[i + 1])

        # 2. Add random noise edges without creating any shortcut < H
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
                    # Created an illegal shortcut, reject edge
                    G.remove_edge(u, v)
            except nx.NetworkXNoPath:
                pass

        assert (
            nx.shortest_path_length(G, src, dst) == H
        ), f"Shortest path mismatch: expected {H}, got {nx.shortest_path_length(G, src, dst)}"
        label = 1

    else:
        # Unreachable pair construction:
        # Partition into two balanced sets S1 (containing src) and S2 (containing dst).
        # For N=48, each partition has 24 nodes, supporting paths up to H=20 (21 nodes).
        all_nodes = list(range(num_nodes))
        rng.shuffle(all_nodes)
        half = num_nodes // 2
        S1, S2 = all_nodes[:half], all_nodes[half:]

        # Plant a path of length H in S1 starting at src
        h_s1 = min(H, len(S1) - 1)
        path1 = S1[: h_s1 + 1]
        src = path1[0]
        for i in range(h_s1):
            G.add_edge(path1[i], path1[i + 1])

        # Plant a path of length H in S2 ending at dst
        h_s2 = min(H, len(S2) - 1)
        path2 = S2[: h_s2 + 1]
        dst = path2[-1]
        for i in range(h_s2):
            G.add_edge(path2[i], path2[i + 1])

        # Add edges within S1, within S2, and directed cross-edges S2 -> S1
        # Strictly NO edges S1 -> S2 are permitted, guaranteeing no path from src to dst.
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

        assert not nx.has_path(
            G, src, dst
        ), "Error: unreachable sample contains a path from src to dst"
        label = 0

    # Prevent positional and node ID shortcut learning:
    # 1. Randomly permute node IDs
    perm = list(range(num_nodes))
    rng.shuffle(perm)
    perm_map = {old: new for old, new in enumerate(perm)}
    new_src = perm_map[src]
    new_dst = perm_map[dst]
    edges = [(perm_map[u], perm_map[v]) for u, v in G.edges()]

    # 2. Randomly shuffle edge sequence order
    rng.shuffle(edges)

    return edges, new_src, new_dst, label, H


class ReachabilityDataset(Dataset):
    """
    Dataset for directed graph reachability with controlled path lengths.
    """

    def __init__(
        self,
        hop_lengths: List[int],
        num_samples: int,
        num_nodes: int = 48,
        target_edges: int = 96,
        seed: int = 42,
    ):
        super().__init__()
        self.hop_lengths = hop_lengths
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.target_edges = target_edges
        self.rng = random.Random(seed)

        self.samples = []
        for i in range(num_samples):
            h = self.rng.choice(hop_lengths)
            reachable = (i % 2 == 0)
            sample = generate_graph_sample(
                H=h,
                reachable=reachable,
                num_nodes=self.num_nodes,
                target_edges=self.target_edges,
                rng=self.rng,
            )
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        edges, src, dst, label, h = self.samples[idx]
        return {
            "edges": edges,
            "src": src,
            "dst": dst,
            "label": label,
            "hop_length": h,
            "num_nodes": self.num_nodes,
        }


def collate_graph_batch(batch_list: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of graph samples into batched PyTorch tensors for model ingestion.
    """
    B = len(batch_list)
    num_nodes = batch_list[0]["num_nodes"]
    num_edges = len(batch_list[0]["edges"])

    node_ids = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(B, 1)
    indicators = torch.zeros(B, num_nodes, 3, dtype=torch.float32)
    adj = torch.zeros(B, num_nodes, num_nodes, dtype=torch.float32)
    edge_src = torch.zeros(B, num_edges, dtype=torch.long)
    edge_dst = torch.zeros(B, num_edges, dtype=torch.long)
    src_idx = torch.zeros(B, dtype=torch.long)
    dst_idx = torch.zeros(B, dtype=torch.long)
    labels = torch.zeros(B, dtype=torch.float32)
    hop_lengths = torch.zeros(B, dtype=torch.long)

    for b, item in enumerate(batch_list):
        s = item["src"]
        d = item["dst"]
        src_idx[b] = s
        dst_idx[b] = d
        labels[b] = float(item["label"])
        hop_lengths[b] = item["hop_length"]

        indicators[b, :, 2] = 1.0
        indicators[b, s, :] = torch.tensor([1.0, 0.0, 0.0])
        indicators[b, d, :] = torch.tensor([0.0, 1.0, 0.0])

        for k, (u, v) in enumerate(item["edges"]):
            adj[b, u, v] = 1.0
            edge_src[b, k] = u
            edge_dst[b, k] = v

    return {
        "node_ids": node_ids,
        "indicators": indicators,
        "adj": adj,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "src_idx": src_idx,
        "dst_idx": dst_idx,
        "label": labels,
        "hop_length": hop_lengths,
    }


def create_dataloader(
    hop_lengths: List[int],
    num_samples: int,
    batch_size: int = 64,
    shuffle: bool = True,
    num_nodes: int = 48,
    target_edges: int = 96,
    seed: int = 42,
) -> DataLoader:
    """Helper to instantiate a ready-to-use DataLoader."""
    dataset = ReachabilityDataset(
        hop_lengths=hop_lengths,
        num_samples=num_samples,
        num_nodes=num_nodes,
        target_edges=target_edges,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_graph_batch,
        pin_memory=True,
    )
