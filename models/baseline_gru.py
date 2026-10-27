"""
Sequential GRU baseline for directed graph reachability.
Parameter matched to ~500k parameters with multi-layer recurrent sequence processing.
"""

from typing import Dict
import torch
import torch.nn as nn


class BaselineGRU(nn.Module):
    """
    Sequential GRU baseline for graph reachability.
    Recurrently processes the sequence of edges following a query token.
    Matched to ~500k parameters.
    """

    def __init__(
        self,
        num_nodes: int = 50,
        node_embed_dim: int = 64,
        d_model: int = 128,
        hidden_dim: int = 158,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embed_dim = node_embed_dim
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node identity embeddings
        self.node_embed = nn.Embedding(num_nodes, node_embed_dim)

        # Input projection for edge and query tokens
        in_dim = node_embed_dim * 2 + 6
        self.edge_proj = nn.Linear(in_dim, d_model)
        self.query_proj = nn.Linear(in_dim, d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Binary classification logit head from pooled hidden representations
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        total_params = self.count_parameters()
        print(f"[Init] BaselineGRU initialized with {total_params:,} parameters.")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Forward pass.
        Args:
            batch: Dictionary with keys 'edge_src', 'edge_dst', 'src_idx', 'dst_idx'.
        Returns:
            logits: Tensor of shape [B] representing reachability logits.
        """
        edge_src = batch["edge_src"]  # [B, M]
        edge_dst = batch["edge_dst"]  # [B, M]
        src_idx = batch["src_idx"]    # [B]
        dst_idx = batch["dst_idx"]    # [B]
        B, M = edge_src.shape

        # 1. Construct Query Token
        src_emb = self.node_embed(src_idx)
        dst_emb = self.node_embed(dst_idx)
        q_ind = torch.zeros(B, 6, device=src_idx.device, dtype=src_emb.dtype)
        q_ind[:, 0] = 1.0  # src is source
        q_ind[:, 4] = 1.0  # dst is destination
        q_token = self.query_proj(torch.cat([src_emb, dst_emb, q_ind], dim=-1)).unsqueeze(1)  # [B, 1, d_model]

        # 2. Construct Edge Tokens
        u_emb = self.node_embed(edge_src)
        v_emb = self.node_embed(edge_dst)

        src_exp = src_idx.unsqueeze(1).expand_as(edge_src)
        dst_exp = dst_idx.unsqueeze(1).expand_as(edge_src)

        u_is_src = (edge_src == src_exp).to(dtype=src_emb.dtype).unsqueeze(-1)
        u_is_dst = (edge_src == dst_exp).to(dtype=src_emb.dtype).unsqueeze(-1)
        u_is_oth = (1.0 - u_is_src - u_is_dst).clamp(min=0.0)

        v_is_src = (edge_dst == src_exp).to(dtype=src_emb.dtype).unsqueeze(-1)
        v_is_dst = (edge_dst == dst_exp).to(dtype=src_emb.dtype).unsqueeze(-1)
        v_is_oth = (1.0 - v_is_src - v_is_dst).clamp(min=0.0)

        edge_ind = torch.cat([u_is_src, u_is_dst, u_is_oth, v_is_src, v_is_dst, v_is_oth], dim=-1)
        edge_tokens = self.edge_proj(torch.cat([u_emb, v_emb, edge_ind], dim=-1))  # [B, M, d_model]

        # 3. Concatenate into full sequence [Query, Edge_1, ..., Edge_M]
        seq = torch.cat([q_token, edge_tokens], dim=1)  # [B, 1 + M, d_model]

        # 4. Multi-layer GRU recurrent sequence processing
        out, _ = self.gru(seq)  # [B, 1 + M, hidden_dim]

        # 5. Mean-pooling across the processed sequence
        pooled = out.mean(dim=1)  # [B, hidden_dim]
        logits = self.head(pooled).squeeze(-1)  # [B]
        return logits
