"""
Standard Feedforward PyTorch Transformer baseline for directed graph reachability.
Parameter matched to ~500k parameters with fixed forward depth.
"""

from typing import Dict
import torch
import torch.nn as nn


class BaselineTransformer(nn.Module):
    """
    Standard PyTorch Transformer Encoder baseline for graph reachability.
    Processes the sequence of graph edges alongside a dedicated query token.
    Fixed forward depth: 3 layers, 4 attention heads.
    """

    def __init__(
        self,
        num_nodes: int = 50,
        node_embed_dim: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 320,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embed_dim = node_embed_dim
        self.d_model = d_model
        self.num_layers = num_layers

        # Node identity embeddings
        self.node_embed = nn.Embedding(num_nodes, node_embed_dim)

        # Input feature dimension:
        # [node_emb(u), node_emb(v), indicator(u), indicator(v)]
        # 64 + 64 + 3 + 3 = 134
        in_dim = node_embed_dim * 2 + 6
        self.edge_proj = nn.Linear(in_dim, d_model)
        self.query_proj = nn.Linear(in_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Binary classification logit head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        total_params = self.count_parameters()
        print(f"[Init] BaselineTransformer initialized with {total_params:,} parameters.")

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
        src_emb = self.node_embed(src_idx)  # [B, node_embed_dim]
        dst_emb = self.node_embed(dst_idx)  # [B, node_embed_dim]
        q_ind = torch.zeros(B, 6, device=src_idx.device, dtype=src_emb.dtype)
        q_ind[:, 0] = 1.0  # src is source
        q_ind[:, 4] = 1.0  # dst is destination
        q_token = self.query_proj(torch.cat([src_emb, dst_emb, q_ind], dim=-1)).unsqueeze(1)  # [B, 1, d_model]

        # 2. Construct Edge Tokens
        u_emb = self.node_embed(edge_src)  # [B, M, node_embed_dim]
        v_emb = self.node_embed(edge_dst)  # [B, M, node_embed_dim]

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

        # 4. Multi-layer Transformer processing
        out = self.transformer(seq)  # [B, 1 + M, d_model]

        # 5. Extract representation from query token at index 0
        logits = self.head(out[:, 0, :]).squeeze(-1)  # [B]
        return logits
