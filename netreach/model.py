"""
Neural Latent Recurrent Core Architecture for NetReach.

Implements ChallengerLatentNet (501,793 parameters):
- Shared-weight transition block with GRU-style highway gating.
- Persistent Source Anchor injection preventing query amnesia over 20+ hops.
- Vectorized fused message passing via torch.bmm and torch.scatter_reduce.
- Zero external C-extension dependencies, fully operational on both CUDA and pure CPU.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class ChallengerLatentNet(nn.Module):
    """
    Recurrent Latent Network with Gated Transitions and Persistent Source Anchor injection.
    Maintains per-node latent state H_t in R^{B x N x d}.
    """

    def __init__(
        self,
        num_nodes: int = 50,
        node_embed_dim: int = 64,
        d_model: int = 160,
        hidden_dim: int = 352,
        train_steps: int = 5,
        use_compile: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_embed_dim = node_embed_dim
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.train_steps = train_steps

        # Node identity embeddings and initial state projection
        self.node_embed = nn.Embedding(num_nodes, node_embed_dim)
        self.node_proj = nn.Linear(node_embed_dim + 3, d_model)

        # Message passing projection
        self.msg_proj = nn.Linear(d_model, d_model)

        # Gated Recurrence with Persistent Source Anchor
        # Input to transitions: [H_t, mean_msg, max_msg, H_0] = 4 * d_model
        in_dim = 4 * d_model
        self.gate_linear = nn.Linear(in_dim, d_model)
        self.cand_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = RMSNorm(d_model)

        # Readout head evaluating reachability logit
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self._transition_fn = self._raw_transition
        if use_compile and hasattr(torch, "compile"):
            try:
                self._transition_fn = torch.compile(self._raw_transition)
            except Exception:
                self._transition_fn = self._raw_transition

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _raw_transition(
        self,
        H: torch.Tensor,
        H_0: torch.Tensor,
        adj: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gated recurrent transition step:
        H_{t+1} = RMSNorm((1 - z_t) * H_t + z_t * H_tilde_t)
        """
        B, N, d = H.shape

        # 1. Message feature projection
        M = self.msg_proj(H)  # [B, N, d]

        # 2. Vectorized Mean Aggregation via fused bmm
        adj_t = adj.transpose(1, 2)
        in_degree = adj_t.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean_msg = torch.bmm(adj_t, M) / in_degree  # [B, N, d]

        # 3. Vectorized Max Aggregation via scatter_reduce over edge buffers (O(B*M*d), NO 4D tensor)
        edge_messages = torch.gather(M, 1, edge_src.unsqueeze(-1).expand(-1, -1, d))  # [B, M, d]
        dst_index = edge_dst.unsqueeze(-1).expand(-1, -1, d)
        max_msg = torch.zeros(B, N, d, device=M.device, dtype=M.dtype)
        max_msg = max_msg.scatter_reduce(1, dst_index, edge_messages, reduce="amax", include_self=False)

        # 4. Gated Recurrent Update with Persistent Source Anchor H_0
        combined = torch.cat([H, mean_msg, max_msg, H_0], dim=-1)  # [B, N, 4*d]

        z = torch.sigmoid(self.gate_linear(combined))  # [B, N, d]
        H_tilde = self.cand_mlp(combined)              # [B, N, d]

        H_next = (1.0 - z) * H + z * H_tilde
        return self.norm(H_next)

    def transition(
        self,
        H: torch.Tensor,
        H_0: torch.Tensor,
        adj: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> torch.Tensor:
        try:
            return self._transition_fn(H, H_0, adj, edge_src, edge_dst)
        except Exception:
            self._transition_fn = self._raw_transition
            return self._raw_transition(H, H_0, adj, edge_src, edge_dst)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        steps: Optional[int] = None,
        return_latents: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass unrolling gated latent recurrence for `steps` iterations.

        Args:
            batch: Dictionary with keys 'node_ids', 'indicators', 'adj', 'edge_src', 'edge_dst', 'src_idx', 'dst_idx'.
            steps: Number of recurrent steps T.
            return_latents: If True, returns (logits, H_T) for cosine similarity diagnostics.
        Returns:
            logits: Tensor of shape [B] representing reachability logits.
            (optional) H_T: Tensor of shape [B, N, d] of final node latents.
        """
        node_ids = batch["node_ids"]
        indicators = batch["indicators"]
        adj = batch["adj"]
        edge_src = batch["edge_src"]
        edge_dst = batch["edge_dst"]
        src_idx = batch["src_idx"]
        dst_idx = batch["dst_idx"]

        B, N = node_ids.shape
        T = steps if steps is not None else self.train_steps

        # 1. Initialize per-node latent tensor and persistent source anchor H_0
        emb = self.node_embed(node_ids)
        H_0 = self.norm(self.node_proj(torch.cat([emb, indicators], dim=-1)))
        H = H_0

        # 2. Recurrent unrolling for T iterations with gated transition and persistent anchor
        for _ in range(T):
            H = self.transition(H, H_0, adj, edge_src, edge_dst)

        # 3. Readout: extract src and dst representations
        batch_idx = torch.arange(B, device=node_ids.device)
        h_src = H[batch_idx, src_idx]  # [B, d_model]
        h_dst = H[batch_idx, dst_idx]  # [B, d_model]

        h_comb = torch.cat([h_src, h_dst, h_src * h_dst], dim=-1)  # [B, 3*d_model]
        logits = self.head(h_comb).squeeze(-1)  # [B]

        if return_latents:
            return logits, H
        return logits
