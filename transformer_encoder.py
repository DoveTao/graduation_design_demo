# transformer_encoder.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(F.gelu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B,N,D]
        attn_mask (optional): MultiheadAttention expects (N,N) or (B*n_heads,N,N) boolean/float mask.
        For MVP we keep it None.
        """
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(h)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class TokenEncoder(nn.Module):
    """
    Lightweight ViT-style encoder over token sequence (cells).
    """
    def __init__(self, dim: int, n_layers: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class PatchEmbed(nn.Module):
    """
    Embed a sampled patch [3,p,p] -> token feature [D]
    Input patches: [B,N,3,p,p]
    Output: [B,N,D]
    """
    def __init__(self, p: int, dim: int, in_ch: int = 3):
        super().__init__()
        self.p = p
        self.dim = dim
        self.proj = nn.Sequential(
            nn.LayerNorm(in_ch * p * p),
            nn.Linear(in_ch * p * p, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, C, p1, p2 = patches.shape
        x = patches.view(B, N, C * p1 * p2)
        x = self.proj(x)
        return x


class BearingPosEnc(nn.Module):
    """
    Map bearing [x,y,z] into a positional embedding [D].
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, bearing: torch.Tensor) -> torch.Tensor:
        # bearing: [B,N,3] or [N,3]
        return self.net(bearing)
