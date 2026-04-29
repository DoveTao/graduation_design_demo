"""
File: transformer_encoder.py
Description:
    Lightweight token encoders and patch embedding layers for panoramic
    matching. The module converts sampled ERP patches into token features and
    provides self-attention and optional cross-context blocks.

Main Components:
    - MLP and TransformerBlock building blocks
    - TokenEncoder for per-image token sequence encoding
    - CrossContextBlock and CrossContextEncoder for ablation experiments
    - PatchEmbed and BearingPosEnc for local appearance and spherical position

Usage / Role:
    Provides feature extraction and token encoding modules used by the main
    model definition.

Notes:
    Includes lightweight pooling variants for limited GPU memory experiments
    and the MVP translation feature branch.
"""

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


class CrossContextBlock(nn.Module):
    """
    Lightweight bidirectional cross-attention between the two panorama token sets.

    This is intentionally small and residual-only: each side keeps its own token
    identity, but can borrow context from the other image before matching logits
    are computed.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        strength: float = 0.25,
    ):
        super().__init__()
        self.strength = float(strength)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def _update(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(x)
        kv = self.norm_ctx(ctx)
        h, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x + self.strength * self.drop(h)
        x = x + self.strength * self.drop(self.mlp(self.norm_mlp(x)))
        return x

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a0, b0 = a, b
        a = self._update(a0, b0)
        b = self._update(b0, a0)
        return a, b


class CrossContextEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        strength: float = 0.25,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossContextBlock(dim, n_heads, mlp_ratio=mlp_ratio, dropout=dropout, strength=strength)
            for _ in range(max(1, int(n_layers)))
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            a, b = blk(a, b)
        return self.norm(a), self.norm(b)


class PatchEmbed(nn.Module):
    """
    CNN patch embedding for sampled ERP/spherical patches.

    Input:
        patches: [B, N, C, p, p]
    Output:
        token features: [B, N, D]

    Compared with the previous flatten+MLP version, this keeps the local 2D
    structure inside each sampled patch and extracts edge/texture-like local
    features before pooling to one token.
    """
    def __init__(
        self,
        p: int,
        dim: int,
        in_ch: int = 3,
        use_coords: bool = False,
        use_avgmax_pool: bool = False,
        pool_mode: str = "avg",
    ):
        super().__init__()
        self.p = int(p)
        self.dim = int(dim)
        self.use_coords = bool(use_coords)
        self.pool_mode = str(pool_mode)
        if bool(use_avgmax_pool):
            self.pool_mode = "avgmax"
        if self.pool_mode not in {"avg", "avgmax", "gated_avgmax"}:
            raise ValueError(f"Unsupported patch pool_mode: {self.pool_mode}")
        conv_in_ch = int(in_ch) + (2 if self.use_coords else 0)
        if self.use_coords:
            coord = torch.linspace(-1.0, 1.0, steps=self.p, dtype=torch.float32)
            yy, xx = torch.meshgrid(coord, coord, indexing="ij")
            self.register_buffer("patch_coords", torch.stack([xx, yy], dim=0).view(1, 1, 2, self.p, self.p), persistent=False)

        # Keep this lightweight because it is applied to B*N patches.
        # For the current setting p=16, this gives:
        #   [C,16,16] -> [32,16,16] -> [64,8,8] -> pooled -> D
        self.cnn = nn.Sequential(
            nn.Conv2d(conv_in_ch, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        if self.pool_mode == "gated_avgmax":
            self.pool_gate = nn.Parameter(torch.full((64,), -2.0))
        proj_in = 128 if self.pool_mode == "avgmax" else 64
        self.proj = nn.Sequential(
            nn.LayerNorm(proj_in),
            nn.Linear(proj_in, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, N, C, p1, p2 = patches.shape
        if self.use_coords:
            coords = self.patch_coords.expand(B, N, -1, -1, -1).to(dtype=patches.dtype, device=patches.device)
            patches = torch.cat([patches, coords], dim=2)
            C = patches.shape[2]
        x = patches.reshape(B * N, C, p1, p2).contiguous()
        x = self.cnn(x)
        if self.pool_mode == "avgmax":
            x_avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
            x_max = F.adaptive_max_pool2d(x, 1).flatten(1)
            x = torch.cat([x_avg, x_max], dim=-1)
        elif self.pool_mode == "gated_avgmax":
            x_avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
            x_max = F.adaptive_max_pool2d(x, 1).flatten(1)
            gate = torch.sigmoid(self.pool_gate).view(1, -1).to(dtype=x_avg.dtype)
            x = x_avg + gate * (x_max - x_avg)
        else:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.proj(x)
        x = x.view(B, N, self.dim)
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
