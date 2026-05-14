"""
src/primitives.py
=================
Shared building blocks used by all three JiT model variants.
Each piece is taken verbatim from the corresponding notebook cell.

Key differences between models are preserved here via separate helpers:
  - RMSNorm:
      attention/vmamba use  x.float() * rsqrt(...)  then  (norm * weight).to(x.dtype)
      vim uses              x.float() * rsqrt(...)  then  (norm * weight).to(x.dtype)
      → identical result, both versions kept as one class (they are the same)

  - get_2d_sincos_pos_embed:
      attention notebook  → helper named _get_1d_sincos, public fn get_2d_sincos_pos_embed
      vmamba notebook     → helper named _get_1d_sincos_pos_embed_from_grid (same math)
      vim notebook        → helpers get_1d_sincos_pos_embed_from_grid / get_2d_sincos_pos_embed_from_grid
      We expose ALL names so each model file can use the exact same call as its notebook.

  - BottleneckPatchEmbed:
      attention notebook  → 3rd arg named pca_dim
      vim/vmamba notebooks → 3rd arg named bottleneck_dim
      → same class, same code; arg name doesn't affect behaviour.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RMSNorm ───────────────────────────────────────────────────────────────────
# Taken from jit-cifar10 notebook (Cell 7). Identical in all three notebooks.

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()
        norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm).type_as(x)


# ── Positional embeddings ─────────────────────────────────────────────────────

# --- Attention notebook (jit-cifar10, Cell 7) ---
def _get_1d_sincos(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Used by attention (jit-cifar10) and vmamba notebooks."""
    gh = np.arange(grid_size, dtype=np.float32)
    gw = np.arange(grid_size, dtype=np.float32)
    g = np.meshgrid(gw, gh)
    g = np.stack(g, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos(embed_dim // 2, g[0])
    emb_w = _get_1d_sincos(embed_dim // 2, g[1])
    return np.concatenate([emb_h, emb_w], axis=1)

# --- VMamba notebook (jit-vmamba-cifar10, Cell 7) — same maths, different names ---
def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)

def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)

# --- ViM notebook (jit-vim-cifar10, Cell 7) — public versions of same helpers ---
get_1d_sincos_pos_embed_from_grid = _get_1d_sincos_pos_embed_from_grid
get_2d_sincos_pos_embed_from_grid = _get_2d_sincos_pos_embed_from_grid

def get_2d_sincos_pos_embed_vim(embed_dim, grid_size):
    """Used by the ViM notebook (same result as get_2d_sincos_pos_embed)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


# ── Shared embedders (identical across all three notebooks) ──────────────────

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """
    Reference LabelEmbedder: NO internal dropout. Label dropout happens
    externally in the flow wrapper (matching JiT's denoiser.py).
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        return self.embedding_table(labels)


# ── BottleneckPatchEmbed (identical across all three notebooks) ───────────────
# Attention notebook uses arg name `pca_dim`; Mamba notebooks use `bottleneck_dim`.
# The code is identical — we use `bottleneck_dim` as the canonical name here.

class BottleneckPatchEmbed(nn.Module):
    """Two-Conv2d bottleneck patch embed, matching model_jit.py exactly."""
    def __init__(self, img_size, patch_size, in_chans, bottleneck_dim, embed_dim, bias=True):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj1 = nn.Conv2d(in_chans, bottleneck_dim, kernel_size=patch_size,
                                stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        x = self.proj2(self.proj1(x))
        return x.flatten(2).transpose(1, 2)   # (B, N, embed_dim)


# ── SwiGLUFFN (identical across all three notebooks) ─────────────────────────

class SwiGLUFFN(nn.Module):
    """
    Matches model_jit.py exactly:
    hidden_dim is passed as mlp_ratio * hidden_size, then internally
    converted to int(hidden_dim * 2/3) for parameter-matched SwiGLU.
    """
    def __init__(self, dim, hidden_dim, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3  = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


# ── modulate + FinalLayer (identical across all three notebooks) ──────────────

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)
