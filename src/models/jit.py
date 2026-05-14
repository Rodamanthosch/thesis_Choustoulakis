"""
src/models/jit.py
=================
JiT-S: Just image Transformer — attention baseline.
Base model extracted exactly from jit-cifar10.ipynb.

Extended with optional CFG and in-context class tokens,
controlled entirely by constructor arguments (all off by default,
matching the original notebook behaviour exactly).

NOTABLE DIFFERENCES vs Mamba variants:
  - Uses QK-Norm multi-head self-attention (not SSM)
  - Uses 2D RoPE (VisionRotaryEmbeddingFast) on Q and K
  - JiTBlock.forward(x, c, feat_rope)  — rope passed per-block
  - No d_state / d_conv / expand / K parameters

ADDED EXTENSIONS (off by default):
  - in_context_len > 0  → prepend in-context class tokens (JiT paper Table 9)
  - in_context_start    → which block to start prepending from
  - CFG is handled in the Denoiser/config layer, not in the model itself
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.primitives import (
    RMSNorm, get_2d_sincos_pos_embed,
    TimestepEmbedder, LabelEmbedder,
    BottleneckPatchEmbed, SwiGLUFFN, FinalLayer, modulate,
)


# ── 2D RoPE ───────────────────────────────────────────────────────────────────
# Exact from jit-cifar10 Cell 7.

def rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)

def broadcat(tensors, dim=-1):
    return torch.cat(tensors, dim=dim)

class VisionRotaryEmbeddingFast(nn.Module):
    """
    2D factorized RoPE for ViT-style transformers.
    Ported from EVA: https://github.com/baaivision/EVA
    Used by LightningDiT and JiT.
    `num_cls_token` lets us skip the first N tokens (they don't get rotated).
    """
    def __init__(self, dim, pt_seq_len, ft_seq_len=None, theta=10000.0, num_cls_token=0):
        super().__init__()
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        self.num_cls_token = num_cls_token

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.outer(t, freqs)
        freqs_w = torch.outer(t, freqs)
        freqs_h = freqs_h[:, None, :].expand(ft_seq_len, ft_seq_len, -1)
        freqs_w = freqs_w[None, :, :].expand(ft_seq_len, ft_seq_len, -1)
        freqs = broadcat((freqs_h, freqs_w), dim=-1)
        freqs = freqs.reshape(-1, freqs.shape[-1])

        freqs_cos = freqs.cos().repeat_interleave(2, dim=-1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=-1)

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t):
        """t: (B, num_heads, N, head_dim)"""
        if self.num_cls_token > 0:
            t_cls = t[:, :, :self.num_cls_token]
            t_img = t[:, :, self.num_cls_token:]
        else:
            t_cls = None
            t_img = t

        cos = self.freqs_cos[: t_img.shape[-2]]
        sin = self.freqs_sin[: t_img.shape[-2]]
        t_img_rot = (t_img * cos) + (rotate_half(t_img) * sin)

        if t_cls is not None:
            return torch.cat([t_cls, t_img_rot], dim=-2)
        return t_img_rot


# ── Attention ─────────────────────────────────────────────────────────────────
# Exact from jit-cifar10 Cell 9.

class Attention(nn.Module):
    """QK-Norm attention with 2D RoPE (applied externally via `rope` callable)."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = rope(q)
        k = rope(k)
        x = F.scaled_dot_product_attention(q, k, v,
                                            dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ── JiT Block ─────────────────────────────────────────────────────────────────
# Exact from jit-cifar10 Cell 13.

class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn  = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                qk_norm=True, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, feat_rope):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ── JiT model ────────────────────────────────────────────────────────────────
# Base: exact from jit-cifar10 Cell 15.
# Extended: in_context_len and in_context_start added (JiT paper Table 9).

class JiT(nn.Module):
    """
    JiT attention baseline.

    Matches jit-cifar10.ipynb exactly when:
        in_context_len = 0  (default)

    Enable in-context class tokens (JiT paper Table 9):
        in_context_len   = 32   # number of repeated class tokens to prepend
        in_context_start = 0    # which block index to start prepending (paper uses 4 for B)
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=10,
        bottleneck_dim=128,
        # ── extensions (off by default = matches notebook exactly) ──
        in_context_len: int = 0,    # 0 = no in-context tokens (notebook default)
        in_context_start: int = 0,  # block index to start prepending in-context tokens
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = in_channels
        self.patch_size   = patch_size
        self.num_heads    = num_heads
        self.hidden_size  = hidden_size
        self.input_size   = input_size
        self.num_classes  = num_classes
        self.in_context_len   = in_context_len
        self.in_context_start = in_context_start

        # embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )

        # fixed 2D sin-cos pos embed
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size),
                                       requires_grad=False)

        # in-context positional embeddings (only if in_context_len > 0)
        if in_context_len > 0:
            self.incontext_pos_embed = nn.Parameter(
                torch.zeros(1, in_context_len, hidden_size)
            )

        # 2D RoPE — num_cls_token=in_context_len so in-context tokens are not rotated
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=in_context_len
        )

        # transformer blocks with middle-half dropout slot
        lo, hi = depth // 4, depth // 4 * 3
        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (lo <= i < hi) else 0.0,
                proj_drop=proj_drop if (lo <= i < hi) else 0.0,
            )
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        pe = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        p = self.patch_size
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    def forward(self, x, t, y):
        """
        x: (B, C, H, W)  noisy input
        t: (B,)           timestep in [0, 1]
        y: (B,)           class labels (pass num_classes for unconditional / CFG null)
        Returns: (B, C, H, W) predicted clean image
        """
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        x = self.x_embedder(x)
        x = x + self.pos_embed

        for i, block in enumerate(self.blocks):
            # Prepend in-context tokens starting at in_context_start
            if self.in_context_len > 0 and i >= self.in_context_start:
                ctx = y_emb[:, None, :].expand(-1, self.in_context_len, -1)
                ctx = ctx + self.incontext_pos_embed
                x = torch.cat([ctx, x], dim=1)

            x = block(x, c, feat_rope=self.feat_rope)

            # Remove in-context tokens after each block
            if self.in_context_len > 0 and i >= self.in_context_start:
                x = x[:, self.in_context_len:, :]

        x = self.final_layer(x, c)
        return self.unpatchify(x)
