"""
src/models/vim.py
=================
JiT-S2-ViM: JiT with BiMamba (Vision Mamba Algorithm 1) mixer.
Extracted exactly from jit-vim-cifar10.ipynb.

NOTABLE DIFFERENCES vs attention and VMamba:
  - BiMamba replaces self-attention — no RoPE, no Q/K
  - JiTBlock.forward(x, c)  — no H/W (BiMamba is a sequence mixer)
  - RMSNorm for output norm (vs LayerNorm in VMamba)
  - ONE shared in_proj / z / out_proj for both directions (Vim Algo 1)
  - expand=1 default (NOT 2 — see Cell 19 comment "NOT E = 2D")
  - d_conv=4 (vs d_conv=3 in VMamba)
  - Requires: mamba-ssm>=2.2.4, causal-conv1d>=1.5.0 (CUDA kernels, no fallback)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from causal_conv1d import causal_conv1d_fn

from src.primitives import (
    RMSNorm, get_2d_sincos_pos_embed_vim,
    TimestepEmbedder, LabelEmbedder,
    BottleneckPatchEmbed, SwiGLUFFN, FinalLayer, modulate,
)


# ── dt initialisation helper (from jit-vim-cifar10 Cell 9) ──────────────────

def _dt_init(dt_proj, d_inner, dt_init_floor=1e-4, dt_min=0.001, dt_max=0.1):
    """Vim/Mamba dt_proj init: bias is softplus^{-1} of uniform[dt_min, dt_max]."""
    dt_init_std = d_inner ** -0.5
    nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
    dt = torch.exp(
        torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    ).clamp(min=dt_init_floor)
    inv_dt = dt + torch.log(-torch.expm1(-dt))   # softplus^{-1}(dt)
    with torch.no_grad():
        dt_proj.bias.copy_(inv_dt)
    dt_proj.bias._no_reinit = True


# ── One scan direction (from jit-vim-cifar10 Cell 9) ─────────────────────────

class _DirectionalSSM(nn.Module):
    """One scan direction: depthwise Conv1d → SiLU → selective_scan."""
    def __init__(self, d_inner, d_state, d_conv, dt_rank):
        super().__init__()
        self.d_inner, self.d_state, self.d_conv, self.dt_rank = d_inner, d_state, d_conv, dt_rank

        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=d_inner, bias=True,
        )
        self.x_proj  = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        _dt_init(self.dt_proj, d_inner)

        # A_log: (E, N), A initialized to -[1..N] (S4 init)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True

    def forward(self, x_inner):
        """x_inner: (B, L, E) → (B, L, E)."""
        B_, L, E = x_inner.shape
        x_t = x_inner.transpose(1, 2).contiguous()        # (B, E, L)

        # Depthwise causal Conv1d with fused SiLU (causal-conv1d kernel)
        x_t = causal_conv1d_fn(
            x_t, self.conv1d.weight.squeeze(1), self.conv1d.bias,
            activation="silu",
        )
        x_after_conv = x_t.transpose(1, 2).contiguous()   # (B, L, E)

        # Project to (Δ, B_ssm, C_ssm)
        x_dbl = self.x_proj(x_after_conv)                                  # (B, L, dt_rank + 2N)
        dt, Bm, Cm = torch.split(x_dbl,
                                  [self.dt_rank, self.d_state, self.d_state],
                                  dim=-1)
        dt = self.dt_proj(dt)                                              # (B, L, E)

        A = -torch.exp(self.A_log.float())                                 # (E, N)

        # Selective-scan CUDA kernel
        # signature: u(B,E,L), delta(B,E,L), A(E,N), B(B,N,L), C(B,N,L), D(E,)
        y = selective_scan_fn(
            x_after_conv.transpose(1, 2).contiguous(),
            dt.transpose(1, 2).contiguous(),
            A,
            Bm.transpose(1, 2).contiguous(),
            Cm.transpose(1, 2).contiguous(),
            self.D.float(),
            z=None,
            delta_bias=None,
            delta_softplus=True,
            return_last_state=False,
        )                                                                  # (B, E, L)

        return y.transpose(1, 2).contiguous()                              # (B, L, E)


# ── BiMambaV2 (from jit-vim-cifar10 Cell 9) ──────────────────────────────────

class BiMambaV2(nn.Module):
    """
    Vim Algorithm 1 — bidirectional Mamba with full-sequence flip semantics.

    Shared in_proj produces a single (x, z). Two _DirectionalSSM stacks scan
    x_inner forward and backward independently. Both outputs are gated by the
    *same* SiLU(z), summed, and projected by a *shared* out_proj.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 dt_rank=None, proj_drop=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # Shared in_proj: D → 2*E (produces x_inner and z gate)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # Per-direction SSM stacks
        self.ssm_fwd = _DirectionalSSM(self.d_inner, d_state, d_conv, self.dt_rank)
        self.ssm_bwd = _DirectionalSSM(self.d_inner, d_state, d_conv, self.dt_rank)
        # Shared out_proj
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: (B, L, D) → (B, L, D)."""
        x = x.contiguous()
        xz = self.in_proj(x)                                # (B, L, 2E)
        x_inner, z = xz.chunk(2, dim=-1)                    # (B, L, E) each

        y_fwd = self.ssm_fwd(x_inner)                       # forward scan
        y_bwd = self.ssm_bwd(torch.flip(x_inner, dims=[1])).flip(dims=[1])

        z_act = F.silu(z)
        y = y_fwd * z_act + y_bwd * z_act                   # shared z gates both
        return self.proj_drop(self.out_proj(y))             # shared out_proj after sum


# ── JiT Block (from jit-vim-cifar10 Cell 13) ─────────────────────────────────

class JiTBlock(nn.Module):
    """JiT block with adaLN-Zero conditioning, BiMamba mixer (Vim Algo. 1), and SwiGLU FFN."""
    def __init__(self, hidden_size, num_heads=None, mlp_ratio=4.0,
                 d_state=16, d_conv=4, expand=2, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        # num_heads kept for signature parity with the attention baseline; unused.
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.mixer = BiMambaV2(
            d_model=hidden_size,
            d_state=d_state, d_conv=d_conv, expand=expand,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.mixer(
            modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ── JiT-ViM model (from jit-vim-cifar10 Cell 15) ─────────────────────────────

class JiTViM(nn.Module):
    """
    JiT with BiMamba mixer. No in-context tokens. Class conditioning via adaLN-Zero only.

    Matches jit-vim-cifar10.ipynb exactly with default arguments.
    Note: Cell 19 uses expand=1 ("NOT E = 2D — paper-faithful Vim").
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=384,
        depth=12,
        num_heads=6,           # kept for signature parity with attention baseline; unused
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=10,
        bottleneck_dim=128,
        # Mamba knobs
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = in_channels
        self.patch_size   = patch_size
        self.hidden_size  = hidden_size
        self.input_size   = input_size
        self.num_classes  = num_classes

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )

        # Fixed 2D sin-cos pos embed (no RoPE)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size),
                                       requires_grad=False)

        # Transformer blocks (with BiMamba mixer); middle-half dropout slot
        lo, hi = depth // 4, depth // 4 * 3
        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
                d_state=d_state, d_conv=d_conv, expand=expand,
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

        # Fixed sin-cos pos embed
        pe = get_2d_sincos_pos_embed_vim(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # Patch embed xavier init on flattened conv weights
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Embeddings
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # adaLN-Zero
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # Zero output
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
        """x: (B, C, H, W) | t: (B,) | y: (B,)  → (B, C, H, W)"""
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        x = self.x_embedder(x)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        return self.unpatchify(x)
