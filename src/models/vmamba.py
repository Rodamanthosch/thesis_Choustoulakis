"""
src/models/vmamba.py
====================
JiT-S2-VMamba: JiT with SS2D (VMamba 2D Selective Scan) mixer.
Extracted exactly from jit-vmamba-cifar10.ipynb.

NOTABLE DIFFERENCES vs ViM and attention:
  - SS2D replaces BiMamba: 4-direction CrossScan, ONE stacked CUDA launch
  - JiTBlock.forward(x, c, H, W) — SS2D needs H, W for 2D CrossScan
  - Output norm is LayerNorm (NOT RMSNorm) — "# matches vmamba.py"
  - expand=1, d_conv=3 (vs expand=2, d_conv=4 in ViM)
  - SSM params stored as stacked tensors (x_proj_weight, dt_projs_weight/bias, A_logs, Ds)
    NOT as ModuleList — this enables the single batched einsum + one CUDA launch
  - Requires: mamba-ssm>=2.2.4 only (no causal-conv1d)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from src.primitives import (
    RMSNorm, get_2d_sincos_pos_embed,
    TimestepEmbedder, LabelEmbedder,
    BottleneckPatchEmbed, SwiGLUFFN, FinalLayer, modulate,
)


# ── CrossScan / CrossMerge (from jit-vmamba-cifar10 Cell 7) ─────────────────
# Path 0: row-major  →→↓
# Path 1: reverse of path 0  ←↑
# Path 2: column-major  ↓→
# Path 3: reverse of path 2  ↑←

def cross_scan(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    L = H * W
    s1 = x.reshape(B, C, L)
    s2 = s1.flip(-1)
    s3 = x.transpose(2, 3).reshape(B, C, L)
    s4 = s3.flip(-1)
    return torch.stack([s1, s2, s3, s4], dim=1)


def cross_merge(ys: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Undo each direction's reordering and sum.
    Verified inverse: with identity SSM, cross_merge(cross_scan(x), H, W) == 4*x.
    """
    B, K, C, L = ys.shape
    s1 = ys[:, 0]
    s2 = ys[:, 1].flip(-1)
    s3 = ys[:, 2].reshape(B, C, W, H).transpose(2, 3).reshape(B, C, L)
    s4 = ys[:, 3].flip(-1).reshape(B, C, W, H).transpose(2, 3).reshape(B, C, L)
    return s1 + s2 + s3 + s4


# ── SS2D (from jit-vmamba-cifar10 Cell 9) ────────────────────────────────────

class SS2D(nn.Module):
    """
    VMamba SS2D faithful to the paper's improved VSS Block (Figure 3(d)).

    Forward:  x: (B, L, D)  → out: (B, L, D)
              The caller passes H, W so SS2D can do its 2D CrossScan.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 1,
        dt_rank: int = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        K: int = 4,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)
        self.K = K

        # ── 1. Input projection (no gate branch) ──────────────────────
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # ── 2. Depthwise 2D conv + SiLU ──────────────────────────────
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv // 2,
            groups=self.d_inner, bias=True,
        )
        self.act = nn.SiLU()

        # ── 3. Per-direction SSM parameters, stored STACKED over K ───
        # x_proj: maps d_inner → (dt_rank + 2*d_state); K copies stacked.
        # Stored as a single Parameter of shape (K, dt_rank+2*d_state, d_inner),
        # used as a batched matmul in forward.
        self.x_proj_weight = nn.Parameter(
            torch.empty(K, self.dt_rank + 2 * d_state, self.d_inner)
        )
        nn.init.kaiming_uniform_(self.x_proj_weight, a=math.sqrt(5))

        # dt_proj: maps dt_rank → d_inner, with bias; K stacked
        self.dt_projs_weight = nn.Parameter(torch.empty(K, self.d_inner, self.dt_rank))
        self.dt_projs_bias   = nn.Parameter(torch.empty(K, self.d_inner))
        # Initialize dt_proj weight (Kaiming-style) and bias via softplus^{-1} sampling
        for k in range(K):
            dt_init_std = self.dt_rank ** -0.5
            nn.init.uniform_(self.dt_projs_weight[k], -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.dt_projs_bias[k].copy_(inv_dt)
        self.dt_projs_bias._no_reinit = True

        # A_log: K × (d_inner, d_state). S4 init: A = -[1..N], stored as log.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        A_log_init = torch.log(A).unsqueeze(0).repeat(K, 1, 1)        # (K, d_inner, d_state)
        self.A_logs = nn.Parameter(A_log_init)
        self.A_logs._no_weight_decay = True

        # D: K × d_inner — skip scalar per channel, per direction
        self.Ds = nn.Parameter(torch.ones(K, self.d_inner))
        self.Ds._no_weight_decay = True

        # ── 4. Output norm + projection ──────────────────────────────
        self.out_norm = nn.LayerNorm(self.d_inner)            # matches vmamba.py
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: (B, L, D) where L == H*W. Returns (B, L, D)."""
        B, L, D = x.shape
        assert L == H * W, f"SS2D got L={L} != H*W={H*W}"
        K = self.K
        d_inner = self.d_inner
        d_state = self.d_state

        # ── 1. in_proj + reshape to 2D ───────────────────────────────
        z = self.in_proj(x)                                     # (B, L, d_inner)
        z2d = z.view(B, H, W, d_inner).permute(0, 3, 1, 2).contiguous()
        #     z2d: (B, d_inner, H, W)

        # ── 2. Depthwise 2D conv + SiLU ──────────────────────────────
        z2d = self.act(self.conv2d(z2d))                        # (B, d_inner, H, W)

        # ── 3. CrossScan: 4 directions ───────────────────────────────
        xs = cross_scan(z2d)                                    # (B, K, d_inner, L)

        # ── 4. STACKED selective scan: one CUDA launch over K*d_inner ─
        # Flatten the K direction into the channel dim: (B, K*d_inner, L)
        xs_flat = xs.view(B, K * d_inner, L)

        # Per-direction x_proj: produces (Δ, B_ssm, C_ssm).
        # xs:             (B, K, d_inner, L)
        # x_proj_weight:  (K, dt_rank+2*d_state, d_inner)
        # x_dbl:          (B, K, dt_rank+2*d_state, L)
        x_dbl = torch.einsum("bkdl,kod->bkol", xs, self.x_proj_weight)

        dt_r, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, d_state, d_state], dim=2
        )
        # dt_r: (B, K, dt_rank, L); B_ssm, C_ssm: (B, K, d_state, L)

        # dt = dt_proj(dt_r): (K, d_inner, dt_rank) @ (B, K, dt_rank, L) → (B, K, d_inner, L)
        dt = torch.einsum("bkrl,kdr->bkdl", dt_r, self.dt_projs_weight)

        # Flatten K into channel for the kernel call.
        dt_flat    = dt.contiguous().view(B, K * d_inner, L)            # (B, K*d_inner, L)
        B_ssm_flat = B_ssm.contiguous().view(B, K, d_state, L)          # (B, K, d_state, L)
        C_ssm_flat = C_ssm.contiguous().view(B, K, d_state, L)          # (B, K, d_state, L)

        # A, D, delta_bias always fp32 (small, no upcast cost).
        # NOTE: do NOT .float() the four large activations below — under autocast they
        # arrive in fp16 and we want them to stay that way through saved-for-backward.
        A          = -torch.exp(self.A_logs.float()).view(K * d_inner, d_state)
        D_param    = self.Ds.float().view(K * d_inner)
        delta_bias = self.dt_projs_bias.float().view(K * d_inner)

        # selective_scan_fn supports grouped B/C: shape (B, G, d_state, L) with G=K
        y = selective_scan_fn(
            xs_flat,                                      # u: (B, K*d_inner, L)
            dt_flat,                                      # delta: (B, K*d_inner, L)
            A,                                            # (K*d_inner, d_state)
            B_ssm_flat,                                   # (B, K, d_state, L)
            C_ssm_flat,                                   # (B, K, d_state, L)
            D_param,                                      # (K*d_inner,)
            z=None,
            delta_bias=delta_bias,                        # (K*d_inner,)
            delta_softplus=True,
            return_last_state=False,
        )                                                  # (B, K*d_inner, L)

        # ── 5. CrossMerge ────────────────────────────────────────────
        ys = y.view(B, K, d_inner, L)                     # un-flatten
        out = cross_merge(ys, H, W)                       # (B, d_inner, L)
        out = out.transpose(1, 2)                         # (B, L, d_inner)

        # ── 6. LayerNorm + out_proj ──────────────────────────────────
        out = self.out_norm(out)
        out = self.out_proj(out)                          # (B, L, D)
        return self.proj_drop(out)


# ── JiT Block (from jit-vmamba-cifar10 Cell 13) ──────────────────────────────

class JiTBlock(nn.Module):
    """JiT block with adaLN-Zero conditioning, SS2D mixer, and SwiGLU FFN."""
    def __init__(self, hidden_size, num_heads=None, mlp_ratio=4.0,
                 d_state=16, d_conv=3, expand=1, K=4,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        # num_heads kept for signature parity with attention baseline; unused.
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.mixer = SS2D(
            d_model=hidden_size,
            d_state=d_state, d_conv=d_conv, expand=expand, K=K,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, H, W):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.mixer(
            modulate(self.norm1(x), shift_msa, scale_msa), H, W)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ── JiT-VMamba model (from jit-vmamba-cifar10 Cell 15) ───────────────────────

class JiTVMamba(nn.Module):
    """JiT with SS2D (VMamba) mixer. No in-context tokens. Class conditioning via adaLN-Zero only."""
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=384,
        depth=12,
        num_heads=6,           # kept for signature parity; unused
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=10,
        bottleneck_dim=128,
        # Mamba / SS2D knobs
        d_state=16,
        d_conv=3,
        expand=1,
        K=4,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = in_channels
        self.patch_size   = patch_size
        self.hidden_size  = hidden_size
        self.input_size   = input_size
        self.num_classes  = num_classes

        # Spatial grid size (used by SS2D mixers)
        self.grid_size = input_size // patch_size

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )

        # Fixed 2D sin-cos pos embed (no RoPE)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size),
                                       requires_grad=False)

        # Transformer blocks (with SS2D mixer); middle-half dropout slot
        lo, hi = depth // 4, depth // 4 * 3
        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
                d_state=d_state, d_conv=d_conv, expand=expand, K=K,
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
        pe = get_2d_sincos_pos_embed(
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

        H = W = self.grid_size
        for block in self.blocks:
            x = block(x, c, H, W)

        x = self.final_layer(x, c)
        return self.unpatchify(x)
