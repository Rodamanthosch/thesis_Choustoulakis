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
        state_init: str = "none",     # "none" | "dimsum" | "learned"
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

        # ── 5. DiMSUM-style state-init conditioning (off by default) ──
        # Realized as ONE virtual write position prepended to every scan
        # direction with condition-derived input u0 = W_u(c), write
        # direction B0, pre-softplus step dt0, and C0 = 0 (output
        # stripped) — algebraically EXACT h_{-1} initialization under
        # Mamba's discretization:  h_{-1} = softplus(dt0 + dt_bias) · B0 ⊗ u0.
        # Verified against selective_scan_ref in tests/test_stateinit_equivalence.py.
        #
        #   "dimsum":  paper-literal h_{-1}[d, n] = W_u(c)[d]  in every
        #              direction (B0 ≡ 1_N; dt0 chosen at runtime so the
        #              effective Δ0 is exactly 1 — see forward()).
        #   "learned": per-direction learnable B0 (K, N) and dt0 (K, d_inner);
        #              strict superset of "dimsum".
        assert state_init in ("none", "dimsum", "learned"), state_init
        self.state_init = state_init
        if state_init != "none":
            # zero-initialized in JiTVMamba.initialize_weights (adaLN-Zero
            # discipline: h_{-1} = 0 at init → exact baseline at step 0).
            self.cond_u_proj = nn.Linear(d_model, self.d_inner, bias=False)
            if state_init == "learned":
                self.B0  = nn.Parameter(torch.ones(K, d_state))
                self.dt0 = nn.Parameter(torch.zeros(K, self.d_inner))
            else:  # "dimsum": fixed all-ones write direction (buffer, not trained)
                self.register_buffer("B0", torch.ones(K, d_state))

    def forward(self, x: torch.Tensor, H: int, W: int,
                cond: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, L_in, D) where L_in == extra_len + H*W, extra_len >= 0.
        cond: (B, D) adaLN condition vector c = t_emb + y_emb; consumed ONLY
              when state_init != "none" (DiMSUM-style scan-state init, fresh
              per block). Ignored otherwise — with state_init == "none" this
              forward is byte-for-byte the previous implementation.

        DiM-style persistent condition prefix (extra_len > 0):
          The leading `extra_len` positions are condition tokens. They skip the
          2D depthwise conv (they have no grid coordinate), are prepended to the
          head of ALL K CrossScan directions so they lead every scan (DiM pins
          the condition at scan-position 0), get their own Δ/B/C from the shared
          x_proj, are scanned together with the grid, then merged across the K
          directions by a PLAIN SUM — which is the exact cross_merge analog:
          cross_merge un-permutes each grid direction back to canonical order
          before summing, but the extras were prepended AFTER cross_scan so they
          were never flipped/transposed and already sit in canonical order in
          every direction; their un-permute is the identity, leaving only the
          sum. The updated extra outputs are returned in-place (positions
          0..extra_len) so they persist & update across blocks like DiM.

        extra_len == 0 recovers the byte-for-byte baseline path.
        """
        B, L_in, D = x.shape
        K = self.K
        d_inner = self.d_inner
        d_state = self.d_state
        HW = H * W
        extra_len = L_in - HW
        assert extra_len >= 0, f"SS2D got L_in={L_in} < H*W={HW}"

        # ── 1. in_proj (extras share in_proj with the grid, as in DiM) ─
        z_all   = self.in_proj(x)                               # (B, L_in, d_inner)
        z_extra = z_all[:, :extra_len]                          # (B, extra_len, d_inner)
        z_grid  = z_all[:, extra_len:]                          # (B, HW, d_inner)

        # ── 2. Grid path: reshape → depthwise 2D conv + SiLU ─────────
        z2d = z_grid.view(B, H, W, d_inner).permute(0, 3, 1, 2).contiguous()
        z2d = self.act(self.conv2d(z2d))                        # (B, d_inner, H, W)

        # ── 3. CrossScan: 4 directions over the pure H×W grid ────────
        xs = cross_scan(z2d)                                    # (B, K, d_inner, HW)

        # ── 3b. Prepend the condition seed to the head of every scan ─
        if extra_len > 0:
            # Same seed fed to all K directions; each direction's x_proj[k]
            # still gives it a distinct Δ/B/C. Seed leads the scan → it seeds
            # the recurrent state for every image token, in every direction.
            seed = z_extra.transpose(1, 2)[:, None]             # (B, 1, d_inner, extra_len)
            seed = seed.expand(B, K, d_inner, extra_len).to(xs.dtype)
            xs = torch.cat([seed, xs], dim=-1)                  # (B, K, d_inner, extra_len+HW)

        # ── 4. Per-direction x_proj (also covers the prepended condition tokens):
        # xs (B,K,d_inner,L) × x_proj_weight (K,dt_rank+2N,d_inner) → (B,K,dt_rank+2N,L)
        x_dbl = torch.einsum("bkdl,kod->bkol", xs, self.x_proj_weight)

        dt_r, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, d_state, d_state], dim=2
        )
        dt = torch.einsum("bkrl,kdr->bkdl", dt_r, self.dt_projs_weight)

        # ── 4b. State-init: prepend ONE virtual write position per direction ─
        # Unlike the in-context prefix (3b), the virtual position does NOT go
        # through x_proj: its Δ/B come from dedicated (condition-derived)
        # parameters and its C is zero, so it contributes ONLY through the
        # recurrent state — a pure h_{-1} injection (DiMSUM Conditional Mamba).
        v = 0
        if self.state_init != "none" and cond is not None:
            assert extra_len == 0, (
                "state_init and the in-context prefix are separate arms; "
                "run them one at a time (in_context_len must be 0)."
            )
            v = 1
            u0 = self.cond_u_proj(cond)                       # (B, d_inner)
            u0k = u0[:, None, :].expand(B, K, d_inner)        # shared across directions
            if self.state_init == "dimsum":
                # Paper-literal: h_{-1}[d,n] = W_u(c)[d] in EVERY direction.
                # The kernel adds dt_projs_bias to all positions, so choose
                # dt0 = softplus^{-1}(1) - bias ⇒ effective Δ0 ≡ 1 exactly
                # (per direction, regardless of the learned bias value; the
                # bias cancels, so no gradient leaks into it through this path).
                dt0 = (math.log(math.e - 1.0) - self.dt_projs_bias)   # (K, d_inner)
            else:  # "learned": free pre-softplus Δ0 (effective softplus(dt0+bias))
                dt0 = self.dt0                                        # (K, d_inner)
            xs = torch.cat(
                [u0k.unsqueeze(-1).to(xs.dtype), xs], dim=-1)         # (B,K,d,1+L)
            dt = torch.cat(
                [dt0[None, :, :, None].expand(B, K, d_inner, 1).to(dt.dtype), dt],
                dim=-1)
            B_ssm = torch.cat(
                [self.B0[None, :, :, None].expand(B, K, d_state, 1).to(B_ssm.dtype),
                 B_ssm], dim=-1)
            C_ssm = torch.cat(
                [C_ssm.new_zeros(B, K, d_state, 1), C_ssm], dim=-1)   # C0 = 0

        L = xs.shape[-1]                                        # v + extra_len + HW

        # ── 4c. STACKED selective scan: one CUDA launch over K*d_inner ─
        xs_flat    = xs.reshape(B, K * d_inner, L)
        dt_flat    = dt.contiguous().view(B, K * d_inner, L)
        B_ssm_flat = B_ssm.contiguous().view(B, K, d_state, L)
        C_ssm_flat = C_ssm.contiguous().view(B, K, d_state, L)

        # A, D, delta_bias always fp32 (small, no upcast cost).
        A          = -torch.exp(self.A_logs.float()).view(K * d_inner, d_state)
        D_param    = self.Ds.float().view(K * d_inner)
        delta_bias = self.dt_projs_bias.float().view(K * d_inner)

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

        ys = y.view(B, K, d_inner, L)                     # (B, K, d_inner, v+extra_len+HW)

        # Strip the virtual write position: its output (C0·h0 + D·u0 = D·u0)
        # is discarded — the condition acted purely through the entering state.
        if v:
            ys = ys[:, :, :, v:]                          # (B, K, d_inner, extra_len+HW)

        # ── 5. Merge. Grid: cross_merge (un-permute + sum). ──────────
        #        Extras: plain sum over directions (cross_merge analog; no
        #        un-permute needed — they were never reordered).
        if extra_len > 0:
            ys_extra = ys[:, :, :, :extra_len]            # (B, K, d_inner, extra_len)
            ys_grid  = ys[:, :, :, extra_len:]            # (B, K, d_inner, HW)
            out_grid  = cross_merge(ys_grid, H, W)        # (B, d_inner, HW)
            out_extra = ys_extra.sum(dim=1)               # (B, d_inner, extra_len)
            out = torch.cat([out_extra, out_grid], dim=-1)  # (B, d_inner, extra_len+HW)
        else:
            out = cross_merge(ys, H, W)                   # (B, d_inner, HW)  [baseline]

        out = out.transpose(1, 2)                         # (B, L_in, d_inner)

        # ── 6. LayerNorm + out_proj ──────────────────────────────────
        out = self.out_norm(out)
        out = self.out_proj(out)                          # (B, L_in, D)
        return self.proj_drop(out)


# ── JiT Block (from jit-vmamba-cifar10 Cell 13) ──────────────────────────────

class JiTBlock(nn.Module):
    """JiT block with adaLN-Zero conditioning, SS2D mixer, and SwiGLU FFN."""
    def __init__(self, hidden_size, num_heads=None, mlp_ratio=4.0,
                 d_state=16, d_conv=3, expand=1, K=4,
                 attn_drop=0.0, proj_drop=0.0, state_init="none"):
        super().__init__()
        # num_heads kept for signature parity with attention baseline; unused.
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.mixer = SS2D(
            d_model=hidden_size,
            d_state=d_state, d_conv=d_conv, expand=expand, K=K,
            proj_drop=proj_drop, state_init=state_init,
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
            modulate(self.norm1(x), shift_msa, scale_msa), H, W, cond=c)
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
        # ── DiM-style persistent in-context condition prefix (off by default) ──
        #   in_context_len=0 → baseline (adaLN-Zero only), byte-for-byte.
        #   Prefix tokens are prepended ONCE at block `in_context_start`, pinned
        #   at the front, scanned in all 4 SS2D directions leading every scan,
        #   updated in-place each block (persist & update, like DiM), and
        #   stripped once after the final block.
        in_context_len: int = 0,
        in_context_start: int = 0,
        in_context_content: str = "time_class",   # "time_class" | "class"
        # ── DiMSUM-style scan-state-init conditioning (off by default) ──
        #   "none"    → baseline / in-context arms (unchanged, byte-for-byte)
        #   "dimsum"  → paper-literal h_{-1} = W_u(c) per direction, per block
        #   "learned" → + learnable per-direction write direction B0 and Δ0
        state_init: str = "none",
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = in_channels
        self.patch_size   = patch_size
        self.hidden_size  = hidden_size
        self.input_size   = input_size
        self.num_classes  = num_classes
        self.in_context_len     = in_context_len
        self.in_context_start   = in_context_start
        self.in_context_content = in_context_content
        self.state_init         = state_init
        assert not (state_init != "none" and in_context_len > 0), (
            "state_init and the in-context prefix are separate conditioning "
            "arms — enable at most one per run."
        )

        # Learnable positional slots for the prefix tokens (DiM additional_embed).
        if in_context_len > 0:
            if in_context_content == "time_class":
                assert in_context_len in (2, 4), (
                    "in_context_content='time_class' expects in_context_len 2 "
                    "([t,y]) or 4 ([t,y,y,t]); got %d" % in_context_len
                )
            elif in_context_content == "class":
                assert in_context_len >= 1
            else:
                raise ValueError(
                    "Unknown in_context_content=%r (use 'time_class' or 'class')"
                    % in_context_content
                )
            self.incontext_pos_embed = nn.Parameter(
                torch.zeros(1, in_context_len, hidden_size)
            )

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
                state_init=state_init,
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

        # State-init conditioning: re-zero the write-content projection
        # (the _basic_init xavier pass above touched it). h_{-1} = 0 at
        # init → the model starts as the exact baseline (adaLN-Zero discipline).
        if self.state_init != "none":
            for block in self.blocks:
                nn.init.constant_(block.mixer.cond_u_proj.weight, 0)

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

    def _build_prefix(self, t_emb, y_emb):
        """Build the DiM-style condition prefix (B, in_context_len, D).

        content="time_class":  len 2 → [t, y];  len 4 → [t, y, y, t]  (DiM mirror)
        content="class":       len n → [y] * n  (JiT Table 9 / DiS style, K-sweep)
        A learnable positional slot is added per token (DiM additional_embed).
        """
        n = self.in_context_len
        if self.in_context_content == "time_class":
            toks = [t_emb, y_emb] if n == 2 else [t_emb, y_emb, y_emb, t_emb]
        else:  # "class"
            toks = [y_emb] * n
        ctx = torch.stack(toks, dim=1)                 # (B, n, D)
        return ctx + self.incontext_pos_embed          # learnable slots

    def forward(self, x, t, y):
        """x: (B, C, H, W) | t: (B,) | y: (B,)  → (B, C, H, W)"""
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        x = self.x_embedder(x)
        x = x + self.pos_embed

        H = W = self.grid_size
        for i, block in enumerate(self.blocks):
            # DiM: prepend the condition prefix ONCE, then let it persist &
            # update through the remaining blocks (each SS2D mixer re-injects
            # it into every scan direction and returns it updated in-place).
            if self.in_context_len > 0 and i == self.in_context_start:
                ctx = self._build_prefix(t_emb, y_emb)
                x = torch.cat([ctx, x], dim=1)         # (B, in_context_len + L, D)
            x = block(x, c, H, W)

        # Strip the prefix once, after the final block.
        if self.in_context_len > 0:
            x = x[:, self.in_context_len:, :]

        x = self.final_layer(x, c)
        return self.unpatchify(x)

