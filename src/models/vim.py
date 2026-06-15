"""
src/models/vim.py
=================
JiT-S2-ViM: JiT with configurable ViM / Mamba3 sequence mixer.

Modes:
  - mixer_impl="vim":
      Original BiMambaV2 / Vision Mamba Algorithm 1 style mixer.
      This is Mamba-1-style selective scan with explicit causal Conv1d.

  - mixer_impl="mamba3":
      ViM-style bidirectional wrapper around the official upstream Mamba3 block.
      This uses official Mamba3 internals, not the old Vim causal-conv block.

NOTABLE DIFFERENCES vs attention and VMamba:
  - Sequence mixer replaces self-attention.
  - JiTBlock.forward(x, c) -- no H/W.
  - RMSNorm for output norm.
  - Fixed 2D sin-cos positional embedding, no attention RoPE.
  - Class/timestep conditioning is still AdaLN-Zero outside the mixer.

OPTIONAL CONDITIONING EXTENSIONS (both OFF by default = baseline byte-for-byte):

  Option A — in-context class tokens (in_context_len > 0):
    Prefix-token conditioning. Plumbing follows DiS (feizc/DiS, bimamba_type="v2"):
    conditioning tokens are concatenated at the FRONT of the sequence, given
    learnable positional slots, scanned by BiMambaV2, and stripped after the
    final block. Token CONTENT follows JiT Table 9 (K repeated class embeddings),
    so JiT-S and JiT-S2-ViM differ only in the mixer -> the controlled test of
    "does the JiT in-context mechanism survive the SSM scan compression (eq. 5)".
      in_context_len   : number of prepended class tokens (DiS uses 2; JiT-B uses K=4)
      in_context_start : block index to prepend at (DiS=0; JiT-B=4)

  Option C — DiMSUM Conditional Mamba (cond_init="conv_state"):
    DiMSUM (VinAIResearch/DiMSUM) "Conditional Mamba" sets the SSM prior from the
    condition. Its RELEASED code does this through a forked causal_conv1d
    (causal_conv1d_fwd_cond) + forked mamba_inner_fn_cond, feeding a d_inner-wide
    projection of c as the conv's initial state (NOT a full d_inner x d_state SSM
    state, and the stock selective_scan path does not inject it at all). We
    reproduce that exact mechanism on STOCK kernels: a shared cond_proj(c) -> d_inner
    seeds the (d_conv-1) causal-conv left-pad window of BOTH scan directions,
    instead of zeros. No forked CUDA op required; cond_init="none" is the baseline.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from causal_conv1d import causal_conv1d_fn

from src.primitives import (
    RMSNorm,
    get_2d_sincos_pos_embed_vim,
    TimestepEmbedder,
    LabelEmbedder,
    BottleneckPatchEmbed,
    SwiGLUFFN,
    FinalLayer,
    modulate,
)


# ── dt initialization helper ────────────────────────────────────────────────

def _dt_init(dt_proj, d_inner, dt_init_floor=1e-4, dt_min=0.001, dt_max=0.1):
    """Vim/Mamba dt_proj init: bias is softplus^{-1} of uniform[dt_min, dt_max]."""
    dt_init_std = d_inner ** -0.5
    nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)

    dt = torch.exp(
        torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=dt_init_floor)

    inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus^{-1}(dt)

    with torch.no_grad():
        dt_proj.bias.copy_(inv_dt)

    dt_proj.bias._no_reinit = True


# ── One ViM / Mamba-1-style scan direction ──────────────────────────────────

class _DirectionalSSM(nn.Module):
    """One scan direction: depthwise causal Conv1d -> SiLU -> selective_scan."""
    def __init__(self, d_inner, d_state, d_conv, dt_rank):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        _dt_init(self.dt_proj, d_inner)

        # A_log: (E, N), A initialized to -[1..N].
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True

    def forward(self, x_inner, seed=None):
        """
        x_inner: (B, L, E)
        seed:    (B, E) | None
                 DiMSUM-style conditional initial conv state (option C). When given,
                 the (d_conv-1) causal-conv left-pad window is filled with this
                 projected condition instead of zeros, so the earliest scan steps
                 carry the condition into the SSM hidden state. When None, the fused
                 causal_conv1d kernel is used (byte-for-byte baseline path).
        returns: (B, L, E)
        """
        x_t = x_inner.transpose(1, 2).contiguous()  # (B, E, L)

        if seed is None:
            # Baseline path: fused depthwise causal Conv1d with fused SiLU.
            x_t = causal_conv1d_fn(
                x_t,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation="silu",
            )
        else:
            # DiMSUM Conditional-Mamba path, stock-kernel faithful.
            # Reproduces causal_conv1d_fwd_cond's intent (condition seeds the conv's
            # initial state) using only F.conv1d — no forked causal-conv1d CUDA op.
            # seed=0 recovers the baseline exactly.
            B, E, L = x_t.shape
            pad = seed[:, :, None].expand(B, E, self.d_conv - 1).to(x_t.dtype)
            x_padded = torch.cat([pad, x_t], dim=-1)               # (B, E, L + d_conv - 1)
            x_t = F.conv1d(x_padded, self.conv1d.weight, self.conv1d.bias,
                           groups=self.d_inner)                    # valid conv -> (B, E, L)
            x_t = F.silu(x_t)

        x_after_conv = x_t.transpose(1, 2).contiguous()  # (B, L, E)

        # Project to (Delta, B_ssm, C_ssm).
        x_dbl = self.x_proj(x_after_conv)  # (B, L, dt_rank + 2N)

        dt, Bm, Cm = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        dt = self.dt_proj(dt)  # (B, L, E)
        A = -torch.exp(self.A_log.float())  # (E, N)

        # selective_scan_fn signature:
        # u(B,E,L), delta(B,E,L), A(E,N), B(B,N,L), C(B,N,L), D(E,)
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
        )  # (B, E, L)

        return y.transpose(1, 2).contiguous()  # (B, L, E)


# ── Original BiMambaV2 / ViM-style mixer ────────────────────────────────────

class BiMambaV2(nn.Module):
    """
    ViM-style bidirectional Mamba.

    Shared in_proj produces one (x_inner, z). Two _DirectionalSSM modules scan
    x_inner forward and backward independently. Both outputs are gated by the
    same SiLU(z), summed, and projected by a shared out_proj.
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank=None,
        proj_drop=0.0,
        d_cond=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # DiMSUM Conditional-Mamba (option C): a single shared projection of the
        # condition c -> d_inner, used to seed BOTH scan directions' conv state.
        # Mirrors DiMSUM mamba_simple.py (one cond_proj, same seed fed to fwd & bwd).
        self.d_cond = d_cond
        if d_cond is not None:
            self.cond_proj = nn.Linear(d_cond, self.d_inner, bias=True)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.ssm_fwd = _DirectionalSSM(
            self.d_inner,
            d_state,
            d_conv,
            self.dt_rank,
        )
        self.ssm_bwd = _DirectionalSSM(
            self.d_inner,
            d_state,
            d_conv,
            self.dt_rank,
        )

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond=None):
        """
        x:    (B, L, D)
        cond: (B, D) | None  — DiMSUM condition embedding (option C). Projected to
              (B, d_inner) and used to seed the conv initial state of BOTH scans.
        returns: (B, L, D)
        """
        x = x.contiguous()

        seed = None
        if cond is not None and self.d_cond is not None:
            seed = self.cond_proj(cond)  # (B, d_inner)

        xz = self.in_proj(x)  # (B, L, 2E)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, L, E)

        y_fwd = self.ssm_fwd(x_inner, seed=seed)

        x_bwd = torch.flip(x_inner, dims=[1])
        y_bwd = self.ssm_bwd(x_bwd, seed=seed)
        y_bwd = torch.flip(y_bwd, dims=[1])

        z_act = F.silu(z)
        y = y_fwd * z_act + y_bwd * z_act

        return self.proj_drop(self.out_proj(y))


# ── Official Mamba3 bidirectional vision wrapper ────────────────────────────

class VisionMamba3Bidirectional(nn.Module):
    """
    ViM-style bidirectional wrapper around the official upstream Mamba3 block.

    This replaces the internal Vim/Mamba-1 mixer with official Mamba3, then
    applies it in forward and backward sequence order.

    This is NOT the old Vim causal-conv selective-scan block.
    It is "vision bidirectionality + official Mamba3 internals".

    Requires an upstream Mamba version that exposes:
        from mamba_ssm import Mamba3

    Example install:
        MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
          git+https://github.com/state-spaces/mamba.git --no-build-isolation
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        headdim: int = 64,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        chunk_size: int = 16,
        is_outproj_norm: bool = False,
        bidirectional: bool = True,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        try:
            from mamba_ssm import Mamba3
        except ImportError as exc:
            raise ImportError(
                "VisionMamba3Bidirectional requires official Mamba3. "
                "Install the upstream state-spaces/mamba package from source."
            ) from exc

        self.bidirectional = bidirectional

        self.fwd = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            is_outproj_norm=is_outproj_norm,
        )

        if bidirectional:
            self.bwd = Mamba3(
                d_model=d_model,
                d_state=d_state,
                headdim=headdim,
                is_mimo=is_mimo,
                mimo_rank=mimo_rank,
                chunk_size=chunk_size,
                is_outproj_norm=is_outproj_norm,
            )
        else:
            self.bwd = None

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        returns: (B, L, D)
        """
        y_fwd = self.fwd(x)

        if self.bwd is None:
            return self.proj_drop(y_fwd)

        x_bwd = torch.flip(x, dims=[1])
        y_bwd = self.bwd(x_bwd)
        y_bwd = torch.flip(y_bwd, dims=[1])

        return self.proj_drop(y_fwd + y_bwd)


# ── JiT block ───────────────────────────────────────────────────────────────

class JiTBlock(nn.Module):
    """
    JiT block with adaLN-Zero conditioning and configurable sequence mixer.

    mixer_impl="vim":
        original BiMambaV2 / ViM-style Mamba-1 mixer.

    mixer_impl="mamba3":
        ViM-style bidirectional wrapper around official Mamba3.
    """
    def __init__(
        self,
        hidden_size,
        num_heads=None,
        mlp_ratio=4.0,
        d_state=16,
        d_conv=4,
        expand=2,
        attn_drop=0.0,
        proj_drop=0.0,
        mixer_impl="vim",
        cond_init="none",
        mamba3_bidirectional=True,
        mamba3_d_state=128,
        mamba3_headdim=64,
        mamba3_is_mimo=True,
        mamba3_mimo_rank=4,
        mamba3_chunk_size=16,
        mamba3_is_outproj_norm=False,
    ):
        super().__init__()

        # num_heads and attn_drop are kept for signature parity with attention baseline.
        del num_heads
        del attn_drop

        # cond_init (option C, DiMSUM Conditional Mamba):
        #   "none"        -> baseline, condition reaches the SSM only via adaLN
        #   "conv_state"  -> seed the conv initial state from c (BiMambaV2 only)
        self.cond_init = cond_init
        self.supports_cond = (mixer_impl == "vim" and cond_init != "none")
        if cond_init not in ("none", "conv_state"):
            raise ValueError(
                f"Unknown cond_init={cond_init!r}. Use 'none' or 'conv_state'."
            )
        if cond_init != "none" and mixer_impl != "vim":
            raise ValueError(
                "cond_init='conv_state' is only implemented for mixer_impl='vim'."
            )

        self.norm1 = RMSNorm(hidden_size, eps=1e-6)

        if mixer_impl == "vim":
            self.mixer = BiMambaV2(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                proj_drop=proj_drop,
                d_cond=hidden_size if cond_init == "conv_state" else None,
            )
        elif mixer_impl == "mamba3":
            self.mixer = VisionMamba3Bidirectional(
                d_model=hidden_size,
                d_state=mamba3_d_state,
                headdim=mamba3_headdim,
                is_mimo=mamba3_is_mimo,
                mimo_rank=mamba3_mimo_rank,
                chunk_size=mamba3_chunk_size,
                is_outproj_norm=mamba3_is_outproj_norm,
                bidirectional=mamba3_bidirectional,
                proj_drop=proj_drop,
            )
        else:
            raise ValueError(
                f"Unknown mixer_impl={mixer_impl!r}. Use 'vim' or 'mamba3'."
            )

        self.norm2 = RMSNorm(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        mixer_in = modulate(self.norm1(x), shift_msa, scale_msa)
        if self.supports_cond:
            mixer_out = self.mixer(mixer_in, cond=c)
        else:
            mixer_out = self.mixer(mixer_in)
        x = x + gate_msa.unsqueeze(1) * mixer_out

        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)

        return x


# ── JiT-ViM model ───────────────────────────────────────────────────────────

class JiTViM(nn.Module):
    """
    JiT with configurable ViM/Mamba3 mixer.

    mixer_impl="vim":
        Original BiMambaV2 mixer.

    mixer_impl="mamba3":
        ViM-style forward/backward wrapper around official Mamba3.
        This uses official Mamba3 internals, not the old Vim causal-conv block.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=384,
        depth=12,
        num_heads=6,  # kept for signature parity with attention baseline; unused
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=10,
        bottleneck_dim=128,

        # Original Vim / Mamba-1 knobs.
        d_state=16,
        d_conv=4,
        expand=2,

        # Mixer selection.
        mixer_impl="vim",

        # ── Option A: in-context class tokens (DiS-style prefix conditioning) ──
        #   Plumbing follows DiS (concat conditioning tokens at the front of the
        #   sequence, learnable positional slots, strip after the final block,
        #   BiMambaV2 mixer). Token CONTENT follows JiT Table 9 (K repeated class
        #   embeddings) so JiT-S and JiT-S2-ViM differ only in the mixer.
        #   in_context_len=0 -> off (byte-for-byte baseline).
        in_context_len: int = 0,
        in_context_start: int = 0,

        # ── Option C: DiMSUM Conditional-Mamba seeding ("none" | "conv_state") ──
        cond_init: str = "none",

        # Official Mamba3 knobs; used only when mixer_impl="mamba3".
        mamba3_bidirectional=True,
        mamba3_d_state=128,
        mamba3_headdim=64,
        mamba3_is_mimo=True,
        mamba3_mimo_rank=4,
        mamba3_chunk_size=16,
        mamba3_is_outproj_norm=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.cond_init = cond_init

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        self.x_embedder = BottleneckPatchEmbed(
            input_size,
            patch_size,
            in_channels,
            bottleneck_dim,
            hidden_size,
            bias=True,
        )

        # Fixed 2D sin-cos positional embedding. No attention RoPE here.
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size),
            requires_grad=False,
        )

        # Learnable positional slots for the in-context tokens (option A).
        # Matches jit.py exactly: zeros-initialised, learnable, only the patch
        # grid uses the fixed sin-cos table so baseline positions are unchanged.
        if in_context_len > 0:
            self.incontext_pos_embed = nn.Parameter(
                torch.zeros(1, in_context_len, hidden_size)
            )

        # Middle-half dropout slot, matching your original structure.
        lo, hi = depth // 4, depth // 4 * 3

        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                attn_drop=attn_drop if (lo <= i < hi) else 0.0,
                proj_drop=proj_drop if (lo <= i < hi) else 0.0,
                mixer_impl=mixer_impl,
                cond_init=cond_init,
                mamba3_bidirectional=mamba3_bidirectional,
                mamba3_d_state=mamba3_d_state,
                mamba3_headdim=mamba3_headdim,
                mamba3_is_mimo=mamba3_is_mimo,
                mamba3_mimo_rank=mamba3_mimo_rank,
                mamba3_chunk_size=mamba3_chunk_size,
                mamba3_is_outproj_norm=mamba3_is_outproj_norm,
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

        # Fixed sin-cos pos embed.
        pe = get_2d_sincos_pos_embed_vim(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5),
        )
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # Patch embed Xavier init on flattened conv weights.
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))

        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Embeddings.
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # adaLN-Zero.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # Zero output.
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
        x: (B, C, H, W)
        t: (B,)
        y: (B,)
        returns: (B, C, H, W)
        """
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        x = self.x_embedder(x)
        x = x + self.pos_embed

        for i, block in enumerate(self.blocks):
            # Option A: prepend JiT-style in-context class tokens ONCE, at
            # block in_context_start. Plumbed like DiS (concat at the front,
            # learnable positional slots); BiMambaV2 scans any L unchanged.
            if self.in_context_len > 0 and i == self.in_context_start:
                ctx = y_emb[:, None, :].expand(-1, self.in_context_len, -1)
                ctx = ctx + self.incontext_pos_embed
                x = torch.cat([ctx, x], dim=1)
            x = block(x, c)

        # Strip the in-context tokens once, after the final block (DiS: x[:, extras:]).
        if self.in_context_len > 0:
            x = x[:, self.in_context_len:, :]

        x = self.final_layer(x, c)

        return self.unpatchify(x)
