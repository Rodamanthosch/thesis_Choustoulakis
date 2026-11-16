"""
src/models/spatial_mamba.py
===========================
JiT-S2-Spatial-Mamba: JiT with Spatial-Mamba's Structure-aware SSM mixer
(Xiao et al., "Spatial-Mamba: Effective Visual State Space Models via
Structure-aware State Fusion", ICLR 2025; EdwardChasel/Spatial-Mamba).

The mixer replaces SS2D's 4-direction CrossScan + CrossMerge with ONE row-major
scan whose hidden states are fused over their 2-D neighbourhood (SASF) BEFORE
the observation equation (paper Eq. 2-3):

    x_t = A_t x_{t-1} + B_t u_t                                  (scan, N = 1)
    h_t = sum_d alpha_d sum_{i,j in {-d,0,d}} k^d_ij x_{t+iw+j}  (SASF)
    y_t = C_t h_t + D u_t

Hidden states on the STOCK kernel. The official repo forks the selective-scan
CUDA kernel to return the states x_t. With d_state N = 1 (the paper's setting)
that is unnecessary: the stock kernel computes y_t = sum_n C_t[n] x_t[n] + D u_t,
so calling it with C = 1 and D = None returns x_t EXACTLY. SASF and the
observation y = h * C + u * D then run in PyTorch, as in the official ssm().
Verified against a reference scan in tests/test_spatial_mamba_cpu.py.

d_state > 1 (an extension; the paper defines N = 1 only): the scan runs over
N * d_inner single-state channels, which is the Mamba-1 N-state recurrence
exactly, and SASF fuses every state map with kernels shared by a channel's N
maps or one set per state (sasf_state_kernels). Verified against a naive
N-state transcription in tests/test_spatial_dstate16_cpu.py.

Conditioning: adaLN-Zero, plus (optionally) in-context class tokens placed
INSIDE the scan (in_context_len / in_context_positions): token e is inserted
just before grid cell (row_e, col_e) of the row-major scan instead of all at
its head, so with SASF every image token can sit within a few scan steps of
one (see JiTSpatialMamba). Off by default and then byte-for-byte the baseline
(tests/test_spatial_incontext_cpu.py check A). Optionally SF-Mamba's
auxiliary patch swapping (aux_tokens: two mean-initialised tokens at the head
and tail of the scan, swapped between blocks; tests/test_spatial_auxswap_cpu.py),
and CAN-style conditional kernels for the mixer DWConv and SASF (dwconv_can /
sasf_can, our adaptation of Cai et al., CVPR 2024; tests/test_spatial_can_cpu.py).
State-init, SSC and adaln_cond
are not ported; src/models/vmamba.py is untouched. The block/model scaffold
mirrors JiTVMamba line for line (proven by tests/test_spatial_mamba_cpu.py
check J), so a JiT-Spatial vs JiT-VMamba run differs only in the mixer and the
optional LPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from src.primitives import (
    RMSNorm, get_2d_sincos_pos_embed,
    TimestepEmbedder, LabelEmbedder,
    BottleneckPatchEmbed, FinalLayer, modulate, _dwconv_tokens,
    NORMS, convert_rmsnorm_to_derf, ssm_rms_norm,
)
from src.feature_rope import apply_feature_rope, validate_feature_rope
from src.models.vmamba import FFN_CLASSES

LPU_MODES = ("none", "zero", "paper")
STATE_KERNEL_MODES = ("shared", "per_state")   # SASF kernels at d_state > 1
SPATIAL_SSM_NORMS = ("none", "dt")             # Jamba's dt_r norm; B/C not ported
AUX_TOKEN_MODES = ("none", "swap", "noswap")   # SF-Mamba aux tokens (+ no-swap control)


# ── CAN-style conditional kernels (dwconv_can / sasf_can) ────────────────────
#
# After CAN (Cai et al., "Condition-Aware Neural Network for Controlled Image
# Generation", CVPR 2024, arXiv 2404.01143): a layer's weight becomes W + W_c,
# W_c generated from the condition, applied per sample by folding the batch
# into the channels of one grouped conv. THIS IS OUR ADAPTATION, not CAN's
# code (never released): the paper does not specify the generator (it only
# prices a "single linear layer"), so ours is one bias-free Linear on SiLU(c),
# zero-init. CAN gives each depthwise conv its OWN generator and shares one
# generator (with per-layer static weights) only across its output
# projections; we use that shared device for both of our sites -- the mixer
# DWConv and the SASF kernels -- so dK = G(c) is the same in every block and
# each block keeps its own static K_l.

class CANGenerator(nn.Module):
    """dK = W SiLU(c), reshaped to out_shape per sample. No bias, W zero-init.

    W is a raw Parameter, not an nn.Linear, so the model's xavier pass skips
    it and building it draws no random numbers. Computed with autocast off, in
    at least fp32 (a float64 c stays float64).
    """
    def __init__(self, in_dim, out_shape):
        super().__init__()
        self.out_shape = tuple(int(s) for s in out_shape)
        self.weight = nn.Parameter(torch.zeros(math.prod(self.out_shape), in_dim))

    def forward(self, c):
        wd = torch.promote_types(c.dtype, torch.float32)
        with torch.autocast(device_type=c.device.type, enabled=False):
            dk = F.linear(F.silu(c.to(wd)), self.weight.to(wd))
        return dk.view(c.shape[0], *self.out_shape)


def can_dwconv(x, weight, dk, bias=None, padding=0, dilation=1):
    """Per-sample depthwise conv: sample b convolves with weight + dk[b].

    x: (B, C, H, W), weight: (C, 1, k, k), dk: (B, C, k, k). CAN's
    batch-to-channel trick: one conv over B*C groups. The kernel sum and the
    conv run in at least fp32 with autocast off; the output is cast back to
    x's dtype, so the block downstream sees the static path's dtype.
    """
    B, C, H, W = x.shape
    k = weight.shape[-2:]
    wd = torch.promote_types(x.dtype, torch.float32)
    with torch.autocast(device_type=x.device.type, enabled=False):
        w = (weight.to(wd).view(1, C, *k) + dk.to(wd)).reshape(B * C, 1, *k)
        b = None if bias is None else bias.to(wd).repeat(B)
        y = F.conv2d(x.to(wd).reshape(1, B * C, H, W), w, b,
                     padding=padding, dilation=dilation, groups=B * C)
    return y.view(B, C, y.shape[-2], y.shape[-1]).to(x.dtype)


# ── Structure-aware state fusion (SASF) ──────────────────────────────────────

class StateFusion(nn.Module):
    """SASF (paper Eq. 3), the training branch of the official StateFusion.

    One depthwise 3x3 kernel per dilation d, applied to the replicate-padded
    state map, combined with learnable weights alpha. Kernels and alpha are
    ones-init as in the official code. With dilations=(1, 3, 5) this is the
    official module op for op (tests/test_spatial_mamba_cpu.py check D).

    The official eval branch (all kernels merged into one 11x11 kernel) is not
    ported: it is a speed trick that zero-pads where training replicate-pads,
    and caches the merged weight once (stale after an EMA swap or a resume).
    Here train and eval compute the same function.
    """
    def __init__(self, dim, dilations=(1, 3, 5)):
        super().__init__()
        self.dim = dim
        self.dilations = tuple(int(d) for d in dilations)
        assert self.dilations and all(d >= 1 for d in self.dilations), self.dilations
        self.kernels = nn.ParameterList(
            [nn.Parameter(torch.ones(dim, 1, 3, 3)) for _ in self.dilations])
        self.alpha = nn.Parameter(torch.ones(len(self.dilations)))

    def forward(self, h, dk=None):
        """h: (B, dim, H, W) hidden states  ->  fused states, same shape.

        dk: optional (B, n_dilations, dim, 3, 3) conditional kernels (sasf_can):
        dilation j uses kernels[j] + dk[:, j] per sample (can_dwconv, fp32);
        alpha is unchanged. None is the original path.
        """
        out = 0
        for j, (a, k, d) in enumerate(zip(self.alpha, self.kernels, self.dilations)):
            hp = F.pad(h, (d, d, d, d), mode="replicate")
            if dk is None:
                out = out + a * F.conv2d(hp, k.to(hp.dtype), padding=0,
                                         dilation=d, groups=self.dim)
            else:
                out = out + a * can_dwconv(hp, k, dk[:, j], dilation=d)
        return out


# ── Structure-aware SSM (the mixer) ──────────────────────────────────────────

class StructureAwareSSM(nn.Module):
    """Spatial-Mamba's Structure-aware SSM (paper Fig. 4, right).

    Forward:  x: (B, H*W, D)  ->  (B, H*W, D)

      in_proj  D -> 2*d_inner, split (u, z)     [gate=False: D -> d_inner]
      u -> DWConv3x3 + SiLU on the H x W grid -> row-major sequence
      x_proj -> (dt, B, C); dt_proj
      x_t  = selective_scan(u, dt, A, B, C=1, D=None)    (states, see module doc)
      h    = SASF(x)
      y    = h * C + u * D
      y    = LayerNorm(y) [* SiLU(z)] -> out_proj

    gate=True is the official structure. gate=False drops the z half of
    in_proj and the multiplicative branch (VMamba's "v05_noz" treatment).
    Parameters and init follow the official StructureAwareSSM (dt_init="random",
    S4-real A, D = 1, x_proj / dt_proj stored as raw Parameters so the model's
    Linear init pass leaves them alone).

    d_state > 1 (not in the paper, which only defines N = 1). Each channel c
    then has N states x_t[c, n] with the Mamba-1 recurrence (per-channel dt,
    per-(channel, state) A, B_t[n] shared by all channels). C = 1 can only read
    out ONE state per channel, so the scan runs over N * d_inner single-state
    channels -- copy n of channel c sees the same u and dt, its own A[c, n] and,
    as B-group n, the same B_t[n] -- which IS the N-state recurrence, exactly.
    SASF then fuses all N * d_inner state maps, and the observation contracts
    them with C:  y_t[c] = sum_n C_t[n] h_t[c, n] + D[c] u_t[c].
    state_kernels picks how SASF treats the N maps of one channel:
      "shared"    -> one set of kernels per channel, applied to all N maps
                     (SASF params as at N = 1)
      "per_state" -> one set of kernels per (channel, state) map
    Both fuse every map, so they cost the same compute. At N = 1 the two are
    the same module, and the forward is the original d_state = 1 code path.

    In-context tokens (in_context_positions, sorted flat grid indices p_e):
    x may then carry P = len(p) condition tokens in front of the H*W grid,
    x: (B, P + H*W, D). Token e is scanned just before grid cell p_e (all p_e
    = 0 is the DiM head prefix). As in SS2D, the tokens share in_proj, skip
    the depthwise conv (no grid cell) and get their own dt/B/C from x_proj.
    SASF fuses the GRID states only; a token's own output is C h + D u of its
    unfused state. Output keeps the input layout [tokens, grid]. With P = 0
    (no tokens, or a block before in_context_start) the forward is the
    original code path. p_e = H*W scans token e after the last cell (the
    SF-Mamba tail token), and token_conv=True gives the tokens the DWConv's
    centre tap + SiLU instead of skipping it (aux tokens; see JiTSpatialMamba).
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 1,
        d_conv: int = 3,
        expand: int = 1,
        gate: bool = True,
        dilations=(1, 3, 5),
        state_kernels: str = "shared",
        in_context_positions=(),
        token_conv: bool = False,
        dt_rank: int = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        proj_drop: float = 0.0,
        ssm_norm: str = "none",
        feature_rope: str = "none",   # "none" | "1d" | "2d_axial"  (src/feature_rope.py)
        feature_rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert d_state >= 1, d_state
        assert ssm_norm in SPATIAL_SSM_NORMS, (
            "Unknown ssm_norm=%r for spatial_mamba (use %s; 'bcdt' is not "
            "ported: at d_state 1 an RMSNorm over one B / C value is its sign)"
            % (ssm_norm, " | ".join(SPATIAL_SSM_NORMS)))
        assert state_kernels in STATE_KERNEL_MODES, (
            "Unknown state_kernels=%r (use %s)"
            % (state_kernels, " | ".join(STATE_KERNEL_MODES))
        )
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)
        self.gate = bool(gate)
        self.state_kernels = state_kernels
        self.in_context_positions = tuple(sorted(int(p) for p in in_context_positions))
        assert all(p >= 0 for p in self.in_context_positions), self.in_context_positions
        # token_conv: the tokens get the DWConv's centre tap (+ bias) and SiLU,
        # as an off-grid 1x1 image (SF-Mamba's aux tokens pass the conv + SiLU).
        # False (class tokens): raw in_proj output, no conv, no SiLU.
        self.token_conv = bool(token_conv)
        self._scan_order_cache = {}

        # ── 1. Input projection (u, and z when gated) ─────────────────
        self.in_proj = nn.Linear(
            d_model, (2 if self.gate else 1) * self.d_inner, bias=False)

        # ── 2. Depthwise 2D conv + SiLU ──────────────────────────────
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv // 2,
            groups=self.d_inner, bias=True,
        )
        self.act = nn.SiLU()

        # ── 3. SSM parameters (one direction) ─────────────────────────
        # x_proj: d_inner -> (dt_rank + 2*d_state); nn.Linear's default init.
        self.x_proj_weight = nn.Parameter(
            torch.empty(self.dt_rank + 2 * d_state, self.d_inner))
        nn.init.kaiming_uniform_(self.x_proj_weight, a=math.sqrt(5))

        # dt_proj: dt_rank -> d_inner. Weight uniform(+-dt_rank^-0.5), bias =
        # softplus^{-1}(dt) with dt log-uniform in [dt_min, dt_max].
        self.dt_projs_weight = nn.Parameter(torch.empty(self.d_inner, self.dt_rank))
        self.dt_projs_bias   = nn.Parameter(torch.empty(self.d_inner))
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_projs_weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_projs_bias.copy_(inv_dt)
        self.dt_projs_bias._no_reinit = True

        # A_log: S4-real init A = -[1..N] (N = 1 -> A = -1), stored as log.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(A))                 # (d_inner, d_state)
        self.A_logs._no_weight_decay = True

        # D: skip scalar per channel
        self.Ds = nn.Parameter(torch.ones(self.d_inner))
        self.Ds._no_weight_decay = True

        # ── 4. Structure-aware state fusion ──────────────────────────
        # "per_state" at d_state > 1: one kernel set per (channel, state) map,
        # channel index n * d_inner + c (the scan's state-major layout).
        n_maps = d_state if (state_kernels == "per_state" and d_state > 1) else 1
        self.state_fusion = StateFusion(self.d_inner * n_maps, dilations)

        # ── 5. Output norm (+ gate) + projection ─────────────────────
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        # ── 6. Jamba's dt_r RMSNorm (off by default) ─────────────────
        # "dt": dt_r <- RMSNorm(dt_r) over its dt_rank channels, learnable
        # weight init 1, BEFORE dt_projs -- JambaMambaMixer's dt_layernorm
        # (src/primitives.py::ssm_rms_norm). B and C are left alone. Ones-init
        # draws no random numbers, so every other weight is the "none" model's.
        self.ssm_norm = ssm_norm
        if ssm_norm == "dt":
            self.dt_norm_weight = nn.Parameter(torch.ones(self.dt_rank))

        # ── 7. Feature-level RoPE (off by default; no params, no buffers) ──
        self.feature_rope = feature_rope
        self.feature_rope_theta = validate_feature_rope(
            feature_rope, feature_rope_theta, self.d_inner)

    def _scan_order(self, HW: int, device):
        """Index maps between the layout [P tokens, HW grid] and the scan order.

        order: scan position -> layout index (token e right before grid cell
        p_e; p_e == HW means after the last cell, the SF-Mamba tail token);
        inv: layout index -> scan position. seq = cat[..., order] and
        cat = seq[..., inv].
        """
        key = (HW, str(device))
        if key not in self._scan_order_cache:
            pos, P = self.in_context_positions, len(self.in_context_positions)
            assert not pos or pos[-1] <= HW, (
                "in-context position %d is outside the %d-cell grid" % (pos[-1], HW))
            order, e = [], 0
            for g in range(HW):
                while e < P and pos[e] == g:
                    order.append(e)
                    e += 1
                order.append(P + g)
            while e < P:                       # pos[e] == HW: after the last cell
                order.append(e)
                e += 1
            order = torch.tensor(order, device=device)
            self._scan_order_cache[key] = (order, torch.argsort(order))
        return self._scan_order_cache[key]

    def forward(self, x: torch.Tensor, H: int, W: int, can=None) -> torch.Tensor:
        """can: None, or (dk_dw, dk_sasf) from JiTSpatialMamba.can_deltas, either
        entry None: per-sample conditional kernels for the DWConv
        (B, d_inner, k, k) and SASF (B, n_dilations, d_inner, 3, 3). Only at
        d_state 1 without tokens (asserted by the model)."""
        B, L, _ = x.shape
        d_inner, d_state = self.d_inner, self.d_state
        dk_dw, dk_sasf = can if can is not None else (None, None)
        HW = H * W
        P = L - HW                     # in-context tokens in front of the grid
        assert P in (0, len(self.in_context_positions)), (
            "StructureAwareSSM got %d tokens for a %dx%d grid; expected the grid "
            "plus 0 or %d in-context tokens"
            % (L, H, W, len(self.in_context_positions))
        )

        # ── 1. in_proj (in-context tokens share it, as in DiM / SS2D) ─
        xz = self.in_proj(x)
        if self.gate:
            u, z = xz.chunk(2, dim=-1)                          # (B, L, d_inner) each
        else:
            u, z = xz, None
        if P:
            u_extra = u[:, :P].transpose(1, 2)                  # (B, d_inner, P): no grid cell,
            u = u[:, P:]                                        # so no conv (as SS2D)
            if self.token_conv:                                 # centre tap + SiLU (aux tokens)
                k = self.d_conv // 2
                u_extra = self.act(
                    u_extra * self.conv2d.weight[:, 0, k, k].unsqueeze(-1)
                    + self.conv2d.bias.unsqueeze(-1))

        # ── 2. depthwise conv + SiLU on the grid, back to row-major ──
        u = u.view(B, H, W, d_inner).permute(0, 3, 1, 2).contiguous()
        if dk_dw is None:
            u = self.act(self.conv2d(u))                        # (B, d_inner, H, W)
        else:   # CAN: per-sample kernel conv2d.weight + dk_dw[b] (fp32)
            u = self.act(can_dwconv(u, self.conv2d.weight, dk_dw, self.conv2d.bias,
                                    padding=self.d_conv // 2))

        # ── 2a. Feature RoPE on the grid only: u feeds x_proj (dt, B, C), the
        # scan and the D skip term, all rotated; the gate branch z does not
        # pass through here, and the tokens have not joined the scan yet.
        if self.feature_rope != "none":
            u = apply_feature_rope(u, self.feature_rope, self.feature_rope_theta)
        u = u.reshape(B, d_inner, HW)                           # (B, d_inner, HW)

        # ── 2b. in-context tokens into the scan, before their grid cells ─
        if P:
            order, inv = self._scan_order(HW, u.device)
            u = torch.cat([u_extra.to(u.dtype), u], dim=-1)[..., order]   # scan order

        # ── 3. data-dependent dt, B, C ───────────────────────────────
        x_dbl = torch.einsum("bdl,od->bol", u, self.x_proj_weight)
        dt_r, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, d_state, d_state], dim=1)     # C_ssm: (B, 1, L)
        if self.ssm_norm == "dt":   # Jamba's dt_layernorm, before dt_proj (§6)
            dt_r = ssm_rms_norm(dt_r, self.dt_norm_weight[None, :, None], dim=1)
        dt = torch.einsum("brl,dr->bdl", dt_r, self.dt_projs_weight)

        if d_state == 1:
            # ── 4. ONE row-major scan, read out as states (C = 1, D = None) ─
            A = -torch.exp(self.A_logs.float())                 # (d_inner, 1)
            B_scan = B_ssm.unsqueeze(1).contiguous()            # (B, 1, 1, L)
            C_one = torch.ones_like(B_scan)
            states = selective_scan_fn(
                u, dt.contiguous(), A, B_scan, C_one, None,
                z=None,
                delta_bias=self.dt_projs_bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )                                                   # (B, d_inner, L) = x_t
            if P:   # back to [tokens, grid]; the tokens' states are not fused
                states, u, C_ssm = (s[..., inv] for s in (states, u, C_ssm))
                s_extra, states = states[..., :P], states[..., P:].contiguous()

            # ── 5. SASF over the 2-D grid of states ──────────────────
            h = self.state_fusion(states.view(B, d_inner, H, W), dk_sasf).reshape(B, d_inner, HW)
            if P:
                h = torch.cat([s_extra, h], dim=-1)             # (B, d_inner, P + HW)

            # ── 6. observation y = C h + D u (official ssm()) ────────
            y = h * C_ssm + u * self.Ds.view(-1, 1)             # (B, d_inner, L)
        else:
            # ── 4'. N-state scan as N * d_inner single-state channels ─
            # Channel n * d_inner + c is state n of channel c: the same u and
            # dt (repeated), A[c, n], and B-group n (the scan's channel groups
            # are contiguous blocks of d_inner) carrying B_t[n].
            N = d_state
            A = -torch.exp(self.A_logs.float()).t().reshape(N * d_inner, 1)
            B_scan = B_ssm.unsqueeze(2).contiguous()            # (B, N, 1, L)
            C_one = torch.ones_like(B_scan)
            states = selective_scan_fn(
                u.repeat(1, N, 1), dt.repeat(1, N, 1).contiguous(), A,
                B_scan, C_one, None,
                z=None,
                delta_bias=self.dt_projs_bias.float().repeat(N),
                delta_softplus=True,
                return_last_state=False,
            )                                                   # (B, N*d_inner, L)
            if P:   # back to [tokens, grid]; the tokens' states are not fused
                states, u, C_ssm = (s[..., inv] for s in (states, u, C_ssm))
                s_extra = states[..., :P].reshape(B, N, d_inner, P)
                states = states[..., P:].contiguous()

            # ── 5'. SASF on every state map ──────────────────────────
            if self.state_kernels == "per_state":
                h = self.state_fusion(states.view(B, N * d_inner, H, W))
            else:   # "shared": the channel's kernels on each of its N maps
                h = self.state_fusion(states.reshape(B * N, d_inner, H, W))
            h = h.reshape(B, N, d_inner, HW)
            if P:
                h = torch.cat([s_extra, h], dim=-1)             # (B, N, d_inner, P + HW)

            # ── 6'. observation y[c] = sum_n C[n] h[c, n] + D[c] u[c] ─
            y = (h * C_ssm.unsqueeze(2)).sum(dim=1) + u * self.Ds.view(-1, 1)

        # ── 7. LayerNorm [* SiLU(z)] + out_proj ─────────────────────
        y = self.out_norm(y.transpose(1, 2))                    # (B, L, d_inner)
        if z is not None:
            y = y * F.silu(z)
        return self.proj_drop(self.out_proj(y))


# ── JiT block with the Structure-aware SSM mixer ─────────────────────────────

class JiTSpatialBlock(nn.Module):
    """JiTBlock (adaLN-Zero, RMSNorm, FFN) with the Structure-aware SSM mixer.

    lpu adds Spatial-Mamba's Local Perception Units (CMT): an ungated residual
    depthwise 3x3 conv over the token grid before the mixer branch (cpe1) and
    before the FFN branch (cpe2), as in the official SpatialMambaBlock.
      "none"  -> no LPU (the scaffold is then exactly JiTBlock's)
      "zero"  -> LPU with zero-init weight and bias: every block is still the
                 identity at init (adaLN-Zero discipline)
      "paper" -> LPU with PyTorch's default conv init (official code)
    In-context tokens (in front of the grid) have no grid cell: the LPU, like
    the conv FFNs, gives them only the kernel's centre tap.
    """
    def __init__(self, hidden_size, num_heads=None, mlp_ratio=4.0,
                 d_state=1, d_conv=3, expand=1, gate=True,
                 sasf_dilations=(1, 3, 5), lpu="none", sasf_state_kernels="shared",
                 grad_checkpoint=False, in_context_positions=(), token_conv=False,
                 attn_drop=0.0, proj_drop=0.0, ffn="swiglu", ssm_norm="none",
                 feature_rope="none", feature_rope_theta=10000.0):
        super().__init__()
        # num_heads kept for signature parity with attention baseline; unused.
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.mixer = StructureAwareSSM(
            d_model=hidden_size, d_state=d_state, d_conv=d_conv, expand=expand,
            gate=gate, dilations=sasf_dilations, state_kernels=sasf_state_kernels,
            in_context_positions=in_context_positions, token_conv=token_conv,
            proj_drop=proj_drop, ssm_norm=ssm_norm,
            feature_rope=feature_rope, feature_rope_theta=feature_rope_theta,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FFN_CLASSES[ffn](hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.ffn_spatial = ffn != "swiglu"   # conv FFNs need the (H, W) grid
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        assert lpu in LPU_MODES, lpu
        self.lpu = lpu
        # Recompute the mixer in the backward pass instead of storing its
        # activations (same maths; trades compute for memory -- needed at
        # d_state 16, where each mixer holds 16x the state tensors).
        self.grad_checkpoint = bool(grad_checkpoint)
        if lpu != "none":
            self.cpe1 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1,
                                  groups=hidden_size)
            self.cpe2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1,
                                  groups=hidden_size)

    @staticmethod
    def _grid_conv(conv, x, H, W):
        """Depthwise conv over the token grid of x: (B, P + H*W, D) -> same
        shape; the P leading in-context tokens see only the centre tap."""
        return _dwconv_tokens(conv, x, H, W)

    def forward(self, x, c, H, W, can=None):
        """can: the model's shared CAN kernel deltas (dk_dw, dk_sasf) or None."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        if self.lpu != "none":
            x = x + self._grid_conv(self.cpe1, x, H, W)
        u = modulate(self.norm1(x), shift_msa, scale_msa)
        if self.grad_checkpoint and torch.is_grad_enabled():
            mix = torch.utils.checkpoint.checkpoint(self.mixer, u, H, W, can,
                                                    use_reentrant=False)
        else:
            mix = self.mixer(u, H, W, can)
        x = x + gate_msa.unsqueeze(1) * mix
        if self.lpu != "none":
            x = x + self._grid_conv(self.cpe2, x, H, W)
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h, H, W) if self.ffn_spatial else self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x


# ── JiT-Spatial-Mamba model ──────────────────────────────────────────────────

class JiTSpatialMamba(nn.Module):
    """JiT with the Structure-aware SSM (Spatial-Mamba) mixer.

    Class conditioning via adaLN-Zero, optionally plus in-context class tokens
    (no SSM-level conditioning). Mirrors JiTVMamba's baseline scaffold exactly;
    the knobs are the mixer's (d_state, expand, gate, sasf_dilations,
    sasf_state_kernels), the LPU, the FFN variant and the in-context tokens.

    In-context class tokens (in_context_len > 0), JiTVMamba's persist & update
    scheme: in_context_len tokens [y_emb + slot_e] are prepended ONCE at block
    in_context_start, updated by every later block and stripped after the
    last. in_context_positions places them in the ONE row-major scan: token e
    is scanned just before grid cell (row_e, col_e). None puts them all before
    cell (0, 0), i.e. the DiM head prefix. Spreading them works because SASF
    reads states up to max(dilations) rows/columns away in every direction:
    on the 8x8 grid with dilations (1, 2, 3), cells (0,3), (3,3), (7,3) put
    every image token within 2 scan steps of a class token (one prefix
    token: 37), +3 tokens of scan length.

    Auxiliary tokens (aux_tokens != "none"), after SF-Mamba's auxiliary
    patch swapping (Yoshimura et al., ICML 2026, arXiv 2603.16423; official
    sony/SF-Mamba HybridStage.forward). At block aux_tokens_start two tokens,
    both the mean of the grid tokens (no parameters, no slot), join the
    sequence: a HEAD token scanned before cell (0, 0) and a TAIL token scanned
    after the last cell (layout [head, tail, grid]). The tail's state has read
    the whole grid; "swap" exchanges the two tokens after every block but the
    last, so that summary is scanned FIRST in the next block and reaches every
    cell through the recurrence (future -> past flow from one forward scan).
    "noswap" is the control: the same tokens, never swapped. Both are
    stripped after the last block. Deviations from SF-Mamba: the tokens get
    the mixer DWConv's centre tap + SiLU (no grid neighbours; SF-Mamba's 1-D
    conv also sees the adjacent patch), their states are not fused by SASF,
    and they start at a chosen block, not at the first block of a stage.

    CAN-style conditional kernels (dwconv_can / sasf_can; see CANGenerator --
    our adaptation of Cai et al., CVPR 2024, not their code). One generator
    per site, shared by ALL blocks: dK = W SiLU(c) with c the adaLN vector
    (t + y), W zero-init, no bias, computed once per forward. Block l uses
    K_l + dK per sample -- the mixer DWConv (bias unchanged) and every SASF
    dilation's kernel (alpha unchanged) -- in fp32. No per-block conditional
    parameter. W_dw: 9 * d_inner x D, W_sasf: 9 * n_dilations * d_inner x D
    (+5,308,416 on the Tiny-IN config). set_can_zero(dwconv=, sasf=) drops a
    site's dK on the same weights (eval ablation): that site then runs the
    static code path exactly. Only with the plain setup: no tokens, d_state 1,
    no feature RoPE / ssm_norm / Derf / conv FFN (asserted).
    """
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
        # Structure-aware SSM knobs
        d_state=1,
        d_conv=3,
        expand=1,
        #   gate=True  -> official Fig. 4 structure: LN(y) * SiLU(z)
        #   gate=False -> no multiplicative branch (VMamba "v05_noz" style)
        gate: bool = True,
        #   SASF dilations: (1, 3, 5) is the paper's; (1, 2, 3) fits an 8x8 grid
        sasf_dilations=(1, 3, 5),
        #   Local Perception Units before the mixer and the FFN: none|zero|paper
        lpu: str = "none",
        #   d_state > 1 only: SASF kernels "shared" by a channel's N state maps,
        #   or one set "per_state" (see StructureAwareSSM)
        sasf_state_kernels: str = "shared",
        #   recompute each mixer in the backward pass (activation checkpointing):
        #   same model and maths, less memory, more compute
        grad_checkpoint: bool = False,
        # FFN variant (same choices as JiTVMamba)
        ffn: str = "swiglu",
        # In-context class tokens (off by default; see the class docstring)
        #   in_context_len=0 -> adaLN-Zero only, byte-for-byte the baseline
        in_context_len: int = 0,
        in_context_start: int = 0,
        in_context_content: str = "class",        # only "class" is ported
        #   [[row, col], ...], one per token: scanned just before that grid
        #   cell; None -> all before (0, 0) (DiM head prefix)
        in_context_positions=None,
        # SF-Mamba auxiliary tokens (off by default; see the class docstring)
        #   "none" -> the baseline, byte-for-byte | "swap" | "noswap" (control)
        aux_tokens: str = "none",
        aux_tokens_start: int = 0,
        # Norm layer (same choices as JiTVMamba): "rmsnorm" | "derf" in
        # place of norm1, norm2, norm_final (the mixer's LN(y) stays)
        norm: str = "rmsnorm",
        derf_alpha_init: float = 0.5,
        # Jamba's RMSNorm on the SSM parameters (as JiTVMamba's ssm_norm):
        #   "none" -> the baseline, byte-for-byte | "dt" -> dt_r only, before
        #   dt_proj. "bcdt" is not ported (B, C are scalars at d_state 1).
        ssm_norm: str = "none",
        # Feature-level RoPE (as JiTVMamba's feature_rope; src/feature_rope.py):
        #   "none" -> the baseline, byte-for-byte | "1d" | "2d_axial"
        feature_rope: str = "none",
        feature_rope_theta: float = 10000.0,
        # CAN-style conditional kernels (off by default; see the class docstring)
        #   shared generator per site -> K_l + G(c) in every block
        dwconv_can: bool = False,
        sasf_can: bool = False,
    ):
        super().__init__()
        validate_feature_rope(feature_rope, feature_rope_theta, int(expand * hidden_size))
        self.dwconv_can, self.sasf_can = bool(dwconv_can), bool(sasf_can)
        if self.dwconv_can or self.sasf_can:
            for bad, why in ((in_context_len > 0, "in-context class tokens"),
                             (aux_tokens != "none", "aux_tokens"),
                             (d_state > 1, "d_state > 1"),
                             (feature_rope != "none", "feature_rope"),
                             (ssm_norm != "none", "ssm_norm"),
                             (norm == "derf", "norm: derf"),
                             (ffn != "swiglu", "conv FFNs")):
                assert not bad, "dwconv_can / sasf_can are not tested with %s" % why
        assert ffn in FFN_CLASSES, (
            "Unknown ffn=%r (use %s)" % (ffn, " | ".join(FFN_CLASSES))
        )
        assert norm in NORMS, (
            "Unknown norm=%r (use %s)" % (norm, " | ".join(NORMS))
        )
        assert ssm_norm in SPATIAL_SSM_NORMS, (
            "Unknown ssm_norm=%r for spatial_mamba (use %s)"
            % (ssm_norm, " | ".join(SPATIAL_SSM_NORMS))
        )
        self.ssm_norm = ssm_norm
        self.norm_type = norm
        assert lpu in LPU_MODES, (
            "Unknown lpu=%r (use %s)" % (lpu, " | ".join(LPU_MODES))
        )
        assert sasf_state_kernels in STATE_KERNEL_MODES, (
            "Unknown sasf_state_kernels=%r (use %s)"
            % (sasf_state_kernels, " | ".join(STATE_KERNEL_MODES))
        )
        self.ffn = ffn
        self.lpu = lpu
        self.gate = bool(gate)
        self.d_state = d_state
        self.sasf_state_kernels = sasf_state_kernels
        self.grad_checkpoint = bool(grad_checkpoint)
        self.sasf_dilations = tuple(sasf_dilations)
        self.in_channels  = in_channels
        self.out_channels = in_channels
        self.patch_size   = patch_size
        self.hidden_size  = hidden_size
        self.input_size   = input_size
        self.num_classes  = num_classes

        # Spatial grid size (used by the mixer, the LPU and the conv FFNs)
        self.grid_size = input_size // patch_size

        # In-context class tokens: validate and resolve their scan positions.
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.in_context_content = in_context_content
        cells = ()
        if in_context_len > 0:
            assert in_context_content == "class", (
                "Unknown in_context_content=%r: only 'class' is ported to "
                "spatial_mamba" % (in_context_content,)
            )
            assert 0 <= in_context_start < depth, (
                "in_context_start=%d must index a block (depth %d)"
                % (in_context_start, depth)
            )
            if in_context_positions is None:
                cells = ((0, 0),) * in_context_len
            else:
                cells = tuple(tuple(int(v) for v in p) for p in in_context_positions)
                assert len(cells) == in_context_len and all(len(p) == 2 for p in cells), (
                    "in_context_positions needs one [row, col] per token "
                    "(in_context_len=%d); got %r" % (in_context_len, in_context_positions)
                )
                assert all(0 <= r < self.grid_size and 0 <= c < self.grid_size
                           for r, c in cells), (
                    "in_context_positions must lie on the %dx%d grid; got %r"
                    % (self.grid_size, self.grid_size, in_context_positions)
                )
                cells = tuple(sorted(cells))      # token e = e-th in scan order
            # Learnable positional slot per token (DiM additional_embed), as JiTVMamba.
            self.incontext_pos_embed = nn.Parameter(
                torch.zeros(1, in_context_len, hidden_size)
            )
            nn.init.normal_(self.incontext_pos_embed, std=0.02)   # LTH14/JiT init
        else:
            assert not in_context_positions, (
                "in_context_positions places in-context tokens; set in_context_len > 0"
            )
        self.in_context_positions = cells
        flat_positions = tuple(r * self.grid_size + c for r, c in cells)

        # SF-Mamba auxiliary tokens: head before cell 0, tail after the last.
        assert aux_tokens in AUX_TOKEN_MODES, (
            "Unknown aux_tokens=%r (use %s)" % (aux_tokens, " | ".join(AUX_TOKEN_MODES))
        )
        self.aux_tokens = aux_tokens
        self.aux_tokens_start = aux_tokens_start
        if aux_tokens != "none":
            assert in_context_len == 0, (
                "aux_tokens and in-context class tokens are mutually exclusive"
            )
            assert 0 <= aux_tokens_start < depth, (
                "aux_tokens_start=%d must index a block (depth %d)"
                % (aux_tokens_start, depth)
            )
            flat_positions = (0, self.grid_size * self.grid_size)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )

        # Fixed 2D sin-cos pos embed (no RoPE)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size),
                                       requires_grad=False)

        # Blocks; middle-half dropout slot (as JiTVMamba)
        lo, hi = depth // 4, depth // 4 * 3
        self.blocks = nn.ModuleList([
            JiTSpatialBlock(
                hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
                d_state=d_state, d_conv=d_conv, expand=expand, gate=gate,
                sasf_dilations=sasf_dilations, lpu=lpu,
                sasf_state_kernels=sasf_state_kernels, grad_checkpoint=grad_checkpoint,
                in_context_positions=flat_positions,
                token_conv=aux_tokens != "none",
                attn_drop=attn_drop if (lo <= i < hi) else 0.0,
                proj_drop=proj_drop if (lo <= i < hi) else 0.0,
                ffn=ffn, ssm_norm=ssm_norm,
                feature_rope=feature_rope, feature_rope_theta=feature_rope_theta,
            )
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # CAN generators: zero Parameters, so no random numbers are drawn and
        # every other weight is the flags-off model's at the same seed.
        d_inner = int(expand * hidden_size)
        if self.dwconv_can:
            self.can_dwconv = CANGenerator(hidden_size, (d_inner, d_conv, d_conv))
        if self.sasf_can:
            self.can_sasf = CANGenerator(hidden_size, (len(sasf_dilations), d_inner, 3, 3))
        self.can_zero = {"dwconv": False, "sasf": False}
        self.initialize_weights()
        if norm == "derf":
            convert_rmsnorm_to_derf(self, derf_alpha_init)

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

        # ConvGLU: TransNeXt's depthwise init, as in JiTVMamba.
        if self.ffn == "convglu":
            for block in self.blocks:
                conv = block.mlp.dwconv
                fan_out = conv.kernel_size[0] * conv.kernel_size[1] \
                    * conv.out_channels // conv.groups
                nn.init.normal_(conv.weight, 0, math.sqrt(2.0 / fan_out))
                nn.init.zeros_(conv.bias)

        # LPU "zero": the ungated residual convs start at 0, so every block is
        # the identity at init (as adaLN-Zero makes the gated branches).
        if self.lpu == "zero":
            for block in self.blocks:
                for conv in (block.cpe1, block.cpe2):
                    nn.init.zeros_(conv.weight)
                    nn.init.zeros_(conv.bias)

        # adaLN-Zero
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # Zero output
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def set_can_zero(self, dwconv: bool = False, sasf: bool = False):
        """Eval ablation: drop dK at a site (static path) on the same weights."""
        assert not dwconv or self.dwconv_can, "dwconv_can is off in this model"
        assert not sasf or self.sasf_can, "sasf_can is off in this model"
        self.can_zero = {"dwconv": bool(dwconv), "sasf": bool(sasf)}

    def can_deltas(self, c):
        """(dk_dw, dk_sasf) for conditioning vector c, None where a site is off
        or zeroed; None when both are. Called once per forward."""
        dk_dw = self.can_dwconv(c) if self.dwconv_can and not self.can_zero["dwconv"] else None
        dk_sasf = self.can_sasf(c) if self.sasf_can and not self.can_zero["sasf"] else None
        return None if dk_dw is None and dk_sasf is None else (dk_dw, dk_sasf)

    def unpatchify(self, x):
        p = self.patch_size
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    def forward(self, x, t, y):
        """x: (B, C, H, W) | t: (B,) | y: (B,)  -> (B, C, H, W)"""
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb
        can = self.can_deltas(c)        # CAN kernel deltas, once for all blocks

        x = self.x_embedder(x)
        x = x + self.pos_embed

        H = W = self.grid_size
        for i, block in enumerate(self.blocks):
            # Prepend the class tokens ONCE, then let them persist & update
            # (each mixer scans them before their grid cells; see the docstring).
            if self.in_context_len > 0 and i == self.in_context_start:
                ctx = torch.stack([y_emb] * self.in_context_len, dim=1)
                x = torch.cat([ctx + self.incontext_pos_embed, x], dim=1)
            # SF-Mamba aux tokens: [head, tail] = mean of the grid tokens, once.
            if self.aux_tokens != "none" and i == self.aux_tokens_start:
                m = x.mean(dim=1, keepdim=True)
                x = torch.cat([m, m, x], dim=1)
            x = block(x, c, H, W, can)
            # Swap head and tail after every block but the last (out of place).
            if (self.aux_tokens == "swap"
                    and self.aux_tokens_start <= i < len(self.blocks) - 1):
                x = torch.cat([x[:, 1:2], x[:, 0:1], x[:, 2:]], dim=1)

        # Strip the class tokens once, after the final block.
        if self.in_context_len > 0:
            x = x[:, self.in_context_len:, :]
        if self.aux_tokens != "none":
            x = x[:, 2:, :]

        x = self.final_layer(x, c)
        return self.unpatchify(x)
