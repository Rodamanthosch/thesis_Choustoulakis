"""
src/models/vmamba_sasf.py
=========================
JiT-S2-VMamba + SASF: the JiT-S2-VMamba model (src/models/vmamba.py) at
d_state = 1, with Spatial-Mamba's structure-aware state fusion (SASF) added
inside each of SS2D's four scan directions -- the "VMamba + SASF" experiment
of Spatial-Mamba's Appendix F (Table 7), in the JiT scaffold.

Per direction k, the SSM states are fused over their 2-D neighbourhood before
that direction's observation, and the four directions are then merged:

    x^k_t = A^k_t x^k_{t-1} + B^k_t u^k_t                 (scan k, d_state 1)
    h^k   = SASF_k( x^k on the H x W grid )               (own kernels per k)
    y     = sum_k ( C^k * h^k + D^k * u )                 (= CrossMerge)

States on the stock kernel, as in src/models/spatial_mamba.py: with one state
per channel, selective_scan_fn with C = 1 and D = None returns x^k_t exactly.
Each direction's states (and its C) are un-permuted back to row-major grid
order, fused there, read out with C and D, and summed over directions -- which
is exactly CrossMerge of the per-direction outputs (cross_merge sums the
un-permuted directions).

Everything else is vmamba.py, unchanged:
  - SS2DSASF subclasses SS2D (same init, same parameter names) and adds only
    `state_fusion`;
  - JiTVMambaSASF subclasses JiTVMamba(d_state=1), builds it as usual, then
    swaps each block's SS2D for an SS2DSASF carrying the SAME weights.
So with identity fusion the model IS JiT-VMamba at d_state 1 (proven by
tests/test_vmamba_sasf_cpu.py). No LPU, no gate (VMamba's v05_noz block).
Baseline conditioning only: none of vmamba.py's conditioning arms are enabled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from src.models.vmamba import SS2D, JiTVMamba, cross_scan


def cross_unscan(ys: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Undo each direction's cross_scan reordering, WITHOUT summing.

    ys: (B, K=4, C, L) in each direction's scan order -> (B, 4, C, L), every
    direction back in row-major grid order. cross_merge(ys) == this.sum(1).
    """
    B, K, C, L = ys.shape
    s1 = ys[:, 0]
    s2 = ys[:, 1].flip(-1)
    s3 = ys[:, 2].reshape(B, C, W, H).transpose(2, 3).reshape(B, C, L)
    s4 = ys[:, 3].flip(-1).reshape(B, C, W, H).transpose(2, 3).reshape(B, C, L)
    return torch.stack([s1, s2, s3, s4], dim=1)


class DirectionalStateFusion(nn.Module):
    """K independent SASF modules, one per scan direction, run as one conv.

    Direction k owns its depthwise 3x3 kernels (channels k*dim .. (k+1)*dim)
    and its own alpha[k]; with K = 1 this is spatial_mamba.StateFusion. Kernels
    and alpha are ones-init and the padding is replicate, as in the official
    Spatial-Mamba StateFusion (training branch).
    """
    def __init__(self, K, dim, dilations=(1, 3, 5)):
        super().__init__()
        self.K, self.dim = K, dim
        self.dilations = tuple(int(d) for d in dilations)
        assert self.dilations and all(d >= 1 for d in self.dilations), self.dilations
        self.kernels = nn.ParameterList(
            [nn.Parameter(torch.ones(K * dim, 1, 3, 3)) for _ in self.dilations])
        self.alpha = nn.Parameter(torch.ones(K, len(self.dilations)))

    def forward(self, h):
        """h: (B, K*dim, H, W) -> fused, same shape."""
        out = 0
        for i, (k, d) in enumerate(zip(self.kernels, self.dilations)):
            hp = F.pad(h, (d, d, d, d), mode="replicate")
            conv = F.conv2d(hp, k.to(hp.dtype), padding=0, dilation=d, groups=self.K * self.dim)
            a = self.alpha[:, i].repeat_interleave(self.dim)[None, :, None, None]
            out = out + a.to(conv.dtype) * conv
        return out


class SS2DSASF(SS2D):
    """vmamba.py's SS2D at d_state = 1 with per-direction state fusion.

    Same parameters and init as SS2D (it calls SS2D.__init__), plus
    `state_fusion`. Baseline path only: no in-context tokens, state-init, SSC.
    """
    def __init__(self, d_model, d_conv=3, expand=1, K=4, dilations=(1, 3, 5),
                 proj_drop=0.0):
        super().__init__(d_model=d_model, d_state=1, d_conv=d_conv, expand=expand,
                         K=K, proj_drop=proj_drop)
        assert K == 4, "cross_unscan assumes VMamba's 4 CrossScan directions"
        self.state_fusion = DirectionalStateFusion(K, self.d_inner, dilations)

    def forward(self, x, H, W, cond=None):
        """x: (B, H*W, D) -> (B, H*W, D). `cond` is accepted (JiTBlock passes
        it) and ignored: this mixer has no SSM-level conditioning."""
        B, L, _ = x.shape
        K, d_inner = self.K, self.d_inner
        assert L == H * W, (
            "SS2DSASF got %d tokens for a %dx%d grid; in-context prefix tokens "
            "are not supported" % (L, H, W)
        )

        # ── 1-2. in_proj, depthwise conv + SiLU (SS2D) ───────────────
        z = self.in_proj(x)
        z2d = z.view(B, H, W, d_inner).permute(0, 3, 1, 2).contiguous()
        z2d = self.act(self.conv2d(z2d))                          # (B, d_inner, H, W)

        # ── 3. CrossScan + per-direction x_proj / dt_proj (SS2D) ─────
        xs = cross_scan(z2d)                                      # (B, K, d_inner, L)
        x_dbl = torch.einsum("bkdl,kod->bkol", xs, self.x_proj_weight)
        dt_r, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, 1, 1], dim=2)
        dt = torch.einsum("bkrl,kdr->bkdl", dt_r, self.dt_projs_weight)

        # ── 4. one stacked scan, read out as states (C = 1, D = None) ─
        A = -torch.exp(self.A_logs.float()).view(K * d_inner, 1)
        B_scan = B_ssm.contiguous().view(B, K, 1, L)
        states = selective_scan_fn(
            xs.reshape(B, K * d_inner, L),
            dt.contiguous().view(B, K * d_inner, L),
            A, B_scan, torch.ones_like(B_scan), None,
            z=None,
            delta_bias=self.dt_projs_bias.float().view(K * d_inner),
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, d_inner, L)                                  # x^k_t, scan order

        # ── 5. back to the grid, SASF per direction ──────────────────
        h = cross_unscan(states, H, W)                            # (B, K, d_inner, L)
        h = self.state_fusion(h.reshape(B, K * d_inner, H, W)).view(B, K, d_inner, L)
        C_grid = cross_unscan(C_ssm, H, W)                        # (B, K, 1, L)

        # ── 6. y = sum_k C^k h^k + D^k u  (== CrossMerge) ────────────
        u = z2d.reshape(B, 1, d_inner, L)                         # row-major, every k
        y = (h * C_grid + u * self.Ds.view(1, K, d_inner, 1)).sum(dim=1)

        # ── 7. LayerNorm + out_proj (SS2D) ───────────────────────────
        out = self.out_norm(y.transpose(1, 2))
        return self.proj_drop(self.out_proj(out))


class JiTVMambaSASF(JiTVMamba):
    """JiT-S2-VMamba at d_state = 1 with SASF in every SS2D direction.

    Built exactly as JiTVMamba(d_state=1) (same init, same RNG stream), then
    each block's SS2D is replaced by an SS2DSASF loaded with that SS2D's
    weights; only `state_fusion` (ones-init) is new. Every conditioning arm of
    JiTVMamba stays off.
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
        d_state=1,
        d_conv=3,
        expand=1,
        K=4,
        #   SASF dilations: (1, 3, 5) is the paper's; (1, 2, 3) fits an 8x8 grid
        sasf_dilations=(1, 3, 5),
        ffn: str = "swiglu",
    ):
        assert d_state == 1, (
            "JiTVMambaSASF reads the SS2D states through C = 1, which needs "
            "d_state = 1 (VMamba-T's setting); got %d" % d_state
        )
        super().__init__(
            input_size=input_size, patch_size=patch_size, in_channels=in_channels,
            hidden_size=hidden_size, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop,
            num_classes=num_classes, bottleneck_dim=bottleneck_dim,
            d_state=1, d_conv=d_conv, expand=expand, K=K, ffn=ffn,
        )
        self.sasf_dilations = tuple(sasf_dilations)
        for block in self.blocks:
            old = block.mixer
            new = SS2DSASF(hidden_size, d_conv=d_conv, expand=expand, K=K,
                           dilations=sasf_dilations, proj_drop=old.proj_drop.p)
            missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
            assert not unexpected and all(k.startswith("state_fusion.") for k in missing), \
                (missing, unexpected)
            block.mixer = new
