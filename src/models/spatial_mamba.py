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

Scope (v1): adaLN-Zero baseline only. None of JiT-VMamba's conditioning arms
(in-context prefix, state-init, SSC, adaln_cond) are ported; src/models/vmamba.py
is untouched. The block/model scaffold mirrors JiTVMamba line for line (proven
by tests/test_spatial_mamba_cpu.py check J), so a JiT-Spatial vs JiT-VMamba run
differs only in the mixer and the optional LPU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from src.primitives import (
    RMSNorm, get_2d_sincos_pos_embed,
    TimestepEmbedder, LabelEmbedder,
    BottleneckPatchEmbed, FinalLayer, modulate,
)
from src.models.vmamba import FFN_CLASSES

LPU_MODES = ("none", "zero", "paper")


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

    def forward(self, h):
        """h: (B, dim, H, W) hidden states  ->  fused states, same shape."""
        out = 0
        for a, k, d in zip(self.alpha, self.kernels, self.dilations):
            hp = F.pad(h, (d, d, d, d), mode="replicate")
            out = out + a * F.conv2d(hp, k.to(hp.dtype), padding=0,
                                     dilation=d, groups=self.dim)
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
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 1,
        d_conv: int = 3,
        expand: int = 1,
        gate: bool = True,
        dilations=(1, 3, 5),
        dt_rank: int = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        # C = 1 returns the states only when there is a single state per channel
        # (y_t = sum_n C_t[n] x_t[n]); it is also the paper's setting.
        assert d_state == 1, (
            "StructureAwareSSM reads the scan states through C = 1, which is "
            "exact only for d_state = 1 (the paper's setting); got %d" % d_state
        )
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)
        self.gate = bool(gate)

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
        self.state_fusion = StateFusion(self.d_inner, dilations)

        # ── 5. Output norm (+ gate) + projection ─────────────────────
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, _ = x.shape
        d_inner, d_state = self.d_inner, self.d_state
        assert L == H * W, (
            "StructureAwareSSM got %d tokens for a %dx%d grid; in-context "
            "prefix tokens are not supported by the spatial mixer" % (L, H, W)
        )

        # ── 1. in_proj ───────────────────────────────────────────────
        xz = self.in_proj(x)
        if self.gate:
            u, z = xz.chunk(2, dim=-1)                          # (B, L, d_inner) each
        else:
            u, z = xz, None

        # ── 2. depthwise conv + SiLU on the grid, back to row-major ──
        u = u.view(B, H, W, d_inner).permute(0, 3, 1, 2).contiguous()
        u = self.act(self.conv2d(u)).reshape(B, d_inner, L)     # (B, d_inner, L)

        # ── 3. data-dependent dt, B, C ───────────────────────────────
        x_dbl = torch.einsum("bdl,od->bol", u, self.x_proj_weight)
        dt_r, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, d_state, d_state], dim=1)     # C_ssm: (B, 1, L)
        dt = torch.einsum("brl,dr->bdl", dt_r, self.dt_projs_weight)

        # ── 4. ONE row-major scan, read out as states (C = 1, D = None) ─
        A = -torch.exp(self.A_logs.float())                     # (d_inner, 1)
        B_scan = B_ssm.unsqueeze(1).contiguous()                # (B, 1, 1, L)
        C_one = torch.ones_like(B_scan)
        states = selective_scan_fn(
            u, dt.contiguous(), A, B_scan, C_one, None,
            z=None,
            delta_bias=self.dt_projs_bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )                                                       # (B, d_inner, L) = x_t

        # ── 5. SASF over the 2-D grid of states ──────────────────────
        h = self.state_fusion(states.view(B, d_inner, H, W)).reshape(B, d_inner, L)

        # ── 6. observation y = C h + D u (official ssm()) ────────────
        y = h * C_ssm + u * self.Ds.view(-1, 1)                 # (B, d_inner, L)

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
    """
    def __init__(self, hidden_size, num_heads=None, mlp_ratio=4.0,
                 d_state=1, d_conv=3, expand=1, gate=True,
                 sasf_dilations=(1, 3, 5), lpu="none",
                 attn_drop=0.0, proj_drop=0.0, ffn="swiglu"):
        super().__init__()
        # num_heads kept for signature parity with attention baseline; unused.
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.mixer = StructureAwareSSM(
            d_model=hidden_size, d_state=d_state, d_conv=d_conv, expand=expand,
            gate=gate, dilations=sasf_dilations, proj_drop=proj_drop,
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
        if lpu != "none":
            self.cpe1 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1,
                                  groups=hidden_size)
            self.cpe2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1,
                                  groups=hidden_size)

    @staticmethod
    def _grid_conv(conv, x, H, W):
        """Depthwise conv over the token grid of x: (B, H*W, D) -> same shape."""
        B, L, D = x.shape
        g = x.transpose(1, 2).reshape(B, D, H, W)
        return conv(g).flatten(2).transpose(1, 2)

    def forward(self, x, c, H, W):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        if self.lpu != "none":
            x = x + self._grid_conv(self.cpe1, x, H, W)
        x = x + gate_msa.unsqueeze(1) * self.mixer(
            modulate(self.norm1(x), shift_msa, scale_msa), H, W)
        if self.lpu != "none":
            x = x + self._grid_conv(self.cpe2, x, H, W)
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h, H, W) if self.ffn_spatial else self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x


# ── JiT-Spatial-Mamba model ──────────────────────────────────────────────────

class JiTSpatialMamba(nn.Module):
    """JiT with the Structure-aware SSM (Spatial-Mamba) mixer.

    Class conditioning via adaLN-Zero only (no in-context tokens, no SSM-level
    conditioning). Mirrors JiTVMamba's baseline scaffold exactly; the knobs are
    the mixer's (expand, gate, sasf_dilations), the LPU, and the FFN variant.
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
        # FFN variant (same choices as JiTVMamba)
        ffn: str = "swiglu",
    ):
        super().__init__()
        assert ffn in FFN_CLASSES, (
            "Unknown ffn=%r (use %s)" % (ffn, " | ".join(FFN_CLASSES))
        )
        assert lpu in LPU_MODES, (
            "Unknown lpu=%r (use %s)" % (lpu, " | ".join(LPU_MODES))
        )
        self.ffn = ffn
        self.lpu = lpu
        self.gate = bool(gate)
        self.sasf_dilations = tuple(sasf_dilations)
        self.in_channels  = in_channels
        self.out_channels = in_channels
        self.patch_size   = patch_size
        self.hidden_size  = hidden_size
        self.input_size   = input_size
        self.num_classes  = num_classes

        # Spatial grid size (used by the mixer, the LPU and the conv FFNs)
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

        # Blocks; middle-half dropout slot (as JiTVMamba)
        lo, hi = depth // 4, depth // 4 * 3
        self.blocks = nn.ModuleList([
            JiTSpatialBlock(
                hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
                d_state=d_state, d_conv=d_conv, expand=expand, gate=gate,
                sasf_dilations=sasf_dilations, lpu=lpu,
                attn_drop=attn_drop if (lo <= i < hi) else 0.0,
                proj_drop=proj_drop if (lo <= i < hi) else 0.0,
                ffn=ffn,
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

        x = self.x_embedder(x)
        x = x + self.pos_embed

        H = W = self.grid_size
        for block in self.blocks:
            x = block(x, c, H, W)

        x = self.final_layer(x, c)
        return self.unpatchify(x)
