"""
src/models/vmamba2.py
=====================
JiT-S2-VMamba2: JiT with a Mamba-2 (SSD) SS2D mixer — VMamba's own Mamba-2
variant ("SS2Dm0", forward_type "m0_noz", classification/models/vmamba.py of
MzeroMiko/VMamba) in the JiT scaffold.

Everything around the SSM is SS2D as in src/models/vmamba.py: in_proj (no gate)
-> depthwise 3x3 conv + SiLU -> CrossScan (4 directions) -> SSM -> CrossMerge
-> LayerNorm -> out_proj. Only the SSM core changes, Mamba-1 -> Mamba-2:

                      SS2D (vmamba.py, Mamba-1)        SS2D-m2 (this file)
    step Delta        per channel (low-rank dt_proj)   per HEAD, straight from x_proj
    decay A           per (channel, state), S4D init   per HEAD (scalar)
    state per dir     d_inner x N                      R heads x (P x N) = same total
    kernel            selective_scan_fn (CUDA)         mamba_chunk_scan_combined (Triton)

Heads follow VMamba's m0: R = ceil(d_model / 16) per direction (VMamba reuses
dt_rank as the head count), headdim P = d_inner / R — 24 heads x 16 at D=384.
B and C stay one group per direction, shared by its R heads. The init is the
Mamba-2 module's (A ~ U[1, 16], Delta log-uniform in [dt_min, dt_max]), so Delta
starts in the same range as the Mamba-1 baseline; D is per channel (Mamba-2's
D_has_hdim=True), exactly the baseline's skip.

Exact relation to the Mamba-1 SS2D (tests/test_vmamba2_cpu.py check C): this
mixer IS SS2D with every channel's Delta tied to its head's (dt_proj selecting
one dt_rank coordinate per head) and A tied per head across channels and
states.

Scope (v1): adaLN-Zero baseline only, like spatial_mamba.py. The conditioning
arms of JiT-VMamba are not ported; src/models/vmamba.py is untouched. The
block/model scaffold mirrors JiTVMamba line for line (proven by
tests/test_vmamba2_cpu.py check G), so a vmamba2 vs vmamba run differs only in
the SSM core.
"""

import math
import torch
import torch.nn as nn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from src.primitives import (
    RMSNorm, get_2d_sincos_pos_embed,
    TimestepEmbedder, LabelEmbedder,
    BottleneckPatchEmbed, FinalLayer, modulate,
)
from src.models.vmamba import FFN_CLASSES, cross_scan, cross_merge


# ── SS2D with a Mamba-2 (SSD) core ───────────────────────────────────────────

class SS2DMamba2(nn.Module):
    """
    VMamba SS2Dm0 ("m0_noz") with the Mamba-2 module's init.

    Forward:  x: (B, H*W, D)  -> out: (B, H*W, D)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 1,
        nheads: int = None,           # per direction; None -> ceil(d_model / 16) (VMamba m0)
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_init_range=(1, 16),
        K: int = 4,
        chunk_size: int = 64,         # VMamba m0's; L = 64 on Tiny-IN is ONE chunk
        force_fp32: bool = False,     # VMamba m0's force_fp32: kernel inputs in fp32
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.nheads = nheads if nheads is not None else math.ceil(d_model / 16)
        assert self.d_inner % self.nheads == 0, (
            "d_inner=%d must split into nheads=%d heads" % (self.d_inner, self.nheads)
        )
        self.headdim = self.d_inner // self.nheads
        self.K = K
        self.chunk_size = chunk_size
        self.force_fp32 = force_fp32
        R = self.nheads

        # ── 1. Input projection (no gate branch) ──────────────────────
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # ── 2. Depthwise 2D conv + SiLU ──────────────────────────────
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv // 2,
            groups=self.d_inner, bias=True,
        )
        self.act = nn.SiLU()

        # ── 3. Per-direction SSM parameters, stacked over K ──────────
        # x_proj: d_inner -> (R Delta's, B(N), C(N)) per direction (VMamba m0:
        # the head Delta's come straight out of x_proj, no dt_proj).
        self.x_proj_weight = nn.Parameter(torch.empty(K, R + 2 * d_state, self.d_inner))
        nn.init.kaiming_uniform_(self.x_proj_weight, a=math.sqrt(5))

        # Delta bias per head: Mamba-2 init, softplus(dt_bias) log-uniform.
        dt = torch.exp(
            torch.rand(K, R) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))    # inverse softplus
        self.dt_bias._no_weight_decay = True

        # A per head: Mamba-2 init, A ~ U[A_init_range], stored as log.
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        self.A_logs = nn.Parameter(torch.empty(K, R).uniform_(*A_init_range).log())
        self.A_logs._no_weight_decay = True

        # D: per channel, per direction — the Mamba-1 SS2D skip, unchanged.
        self.Ds = nn.Parameter(torch.ones(K, self.d_inner))
        self.Ds._no_weight_decay = True

        # ── 4. Output norm + projection ──────────────────────────────
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        Bsz, L, _ = x.shape
        assert L == H * W, f"SS2DMamba2 got L={L} != H*W={H * W}"
        K, R, P, N = self.K, self.nheads, self.headdim, self.d_state
        d_inner = self.d_inner

        # ── 1-2. in_proj, conv + SiLU on the grid ────────────────────
        z = self.in_proj(x)                                           # (B, L, d_inner)
        z2d = z.view(Bsz, H, W, d_inner).permute(0, 3, 1, 2).contiguous()
        z2d = self.act(self.conv2d(z2d))                              # (B, d_inner, H, W)

        # ── 3. CrossScan + per-direction x_proj ──────────────────────
        xs = cross_scan(z2d)                                          # (B, K, d_inner, L)
        x_dbl = torch.einsum("bkdl,kod->bkol", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)            # (B, K, ., L)

        # ── 4. SSD layout: direction k's channels d -> head k*R + d // P ──
        # (the kernel maps head h to B/C group h // R = direction k)
        x_ssd = xs.permute(0, 3, 1, 2).reshape(Bsz, L, K * R, P)
        dt_ssd = dts.permute(0, 3, 1, 2).reshape(Bsz, L, K * R)
        B_ssd = Bs.permute(0, 3, 1, 2).contiguous()                   # (B, L, K, N)
        C_ssd = Cs.permute(0, 3, 1, 2).contiguous()
        if self.force_fp32:
            x_ssd, dt_ssd, B_ssd, C_ssd = (t.float() for t in (x_ssd, dt_ssd, B_ssd, C_ssd))

        A = -torch.exp(self.A_logs.float()).view(K * R)               # (K*R,)
        D = self.Ds.float().view(K * R, P)                            # per channel
        y = mamba_chunk_scan_combined(
            x_ssd, dt_ssd, A, B_ssd, C_ssd, self.chunk_size,
            D=D, dt_bias=self.dt_bias.float().view(K * R), dt_softplus=True,
        )                                                             # (B, L, K*R, P)

        # ── 5. CrossMerge, LayerNorm, out_proj ───────────────────────
        ys = y.reshape(Bsz, L, K, d_inner).permute(0, 2, 3, 1)        # (B, K, d_inner, L)
        out = cross_merge(ys, H, W).transpose(1, 2)                   # (B, L, d_inner)
        out = self.out_norm(out)
        return self.proj_drop(self.out_proj(out))


# ── JiT block with the Mamba-2 SS2D mixer ────────────────────────────────────

class JiTVMamba2Block(nn.Module):
    """JiTBlock (adaLN-Zero, RMSNorm, FFN) with the Mamba-2 SS2D mixer."""
    def __init__(self, hidden_size, num_heads=None, mlp_ratio=4.0,
                 d_state=16, d_conv=3, expand=1, K=4, nheads=None,
                 chunk_size=64, force_fp32=False,
                 attn_drop=0.0, proj_drop=0.0, ffn="swiglu"):
        super().__init__()
        # num_heads kept for signature parity with attention baseline; unused.
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.mixer = SS2DMamba2(
            d_model=hidden_size, d_state=d_state, d_conv=d_conv, expand=expand,
            nheads=nheads, K=K, chunk_size=chunk_size, force_fp32=force_fp32,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FFN_CLASSES[ffn](hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.ffn_spatial = ffn != "swiglu"   # conv FFNs need the (H, W) grid
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, H, W):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.mixer(
            modulate(self.norm1(x), shift_msa, scale_msa), H, W)
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h, H, W) if self.ffn_spatial else self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x


# ── JiT-VMamba2 model ────────────────────────────────────────────────────────

class JiTVMamba2(nn.Module):
    """JiT with the Mamba-2 SS2D mixer (VMamba m0).

    Class conditioning via adaLN-Zero only. Mirrors JiTVMamba's baseline
    scaffold exactly; the knobs are the mixer's (d_state, nheads, chunk_size,
    force_fp32) and the FFN variant.
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
        # Mamba-2 SS2D knobs
        d_state=16,
        d_conv=3,
        expand=1,
        K=4,
        #   SSM heads per direction; None -> ceil(hidden_size / 16) (VMamba m0)
        nheads=None,
        #   SSD chunk length (VMamba m0: 64)
        chunk_size=64,
        #   cast the kernel's inputs to fp32 (VMamba m0's force_fp32); the
        #   fallback for GPUs whose Triton cannot do bf16 tl.dot (e.g. T4)
        force_fp32: bool = False,
        # FFN variant (same choices as JiTVMamba)
        ffn: str = "swiglu",
    ):
        super().__init__()
        assert ffn in FFN_CLASSES, (
            "Unknown ffn=%r (use %s)" % (ffn, " | ".join(FFN_CLASSES))
        )
        self.ffn = ffn
        self.force_fp32 = force_fp32
        self.in_channels  = in_channels
        self.out_channels = in_channels
        self.patch_size   = patch_size
        self.hidden_size  = hidden_size
        self.input_size   = input_size
        self.num_classes  = num_classes

        # Spatial grid size (used by the mixer and the conv FFNs)
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
            JiTVMamba2Block(
                hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
                d_state=d_state, d_conv=d_conv, expand=expand, K=K,
                nheads=nheads, chunk_size=chunk_size, force_fp32=force_fp32,
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
