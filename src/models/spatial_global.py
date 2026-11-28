"""
src/models/spatial_global.py
============================
JiT-S2-Spatial-Global: JiT-S2-Spatial-Mamba (src/models/spatial_mamba.py: ONE
row-major scan, SASF state fusion, d_state 1) with JiT's persistent in-context
class tokens, connected to the image WITHOUT entering the scan:

  - image <-> image mixing is the stock scan + SASF, unchanged;
  - every interaction that involves a condition token is linear, undecayed
    and order-free, through the block's OWN x_proj / dt_proj (B, C, Delta).

Tokens (JiTSpatialMamba's in-context scaffolding, reused unchanged): M =
global_len copies of y_emb + a learned slot (init std 0.02), prepended once at
block global_start, carried & updated by every later block (same adaLN
modulation, residuals, LPU / FFN as the image tokens), stripped before the
final layer.

Per block (u: image features after in_proj -> DWConv -> SiLU; u^e: token
features after in_proj -> DWConv CENTRE TAP + bias -> SiLU; d_state N = 1):

    dt_r, B, C = split(x_proj u)          Delta = softplus(dt_proj dt_r + b)
    v = Delta * u                         (both groups, same weights)
    S = (1/M) sum_m v^e_m B^e_m           (B, d_inner, 1)  tokens -> memory
    G = (1/L) sum_s v_s   B_s             (B, d_inner, 1)  image  -> memory

    image   x  = scan(u)  (states, C = 1, D = None: the stock kernel, unchanged)
            h  = SASF(x + a_xe * S)                  <- memory BEFORE the fusion
            y  = C_t h_t + D u_t
    tokens  y^e_m = (a_ee * S + a_ex * G) C^e_m + D u^e_m     (not fused)

then LayerNorm [* SiLU(z)] + out_proj for all rows. Everything is computed from
the same block input (synchronous). S, G, v and the anchor run in fp32 with
autocast off and are cast back to the scan's dtype.

Why no kernel change is needed (the anchored recurrence). With a = a_xe and
Abar_t = exp(Delta_t A) elementwise per channel,

    x'_t = Abar_t * x'_{t-1} + (1 - Abar_t) * (a S) + v_t B_t,   x'_0 = a S

gives x'_t - a S = Abar_t (x'_{t-1} - a S) + v_t B_t: the stock state. So the
anchored state is x + a S, and SASF fuses it. SASF is linear and pads by
replication, so a spatially constant map comes out as kappa * map, kappa_c =
sum_j alpha_j sum(taps of kernel j, channel c) (27 at the ones-init): the
image reads C_t * kappa * a_xe * S, undecayed and the same at every cell.

global_readback=False is "write only": no G, no a_ex, the tokens read S only.
global_len=0 builds JiTSpatialMamba's baseline exactly.

Gains a_xe, a_ee, a_ex: (d_inner,), ones-init, flat 1-D (no weight decay).
Only blocks that carry tokens get them. The LAST block (whose token rows are
stripped right after it) holds a_xe only: its token outputs reach nothing, so
it computes neither G nor the token read, and its token rows are D u^e. Every
parameter therefore receives a non-zero gradient.

Token features differ from spatial_mamba's in-context class tokens, which are
the raw in_proj output (no conv, no SiLU): these get the centre tap + bias and
SiLU (spatial_mamba's token_conv convention for the aux tokens), so u^e lives on
the same distribution as the image u that v = Delta * u is built from.

Proven by tests/test_spatial_global_cpu.py; scripts/check_spatial_global_cuda.py
is the GPU check.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from src.models.spatial_mamba import StructureAwareSSM, JiTSpatialMamba


class SpatialGlobalSSM(StructureAwareSSM):
    """spatial_mamba's Structure-aware SSM (d_state 1) with undecayed global
    condition-token memories.

    Same parameters and init as StructureAwareSSM (it calls its __init__ with
    no tokens and every option off), plus the gains. x: (B, M + H*W, D) with the
    M condition tokens in front; M = 0 is StructureAwareSSM.forward exactly.

    emit_tokens=False is the last token-carrying block: only alpha_xe exists,
    G and the token read are skipped, and the token rows return the D term.
    """
    def __init__(self, d_model, d_conv=3, expand=1, gate=True, dilations=(1, 3, 5),
                 proj_drop=0.0, readback=True, emit_tokens=True):
        super().__init__(d_model=d_model, d_state=1, d_conv=d_conv, expand=expand,
                         gate=gate, dilations=dilations, proj_drop=proj_drop)
        self.readback = bool(readback)
        self.emit_tokens = bool(emit_tokens)
        # Ones-init, flat 1-D: no random numbers, no weight decay.
        self.alpha_xe = nn.Parameter(torch.ones(self.d_inner))
        if self.emit_tokens:
            self.alpha_ee = nn.Parameter(torch.ones(self.d_inner))
            if self.readback:
                self.alpha_ex = nn.Parameter(torch.ones(self.d_inner))

    def _ssm_params(self, u):
        """u: (B, d_inner, L) -> dt (raw, pre-bias) (B, d_inner, L), B, C (B, 1, L)."""
        x_dbl = torch.einsum("bdl,od->bol", u, self.x_proj_weight)
        dt_r, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, 1, 1], dim=1)
        dt = torch.einsum("brl,dr->bdl", dt_r, self.dt_projs_weight)
        return dt, B_ssm, C_ssm

    def terms(self, x, H, W):
        """Everything before the output norm, as a dict (the forward, the proofs
        and the diagnostic share it): z, u_e, u_g, dt, B_x, C_x, dt_e, B_e, C_e,
        ve, S, G, states (the kernel's, before the anchor), anchor (a_xe * S),
        h (fused anchored states), y_g (image rows), y_e (token rows, fp32)."""
        B, L_in, _ = x.shape
        d_inner, HW = self.d_inner, H * W
        M = L_in - HW
        assert M > 0, "terms() needs condition tokens; got L_in=%d for %dx%d" % (L_in, H, W)

        # 1. in_proj, then split: the tokens never enter the scan
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1) if self.gate else (xz, None)
        # 2. features: grid DWConv + SiLU (unchanged); tokens centre tap + SiLU
        k = self.d_conv // 2
        u_e = self.act(u[:, :M].transpose(1, 2) * self.conv2d.weight[:, 0, k, k].unsqueeze(-1)
                       + self.conv2d.bias.unsqueeze(-1))                      # (B, d_inner, M)
        u_g = u[:, M:].reshape(B, H, W, d_inner).permute(0, 3, 1, 2).contiguous()
        u_g = self.act(self.conv2d(u_g)).reshape(B, d_inner, HW)               # (B, d_inner, HW)
        u_e = u_e.to(u_g.dtype)
        # 3. dt / B / C, both groups through the same weights
        dt, B_x, C_x = self._ssm_params(u_g)
        dt_e, B_e, C_e = self._ssm_params(u_e)
        # 4. memories, fp32 with autocast off, from this block's features
        ct = torch.promote_types(u_g.dtype, torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            bias = self.dt_projs_bias.to(ct)[None, :, None]
            ve = F.softplus(dt_e.to(ct) + bias) * u_e.to(ct)                    # (B, d_inner, M)
            S = torch.einsum("bdm,bnm->bdn", ve, B_e.to(ct)) / M               # (B, d_inner, 1)
            G = None
            if self.readback and self.emit_tokens:
                vx = F.softplus(dt.to(ct) + bias) * u_g.to(ct)
                G = torch.einsum("bdl,bnl->bdn", vx, B_x.to(ct)) / HW          # (B, d_inner, 1)
        # 5. the stock scan, read out as states (C = 1, D = None), as the base
        A = -torch.exp(self.A_logs.to(ct))                                     # (d_inner, 1)
        B_scan = B_x.unsqueeze(1).contiguous()                                 # (B, 1, 1, HW)
        states = selective_scan_fn(
            u_g, dt.contiguous(), A, B_scan, torch.ones_like(B_scan), None,
            z=None,
            delta_bias=self.dt_projs_bias.to(ct),
            delta_softplus=True,
            return_last_state=False,
        )                                                                      # (B, d_inner, HW)
        # 6. anchor BEFORE SASF, fuse, observe
        with torch.autocast(device_type=x.device.type, enabled=False):
            anchor = self.alpha_xe.to(ct)[None, :, None] * S                    # (B, d_inner, 1)
            anchored = (states.to(ct) + anchor).to(states.dtype)
        h = self.state_fusion(anchored.view(B, d_inner, H, W)).reshape(B, d_inner, HW)
        y_g = h * C_x + u_g * self.Ds.view(-1, 1)                              # (B, d_inner, HW)
        # 7. token rows (not fused): (a_ee S + a_ex G) C^e + D u^e
        with torch.autocast(device_type=x.device.type, enabled=False):
            y_e = self.Ds.to(ct)[None, :, None] * u_e.to(ct)
            if self.emit_tokens:
                mem = self.alpha_ee.to(ct)[None, :, None] * S
                if self.readback:
                    mem = mem + self.alpha_ex.to(ct)[None, :, None] * G
                y_e = y_e + mem * C_e.to(ct)                                   # (B, d_inner, M)
        return dict(z=z, u_e=u_e, u_g=u_g, dt=dt, B_x=B_x, C_x=C_x, dt_e=dt_e, B_e=B_e,
                    C_e=C_e, ve=ve, S=S, G=G, states=states, anchor=anchor, h=h,
                    y_g=y_g, y_e=y_e)

    def forward(self, x, H, W, can=None):
        """x: (B, M + H*W, D) -> same shape. `can` is accepted (JiTSpatialBlock
        passes it) and must be None: CAN kernels are not combined with this."""
        assert can is None, "spatial_global does not support the CAN kernels"
        if x.shape[1] == H * W:
            return StructureAwareSSM.forward(self, x, H, W)
        t = self.terms(x, H, W)
        y = torch.cat([t["y_e"].to(t["y_g"].dtype), t["y_g"]], dim=-1)          # (B, d_inner, M+HW)
        y = self.out_norm(y.transpose(1, 2))
        if t["z"] is not None:
            y = y * F.silu(t["z"])
        return self.proj_drop(self.out_proj(y))


class JiTSpatialGlobal(JiTSpatialMamba):
    """JiT-S2-Spatial-Mamba with M global condition tokens outside the scan.

    Built as JiTSpatialMamba(in_context_len=M, in_context_positions=None,
    content "class", start global_start) -- the same slots, insertion, persist
    & update and strip -- then the mixers of blocks global_start .. depth-1 are
    replaced by SpatialGlobalSSM carrying the SAME weights (only the gains are
    new). The swap runs under fork_rng, so the RNG state after construction
    (and every shared weight) equals the Spatial prefix model's at the same
    seed. global_len=0 is JiTSpatialMamba's baseline, unchanged.

    spatial_mamba's other token / conditioning options are accepted only to be
    asserted off, so a config that sets one fails loudly instead of being
    silently ignored.
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
        gate=True,
        sasf_dilations=(1, 3, 5),
        lpu="none",
        grad_checkpoint=False,
        ffn="swiglu",
        norm="rmsnorm",
        derf_alpha_init=0.5,
        # ── global condition tokens ──
        global_len=8,                  # M; 0 = the JiTSpatialMamba baseline
        global_start=4,                # block the tokens join at
        global_readback=True,          # False = "write only": no G, no alpha_ex
        # ── spatial_mamba options: accepted ONLY to be asserted off ──
        in_context_len=0,
        in_context_positions=None,
        in_context_content="class",
        aux_tokens="none",
        dwconv_can=False,
        sasf_can=False,
        ssm_norm="none",
        feature_rope="none",
        sasf_state_kernels="shared",
    ):
        off = {
            "in_context_len": (in_context_len, 0),
            "in_context_positions": (in_context_positions, None),
            "in_context_content": (in_context_content, "class"),
            "aux_tokens": (aux_tokens, "none"),
            "dwconv_can": (bool(dwconv_can), False),
            "sasf_can": (bool(sasf_can), False),
            "ssm_norm": (ssm_norm, "none"),
            "feature_rope": (feature_rope, "none"),
            "sasf_state_kernels": (sasf_state_kernels, "shared"),
        }
        for key, (value, default) in off.items():
            assert value == default, (
                "model=spatial_global does not support %s=%r: the pairing with the "
                "global condition tokens is untested (the tokens are set by "
                "global_len / global_start)" % (key, value))
        assert d_state == 1, (
            "model=spatial_global is defined at d_state 1 (the memory is one value "
            "per channel); got d_state=%r" % (d_state,))
        assert isinstance(global_len, int) and not isinstance(global_len, bool) \
            and global_len >= 0, "global_len must be an int >= 0; got %r" % (global_len,)
        assert isinstance(global_readback, bool), (
            "global_readback must be true or false; got %r" % (global_readback,))
        if global_len > 0:
            assert isinstance(global_start, int) and 0 <= global_start < depth, (
                "global_start must be a block index in [0, %d); got %r"
                % (depth, global_start))

        super().__init__(
            input_size=input_size, patch_size=patch_size, in_channels=in_channels,
            hidden_size=hidden_size, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop,
            num_classes=num_classes, bottleneck_dim=bottleneck_dim,
            d_state=1, d_conv=d_conv, expand=expand, gate=gate,
            sasf_dilations=sasf_dilations, lpu=lpu, grad_checkpoint=grad_checkpoint,
            ffn=ffn, norm=norm, derf_alpha_init=derf_alpha_init,
            in_context_len=global_len,
            in_context_start=global_start if global_len > 0 else 0,
            in_context_content="class", in_context_positions=None,
        )
        self.global_len = global_len
        self.global_start = global_start
        self.global_readback = global_readback

        if global_len > 0:
            # StructureAwareSSM.__init__ draws random numbers (x_proj, dt init);
            # fork so the RNG after construction equals the prefix model's.
            with torch.random.fork_rng(devices=[]):
                for i in range(global_start, depth):
                    block = self.blocks[i]
                    old = block.mixer
                    new = SpatialGlobalSSM(
                        hidden_size, d_conv=d_conv, expand=expand, gate=gate,
                        dilations=sasf_dilations, proj_drop=old.proj_drop.p,
                        readback=global_readback, emit_tokens=(i < depth - 1))
                    new = new.to(device=old.in_proj.weight.device)
                    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
                    assert not unexpected and all(k.startswith("alpha_") for k in missing), \
                        (missing, unexpected)
                    block.mixer = new
