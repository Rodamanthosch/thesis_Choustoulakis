"""
src/models/vmamba_global.py
===========================
JiT-S2-VMamba-Global: JiT-S2-VMamba (src/models/vmamba.py: SS2D, 4 directions,
Mamba-1, expand 1, no gate) with JiT's persistent in-context class tokens,
connected to the image WITHOUT entering the selective scan:

  - image <-> image mixing is the stock scan, unchanged;
  - every interaction that involves a condition token is linear, undecayed
    and order-free, through the block's OWN x_proj / dt_proj (B, C, Delta).

Tokens (as JiT, reusing JiTVMamba's scaffolding unchanged): M = global_len
copies of y_emb + a learned slot (init std 0.02), prepended once at block
global_start, carried & updated by every later block (same adaLN modulation,
residuals, FFN as the image tokens), stripped before the final layer.

Per block and direction k (u: image features after in_proj -> DWConv -> SiLU;
u^e: token features after in_proj -> DWConv CENTRE TAP + bias -> SiLU):

    dt_r, B, C = split(x_proj[k] u)          Delta = softplus(dt_proj[k] dt_r + b[k])
    v = Delta * u                            (both groups, same weights)
    S_k = (1/M) sum_m v^e_m B^e_m^T          (B, d_inner, N)  tokens -> memory
    G_k = (1/L) sum_s v_s   B_s^T            (B, d_inner, N)  image  -> memory

    image  y_k   = scan_k(u)  +  a_xe * (S_k C_t)                 -> cross_merge
           (softmax: a_xe * sum_m softmax_m(C_t . B^e_m / sqrt N) v^e_m, no 1/M)
    tokens y^e_m = sum_k [ (a_ee * S_k + a_ex * G_k) C^e_m + D_k * u^e_m ]

then out_norm + out_proj for all tokens. Everything is computed from the same
block input (synchronous). S, G, v and the reads run in fp32 with autocast
off and are cast back to the scan's dtype.

Why no kernel change is needed (the anchored recurrence). With a = a_xe and
Abar_t = exp(Delta_t A) elementwise per (channel, state),

    h_t = Abar_t * h_{t-1} + (1 - Abar_t) * (a S) + v_t B_t^T,   h_0 = a S

gives g_t = h_t - a S = Abar_t * g_{t-1} + v_t B_t^T, g_0 = 0: the stock scan
state. So y_t = C_t h_t + D u_t = scan_t + (a S) C_t. Only elementwise decay
is used, so this holds for Mamba-1's free A[d, n] (no tying). Not for softmax.

Switches: global_readback=False ("write only": no G, no a_ex, tokens read S
only); global_kernel_xe="softmax" (image <- token term only; token rows stay
linear). global_len=0 builds JiTVMamba's baseline exactly.

Gains a_xe, a_ee, a_ex: (d_inner,), ones-init, flat 1-D (no weight decay).
Only blocks that carry tokens get them, and the LAST block (whose token rows
are stripped right after it) gets a_xe only: its token outputs reach nothing,
so it computes neither G nor the token read (nor S under softmax), and its
token rows are just sum_k D_k u^e. Every parameter therefore receives a
gradient (DDP runs without find_unused_parameters).

Token features differ from vmamba.py's prefix arm, whose tokens skip the
DWConv AND the SiLU (raw in_proj output). Here they get the centre tap + bias
and SiLU (Spatial-Mamba's token_conv convention), so u^e lives on the same
distribution as the image u that v = Delta * u is built from.

Proven by tests/test_vmamba_global_cpu.py; scripts/check_vmamba_global_cuda.py
is the GPU check.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from src.models.vmamba import SS2D, JiTVMamba, cross_scan, cross_merge

GLOBAL_KERNELS = ("linear", "softmax")


class SS2DGlobal(SS2D):
    """vmamba.py's SS2D with undecayed global condition-token memories.

    Same parameters and init as SS2D (it calls SS2D.__init__ with every arm
    off), plus the gains. x: (B, M + H*W, D) with the M condition tokens in
    front; M = 0 is SS2D.forward exactly.

    emit_tokens=False is the last token-carrying block: only alpha_xe exists,
    G and the token read are skipped, and the token rows return the D term.
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=1, K=4,
                 proj_drop=0.0, readback=True, kernel_xe="linear",
                 emit_tokens=True):
        super().__init__(d_model=d_model, d_state=d_state, d_conv=d_conv,
                         expand=expand, K=K, proj_drop=proj_drop)
        assert K == 4, "SS2DGlobal merges with cross_merge's 4 directions"
        assert kernel_xe in GLOBAL_KERNELS, (
            "Unknown global_kernel_xe=%r (use %s)" % (kernel_xe, " | ".join(GLOBAL_KERNELS)))
        self.readback = bool(readback)
        self.kernel_xe = kernel_xe
        self.emit_tokens = bool(emit_tokens)
        # Ones-init, flat 1-D: no random numbers, no weight decay.
        self.alpha_xe = nn.Parameter(torch.ones(self.d_inner))
        if self.emit_tokens:
            self.alpha_ee = nn.Parameter(torch.ones(self.d_inner))
            if self.readback:
                self.alpha_ex = nn.Parameter(torch.ones(self.d_inner))

    # ── pieces (separate so the tests and the diagnostic can reach them) ──

    def _token_features(self, z_e, dtype):
        """z_e: (B, M, d_inner) in_proj output -> u^e (B, d_inner, M):
        DWConv centre tap + bias, then SiLU (a token has no grid neighbours)."""
        c = self.d_conv // 2
        w = self.conv2d.weight[:, 0, c, c]
        u_e = self.act(z_e.transpose(1, 2) * w[:, None] + self.conv2d.bias[:, None])
        return u_e.to(dtype)

    def _ssm_params(self, u, tokens):
        """u: (B, K, d_inner, L) image paths or (B, d_inner, M) tokens (shared
        by every direction). Returns dt (raw, pre-bias), B, C per direction."""
        eq = "bdl,kod->bkol" if tokens else "bkdl,kod->bkol"
        x_dbl = torch.einsum(eq, u, self.x_proj_weight)
        dt_r, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dt = torch.einsum("bkrl,kdr->bkdl", dt_r, self.dt_projs_weight)
        return dt, B_ssm, C_ssm

    def _memories(self, u_e, dt_e, B_e, xs, dt, B_x, need_s, need_g):
        """fp32, autocast off. ve (B, K, d_inner, M); S, G (B, K, d_inner, N)."""
        ct = torch.promote_types(xs.dtype, torch.float32)
        with torch.autocast(device_type=xs.device.type, enabled=False):
            bias = self.dt_projs_bias.to(ct)[None, :, :, None]
            ve = F.softplus(dt_e.to(ct) + bias) * u_e.to(ct)[:, None]
            S = G = None
            if need_s:
                S = torch.einsum("bkdm,bknm->bkdn", ve, B_e.to(ct)) / u_e.shape[-1]
            if need_g:
                vx = F.softplus(dt.to(ct) + bias) * xs.to(ct)
                G = torch.einsum("bkdl,bknl->bkdn", vx, B_x.to(ct)) / xs.shape[-1]
        return ve, S, G

    def _scan(self, xs, dt, B_ssm, C_ssm):
        """The stock stacked scan, called as SS2D's baseline path. A, D and
        delta_bias are fp32 for fp32 / bf16 inputs (exactly SS2D's call) and
        float64 for float64 inputs (so the CPU proofs are exact)."""
        B, K, d_inner, L = xs.shape
        d_state = self.d_state
        pt = torch.promote_types(xs.dtype, torch.float32)
        A = -torch.exp(self.A_logs.to(pt)).view(K * d_inner, d_state)
        y = selective_scan_fn(
            xs.reshape(B, K * d_inner, L),
            dt.contiguous().view(B, K * d_inner, L),
            A,
            B_ssm.contiguous().view(B, K, d_state, L),
            C_ssm.contiguous().view(B, K, d_state, L),
            self.Ds.to(pt).view(K * d_inner),
            z=None,
            delta_bias=self.dt_projs_bias.to(pt).view(K * d_inner),
            delta_softplus=True,
            return_last_state=False,
        )
        return y.view(B, K, d_inner, L)

    def _image_read(self, S, ve, B_e, C_x):
        """image <- token term, (B, K, d_inner, L), fp32 (before alpha_xe)."""
        ct = ve.dtype
        with torch.autocast(device_type=C_x.device.type, enabled=False):
            if self.kernel_xe == "linear":
                return torch.einsum("bkdn,bknl->bkdl", S, C_x.to(ct))
            logits = torch.einsum("bknl,bknm->bklm", C_x.to(ct), B_e.to(ct))
            P = torch.softmax(logits / math.sqrt(self.d_state), dim=-1)
            return torch.einsum("bklm,bkdm->bkdl", P, ve)

    def _token_outputs(self, u_e, C_e, S, G):
        """y^e (B, d_inner, M), fp32: sum_k [(a_ee S_k + a_ex G_k) C^e + D_k u^e]."""
        ct = torch.promote_types(u_e.dtype, torch.float32)
        with torch.autocast(device_type=u_e.device.type, enabled=False):
            y_e = self.Ds.to(ct).sum(0)[None, :, None] * u_e.to(ct)
            if self.emit_tokens:
                mem = self.alpha_ee.to(ct)[:, None] * S
                if self.readback:
                    mem = mem + self.alpha_ex.to(ct)[:, None] * G
                y_e = y_e + torch.einsum("bkdn,bknm->bdm", mem, C_e.to(ct))
        return y_e

    def terms(self, x, H, W):
        """Everything up to the merge, as a dict (the forward and the proofs
        share it): u_e, xs, dt, B_x, C_x, dt_e, B_e, C_e, ve, S, G, y_scan
        (kernel output, pre-merge, D included), read (before alpha_xe), ys
        (pre-merge image paths), y_e (token rows, fp32)."""
        B, L_in, _ = x.shape
        d_inner = self.d_inner
        HW = H * W
        M = L_in - HW
        assert M > 0, "terms() needs condition tokens; got L_in=%d for %dx%d" % (L_in, H, W)

        # 1. in_proj, then split: the tokens never enter the scan
        z = self.in_proj(x)                                     # (B, M+HW, d_inner)
        z_e, z_g = z[:, :M], z[:, M:]
        # 2. features: grid DWConv + SiLU (unchanged); tokens centre tap + SiLU
        u_g = z_g.view(B, H, W, d_inner).permute(0, 3, 1, 2).contiguous()
        u_g = self.act(self.conv2d(u_g))                        # (B, d_inner, H, W)
        u_e = self._token_features(z_e, u_g.dtype)              # (B, d_inner, M)
        # 3. CrossScan over the grid only
        xs = cross_scan(u_g)                                    # (B, K, d_inner, HW)
        # 4. dt / B / C, both groups through the same per-direction weights
        dt, B_x, C_x = self._ssm_params(xs, tokens=False)       # (B,K,d,HW) (B,K,N,HW) x2
        dt_e, B_e, C_e = self._ssm_params(u_e, tokens=True)     # (B,K,d,M)  (B,K,N,M)  x2
        # 5. memories (fp32), from this block's features
        need_s = self.kernel_xe == "linear" or self.emit_tokens
        need_g = self.readback and self.emit_tokens
        ve, S, G = self._memories(u_e, dt_e, B_e, xs, dt, B_x, need_s, need_g)
        # 6. image paths: stock scan + alpha_xe * read, then (in forward) merge
        y_scan = self._scan(xs, dt, B_x, C_x)                   # (B, K, d_inner, HW)
        read = self._image_read(S, ve, B_e, C_x)                # (B, K, d_inner, HW) fp32
        with torch.autocast(device_type=x.device.type, enabled=False):
            ys = (y_scan.to(read.dtype)
                  + self.alpha_xe.to(read.dtype)[None, None, :, None] * read).to(y_scan.dtype)
        # 7. token rows
        y_e = self._token_outputs(u_e, C_e, S, G)               # (B, d_inner, M) fp32
        return dict(u_e=u_e, xs=xs, dt=dt, B_x=B_x, C_x=C_x, dt_e=dt_e, B_e=B_e,
                    C_e=C_e, ve=ve, S=S, G=G, y_scan=y_scan, read=read, ys=ys, y_e=y_e)

    def forward(self, x, H, W, cond=None):
        """x: (B, M + H*W, D) -> same shape. `cond` is accepted (JiTBlock passes
        it) and ignored: this mixer has no SSM-level conditioning arm."""
        if x.shape[1] == H * W:
            return SS2D.forward(self, x, H, W)
        t = self.terms(x, H, W)
        out_g = cross_merge(t["ys"], H, W)                      # (B, d_inner, HW)
        out = torch.cat([t["y_e"].to(out_g.dtype), out_g], dim=-1).transpose(1, 2)
        out = self.out_proj(self.out_norm(out))                 # (B, M+HW, D)
        return self.proj_drop(out)


class JiTVMambaGlobal(JiTVMamba):
    """JiT-S2-VMamba with M global condition tokens outside the scan.

    Built as JiTVMamba(in_context_len=M, content "class", layout "prefix",
    start global_start) -- the same slots, insertion, persist & update and
    strip -- then the mixers of blocks global_start .. depth-1 are replaced by
    SS2DGlobal carrying the SAME weights (only the gains are new). The swap
    runs under fork_rng, so the RNG state after construction (and every
    shared weight) equals the prefix model's at the same seed. global_len=0
    is JiTVMamba's baseline, unchanged.

    The vmamba.py arm keys are accepted only to be asserted off, so a config
    that sets one fails loudly instead of being silently ignored.
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
        d_state=16,
        d_conv=3,
        expand=1,
        K=4,
        # ── global condition tokens ──
        global_len=16,                 # M; 0 = the JiTVMamba baseline
        global_start=4,                # block the tokens join at
        global_readback=True,          # False = "write only": no G, no alpha_ex
        global_kernel_xe="linear",     # "linear" | "softmax" (image <- token term)
        # ── orthogonal modifiers that are safe with the tokens ──
        ffn="swiglu",
        norm="rmsnorm",
        derf_alpha_init=0.5,
        # ── vmamba.py arms: accepted ONLY to be asserted off ──
        in_context_len=0,
        in_context_layout="prefix",
        in_context_content="class",
        state_init="none",
        ssc="none",
        adaln_cond="full",
        ssc_z_mlp=False,
        ssm_norm="none",
        feature_rope="none",
        a_tie="none",
        bc_rope="none",
    ):
        off = {
            "in_context_len": (in_context_len, 0),
            "in_context_layout": (in_context_layout, "prefix"),
            "in_context_content": (in_context_content, "class"),
            "state_init": (state_init, "none"),
            "ssc": (ssc, "none"),
            "ssc_z_mlp": (bool(ssc_z_mlp), False),
            "ssm_norm": (ssm_norm, "none"),
            "feature_rope": (feature_rope, "none"),
            "a_tie": (a_tie, "none"),
            "bc_rope": (bc_rope, "none"),
        }
        for key, (value, default) in off.items():
            assert value == default, (
                "model=vmamba_global does not support %s=%r: the pairing with the "
                "global condition tokens is untested (the tokens are set by "
                "global_len / global_start)" % (key, value))
        assert adaln_cond in ("full", True), (
            "model=vmamba_global keeps adaLN-Zero on both branches; got "
            "adaln_cond=%r" % (adaln_cond,))
        assert isinstance(global_len, int) and not isinstance(global_len, bool) \
            and global_len >= 0, "global_len must be an int >= 0; got %r" % (global_len,)
        assert isinstance(global_readback, bool), (
            "global_readback must be true or false; got %r" % (global_readback,))
        assert global_kernel_xe in GLOBAL_KERNELS, (
            "Unknown global_kernel_xe=%r (use %s)"
            % (global_kernel_xe, " | ".join(GLOBAL_KERNELS)))
        assert K == 4, "SS2DGlobal merges with cross_merge's 4 directions; got K=%d" % K
        if global_len > 0:
            assert isinstance(global_start, int) and 0 <= global_start < depth, (
                "global_start must be a block index in [0, %d); got %r"
                % (depth, global_start))

        super().__init__(
            input_size=input_size, patch_size=patch_size, in_channels=in_channels,
            hidden_size=hidden_size, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop,
            num_classes=num_classes, bottleneck_dim=bottleneck_dim,
            d_state=d_state, d_conv=d_conv, expand=expand, K=K,
            in_context_len=global_len,
            in_context_start=global_start if global_len > 0 else 0,
            in_context_content="class", in_context_layout="prefix",
            ffn=ffn, norm=norm, derf_alpha_init=derf_alpha_init,
        )
        self.global_len = global_len
        self.global_start = global_start
        self.global_readback = global_readback
        self.global_kernel_xe = global_kernel_xe

        if global_len > 0:
            # SS2D.__init__ draws random numbers (x_proj, dt init); fork so the
            # RNG after construction equals the prefix model's at this seed.
            with torch.random.fork_rng(devices=[]):
                for i in range(global_start, depth):
                    block = self.blocks[i]
                    old = block.mixer
                    new = SS2DGlobal(
                        hidden_size, d_state=d_state, d_conv=d_conv, expand=expand,
                        K=K, proj_drop=old.proj_drop.p, readback=global_readback,
                        kernel_xe=global_kernel_xe, emit_tokens=(i < depth - 1))
                    new = new.to(device=old.in_proj.weight.device)
                    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
                    assert not unexpected and all(k.startswith("alpha_") for k in missing), \
                        (missing, unexpected)
                    block.mixer = new
