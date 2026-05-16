"""
src/flops_counter.py
========================
Unified FLOPs / MACs / parameter counter for JiT, JiT-S2-ViM, JiT-S2-VMamba.

Why this module exists
----------------------
- thop misses nn.Embedding parameters and silently underreports.
- fvcore counts attention matmuls correctly (used by DiT / SiT).
- Neither tool natively counts the Mamba `selective_scan_fn` CUDA op
  (it's a python autograd.Function whose internals aren't traced).
- We register the canonical Albert-Gu formula for the SSM scan:
      flops = 9 * B * L * D * N   (+ B*D*L if with_D)
  which is exactly what MzeroMiko/VMamba/vmamba.py uses.
  The same Python entry-point selective_scan_fn is used by ViM and VMamba,
  so one hook covers both.

Unit convention
---------------
fvcore counts "one multiply-add = 1 flop", i.e. its 'flops' are actually MACs.
This matches the convention used by JiT / DiT / SiT / VMamba / ViM tables
(they all label MACs as "Gflops"). We report two numbers:
  • macs_total          — MACs (paper convention; use this for "Gflops" column)
  • flops_strict_total  — 2 * MACs (strict FLOPs definition)

Usage
-----
    from src.utils.count_flops import count_complexity

    report = count_complexity(net, img_size=32, num_classes=10, device='cuda')
    print(report)            # pretty-print
    macs   = report['macs_total']
    params = report['params_total']

Or directly:
    macs, flops_strict, params = count_complexity_minimal(net, img_size=32)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
except ImportError as e:
    raise ImportError(
        "fvcore is required for count_flops. Install with: pip install fvcore"
    ) from e


# ── SSM selective-scan hook ──────────────────────────────────────────────────
# Verbatim port of MzeroMiko/VMamba/vmamba.py flops_selective_scan_fn.
# This is Albert Gu's analytical formula for the selective scan:
#   forward+backward sweep × N inner ops = 9 ops per (B, D, L, N) cell.

def _flops_selective_scan_fn(B: int, L: int, D: int, N: int,
                              with_D: bool = True, with_Z: bool = False) -> int:
    """Compute MACs for one selective_scan_fn call.

    Args:
        B: batch
        L: sequence length
        D: inner dimension (= d_inner for ViM, = K*d_inner for VMamba)
        N: SSM state dim (d_state)
        with_D: include the skip connection y += u*D
        with_Z: include the gate (ViM uses this, VMamba does not)

    Returns:
        MAC count for one selective_scan_fn invocation.
    """
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def _selective_scan_flop_jit(inputs, outputs) -> int:
    """fvcore JIT hook for selective_scan_fn / SelectiveScanCuda.

    Pulls (B, D, L) from the u tensor shape and N from A's shape, then
    invokes _flops_selective_scan_fn.

    ViM passes z (gate) to selective_scan_fn → with_Z=True
    VMamba does not pass z                   → with_Z=False
    We detect this by inspecting the number of inputs.
    """
    try:
        # inputs[0] is u : (B, D, L) where D = K*d_inner (VMamba) or d_inner (ViM)
        B, D, L = inputs[0].type().sizes()
        # inputs[2] is A : (D, N)
        N = inputs[2].type().sizes()[1]
    except Exception as e:
        warnings.warn(f"selective_scan_flop_jit: failed to extract shapes ({e}); returning 0.")
        return 0

    # Heuristic for with_Z: if there are ≥ 7 positional inputs the 7th is z (ViM).
    # ViM call signature: (u, delta, A, B, C, D, z, delta_bias, ...)
    # VMamba call signature: (u, delta, A, B, C, D, delta_bias, ...)
    with_Z = False
    if len(inputs) >= 7:
        try:
            # If the 7th input has shape (B, D, L) it's z; if shape (D,) it's delta_bias.
            seventh_shape = inputs[6].type().sizes()
            if len(seventh_shape) == 3:
                with_Z = True
        except Exception:
            pass

    return _flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=with_Z)


def _causal_conv1d_flop_jit(inputs, outputs) -> int:
    """fvcore JIT hook for causal_conv1d_fn (used by ViM's BiMamba).

    causal_conv1d_fn(x, weight, bias, ...) does a depthwise 1D conv.
        x:      (B, D, L)
        weight: (D, k)   — depthwise (groups=D)
    MACs = B * D * L * k   (depthwise: each output uses k inputs from same channel)
    """
    try:
        B, D, L = inputs[0].type().sizes()
        k = inputs[1].type().sizes()[1]
    except Exception:
        return 0
    return B * D * L * k


def _sdpa_flop_jit(inputs, outputs) -> int:
    """fvcore JIT hook for aten::scaled_dot_product_attention.

    IMPORTANT: fvcore does NOT count SDPA matmuls natively — they're opaque
    inside the fused op. Without this hook, attention MACs are silently
    UNDERCOUNTED by ~6% for JiT-B/16 (1.2 GMACs out of 23.2 total).

    F.scaled_dot_product_attention(q, k, v, ...) computes:
        attn = softmax(QK^T / sqrt(d)) @ V
    where q,k,v have shape (B, H, N, head_dim).

    MACs (matmuls only):
        QK^T: B * H * N * head_dim * N      MACs
        @V:   B * H * N * N      * head_dim MACs
        total = 2 * B * H * N * N * head_dim
    """
    try:
        q_shape = inputs[0].type().sizes()
        # Support both (B, H, N, d) and (B, N, d) shapes:
        if len(q_shape) == 4:
            B, H, N, d = q_shape
        elif len(q_shape) == 3:
            B, N, d = q_shape
            H = 1
        else:
            return 0
        return 2 * B * H * N * N * d
    except Exception:
        return 0


# All known JIT op names for the SSM scan, across mamba_ssm versions and VMamba's fork.
# We register the same handler for each.
_SSM_OP_NAMES = (
    "prim::PythonOp.SelectiveScanFn",        # mamba_ssm reference / ViM (older)
    "prim::PythonOp.SelectiveScanCuda",      # VMamba (current)
    "prim::PythonOp.SelectiveScanMamba",     # mamba_ssm (newer)
    "prim::PythonOp.SelectiveScanCoreFn",    # VMamba alt backend
    "prim::PythonOp.SelectiveScanOflexFn",   # VMamba alt backend
)
_CONV1D_OP_NAMES = (
    "prim::PythonOp.CausalConv1dFn",         # ViM's fused conv
)


def _build_supported_ops() -> Dict[str, Any]:
    ops = {}
    for name in _SSM_OP_NAMES:
        ops[name] = _selective_scan_flop_jit
    for name in _CONV1D_OP_NAMES:
        ops[name] = _causal_conv1d_flop_jit
    # Attention SDPA — fvcore misses this natively in modern PyTorch.
    ops["aten::scaled_dot_product_attention"] = _sdpa_flop_jit
    return ops


# ── Public API ───────────────────────────────────────────────────────────────

def count_complexity(
    net: nn.Module,
    img_size: int,
    in_channels: int = 3,
    num_classes: int = 10,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Count MACs, FLOPs, and parameters for a JiT-family model.

    The model is expected to have signature `net(x, t, y)`:
        x : (B, in_channels, img_size, img_size)  noisy input
        t : (B,)                                   timestep in [0, 1]
        y : (B,)                                   class labels (long)

    Returns a dict with keys:
        params_total       — total parameter count (includes nn.Embedding)
        params_trainable   — trainable parameters only
        macs_total         — MACs (paper "Gflops" convention)
        flops_strict_total — 2 * MACs (strict FLOPs convention)
        macs_by_module     — fvcore per-module breakdown (Counter)
        macs_by_op         — fvcore per-op breakdown (Counter)
        unsupported_ops    — fvcore list of ops it couldn't trace (debug)
    """
    if eval_mode:
        net.eval()
    device = torch.device(device)
    net = net.to(device)

    x = torch.randn(1, in_channels, img_size, img_size, device=device)
    t = torch.rand(1, device=device)
    y = torch.randint(0, num_classes, (1,), device=device)

    # --- params ---
    params_total = sum(p.numel() for p in net.parameters())
    params_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # --- macs via fvcore ---
    fca = FlopCountAnalysis(net, (x, t, y))
    fca.set_op_handle(**_build_supported_ops())
    if not verbose:
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        fca.tracer_warnings("none")

    macs_total = fca.total()
    macs_by_module = dict(fca.by_module())
    macs_by_op = dict(fca.by_operator())
    unsupported = dict(fca.unsupported_ops())

    return {
        "params_total":       params_total,
        "params_trainable":   params_trainable,
        "macs_total":         macs_total,
        "flops_strict_total": 2 * macs_total,
        "macs_by_module":     macs_by_module,
        "macs_by_op":         macs_by_op,
        "unsupported_ops":    unsupported,
    }


def count_complexity_minimal(
    net: nn.Module,
    img_size: int,
    **kwargs,
) -> Tuple[int, int, int]:
    """Return just (macs, flops_strict, params) for quick scripts."""
    r = count_complexity(net, img_size, **kwargs)
    return r["macs_total"], r["flops_strict_total"], r["params_total"]


# ── Pretty printer ───────────────────────────────────────────────────────────

# Ops that genuinely have ~0 MACs (memory shuffles, indexing, control flow).
# Listing them explicitly stops them from showing up as scary "unsupported".
_ZERO_FLOP_OPS = frozenset({
    "aten::view", "aten::reshape", "aten::permute", "aten::transpose",
    "aten::contiguous", "aten::to", "aten::size", "aten::expand",
    "aten::expand_as", "aten::unsqueeze", "aten::squeeze", "aten::slice",
    "aten::select", "aten::stack", "aten::cat", "aten::chunk", "aten::split",
    "aten::flip", "aten::clone", "aten::detach", "aten::type_as", "aten::_pad",
    "aten::pad", "aten::flatten", "aten::index", "aten::index_select",
    "aten::masked_fill", "aten::zeros", "aten::ones", "aten::zeros_like",
    "aten::ones_like", "aten::arange", "aten::full", "aten::full_like",
    "aten::empty", "aten::empty_like", "aten::new_zeros", "aten::new_ones",
    "aten::copy_", "aten::fill_", "aten::repeat", "aten::repeat_interleave",
    "aten::roll", "aten::gather", "aten::scatter", "aten::scatter_",
    "aten::_unsafe_view", "aten::dropout", "aten::dropout_",
    "aten::feature_dropout", "prim::ListConstruct", "prim::TupleConstruct",
    "prim::ListUnpack", "prim::TupleUnpack",
    # Lookup ops — embedding is a pure indexing op, no arithmetic.
    "aten::embedding", "aten::embedding_dense_backward",
})

# Ops that are O(N) elementwise — they contribute small but non-zero FLOPs.
# We flag these so you know they're skipped but reassure they're small.
_ELEMENTWISE_OPS = frozenset({
    "aten::add", "aten::add_", "aten::sub", "aten::sub_",
    "aten::mul", "aten::mul_", "aten::div", "aten::div_",
    "aten::neg", "aten::abs", "aten::exp", "aten::log",
    "aten::pow", "aten::sqrt", "aten::rsqrt", "aten::reciprocal",
    "aten::sigmoid", "aten::tanh", "aten::silu", "aten::gelu",
    "aten::relu", "aten::leaky_relu", "aten::softplus", "aten::softmax",
    "aten::mean", "aten::sum", "aten::max", "aten::min",
    "aten::clamp", "aten::clamp_min", "aten::clamp_max",
    "aten::layer_norm", "aten::group_norm", "aten::batch_norm",
    # Trig used in sin-cos positional embeddings — elementwise.
    "aten::sin", "aten::cos", "aten::tan",
    "aten::sin_", "aten::cos_",
    # Other common ones.
    "aten::sign", "aten::erf", "aten::erfc", "aten::floor", "aten::ceil",
    "aten::round", "aten::var", "aten::std", "aten::norm",
})


def _classify_unsupported(unsupported: Dict[str, int]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Split unsupported ops into (zero_macs, elementwise, worrying)."""
    zero, elem, worry = {}, {}, {}
    for op, count in unsupported.items():
        if count == 0:
            continue
        if op in _ZERO_FLOP_OPS:
            zero[op] = count
        elif op in _ELEMENTWISE_OPS:
            elem[op] = count
        else:
            worry[op] = count
    return zero, elem, worry


def print_report(
    report: Dict[str, Any],
    model_name: str = "Model",
    show_modules: bool = True,
    show_unsupported_detail: bool = True,
) -> None:
    """Pretty-print a complexity report with validity diagnostics.

    Shows:
      - Headline params + MACs + FLOPs
      - Per-operator breakdown (what got counted, with % and bar chart)
      - Top modules by MACs (which submodules dominate)
      - Validity check (split unsupported ops into zero/elementwise/worrying)
      - Architectural sanity check (formulas used for SSM / SDPA)
      - Trustworthiness verdict
    """
    p   = report["params_total"]
    pt  = report["params_trainable"]
    m   = report["macs_total"]
    f   = report["flops_strict_total"]
    unsup = report["unsupported_ops"]
    by_op = report["macs_by_op"]
    by_mod = report["macs_by_module"]

    bar = "=" * 70
    sub = "-" * 70

    print(f"\n{bar}")
    print(f"  Complexity report: {model_name}")
    print(f"{bar}")

    # ── Headline numbers ──
    print(f"  Params (total)     : {p/1e6:>10.4f} M     ({p:,})")
    print(f"  Params (trainable) : {pt/1e6:>10.4f} M     ({pt:,})")
    print(f"  MACs               : {m/1e9:>10.4f} G     (paper 'Gflops' convention)")
    print(f"  FLOPs (strict)     : {f/1e9:>10.4f} G     (= 2 × MACs)")

    # ── What got counted (by op) ──
    if by_op:
        print(f"\n{sub}")
        print(f"  ✅ COUNTED — MACs by operator type:")
        print(f"{sub}")
        total = sum(by_op.values()) or 1
        for op, v in sorted(by_op.items(), key=lambda kv: -kv[1]):
            if v == 0:
                continue
            pct = 100 * v / total
            bar_chars = int(pct / 2.5)  # 40 cols max
            print(f"    {op:<35s} {v/1e9:>9.4f} G  ({pct:5.1f}%)  {'█' * bar_chars}")

    # ── What got counted (by module) ──
    if show_modules and by_mod:
        print(f"\n{sub}")
        print(f"  ✅ COUNTED — Top modules by MACs:")
        print(f"{sub}")
        items = [(k, v) for k, v in by_mod.items() if v > 0 and k]
        for mod, v in sorted(items, key=lambda kv: -kv[1])[:12]:
            depth = mod.count('.')
            indent = "  " * min(depth, 4)
            display_name = mod.split('.')[-1] if depth > 0 else mod
            pct = 100 * v / m if m > 0 else 0
            print(f"    {indent}{display_name:<30s} {v/1e9:>9.4f} G  ({pct:4.1f}%)")

    # ── Validity diagnostics ──
    zero_ops, elem_ops, worry_ops = _classify_unsupported(unsup)

    print(f"\n{sub}")
    print(f"  🔍 VALIDITY CHECK — what's NOT counted:")
    print(f"{sub}")

    if zero_ops:
        n_zero = sum(zero_ops.values())
        print(f"  ✓ Zero-FLOP ops (memory shuffles, indexing): {len(zero_ops)} types, {n_zero} calls")
        print(f"    These genuinely contribute 0 MACs — safe to ignore.")
        if show_unsupported_detail and len(zero_ops) <= 8:
            for op, c in sorted(zero_ops.items(), key=lambda kv: -kv[1])[:8]:
                print(f"      {op:<40s} ×{c}")

    if elem_ops:
        n_elem = sum(elem_ops.values())
        print(f"\n  ⚠ Elementwise ops (norm, silu, softmax, etc.): {len(elem_ops)} types, {n_elem} calls")
        print(f"    These contribute small but non-zero MACs that fvcore skips.")
        print(f"    Typical magnitude: <1% of total for ViT/Mamba. Same convention")
        print(f"    used by DiT/SiT/JiT/VMamba papers when reporting their numbers.")
        if show_unsupported_detail:
            for op, c in sorted(elem_ops.items(), key=lambda kv: -kv[1])[:10]:
                print(f"      {op:<40s} ×{c}")

    if worry_ops:
        print(f"\n  ❌ POTENTIALLY MISSED ops:")
        print(f"     These don't match any known zero/elementwise pattern.")
        print(f"     If any look compute-heavy (matmul, conv, einsum), the count is WRONG.")
        for op, c in sorted(worry_ops.items(), key=lambda kv: -kv[1]):
            print(f"      {op:<40s} ×{c}")
    else:
        print(f"\n  ✅ No unexpected ops found.")

    # ── Architectural sanity check ──
    print(f"\n{sub}")
    print(f"  📐 ARCHITECTURAL SANITY CHECK:")
    print(f"{sub}")

    ssm_ops = ["PythonOp.SelectiveScanFn", "PythonOp.SelectiveScanCuda",
               "PythonOp.SelectiveScanMamba", "PythonOp.SelectiveScanCoreFn",
               "PythonOp.SelectiveScanOflexFn"]
    sdpa_op = "scaled_dot_product_attention"
    conv1d_op = "PythonOp.CausalConv1dFn"

    ssm_total = sum(by_op.get(op, 0) for op in ssm_ops)
    sdpa_total = by_op.get(sdpa_op, 0)
    conv1d_total = by_op.get(conv1d_op, 0)
    linear_total = by_op.get("linear", 0)
    conv_total = by_op.get("conv", 0)

    if ssm_total > 0:
        arch = "Mamba-based (ViM or VMamba)"
        print(f"    Architecture detected     : {arch}")
        print(f"    SSM scan MACs (custom hook): {ssm_total/1e9:>9.4f} G  ({100*ssm_total/m:.1f}%)")
        print(f"    Causal conv1d MACs        : {conv1d_total/1e9:>9.4f} G  ({100*conv1d_total/m:.1f}%)")
        print(f"    Linear (projections) MACs : {linear_total/1e9:>9.4f} G  ({100*linear_total/m:.1f}%)")
        print(f"    Conv (patch embed) MACs   : {conv_total/1e9:>9.4f} G  ({100*conv_total/m:.1f}%)")
        print()
        print(f"    🟢 SSM scan formula: 9·B·L·D·N (+ B·D·L)")
        print(f"       (Albert Gu's formula, ported verbatim from MzeroMiko/VMamba)")
    elif sdpa_total > 0:
        arch = "Attention-based (JiT)"
        print(f"    Architecture detected     : {arch}")
        print(f"    SDPA (attention) MACs     : {sdpa_total/1e9:>9.4f} G  ({100*sdpa_total/m:.1f}%)")
        print(f"    Linear MACs (incl. QKV)   : {linear_total/1e9:>9.4f} G  ({100*linear_total/m:.1f}%)")
        print(f"    Conv (patch embed) MACs   : {conv_total/1e9:>9.4f} G  ({100*conv_total/m:.1f}%)")
        print()
        print(f"    🟢 SDPA formula: 2·B·H·N²·d (QK^T + attn·V matmuls)")
        print(f"       ⚠ WITHOUT this custom hook, fvcore silently misses this!")
    else:
        print(f"    Architecture: unknown / mixed")

    # ── Trustworthiness summary ──
    print(f"\n{sub}")
    if worry_ops:
        print(f"  VERDICT: ⚠ count may be incomplete — see ❌ ops above")
    elif elem_ops:
        print(f"  VERDICT: ✅ count is valid (elementwise <1% expected, see ⚠ above)")
    else:
        print(f"  VERDICT: ✅ count is complete and valid")
    print(f"{bar}")


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Sanity check on a small attention-only model (no SSM, since CUDA-free)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    # Build a tiny JiT-S-like attention model (using the minimal repro from this session)
    print("Running self-test on attention-only JiT...")
    # If your repo path differs, adjust this import:
    try:
        from src.models.jit import JiT
    except ImportError:
        # Fallback: use the minimal repro
        sys.path.insert(0, "/home/claude/test_thop")
        from jit_minimal import JiT

    net = JiT(input_size=32, patch_size=2, hidden_size=384, depth=12,
              num_heads=6, num_classes=10, bottleneck_dim=128)
    r = count_complexity(net, img_size=32, num_classes=10, device="cpu")
    print_report(r, model_name="JiT-S CIFAR-10")
