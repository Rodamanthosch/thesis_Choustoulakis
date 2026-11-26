"""Inference-only internal diagnostics for JiT-Spatial-Mamba (model: spatial_mamba).

The Spatial-Mamba counterpart of src/internals.py, for the experiments of
"Registers Matter for Pixel-Space Diffusion Transformers" (arXiv 2605.16147)
and "Vision Mamba Also Needs Registers" (Mamba-R, arXiv 2405.14858).
Supported: d_state 1 (the paper's setting), with or without the gate, with or
without in-context class tokens; not CAN, aux tokens, feature RoPE or ssm_norm
(check_supported).

Exact decomposition of one mixer forward. Given the block's own Delta, B and C
(they depend on the input, so they are frozen at their actual values), the
Structure-aware SSM is linear in its scan input u:

    x_s[c] = sum_{j <= s} prod_{q=j+1..s} exp(Delta_q[c] A[c]) Delta_j[c] B_j u_j[c]   (scan order)
    h      = SASF(x) on the grid cells; the tokens keep their own x (not fused)
    y_i[c] = C_i h_i[c] + D[c] u_i[c]

so the term of source j in output i is C_i * [SASF o scan](j -> i), and the
same without SASF is the direct scan path. Influence is its L2 norm over
channels, measured before LayerNorm, the gate, out_proj and the adaLN gate
(as src/internals.py does for SS2D). Everything is recomputed in float64 and
checked against the module's own forward.
"""

from contextlib import contextmanager
import math

import torch
import torch.nn.functional as F

from src.internals import _Stop, _check_eval, _hooks, mixer_inputs


def check_supported(net):
    """Raise unless the diagnostics below are exact for this model."""
    reasons = []
    if getattr(net, "d_state", 1) != 1:
        reasons.append("d_state > 1")
    if getattr(net, "dwconv_can", False) or getattr(net, "sasf_can", False):
        reasons.append("CAN kernels")
    if getattr(net, "aux_tokens", "none") != "none":
        reasons.append("aux tokens")
    for block in net.blocks:
        mixer = block.mixer
        if mixer.feature_rope != "none":
            reasons.append("feature RoPE")
        if mixer.ssm_norm != "none":
            reasons.append("ssm_norm")
        if mixer.token_conv:
            reasons.append("token_conv")
        break
    if reasons:
        raise ValueError("Spatial-Mamba diagnostics do not support: " + ", ".join(sorted(set(reasons))))


def scan_order(mixer, HW, P, device):
    """(L,) scan position -> layout index, layout [P tokens, HW grid].

    Uses the module's own _scan_order on an empty cache and restores the old
    one, so tensors made here (possibly in inference mode) never land in the
    module's cache, where a later forward with autograd would trip on them.
    """
    if P == 0:
        return torch.arange(HW, device=device)
    saved, mixer._scan_order_cache = mixer._scan_order_cache, {}
    try:
        return mixer._scan_order(HW, device)[0]
    finally:
        mixer._scan_order_cache = saved


def _finish(mixer, y, z):
    """LayerNorm [* SiLU(z)] + out_proj of a pre-norm y, in y's dtype."""
    norm = mixer.out_norm
    out = F.layer_norm(y, norm.normalized_shape, norm.weight.to(y.dtype), norm.bias.to(y.dtype), norm.eps)
    if z is not None:
        out = out * F.silu(z)
    return mixer.proj_drop(F.linear(out, mixer.out_proj.weight.to(y.dtype)))


def _scan_inputs(mixer, x, H, W, dtype):
    """u, z, Delta, B, C of one forward, all in layout order [tokens, grid]."""
    batch, length, _ = x.shape
    HW, d = H * W, mixer.d_inner
    P = length - HW
    if P not in (0, len(mixer.in_context_positions)):
        raise ValueError(f"Mixer got {P} tokens; its positions list {len(mixer.in_context_positions)}")
    xz = F.linear(x.to(dtype), mixer.in_proj.weight.to(dtype))
    u, z = xz.chunk(2, dim=-1) if mixer.gate else (xz, None)
    grid = u[:, P:].reshape(batch, H, W, d).permute(0, 3, 1, 2)
    grid = F.silu(F.conv2d(grid, mixer.conv2d.weight.to(dtype), mixer.conv2d.bias.to(dtype),
                           padding=mixer.d_conv // 2, groups=d))
    u = torch.cat([u[:, :P], grid.reshape(batch, d, HW).transpose(1, 2)], dim=1)     # (B, L, d)
    x_dbl = u @ mixer.x_proj_weight.to(dtype).T
    dt_r, B_ssm, C_ssm = torch.split(x_dbl, [mixer.dt_rank, 1, 1], dim=-1)
    # The module hands the kernel A = -exp(A_logs) and the dt bias in float32.
    delta = F.softplus(dt_r @ mixer.dt_projs_weight.to(dtype).T + mixer.dt_projs_bias.float().to(dtype))
    A = -torch.exp(mixer.A_logs.float()).to(dtype)[:, 0]                               # (d,)
    return dict(u=u, z=z, delta=delta, B=B_ssm[..., 0], C=C_ssm[..., 0], A=A, P=P,
                order=scan_order(mixer, HW, P, x.device))


def sasf_terms(fusion, states):
    """Per-dilation SASF terms alpha_d * conv_d(replicate-pad(states)), (n_dil, B, d, H, W)."""
    terms = []
    for alpha, kernel, dilation in zip(fusion.alpha, fusion.kernels, fusion.dilations):
        padded = F.pad(states, (dilation,) * 4, mode="replicate")
        terms.append(alpha.to(states.dtype) * F.conv2d(padded, kernel.to(states.dtype), dilation=dilation,
                                                       groups=fusion.dim))
    return torch.stack(terms)


@torch.inference_mode()
def mixer_quantities(mixer, x, H, W, dtype=torch.float64):
    """One StructureAwareSSM forward recomputed in `dtype`, with its internals.

    Layout order [P tokens, HW grid] throughout. Returns u, z, delta, B, C, A
    (as the module computes them), states (B, L, d) before SASF, fused
    (B, HW, d) grid states after it, terms (n_dil, B, HW, d) per dilation,
    y (B, L, d) before LayerNorm and out (B, L, D), the mixer's output.
    """
    q = _scan_inputs(mixer, x, H, W, dtype)
    u, delta, A, P, order = q["u"], q["delta"], q["A"], q["P"], q["order"]
    batch, length, d = u.shape
    HW = length - P
    retention = torch.exp(delta * A)                                 # (B, L, d)
    write = delta * q["B"][..., None] * u
    states = torch.empty_like(u)
    state = torch.zeros(batch, d, dtype=dtype, device=u.device)
    for i in order.tolist():
        state = retention[:, i] * state + write[:, i]
        states[:, i] = state
    grid = states[:, P:].transpose(1, 2).reshape(batch, d, H, W)
    terms = sasf_terms(mixer.state_fusion, grid)
    fused = terms.sum(dim=0).reshape(batch, d, HW).transpose(1, 2)
    h = torch.cat([states[:, :P], fused], dim=1)
    y = h * q["C"][..., None] + u * mixer.Ds.to(dtype)
    q.update(states=states, fused=fused, terms=terms.flatten(3).transpose(2, 3), y=y,
             out=_finish(mixer, y, q["z"]), retention=retention)
    return q


@torch.inference_mode()
def spatial_contributions(mixer, x, H, W):
    """Exact per-source decomposition of one StructureAwareSSM forward.

    x: the mixer's input (B, P + HW, D). Returns
      influence (B, L, L): L2 norm over channels of source j's term in output i
                           (scan + SASF, the actual path); diagonal zeroed,
      direct    (B, L, L): the same through the scan alone (SASF skipped),
      output    (B, L, D): the mixer's output rebuilt from the decomposition.
    Sources are the scan inputs u (after in_proj and, on the grid, DWConv + SiLU).
    """
    q = _scan_inputs(mixer, x, H, W, torch.float64)
    u, delta, A, P, order = q["u"], q["delta"], q["A"], q["P"], q["order"]
    batch, length, d = u.shape
    HW = length - P
    inverse = torch.argsort(order)

    # Scan order: weight[s, j] = exp(sum_{q=j+1..s} Delta_q A) Delta_j B_j, for j <= s.
    log_decay = torch.cumsum((delta * A)[:, order], dim=1)                       # (B, L, d)
    causal = torch.ones(length, length, dtype=torch.bool, device=u.device).tril()
    exponent = (log_decay[:, :, None] - log_decay[:, None]).masked_fill(~causal[None, :, :, None], -math.inf)
    source = (delta * q["B"][..., None] * u)[:, order]                            # (B, L, d)
    direct = exponent.exp() * source[:, None]                                     # (B, s, j, d)
    del exponent
    direct = direct[:, inverse][:, :, inverse]                                    # (B, i, j, d) layout

    grid = direct[:, P:].permute(0, 2, 3, 1).reshape(batch * length, d, H, W)
    fused = sasf_terms(mixer.state_fusion, grid).sum(dim=0)
    fused = fused.reshape(batch, length, d, HW).permute(0, 3, 1, 2)               # (B, HW, j, d)
    C = q["C"][:, :, None, None]
    through = torch.cat([direct[:, :P], fused], dim=1) * C
    del fused, grid
    y = through.sum(dim=2) + u * mixer.Ds.double()
    output = _finish(mixer, y, q["z"])

    diagonal = torch.arange(length, device=u.device)
    influence = through.norm(dim=-1)
    influence[:, diagonal, diagonal] = 0
    direct = (direct * C).norm(dim=-1)
    direct[:, diagonal, diagonal] = 0
    return influence, direct, output


@torch.inference_mode()
def mixer_dynamics(net, z, t, y, blocks, dtype=torch.float32):
    """Per-image scan and SASF statistics of each requested block.

    {block: dict} with, per layout position (B, L): retention = mean over
    channels of exp(Delta A) (1 = keeps the state, 0 = resets it),
    half_life = median over channels of ln 2 / (-Delta A) in scan steps,
    write = mean |Delta B| over channels, read = |C|; per image (B,):
    sasf_change = ||fused - states|| / ||states|| on the grid; and
    sasf_share (B, n_dil): ||term_d|| / sum_d ||term_d||.
    """
    check_supported(net)
    H = W = net.grid_size
    inputs = mixer_inputs(net, z, t, y, blocks)
    out = {}
    for block in sorted(set(blocks)):
        mixer = net.blocks[block].mixer
        q = mixer_quantities(mixer, inputs[block], H, W, dtype)
        P = q["P"]
        rate = -(q["delta"] * q["A"])                                    # (B, L, d), > 0
        grid = q["states"][:, P:]
        term_norms = q["terms"].flatten(2).norm(dim=-1).T                 # (B, n_dil)
        out[block] = dict(
            retention=q["retention"].mean(dim=-1).cpu(),
            half_life=(math.log(2) / rate.clamp_min(1e-12)).median(dim=-1).values.cpu(),
            write=q["delta"].mul(q["B"][..., None]).abs().mean(dim=-1).cpu(),
            read=q["C"].abs().cpu(),
            sasf_change=((q["fused"] - grid).flatten(1).norm(dim=1)
                         / grid.flatten(1).norm(dim=1).clamp_min(1e-12)).cpu(),
            sasf_share=(term_norms / term_norms.sum(dim=1, keepdim=True).clamp_min(1e-12)).cpu(),
            tokens=P)
    return out


@torch.inference_mode()
def branch_norms(net, z, t, y, blocks):
    """{block: (mixer, ffn)} per-token norms (B, L) of the residual increments
    gate_msa * mixer(...) and gate_mlp * FFN(...); later blocks are not computed."""
    blocks = _check_eval(net, blocks)
    c = net.t_embedder(t) + net.y_embedder(y)
    captured = {}

    def hook(index, kind, chunk):
        gate = net.blocks[index].adaLN_modulation(c).chunk(6, dim=-1)[chunk].unsqueeze(1)

        def run(_module, _inputs, output):
            captured.setdefault(index, {})[kind] = (gate * output).float().norm(dim=-1)
            if index == blocks[-1] and kind == "ffn":
                raise _Stop
        return run

    handles = []
    for i in blocks:
        handles.append(net.blocks[i].mixer.register_forward_hook(hook(i, "mixer", 2)))
        handles.append(net.blocks[i].mlp.register_forward_hook(hook(i, "ffn", 5)))
    with _hooks(handles):
        try:
            net(z, t, y)
        except _Stop:
            pass
    return {i: (captured[i]["mixer"], captured[i]["ffn"]) for i in blocks}


def pooled_image_tokens(grid, fraction, generator):
    """(B, 4, D): mean of all, top-`fraction` by norm, a fixed random `fraction`
    and bottom-`fraction` image tokens of each image (Mamba-R Table 1 pooling)."""
    batch, count, _ = grid.shape
    k = max(1, round(fraction * count))
    order = grid.norm(dim=-1).argsort(dim=1, descending=True)
    gather = lambda idx: grid.gather(1, idx[..., None].expand(-1, -1, grid.shape[-1])).mean(dim=1)  # noqa: E731
    random_idx = torch.randperm(count, generator=generator)[:k].to(grid.device).expand(batch, -1)
    return torch.stack([grid.mean(dim=1), gather(order[:, :k]), gather(random_idx), gather(order[:, -k:])], dim=1)


POOLED_NAMES = ("all image tokens", "top 5% norm", "random 5%", "bottom 5% norm")


@contextmanager
def token_subset(net, keep):
    """Temporarily give every mixer only the scan positions of tokens `keep`."""
    flat = [r * net.grid_size + c for r, c in net.in_context_positions]
    saved = [(block.mixer.in_context_positions, block.mixer._scan_order_cache) for block in net.blocks]
    try:
        for block in net.blocks:
            block.mixer.in_context_positions = tuple(flat[e] for e in keep)
            block.mixer._scan_order_cache = {}
        yield
    finally:
        for block, (positions, cache) in zip(net.blocks, saved):
            block.mixer.in_context_positions, block.mixer._scan_order_cache = positions, cache


@torch.inference_mode()
def forward_variant(net, x, t, y_adaln, y_tokens=None, keep=None):
    """JiTSpatialMamba.forward with the class routed separately.

    y_adaln conditions adaLN (and the final layer); y_tokens fills the class
    tokens (default y_adaln); keep lists the class tokens kept (default all,
    [] = none). With the defaults this is the model's forward, bit for bit.
    """
    check_supported(net)
    n = net.in_context_len
    keep = list(range(n)) if keep is None else sorted(set(int(e) for e in keep))
    if any(not 0 <= e < n for e in keep):
        raise ValueError(f"keep must index the {n} class tokens: {keep}")
    t_emb = net.t_embedder(t)
    y_emb = net.y_embedder(y_adaln)
    c = t_emb + y_emb
    token_emb = y_emb if y_tokens is None else net.y_embedder(y_tokens)
    x = net.x_embedder(x) + net.pos_embed
    H = W = net.grid_size
    with token_subset(net, keep):
        for i, block in enumerate(net.blocks):
            if keep and i == net.in_context_start:
                ctx = torch.stack([token_emb] * len(keep), dim=1)
                x = torch.cat([ctx + net.incontext_pos_embed[:, keep], x], dim=1)
            x = block(x, c, H, W, None)
    if keep:
        x = x[:, len(keep):]
    return net.unpatchify(net.final_layer(x, c))


def ablation_variants(n, num_classes):
    """{name: (route, keep)}: route picks the labels of adaLN and the tokens."""
    variants = {"full": ("true", None)}
    if not n:
        return variants
    variants["no_tokens"] = ("true", [])
    for e in range(n):
        variants[f"drop_{e}"] = ("true", [k for k in range(n) if k != e])
    for route in ("tokens_null", "adaln_null", "all_null", "tokens_other", "adaln_other", "all_other"):
        variants[route] = (route, None)
    return variants


def variant_labels(route, y, num_classes):
    """(y_adaln, y_tokens) for a route; 'other' is a different real class."""
    null = torch.full_like(y, num_classes)
    other = (y + num_classes // 2) % num_classes
    table = {"true": (y, y), "tokens_null": (y, null), "adaln_null": (null, y), "all_null": (null, null),
             "tokens_other": (y, other), "adaln_other": (other, y), "all_other": (other, other)}
    return table[route]
