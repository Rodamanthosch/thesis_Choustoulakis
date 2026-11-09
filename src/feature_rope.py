"""
src/feature_rope.py
===================
Feature-level rotary position encoding for the Mamba mixers (`feature_rope`).

A fixed, parameter-free rotation of the mixer's image features, applied once on
the canonical H x W grid right after DWConv + SiLU and before anything is
scanned:

    rotated_features[position] = R(position) @ features[position]

R(position) is block-diagonal: each adjacent channel pair (x0, x1) turns by its
own angle a,

    [x0, x1] -> [x0 cos(a) - x1 sin(a),  x0 sin(a) + x1 cos(a)]

"1d"       p = r * W + c  (the ORIGINAL row-major grid index)
           omega[j] = theta ** (-2 j / D),  j = 0 .. D/2 - 1
           angle[p, j] = p * omega[j]                              (D even)

"2d_axial" first D/2 channels turn with the row, last D/2 with the column:
           omega[j] = theta ** (-4 j / D),  j = 0 .. D/4 - 1
           row_angle[r, j] = r * omega[j],  col_angle[c, j] = c * omega[j]
                                                                   (4 divides D)

Coordinates are integers from 0: no normalisation, no resolution interpolation.
The frequencies and the pair convention are those of JiT's
VisionRotaryEmbeddingFast (src/models/jit.py), taken over D feature channels
instead of an attention head's dimensions.

ADAPTATION, NOT A REPRODUCTION. This is inspired by RRG-Mamba (Hou et al.,
IJCAI 2025), whose Figure 2 puts RoPE after DWConv + SiLU and before SS2D. Its
Eq. (10) writes a query-key inner product without saying how that becomes the
feature sequence SS2D consumes, and the GDLM code was not available to us, so
rotating the features themselves is OUR reading. There is no query-key
similarity matrix, no Q/K projection, and nothing rotates the SSM state, B or C
directly (B, C and dt merely see rotated inputs through x_proj). RoFormer's
guarantee that a Q-K inner product depends only on relative position belongs to
attention: it does NOT carry over to the Mamba block as a whole, where the
rotation is a position-dependent change of the input basis followed by a scan
whose learned projections are not rotation-equivariant.

Conditioning tokens are never rotated and never shift an image coordinate: the
mixers call this on the grid alone, before the tokens enter the scan.
"""

import functools
import math

import torch

FEATURE_ROPE_MODES = ("none", "1d", "2d_axial")


def validate_feature_rope(mode, theta, dim):
    """Check a (mode, theta) pair for a mixer of feature width `dim`.

    Returns theta as a float for an active mode and None for "none", whose
    theta is never read. Numeric strings are accepted because a YAML `1e4` and
    a CLI override of an absent key both arrive as strings.
    """
    if mode not in FEATURE_ROPE_MODES:
        raise ValueError(
            "Unknown feature_rope=%r (use %s)" % (mode, " | ".join(FEATURE_ROPE_MODES)))
    if mode == "none":
        return None
    try:
        if isinstance(theta, bool):
            raise TypeError
        value = float(theta)
    except (TypeError, ValueError):
        raise ValueError(
            "feature_rope_theta must be a finite, positive number; got %r" % (theta,)
        ) from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "feature_rope_theta must be a finite, positive number; got %r" % (theta,))
    divisor = 2 if mode == "1d" else 4
    if dim % divisor != 0:
        raise ValueError(
            "feature_rope=%r needs the mixer's feature width d_inner to be "
            "divisible by %d; got %d" % (mode, divisor, dim))
    return value


def _frequencies(dim, theta, device):
    """omega[j] = theta ** (-2 j / dim), j < dim / 2 (JiT's expression), fp32."""
    exponents = torch.arange(0, dim, 2, device=device, dtype=torch.float32)[:dim // 2] / dim
    return 1.0 / (theta ** exponents)


@functools.lru_cache(maxsize=128)
def _rotation_tables(mode, theta, dim, H, W, device, dtype):
    """(cos, sin), each (1, dim / 2, H, W): one angle per channel pair per cell.

    Angles and their cos / sin are always computed in fp32, then cast to the
    working dtype. The cache lives at module level, keyed by everything the
    tables depend on, so blocks share it and nothing reaches a state_dict, and
    deep copies (the EMA nets) stay cheap. Building outside inference mode
    matters: an inference tensor cached by a sampling or measurement call would
    otherwise break every later backward pass.
    """
    validate_feature_rope(mode, theta, dim)
    assert mode != "none", "the mixers skip the rotation entirely when it is off"
    with torch.inference_mode(False), torch.no_grad(), \
            torch.autocast(device_type=device.type, enabled=False):
        if mode == "1d":
            positions = torch.arange(H * W, device=device, dtype=torch.float32)
            angles = torch.outer(positions, _frequencies(dim, theta, device)).view(H, W, -1)
        else:   # "2d_axial": the first half of the pairs per row, the second per column
            freqs = _frequencies(dim // 2, theta, device)
            rows = torch.arange(H, device=device, dtype=torch.float32)
            cols = torch.arange(W, device=device, dtype=torch.float32)
            row_angles = torch.outer(rows, freqs)[:, None, :].expand(H, W, -1)
            col_angles = torch.outer(cols, freqs)[None, :, :].expand(H, W, -1)
            angles = torch.cat([row_angles, col_angles], dim=-1)
        angles = angles.permute(2, 0, 1)[None]                  # (1, dim / 2, H, W)
        return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_feature_rope(x, mode, theta):
    """Rotate adjacent channel pairs of a grid of features.

    x: (B, D, H, W), the canonical image grid (rectangular grids are fine).
    Returns the same shape, device and dtype. The product runs in at least
    fp32, so a bf16 input is rounded only once, on the way out.
    """
    _, dim, H, W = x.shape
    work_dtype = torch.promote_types(x.dtype, torch.float32)
    cos, sin = _rotation_tables(mode, theta, dim, H, W, x.device, work_dtype)
    xf = x.to(work_dtype)
    x0, x1 = xf[:, 0::2], xf[:, 1::2]
    rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=2)
    return rotated.flatten(1, 2).to(x.dtype)
