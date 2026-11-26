"""Paired, inference-only block-removal diagnostics. No model flags or retraining."""

from contextlib import contextmanager
import hashlib
import math

import numpy as np
import torch
from torch import nn


class _Bypass(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


@contextmanager
def bypass_blocks(net, indices):
    """Actually bypass computation; preserve original indices and prefix insertion.

    Restore the exact original modules even if inference raises. Use only on an
    unwrapped evaluation model, without concurrent forwards.
    """
    indices = tuple(indices)
    if net.training:
        raise ValueError("Block ablation is evaluation-only; call net.eval() first")
    if len(set(indices)) != len(indices) or any(
        not isinstance(i, int) or i < 0 or i >= len(net.blocks) for i in indices
    ):
        raise ValueError(f"Invalid block indices: {indices}")
    original = {i: net.blocks[i] for i in indices}
    try:
        for i in indices:
            net.blocks[i] = _Bypass().eval()
        yield
    finally:
        for i, block in original.items():
            net.blocks[i] = block


def removal_variants(depth, window_sizes=(1, 2, 4)):
    """Baseline plus every contiguous window; names and indices are ZERO-based."""
    variants = {"baseline": ()}
    for width in sorted(set(window_sizes)):
        if not 1 <= width < depth:
            raise ValueError(f"Window {width} must be in [1, {depth - 1}]")
        for start in range(depth - width + 1):
            indices = tuple(range(start, start + width))
            variants["skip_" + "_".join(map(str, indices))] = indices
    return variants


def image_noise(shape, image_id, seed):
    """Noise is independent of batch size, dataloader workers and ablation order."""
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    sample_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    return torch.randn(shape, generator=torch.Generator().manual_seed(sample_seed))


def predictions(net, z, time, labels, num_classes, cfg_scale, cfg_interval):
    """Return conditional and CFG x-predictions with the sampler's interval rule."""
    low, high = cfg_interval
    active = (time < high) & ((time > low) | (low == 0))
    if cfg_scale == 1 or not active.any():
        conditional = net(z, time, labels).float()
        return conditional, conditional
    # Same retained network for both halves, matching the existing CFG sampler.
    out = net(torch.cat((z, z)), torch.cat((time, time)),
              torch.cat((labels, torch.full_like(labels, num_classes)))).float()
    conditional, unconditional = out.chunk(2)
    scale = torch.where(active, cfg_scale, 1.0).view(-1, 1, 1, 1)
    return conditional, unconditional + scale * (conditional - unconditional)


@torch.inference_mode()
def measure_batch(net, images, labels, noise, timesteps, variants, *,
                  num_classes, noise_scale, t_eps, cfg_scale, cfg_interval):
    """Arrays [variant, mode, timestep, image]; all comparisons share inputs.

    x_error: MSE to clean target. x_drift: MSE to the FULL model at the same z,t.
    No clipping and no resampling. This is teacher-forced screening, not FID.
    """
    names = list(variants)
    if not names or names[0] != "baseline" or variants["baseline"]:
        raise ValueError("The first variant must be the unmodified baseline")
    if t_eps <= 0 or not timesteps or any(not 0 <= t <= 1 for t in timesteps):
        raise ValueError("Require positive t_eps and timesteps in [0, 1]")
    shape = (len(names), 2, len(timesteps), len(images))
    errors, drift = np.empty(shape, np.float64), np.empty(shape, np.float64)
    for ti, t in enumerate(timesteps):
        z = t * images + (1 - t) * noise * noise_scale
        time = torch.full((len(images),), t, device=images.device)
        reference = None
        for vi, name in enumerate(names):
            with bypass_blocks(net, variants[name]):
                pred = predictions(net, z, time, labels, num_classes, cfg_scale, cfg_interval)
            if reference is None:
                reference = pred
            for mi, p in enumerate(pred):
                if not torch.isfinite(p).all():
                    raise RuntimeError(f"Nonfinite prediction: {name}, mode={mi}, t={t}")
                errors[vi, mi, ti] = (p - images).square().flatten(1).mean(1).cpu().numpy()
                drift[vi, mi, ti] = (p - reference[mi]).square().flatten(1).mean(1).cpu().numpy()
    return errors, drift


def _sem(values):
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None


def summarize(errors, drift, variants, timesteps, t_eps):
    """Paired image-level uncertainty; do not treat timesteps as independent images."""
    denominator = np.maximum(1 - np.asarray(timesteps, dtype=float), t_eps)**2
    rows, ranking = [], []
    for vi, (name, skipped) in enumerate(variants.items()):
        for mi, mode in enumerate(("conditional", "cfg")):
            loss = errors[vi, mi] / denominator[:, None]
            baseline = errors[0, mi] / denominator[:, None]
            difference = loss - baseline
            for ti, t in enumerate(timesteps):
                rows.append(dict(variant=name, skipped_blocks=list(skipped), mode=mode, t=t,
                                 n=loss.shape[1], x_mse=float(errors[vi, mi, ti].mean()),
                                 v_mse=float(loss[ti].mean()),
                                 baseline_v_mse=float(baseline[ti].mean()),
                                 delta_v_mse=float(difference[ti].mean()),
                                 delta_v_sem=_sem(difference[ti]),
                                 x_drift_mse=float(drift[vi, mi, ti].mean()),
                                 v_drift_mse=float(drift[vi, mi, ti].mean()/denominator[ti])))
            paired = difference.mean(axis=0)
            base = float(baseline.mean())
            ranking.append(dict(variant=name, skipped_blocks=list(skipped),
                                removed=len(skipped), mode=mode, n=loss.shape[1],
                                mean_v_mse=float(loss.mean()), baseline_v_mse=base,
                                mean_delta_v_mse=float(paired.mean()),
                                paired_image_sem=_sem(paired),
                                relative_delta_v_mse=float(paired.mean()/base) if base else None,
                                worst_t_delta_v_mse=float(difference.mean(axis=1).max()),
                                mean_v_drift_mse=float((drift[vi, mi]/denominator[:, None]).mean())))
    return rows, ranking
