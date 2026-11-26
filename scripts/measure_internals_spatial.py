"""Register-style internal diagnostics for JiT-Spatial-Mamba checkpoints.

The Spatial-Mamba counterpart of scripts/measure_internals.py (VMamba): the
experiments of "Registers Matter for Pixel-Space Diffusion Transformers"
(arXiv 2605.16147, Figs. 1-7) and "Vision Mamba Also Needs Registers"
(Mamba-R, arXiv 2405.14858, Sec. 3.2 / Table 1), plus Spatial-Mamba specific
ones. Inference only: no training, no sampling, no FID. Pass one --model per
checkpoint; the first is the reference of the TV and norm ratios:

  python scripts/measure_internals_spatial.py --data-dir DATA --out-dir OUT \\
    --model base    configs/imagenet/jit-b16-spatial-nogate-imagenet.yaml CKPT_A \\
    --model class10 configs/imagenet/jit-b16-spatial-nogate-incontext-class10-imagenet.yaml CKPT_B

Measured per model (figure files in OUT):
  1  token norms per block and t, where the top-1% norms sit     fig2, fig3, fig13
     and which residual branch (mixer / FFN) makes them          fig12
  2  image-token norm and TV ratios against the reference       fig4a (summary.json)
  3  linear probes on pooled image tokens: all / top 5% norm /   fig5
     random 5% / bottom 5% (Mamba-R Table 1)
  4  probes on every class token, null label (image evidence     fig5, fig7
     only) and true label
  5  exact influence maps: what each token reads / writes,       fig1, fig6, fig11
     through scan + SASF and through the scan alone
  6  scan memory: retention exp(Delta A), half-life, |Delta B|    fig8
     at grid cells, row starts and tokens
  7  SASF use: ||fused - states|| / ||states||, share per dilation fig9
  8  class-token ablations (token models): drop one / all        fig10
     tokens, null or another class in the tokens or in adaLN;
     teacher-forced velocity MSE, not FID
  9  PCA of image tokens                                          fig4b

`auto` as CHECKPOINT searches --search-root for the only run whose weights fit
that model; `auto:TEXT` also requires TEXT in the path.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.measure_internals import (BLOCK_COLORS, INK, MODEL_COLORS, MUTED, NEUTRAL, apply_overrides, batches,
                                       colormaps, plot_fig1, plot_fig2, plot_fig3, plot_fig4a, plot_fig4b, plot_fig6,
                                       plot_fig7, ratio_tables, save, setup_style, show_image, to_jsonable,
                                       train_records)
from scripts.measure_redundancy import load_weights, validation_records
from src.internals import (block_features, discover_checkpoint, feature_statistics, fit_probes, influence_maps,
                           mixer_inputs, noised, pca_rgb, repack_checkpoint, tokens_at)
from src.internals_spatial import (POOLED_NAMES, ablation_variants, branch_norms, check_supported, forward_variant,
                                   mixer_dynamics, pooled_image_tokens, spatial_contributions, variant_labels)
from src.redundancy import image_noise


# Data: uint8 images in memory, noise drawn per image id (src/redundancy.py) ─────


class Images(torch.utils.data.Dataset):
    """ADM centre crop (the ImageNet training crop), kept as uint8."""

    def __init__(self, records, size):
        self.records, self.size = records, size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        from src.utils import center_crop_arr
        path, label, _ = self.records[index]
        with Image.open(path) as image:
            array = np.array(center_crop_arr(image.convert("RGB"), self.size))
        return torch.from_numpy(array).permute(2, 0, 1).contiguous(), label


def load_images(records, size, workers):
    loader = torch.utils.data.DataLoader(Images(records, size), batch_size=256, num_workers=workers)
    pixels, labels = zip(*loader)
    return dict(pixels=torch.cat(pixels), labels=torch.cat(labels).long(), ids=[r[2] for r in records])


def take(data, indices, seed, device):
    """Images in [-1, 1], labels and the per-image noise of src/redundancy.py::image_noise."""
    indices = [int(i) for i in indices]
    images = data["pixels"][indices].to(device).float().div(255).mul(2).sub(1)
    noise = torch.stack([image_noise(images.shape[1:], data["ids"][i], seed) for i in indices]).to(device)
    return images, data["labels"][indices].to(device), noise


def per_class(records_per_class, count, total):
    """Indices of the first `count` images of every class (records are grouped by class)."""
    return [i for i in range(total) if i % records_per_class < count]


def padded(values, width):
    """(B, L) -> (B, width) float32, NaN in front for tokens not present yet."""
    out = np.full((values.shape[0], width), np.nan, np.float32)
    out[:, width - values.shape[1]:] = values.float().cpu().numpy()
    return out


# Measurements ───────────────────────────────────────────────────────────────


def measure_statistics(net, data, indices, args, noise_scale, device):
    collected = []
    for start, stop in batches(len(indices), args.batch_size):
        x, y, e = take(data, indices[start:stop], args.seed, device)
        collected.append(feature_statistics(net, x, y, e, args.timesteps, list(range(len(net.blocks))), noise_scale))
    return {key: np.concatenate([c[key] for c in collected], axis=2) for key in collected[0]}


def measure_branches(net, data, indices, args, noise_scale, device):
    """[block, image, n + HW] norms of the mixer and FFN residual increments at --norm-t."""
    width = net.in_context_len + net.grid_size ** 2
    blocks = list(range(len(net.blocks)))
    mixer, ffn = [], []
    for start, stop in batches(len(indices), args.batch_size):
        x, y, e = take(data, indices[start:stop], args.seed, device)
        time = torch.full((len(x),), args.norm_t, device=device)
        norms = branch_norms(net, noised(x, e, args.norm_t, noise_scale), time, y, blocks)
        mixer.append(np.stack([padded(norms[b][0], width) for b in blocks]))
        ffn.append(np.stack([padded(norms[b][1], width) for b in blocks]))
    return dict(mixer=np.concatenate(mixer, axis=1), ffn=np.concatenate(ffn, axis=1))


def measure_pca(net, data, indices, args, noise_scale, device):
    H = W = net.grid_size
    x, y, e = take(data, indices, args.seed, device)
    time = torch.full((len(x),), args.pca_t, device=device)
    features = block_features(net, noised(x, e, args.pca_t, noise_scale), time, y, args.pca_blocks)
    return np.stack([np.stack([pca_rgb(features[b][i, tokens_at(net, b):], H, W).cpu().numpy()
                               for b in args.pca_blocks]) for i in range(len(x))])


def probe_features(net, data, t, mode, args, noise_scale, device, num_classes):
    """(N, n + 4, D): the class tokens, then POOLED_NAMES of the image tokens.
    The random 5% are the same cells for every image (one seeded draw)."""
    collected = []
    for start, stop in batches(len(data["labels"]), args.batch_size):
        x, y, e = take(data, range(start, stop), args.seed, device)
        if mode == "null":
            y = torch.full_like(y, num_classes)
        time = torch.full((len(x),), t, device=device)
        features = block_features(net, noised(x, e, t, noise_scale), time, y, [args.probe_block])[args.probe_block]
        present = tokens_at(net, args.probe_block)
        pooled = pooled_image_tokens(features[:, present:], .05, torch.Generator().manual_seed(args.seed))
        collected.append(torch.cat([features[:, :present], pooled], dim=1).cpu())
    return torch.cat(collected)


def measure_probes(net, train, val, args, noise_scale, device, num_classes, log):
    """Null label: the class enters neither adaLN nor the tokens, so accuracy
    comes from the image alone. True label: the class is handed in."""
    results = {}
    for t in args.probe_t:
        for mode in ("null", "cond"):
            log(f"  probes t={t} labels={mode}")
            x_train = probe_features(net, train, t, mode, args, noise_scale, device, num_classes)
            x_val = probe_features(net, val, t, mode, args, noise_scale, device, num_classes)
            accuracy, decay = fit_probes(x_train, train["labels"], x_val, val["labels"], num_classes,
                                         iterations=args.probe_iterations, seed=args.seed, device=device)
            results[f"t{t}_{mode}"] = dict(accuracy=accuracy, l2=decay, norm=x_val.norm(dim=-1).mean(dim=0).numpy(),
                                           spread=x_val.std(dim=0).mean(dim=-1).numpy())
            del x_train, x_val
    return results


def measure_influence(net, data, indices, args, noise_scale, device):
    """Maps [block, image, ...] through scan + SASF and through the scan alone."""
    H = W = net.grid_size
    keys = ("cells", "read", "write", "direct_cells", "direct_read", "direct_write")
    collected, worst = {k: [] for k in keys}, 0.0
    for start, stop in batches(len(indices), args.attn_batch_size):
        x, y, e = take(data, indices[start:stop], args.seed, device)
        time = torch.full((len(x),), args.attn_t, device=device)
        inputs = mixer_inputs(net, noised(x, e, args.attn_t, noise_scale), time, y, args.attn_blocks)
        per_block = {k: [] for k in keys}
        for block in args.attn_blocks:
            mixer = net.blocks[block].mixer
            influence, direct, rebuilt = spatial_contributions(mixer, inputs[block], H, W)
            with torch.inference_mode():
                actual = mixer(inputs[block], H, W).double()
            worst = max(worst, float((rebuilt - actual).abs().max() / actual.abs().max().clamp_min(1e-12)))
            present = tokens_at(net, block)
            for prefix, matrix in (("", influence), ("direct_", direct)):
                maps = influence_maps(matrix.float(), present, H, W)
                for key, value in zip(("cells", "read", "write"), maps):
                    per_block[prefix + key].append(value.cpu().numpy())
            del influence, direct
        for key in keys:
            collected[key].append(np.stack(per_block[key]))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if worst > 1e-3:
        raise RuntimeError(f"Scan decomposition disagrees with the mixer: relative error {worst:.3g}")
    return {k: np.concatenate(v, axis=1) for k, v in collected.items()}, worst


def measure_dynamics(net, data, indices, args, noise_scale, device):
    """Scan memory and SASF statistics [t, block, image, ...]."""
    width = net.in_context_len + net.grid_size ** 2
    blocks = list(range(len(net.blocks)))
    out = {k: [] for k in ("retention", "half_life", "write", "read", "sasf_change", "sasf_share")}
    for t in args.dynamics_t:
        per_t = {k: [] for k in out}
        for start, stop in batches(len(indices), args.batch_size):
            x, y, e = take(data, indices[start:stop], args.seed, device)
            time = torch.full((len(x),), t, device=device)
            stats = mixer_dynamics(net, noised(x, e, t, noise_scale), time, y, blocks)
            for key in ("retention", "half_life", "write", "read"):
                per_t[key].append(np.stack([padded(stats[b][key], width) for b in blocks]))
            per_t["sasf_change"].append(np.stack([stats[b]["sasf_change"].numpy() for b in blocks]))
            per_t["sasf_share"].append(np.stack([stats[b]["sasf_share"].numpy() for b in blocks]))
        for key in out:
            out[key].append(np.concatenate(per_t[key], axis=1))
    return {k: np.stack(v) for k, v in out.items()}


def measure_ablation(net, data, indices, args, noise_scale, t_eps, device, num_classes):
    """Velocity MSE [variant, t, image], drift to the full model, and per-image
    route shares [t, image] = ||route_other - full||^2 / ||all_other - full||^2."""
    variants = ablation_variants(net.in_context_len, num_classes)
    names = list(variants)
    shape = (len(names), len(args.ablation_t), len(indices))
    v_mse, drift = np.empty(shape), np.empty(shape)
    worst = 0.0
    for ti, t in enumerate(args.ablation_t):
        scale = max(1 - t, t_eps) ** 2
        for start, stop in batches(len(indices), args.batch_size):
            x, y, e = take(data, indices[start:stop], args.seed, device)
            time = torch.full((len(x),), t, device=device)
            z = noised(x, e, t, noise_scale)
            full = None
            for vi, name in enumerate(names):
                route, keep = variants[name]
                y_adaln, y_tokens = variant_labels(route, y, num_classes)
                pred = forward_variant(net, z, time, y_adaln, y_tokens, keep).double()
                if full is None:
                    full = pred
                    with torch.inference_mode():
                        reference = net(z, time, y).double()
                    worst = max(worst, float((pred - reference).abs().max()))
                v_mse[vi, ti, start:stop] = ((pred - x.double()).square().flatten(1).mean(1) / scale).cpu().numpy()
                drift[vi, ti, start:stop] = ((pred - full).square().flatten(1).mean(1) / scale).cpu().numpy()
    if worst > 1e-4:
        raise RuntimeError(f"forward_variant('full') differs from the model's forward by {worst:.3g}")
    shares = {}
    if "all_other" in names:
        total = drift[names.index("all_other")]
        for route in ("tokens_other", "adaln_other"):
            shares[route] = drift[names.index(route)] / np.maximum(total, 1e-30)
    return dict(names=names, v_mse=v_mse, drift=drift, shares=shares, full_vs_forward=worst)


def sasf_kernels(net):
    """Per block: alpha, ||alpha_d k_d|| (mean over channels) and the centre-tap energy share."""
    rows = []
    for block in net.blocks:
        fusion = block.mixer.state_fusion
        weight = torch.stack([a * k[:, 0] for a, k in zip(fusion.alpha, fusion.kernels)]).detach().float().cpu()
        energy = weight.square().flatten(2).sum(-1)                                       # (n_dil, d)
        rows.append(dict(alpha=fusion.alpha.detach().float().cpu().numpy(),
                         kernel_norm=energy.sqrt().mean(dim=1).numpy(),
                         centre_share=(weight[:, :, 1, 1].square() / energy.clamp_min(1e-30)).mean(dim=1).numpy()))
    return rows


# Summaries ──────────────────────────────────────────────────────────────────


def ratio(a, b):
    return float(a / b) if b else float("nan")


def outlier_maps(norm, n, H, W, fraction=.01):
    """[block, H, W] fraction of images whose top-`fraction` image-token norms include each cell."""
    grid = norm[:, :, n:]                                                    # (blocks, images, HW)
    k = max(1, int(round(fraction * grid.shape[-1])))
    top = np.argsort(-grid, axis=-1)[..., :k]
    counts = np.zeros(grid.shape[0] * H * W)
    offsets = (np.arange(grid.shape[0]) * H * W)[:, None, None]
    np.add.at(counts, (top + offsets).ravel(), 1)
    return counts.reshape(grid.shape[0], H, W) / grid.shape[1]


def reach(maps, fraction=.1):
    """Cells whose influence is >= `fraction` of the map's maximum, [block, image, token]."""
    flat = maps.reshape(*maps.shape[:3], -1)
    top = flat.max(axis=-1, keepdims=True)
    return (flat >= fraction * np.maximum(top, 1e-30)).sum(axis=-1)


def model_summary(result, args, H, W):
    n, start = result["tokens"], result["start"]
    t_index = args.timesteps.index(args.norm_t)
    norms, branches = {}, {}
    for block in range(result["stats"]["norm"].shape[1]):
        values = np.nanmean(result["stats"]["norm"][t_index, block], axis=0)
        patch = float(np.median(values[n:]))
        entry = dict(median_patch_norm=patch, max_over_median=ratio(values[n:].max(), patch),
                     p99_over_median=ratio(np.percentile(result["stats"]["norm"][t_index, block, :, n:], 99), patch))
        if n and block >= start:
            entry["token_norm_over_median_patch"] = values[:n] / patch
        norms[f"block_{block}"] = entry
        branch = {}
        for kind in ("mixer", "ffn"):
            image = result["branches"][kind][block, :, n:]
            branch[kind] = dict(median=float(np.median(image)), max_over_median=ratio(image.max(), np.median(image)),
                                p99_over_median=ratio(np.percentile(image, 99), np.median(image)))
        branches[f"block_{block}"] = branch
    probes = {key: dict(class_tokens=p["accuracy"][:n], **{name: float(p["accuracy"][n + i])
                                                           for i, name in enumerate(POOLED_NAMES)})
              for key, p in result["probes"].items()}
    influence = {}
    for bi, block in enumerate(args.attn_blocks):
        if n and block >= start:
            write, direct = result["influence"]["write"][bi], result["influence"]["direct_write"][bi]
            influence[f"block_{block}"] = dict(
                write_mass_per_token=write.reshape(*write.shape[:2], -1).sum(-1).mean(0),
                write_mass_sasf_over_scan=ratio(write.sum(), direct.sum()),
                write_reach_cells=reach(write[None])[0].mean(0),
                write_reach_cells_scan_only=reach(direct[None])[0].mean(0),
                read_mass_per_token=result["influence"]["read"][bi].reshape(*write.shape[:2], -1).sum(-1).mean(0))
    dynamics = {}
    d = result["dynamics"]
    row_start = np.array([r * W for r in range(1, H)])
    interior = np.array([r * W + c for r in range(H) for c in range(1, W)])
    for ti, t in enumerate(args.dynamics_t):
        rows = {}
        for block in range(d["retention"].shape[1]):
            grid = d["retention"][ti, block][:, n:]
            entry = dict(retention_grid=float(np.nanmean(grid)), retention_row_start=float(np.nanmean(grid[:, row_start])),
                         retention_interior=float(np.nanmean(grid[:, interior])),
                         half_life_median=float(np.nanmedian(d["half_life"][ti, block][:, n:])),
                         sasf_change=float(d["sasf_change"][ti, block].mean()),
                         sasf_share=d["sasf_share"][ti, block].mean(axis=0))
            if n and block >= start:
                entry.update(retention_tokens=np.nanmean(d["retention"][ti, block][:, :n], axis=0),
                             write_tokens_over_grid=ratio(np.nanmean(d["write"][ti, block][:, :n]),
                                                          np.nanmean(d["write"][ti, block][:, n:])))
            rows[f"block_{block}"] = entry
        dynamics[f"t{t}"] = rows
    summary = dict(tokens=n, start_block=start, positions=result["positions"], checkpoint=result["checkpoint"],
                   epoch=result["info"].get("epoch"), norms=norms, branch_norms=branches, probes=probes,
                   influence=influence, influence_rebuild_relative_error=result["influence_error"],
                   dynamics=dynamics, sasf_kernels=result["kernels"])
    ab = result.get("ablation")
    if ab:
        base = ab["v_mse"][0]
        rows = {}
        for vi, name in enumerate(ab["names"][1:], start=1):
            paired = (ab["v_mse"][vi] - base).mean(axis=0)
            rows[name] = dict(delta_v_mse=float(paired.mean()), paired_sem=float(paired.std(ddof=1) / np.sqrt(len(paired)))
                              if len(paired) > 1 else None, relative=ratio(paired.mean(), base.mean()),
                              delta_v_mse_per_t=(ab["v_mse"][vi] - base).mean(axis=1))
        summary["ablation"] = dict(full_v_mse=float(base.mean()), variants=rows,
                                   route_share_median_per_t={k: np.median(v, axis=1) for k, v in ab["shares"].items()},
                                   full_vs_forward_max_abs=ab["full_vs_forward"])
    return summary


# Figures ────────────────────────────────────────────────────────────────────


def plot_probes(plt, output, results, args):
    names = list(results)
    for t in args.probe_t:
        fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.8 * len(names)), squeeze=False)
        for row, name in enumerate(names):
            ax, result = axes[row, 0], results[name]
            n = result["tokens"]
            labels = [f"tok {e}" for e in range(n)] + list(POOLED_NAMES)
            x = np.arange(len(labels))
            for offset, (mode, color) in enumerate((("null", MODEL_COLORS[0]), ("cond", MODEL_COLORS[1]))):
                ax.bar(x + (offset - .5) * .38, result["probes"][f"t{t}_{mode}"]["accuracy"], width=.38, color=color,
                       label="null label (image only)" if mode == "null" else "true label")
            ax.axhline(1 / result["num_classes"], color=MUTED, linewidth=.8, linestyle=":")
            if n:
                ax.axvline(n - .5, color=INK, linewidth=.8)
            ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=7)
            ax.set(ylim=(0, 1), title=name, ylabel="accuracy")
            ax.legend(fontsize=7)
        fig.suptitle(f"Probes after block {args.probe_block}, t={t}: class tokens, then pooled image tokens "
                     "(Mamba-R Table 1). Dotted line: chance.", fontsize=9)
        save(fig, output, f"fig5_probes_t{t}.png")


def plot_dynamics(plt, output, results, args, H, W):
    seq, _ = colormaps()
    names = list(results)
    fig, axes = plt.subplots(len(names), len(args.dynamics_t), figsize=(4.2 * len(args.dynamics_t), 3 * len(names)),
                             squeeze=False)
    for row, name in enumerate(names):
        result, n = results[name], results[name]["tokens"]
        d = result["dynamics"]
        blocks = np.arange(d["retention"].shape[1])
        for col, t in enumerate(args.dynamics_t):
            ax = axes[row, col]
            grid = d["retention"][col][:, :, n:]
            ax.plot(blocks, np.nanmean(grid[:, :, [r * W for r in range(1, H)]], axis=(1, 2)), color=MODEL_COLORS[1],
                    label="row-start cells (r, 0)")
            ax.plot(blocks, np.nanmean(grid[:, :, [r * W + c for r in range(H) for c in range(1, W)]], axis=(1, 2)),
                    color=MODEL_COLORS[0], label="other cells")
            if n:
                ax.plot(blocks, np.nanmean(d["retention"][col][:, :, :n], axis=(1, 2)), color=INK, marker="o",
                        markersize=3, label="class tokens")
            ax.set(ylim=(0, 1), title=f"{name}, t={t}", xlabel="block", ylabel="mean exp(ΔA)")
            ax.legend(fontsize=7)
    fig.suptitle("Scan memory: retention 1 keeps the state, 0 resets it", fontsize=10)
    save(fig, output, "fig8a_retention.png")

    t_index = len(args.dynamics_t) // 2
    for name in names:
        result, n = results[name], results[name]["tokens"]
        retention = np.nanmean(result["dynamics"]["retention"][t_index][:, :, n:], axis=1)   # (blocks, HW)
        depth = retention.shape[0]
        fig, axes = plt.subplots(2, (depth + 1) // 2, figsize=(1.9 * ((depth + 1) // 2), 4.2), squeeze=False)
        for block, ax in enumerate(axes.flat):
            if block >= depth:
                ax.axis("off")
                continue
            show_image(ax, retention[block].reshape(H, W), f"block {block}", cmap=seq, vmin=0, vmax=1)
            for r, c in (result["positions"] if block >= result["start"] else ()):
                ax.plot(c - .5, r, marker=">", color=INK, markersize=3)
        fig.suptitle(f"{name}: retention per cell, t={args.dynamics_t[t_index]} (markers: class tokens, "
                     "scanned just before their cell)", fontsize=9)
        save(fig, output, f"fig8b_retention_maps_{name}.png")


def plot_sasf(plt, output, results, args):
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), squeeze=False)
    for (name, result), color in zip(results.items(), MODEL_COLORS):
        d = result["dynamics"]
        blocks = np.arange(d["sasf_change"].shape[1])
        for ti, (t, style) in enumerate(zip(args.dynamics_t, ("-", "--", ":", "-."))):
            axes[0, 0].plot(blocks, d["sasf_change"][ti].mean(axis=1), color=color, linestyle=style,
                            label=f"{name}, t={t}")
    axes[0, 0].set(title="||SASF(x) - x|| / ||x|| on the grid states", xlabel="block")
    axes[0, 0].legend(fontsize=7)
    ax = axes[0, 1]
    width = .8 / len(names)
    for mi, (name, result) in enumerate(results.items()):
        share = result["dynamics"]["sasf_share"][len(args.dynamics_t) // 2].mean(axis=1)    # (blocks, n_dil)
        bottom = np.zeros(share.shape[0])
        for di in range(share.shape[1]):
            ax.bar(np.arange(share.shape[0]) + mi * width, share[:, di], width=width, bottom=bottom,
                   color=BLOCK_COLORS[di % 2] if di < 2 else NEUTRAL, edgecolor=MODEL_COLORS[mi], linewidth=.6,
                   label=f"{name}: dilation {result['dilations'][di]}")
            bottom += share[:, di]
    ax.set(title=f"share of each dilation, t={args.dynamics_t[len(args.dynamics_t) // 2]}", xlabel="block", ylim=(0, 1))
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1))
    save(fig, output, "fig9_sasf.png")


def plot_ablation(plt, output, name, result, args):
    ab = result["ablation"]
    names, base = ab["names"], ab["v_mse"][0]
    rows = [(n, (ab["v_mse"][i] - base).mean(axis=0)) for i, n in enumerate(names) if i]
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, .25 * len(rows))), squeeze=False)
    ax = axes[0, 0]
    means = [r[1].mean() for r in rows]
    errors = [1.96 * r[1].std(ddof=1) / np.sqrt(len(r[1])) if len(r[1]) > 1 else 0 for r in rows]
    ax.barh(range(len(rows)), means, xerr=errors, color=MODEL_COLORS[0])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, linewidth=.8)
    ax.set(xlabel="Δ velocity MSE vs full (mean over t), ±1.96 SEM", title=f"{name}: class-token ablations")
    ax = axes[0, 1]
    for (route, values), color in zip(ab["shares"].items(), MODEL_COLORS):
        ax.plot(args.ablation_t, np.median(values, axis=1), marker="o", color=color, label=route.replace("_other", ""))
    ax.set(xlabel="t (0 = noise, 1 = clean)", ylabel="median share",
           title="class change carried by each route:\n||route→other − full||² / ||all→other − full||²")
    ax.legend(fontsize=7)
    save(fig, output, f"fig10_ablation_{name}.png")


def plot_reach(plt, output, name, result, args, H, W):
    seq, _ = colormaps()
    n = result["tokens"]
    bi = len(args.attn_blocks) - 1
    if not n or args.attn_blocks[bi] < result["start"]:
        return
    write, direct = result["influence"]["write"][bi], result["influence"]["direct_write"][bi]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), squeeze=False)
    for ax, maps, title in ((axes[0, 0], write, "writes: scan + SASF"), (axes[0, 1], direct, "writes: scan alone")):
        show_image(ax, maps.sum(axis=1).mean(axis=0), title, cmap=seq)
        for r, c in result["positions"]:
            ax.plot(c - .5, r, marker=">", color=INK, markersize=4)
    ax = axes[0, 2]
    x = np.arange(n)
    ax.bar(x - .2, reach(write[None])[0].mean(0), width=.4, color=MODEL_COLORS[0], label="scan + SASF")
    ax.bar(x + .2, reach(direct[None])[0].mean(0), width=.4, color=MODEL_COLORS[1], label="scan alone")
    ax.set(xlabel="class token", ylabel="cells ≥ 10% of its max", title="reach per token", xticks=x)
    ax.legend(fontsize=7)
    fig.suptitle(f"{name}: class-token influence on the image, block {args.attn_blocks[bi]}, t={args.attn_t}", fontsize=9)
    save(fig, output, f"fig11_reach_{name}.png")


def plot_branches(plt, output, results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), squeeze=False)
    for (name, result), color in zip(results.items(), MODEL_COLORS):
        n = result["tokens"]
        for ax, kind in zip(axes[0], ("mixer", "ffn")):
            image = result["branches"][kind][:, :, n:]
            median = np.median(image.reshape(image.shape[0], -1), axis=1)
            ax.plot(image.reshape(image.shape[0], -1).max(axis=1) / median, color=color, label=f"{name} max")
            ax.plot(np.percentile(image.reshape(image.shape[0], -1), 99, axis=1) / median, color=color,
                    linestyle="--", label=f"{name} p99")
            ax.set(title=f"{kind} branch increment, image tokens", xlabel="block", ylabel="/ median")
            ax.legend(fontsize=7)
    fig.suptitle("Which branch makes high-norm tokens (Registers Matter App. C.1: after the MLP in DiTs)", fontsize=9)
    save(fig, output, "fig12_branch_norms.png")


def plot_outliers(plt, output, results, args, H, W):
    seq, _ = colormaps()
    names = list(results)
    t_index = args.timesteps.index(args.norm_t)
    fig, axes = plt.subplots(len(names), len(args.norm_blocks), figsize=(3 * len(args.norm_blocks), 3 * len(names)),
                             squeeze=False)
    for row, name in enumerate(names):
        maps = outlier_maps(results[name]["stats"]["norm"][t_index], results[name]["tokens"], H, W)
        for col, block in enumerate(args.norm_blocks):
            show_image(axes[row, col], maps[block], f"{name}, block {block}", cmap=seq, vmin=0)
    fig.suptitle(f"Where the top-1% image-token norms sit (fraction of images), t={args.norm_t}", fontsize=9)
    save(fig, output, "fig13_outlier_locations.png")


# Main ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", action="append", required=True,
                        metavar="NAME_CONFIG_CHECKPOINT", help="NAME CONFIG CHECKPOINT|auto[:TEXT] [KEY=VALUE ...]")
    parser.add_argument("--data-dir", required=True, help="Root containing train/ and val/ class folders")
    parser.add_argument("--out-dir", required=True, help="New or empty directory")
    parser.add_argument("--search-root", default=".")
    parser.add_argument("--ema", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--attn-batch-size", type=int, default=2, help="Images per exact decomposition (float64)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stats-per-class", type=int, default=5, help="Val images per class for norms, TV, branches")
    parser.add_argument("--probe-train-per-class", type=int, default=20)
    parser.add_argument("--probe-val-per-class", type=int, default=10)
    parser.add_argument("--probe-iterations", type=int, default=2000)
    parser.add_argument("--timesteps", type=float, nargs="+", default=[round(i / 10, 1) for i in range(11)] + [.98])
    parser.add_argument("--norm-t", type=float, default=.5)
    parser.add_argument("--norm-blocks", type=int, nargs=2, default=[5, 10])
    parser.add_argument("--patch-t", type=float, nargs="+", default=[.1, .5, .98])
    parser.add_argument("--patch-block", type=int, default=10)
    parser.add_argument("--pca-t", type=float, default=.1)
    parser.add_argument("--pca-blocks", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--probe-block", type=int, default=5)
    parser.add_argument("--probe-t", type=float, nargs="+", default=[.1, .3])
    parser.add_argument("--attn-t", type=float, default=.5)
    parser.add_argument("--attn-blocks", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--attn-images", type=int, default=32)
    parser.add_argument("--figure-images", type=int, default=6)
    parser.add_argument("--dynamics-per-class", type=int, default=1)
    parser.add_argument("--dynamics-t", type=float, nargs="+", default=[.1, .5, .9])
    parser.add_argument("--ablation-per-class", type=int, default=2)
    parser.add_argument("--ablation-t", type=float, nargs="+", default=[.1, .3, .5, .7, .9])
    args = parser.parse_args()
    for spec in args.model:
        if len(spec) < 3:
            parser.error("--model needs NAME CONFIG CHECKPOINT")
    if len(args.model) > len(MODEL_COLORS):
        parser.error(f"At most {len(MODEL_COLORS)} models keep distinguishable colours")
    for t in [args.norm_t, *args.patch_t]:
        if t not in args.timesteps:
            parser.error(f"t={t} must be listed in --timesteps")
    for key in ("stats_per_class", "dynamics_per_class", "ablation_per_class"):
        if not 0 < getattr(args, key) <= args.probe_val_per_class:
            parser.error(f"--{key.replace('_', '-')} must be in [1, --probe-val-per-class]")
    if args.figure_images > args.attn_images:
        parser.error("--figure-images must not exceed --attn-images")
    return parser, args


def main():
    parser, args = parse_args()
    # Token slots are NaN in blocks before the tokens enter; averaging them is expected.
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("out-dir must be empty; choose a new directory to preserve previous results")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from scripts.evaluate import build_model
    from src.utils import load_config

    def log(text):
        print(text, flush=True)

    built = []
    for name, config, checkpoint, *overrides in args.model:
        cfg = apply_overrides(load_config(config), overrides)
        if cfg["experiment"]["model"] != "spatial_mamba":
            parser.error(f"{name}: this diagnostic is for model: spatial_mamba (use measure_internals.py for vmamba)")
        net = build_model(cfg)
        try:
            check_supported(net)
        except ValueError as error:
            parser.error(f"{name}: {error}")
        if net.in_context_len and min(args.probe_block, *args.attn_blocks) < net.in_context_start:
            parser.error(f"{name}: --probe-block and --attn-blocks must be >= {net.in_context_start}, "
                         "the block where the tokens enter")
        if checkpoint == "auto" or checkpoint.startswith("auto:"):
            checkpoint = discover_checkpoint(net, args.search_root, checkpoint.partition(":")[2])
        elif (Path(checkpoint) / "data.pkl").is_file():
            checkpoint = str(repack_checkpoint(checkpoint))
        elif not Path(checkpoint).is_file():
            parser.error(f"{name}: checkpoint not found: {checkpoint}")
        log(f"{name}: {checkpoint}")
        built.append((name, cfg, net, checkpoint))

    first = built[0][1]["model"]
    size, num_classes, depth = first["input_size"], first["num_classes"], first["depth"]
    if any((c["model"]["input_size"], c["model"]["num_classes"], c["model"]["depth"]) != (size, num_classes, depth)
           for _, c, _, _ in built):
        parser.error("All models must share input_size, num_classes and depth")
    if any(b >= depth for b in [*args.norm_blocks, *args.pca_blocks, *args.attn_blocks, args.probe_block, args.patch_block]):
        parser.error(f"Block indices must be < depth {depth}")

    log("Loading images")
    val_records, classes = validation_records(args.data_dir, num_classes, args.probe_val_per_class, args.seed)
    val = load_images(val_records, size, args.workers)
    train = load_images(train_records(args.data_dir, classes, args.probe_train_per_class, args.seed), size, args.workers)
    every = args.probe_val_per_class
    stats_indices = per_class(every, args.stats_per_class, len(val_records))
    dynamics_indices = per_class(every, args.dynamics_per_class, len(val_records))
    ablation_indices = per_class(every, args.ablation_per_class, len(val_records))
    attn_indices = [stats_indices[i] for i in np.linspace(0, len(stats_indices) - 1, args.attn_images).round().astype(int)]
    figure_positions = np.linspace(0, len(attn_indices) - 1, args.figure_images).round().astype(int).tolist()
    figure_indices = [attn_indices[i] for i in figure_positions]
    display_images = (val["pixels"][figure_indices].float() / 255).permute(0, 2, 3, 1).numpy()

    results = {}
    for name, cfg, net, checkpoint in built:
        log(f"\n=== {name} ===")
        info = load_weights(net, checkpoint, args.ema)
        log(f"  checkpoint epoch {info.get('epoch')}, EMA{args.ema}")
        net = net.to(device).eval()
        H = W = net.grid_size
        noise_scale, t_eps = cfg["diffusion"]["noise_scale"], cfg["diffusion"]["t_eps"]
        log("  norms and total variation")
        stats = measure_statistics(net, val, stats_indices, args, noise_scale, device)
        log("  branch norms")
        branches = measure_branches(net, val, stats_indices, args, noise_scale, device)
        log("  PCA maps")
        pca = measure_pca(net, val, figure_indices, args, noise_scale, device)
        probes = measure_probes(net, train, val, args, noise_scale, device, num_classes, log)
        log("  exact influence (scan + SASF)")
        influence, influence_error = measure_influence(net, val, attn_indices, args, noise_scale, device)
        log(f"    decomposition vs mixer: max relative error {influence_error:.2e}")
        log("  scan memory and SASF")
        dynamics = measure_dynamics(net, val, dynamics_indices, args, noise_scale, device)
        ablation = None
        if net.in_context_len:
            log("  class-token ablations")
            ablation = measure_ablation(net, val, ablation_indices, args, noise_scale, t_eps, device, num_classes)
        results[name] = dict(stats=stats, branches=branches, pca=pca, probes=probes, influence=influence,
                             influence_error=influence_error, dynamics=dynamics, ablation=ablation,
                             kernels=sasf_kernels(net), dilations=list(net.sasf_dilations),
                             tokens=net.in_context_len, start=net.in_context_start, layout="positions",
                             positions=[list(p) for p in net.in_context_positions], checkpoint=checkpoint,
                             info=info, num_classes=num_classes, figure_positions=figure_positions)
        arrays = dict(stats, pca=pca, **{f"branch_{k}": v for k, v in branches.items()},
                      **{f"influence_{k}": v for k, v in influence.items()},
                      **{f"dynamics_{k}": v for k, v in dynamics.items()},
                      **{f"probe_{p}_{k}": v for p, entry in probes.items() for k, v in entry.items()})
        if ablation:
            arrays.update(ablation_v_mse=ablation["v_mse"], ablation_drift=ablation["drift"],
                          **{f"ablation_share_{k}": v for k, v in ablation["shares"].items()})
        np.savez_compressed(output / f"metrics_{name}.npz", **arrays)
        net.to("cpu")
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    epochs = {name: result["info"].get("epoch") for name, result in results.items()}
    if len(set(epochs.values())) > 1:
        log(f"WARNING: checkpoints come from different epochs, so ratios mix training length: {epochs}")
    reference = built[0][0]
    tables, intervals = ratio_tables(results, reference, args)
    np.savez_compressed(output / "tv_ratio_tables.npz",
                        **{f"{arm}_{key}": value for arm, entry in tables.items() for key, value in entry.items()})
    source_root = Path(__file__).resolve().parents[1]
    files = ("src/internals.py", "src/internals_spatial.py", "scripts/measure_internals.py",
             "scripts/measure_internals_spatial.py", "src/models/spatial_mamba.py")
    summary = dict(papers=["arXiv 2605.16147, Figures 1-7", "arXiv 2405.14858, Sec. 3.2 and Table 1"],
                   arguments=vars(args), reference=reference, epochs=epochs,
                   images_input_sha256=hashlib.sha256(val["pixels"][stats_indices].numpy().tobytes()).hexdigest(),
                   source_sha256={p: hashlib.sha256((source_root / p).read_bytes()).hexdigest() for p in files},
                   ratios_vs_reference=intervals,
                   models={name: model_summary(result, args, net_grid, net_grid)
                           for (name, result), net_grid in zip(results.items(), [size // c["model"]["patch_size"]
                                                                                for _, c, _, _ in built])},
                   caveats=["Spatial-Mamba has no attention: influence maps are the exact linear decomposition of "
                            "the scan + SASF with Delta, B, C frozen at their actual values, not attention",
                            "Blocks are zero-based",
                            "Each model is one training run; differences include training-run variation",
                            "Ablations are teacher-forced velocity MSE at fixed t, not generation quality or FID",
                            "Probes use the val split for scoring and train images for fitting"])
    (output / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2, allow_nan=False), encoding="utf-8")

    plt = setup_style()
    H = W = size // first["patch_size"]
    plot_fig1(plt, output, results, display_images, args)
    plot_fig2(plt, output, results, args)
    plot_fig3(plt, output, results, args)
    plot_fig4a(plt, output, tables, args)
    plot_fig4b(plt, output, results, display_images, args)
    plot_probes(plt, output, results, args)
    plot_fig6(plt, output, results, display_images, args)
    plot_fig7(plt, output, results, args)
    plot_dynamics(plt, output, results, args, H, W)
    plot_sasf(plt, output, results, args)
    plot_branches(plt, output, results)
    plot_outliers(plt, output, results, args, H, W)
    for name, result in results.items():
        if result["ablation"]:
            plot_ablation(plt, output, name, result, args)
        plot_reach(plt, output, name, result, args, H, W)
    log(f"\nSaved {output.resolve()}")


if __name__ == "__main__":
    main()
