"""Reproduce Figures 1-7 of "Registers Matter" (arXiv 2605.16147) on JiT-VMamba checkpoints.

Inference only: no training, no sampling, no FID. Run with --help. Pass one
--model per checkpoint; the first one is the reference of the TV and norm ratios:

  python scripts/measure_internals.py --data-dir DATA --out-dir OUT \
    --model baseline CONFIG_A CKPT_A \
    --model prefix1 CONFIG_B auto model.in_context_len=1 \
    --model row8 CONFIG_C auto:row

`auto` searches --search-root for the only run whose weights fit that model;
`auto:TEXT` also requires TEXT in the path (row and prefix K=8 have equal shapes).
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from scripts.measure_redundancy import ValidationImages, load_weights, validation_records
from src.internals import (block_features, discover_checkpoint, feature_statistics, fit_probes, influence_maps,
                           mixer_inputs, noised, pca_rgb, repack_checkpoint, scan_contributions, tokens_at)

MODEL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")   # entity colours, fixed by --model order
BLOCK_COLORS = ("#6250d6", "#eda100")
SURFACE, INK, MUTED, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#e8e7e4"


def apply_overrides(cfg, overrides):
    """KEY=VALUE on existing dotted config keys; a typo is an error, not a new key."""
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"Override must be KEY=VALUE: {item}")
        node, parts = cfg, key.split(".")
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(f"Unknown config key: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def train_records(root, classes, per_class, seed):
    """Balanced deterministic training subset, labelled like training's ImageFolder."""
    rng, records = random.Random(seed), []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for index, label in enumerate(classes):
        paths = sorted(p for p in (Path(root) / "train" / label).rglob("*")
                       if p.is_file() and p.suffix.lower() in extensions)
        if len(paths) < per_class:
            raise ValueError(f"{label}: need {per_class} train images, found {len(paths)}")
        for path in sorted(rng.sample(paths, per_class)):
            records.append((str(path), index, f"train/{label}/{path.name}"))
    return records


def load_tensors(records, size, seed):
    loader = torch.utils.data.DataLoader(ValidationImages(records, size, seed), batch_size=256, num_workers=2)
    parts = list(loader)
    return tuple(torch.cat([p[i] for p in parts]) for i in range(3))


def batches(total, size):
    return [(i, min(i + size, total)) for i in range(0, total, size)]


def measure_statistics(net, data, indices, args, noise_scale, device):
    images, labels, noise = data
    collected = []
    for start, stop in batches(len(indices), args.batch_size):
        chosen = indices[start:stop]
        collected.append(feature_statistics(
            net, images[chosen].to(device), labels[chosen].to(device), noise[chosen].to(device),
            args.timesteps, list(range(len(net.blocks))), noise_scale))
    return {key: np.concatenate([c[key] for c in collected], axis=2) for key in collected[0]}


def measure_pca(net, data, indices, args, noise_scale, device):
    images, labels, noise = data
    H = W = net.grid_size
    chosen = indices
    x, y, e = images[chosen].to(device), labels[chosen].to(device), noise[chosen].to(device)
    time = torch.full((len(chosen),), args.pca_t, device=device)
    features = block_features(net, noised(x, e, args.pca_t, noise_scale), time, y, args.pca_blocks)
    return np.stack([np.stack([pca_rgb(features[b][i, tokens_at(net, b):], H, W).cpu().numpy()
                               for b in args.pca_blocks]) for i in range(len(chosen))])


def extract_tokens(net, data, t, mode, block, noise_scale, batch_size, device, num_classes):
    images, labels, noise = data
    collected = []
    for start, stop in batches(len(images), batch_size):
        x, e = images[start:stop].to(device), noise[start:stop].to(device)
        y = labels[start:stop].to(device) if mode == "cond" else torch.full((stop - start,), num_classes, device=device)
        time = torch.full((stop - start,), t, device=device)
        collected.append(block_features(net, noised(x, e, t, noise_scale), time, y, [block])[block].cpu())
    return torch.cat(collected)


def measure_probes(net, train, val, args, noise_scale, device, num_classes, log):
    """Per-token probes, at each t, with the true label and with the null label.

    "null" is the honest reading: the class never enters through adaLN or the
    tokens, so accuracy can only come from the image. "cond" hands the class in.
    """
    results = {}
    for t in args.probe_t:
        for mode in ("null", "cond"):
            log(f"  probes t={t} labels={mode}")
            x_train = extract_tokens(net, train, t, mode, args.probe_block, noise_scale, args.batch_size, device, num_classes)
            x_val = extract_tokens(net, val, t, mode, args.probe_block, noise_scale, args.batch_size, device, num_classes)
            accuracy, decay = fit_probes(x_train, train[1], x_val, val[1], num_classes,
                                         iterations=args.probe_iterations, seed=args.seed, device=device)
            results[f"t{t}_{mode}"] = dict(accuracy=accuracy, l2=decay,
                                           norm=x_val.norm(dim=-1).mean(dim=0).numpy(),
                                           spread=x_val.std(dim=0).mean(dim=-1).numpy())
            del x_train, x_val
    return results


def measure_influence(net, data, indices, args, noise_scale, device):
    images, labels, noise = data
    H = W = net.grid_size
    n = net.in_context_len
    cells, read, write, worst = [], [], [], 0.0
    for start, stop in batches(len(indices), args.attn_batch_size):
        chosen = indices[start:stop]
        x, y, e = images[chosen].to(device), labels[chosen].to(device), noise[chosen].to(device)
        time = torch.full((len(chosen),), args.attn_t, device=device)
        inputs = mixer_inputs(net, noised(x, e, args.attn_t, noise_scale), time, y, args.attn_blocks)
        per_block = []
        for block in args.attn_blocks:
            mixer = net.blocks[block].mixer
            influence, rebuilt = scan_contributions(mixer, inputs[block], H, W)
            with torch.inference_mode():
                actual = mixer(inputs[block], H, W).float()
            worst = max(worst, float((rebuilt - actual).abs().max() / actual.abs().max().clamp_min(1e-12)))
            per_block.append([m.cpu().numpy() for m in influence_maps(influence, tokens_at(net, block), H, W)])
        cells.append(np.stack([p[0] for p in per_block]))
        read.append(np.stack([p[1] for p in per_block]))
        write.append(np.stack([p[2] for p in per_block]))
    if worst > 1e-3:
        raise RuntimeError(f"Scan decomposition disagrees with the mixer: relative error {worst:.3g}")
    return dict(cells=np.concatenate(cells, axis=1), read=np.concatenate(read, axis=1),
                write=np.concatenate(write, axis=1)), worst


def bootstrap_ratio(reference, other, seed, rounds=2000):
    """95% interval of mean(other) / mean(reference); same images in both."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(reference), (rounds, len(reference)))
    ratios = other[draws].mean(axis=1) / reference[draws].mean(axis=1)
    return [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]


def tv_grid(args):
    return [i for i, t in enumerate(args.timesteps) if abs(t * 10 - round(t * 10)) < 1e-9]


def ratio_tables(results, reference, args):
    """{arm: {raw, scale_free}} of [t, block + avg], and interval rows."""
    grid = tv_grid(args)
    base = results[reference]["stats"]
    tables, intervals = {}, {}
    for name, result in results.items():
        if name == reference:
            continue
        stats = result["stats"]
        raw = stats["tv"][grid].mean(axis=2) / base["tv"][grid].mean(axis=2)
        scaled = (stats["tv"][grid] / stats["grid_norm"][grid]).mean(axis=2) / \
            (base["tv"][grid] / base["grid_norm"][grid]).mean(axis=2)
        tables[name] = dict(raw=np.column_stack([raw, raw.mean(axis=1)]),
                            scale_free=np.column_stack([scaled, scaled.mean(axis=1)]))
        low = [i for i, t in enumerate(args.timesteps) if t <= 0.2]
        intervals[name] = dict(
            raw_all_t=float(stats["tv"].mean() / base["tv"].mean()),
            raw_all_t_ci95=bootstrap_ratio(base["tv"].mean(axis=(0, 1)), stats["tv"].mean(axis=(0, 1)), args.seed),
            raw_t_le_0_2=float(stats["tv"][low].mean() / base["tv"][low].mean()),
            raw_t_le_0_2_ci95=bootstrap_ratio(base["tv"][low].mean(axis=(0, 1)), stats["tv"][low].mean(axis=(0, 1)), args.seed),
            scale_free_t_le_0_2=float((stats["tv"][low] / stats["grid_norm"][low]).mean() /
                                      (base["tv"][low] / base["grid_norm"][low]).mean()),
            patch_norm_ratio_all_t=float(stats["grid_norm"].mean() / base["grid_norm"].mean()),
            patch_norm_ratio_ci95=bootstrap_ratio(base["grid_norm"].mean(axis=(0, 1)),
                                                  stats["grid_norm"].mean(axis=(0, 1)), args.seed))
    return tables, intervals


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) or np.isinf(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def model_summary(name, result, args):
    stats, n = result["stats"], result["tokens"]
    t_index = args.timesteps.index(args.norm_t)
    norms = {}
    for block in args.norm_blocks:
        mean = np.nanmean(stats["norm"][t_index, block], axis=0)
        patch = float(np.median(mean[n:]))
        entry = dict(median_patch_norm=patch, max_patch_norm_over_median=float(mean[n:].max() / patch))
        if n and block >= result["start"]:
            entry.update(token_norms=mean[:n], max_token_norm_over_median_patch=float(mean[:n].max() / patch),
                         min_token_norm_over_median_patch=float(mean[:n].min() / patch))
        norms[f"block_{block}"] = entry
    cells = result["influence"]["cells"]
    peak = cells.reshape(*cells.shape[:2], -1)
    probes = {}
    for key, probe in result["probes"].items():
        probes[key] = dict(token_accuracy=probe["accuracy"][:n], token_norm=probe["norm"][:n],
                           token_image_spread=probe["spread"][:n],
                           patch_accuracy_mean=float(probe["accuracy"][n:].mean()),
                           patch_accuracy_max=float(probe["accuracy"][n:].max()))
    return dict(tokens=n, start_block=result["start"], layout=result["layout"], checkpoint=result["checkpoint"],
                epoch=result["info"].get("epoch"), fig2_7a_norms=norms, probes=probes,
                fig1_cell_map_max_over_median=float((peak.max(axis=-1) / np.median(peak, axis=-1)).mean()),
                influence_rebuild_relative_error=result["influence_error"])


# Figures ───────────────────────────────────────────────────────────────────


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
                         "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 1.4})
    return plt


def colormaps():
    from matplotlib.colors import LinearSegmentedColormap
    sequential = LinearSegmentedColormap.from_list("seq", ["#e7f0fb", "#7fb0ea", "#2a78d6", "#123f7a"])
    diverging = LinearSegmentedColormap.from_list("div", ["#2a78d6", NEUTRAL, "#e34948"])
    return sequential, diverging


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / name, dpi=170)
    import matplotlib.pyplot as plt
    plt.close(fig)


def show_image(ax, array, title=None, **kwargs):
    ax.imshow(array, interpolation="nearest", **kwargs)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)


def plot_fig1(plt, output, results, images, args):
    seq, _ = colormaps()
    names = list(results)
    block = args.attn_blocks[-1]
    block_index = args.attn_blocks.index(block)
    figure_slots = results[names[0]]["figure_positions"]
    fig, axes = plt.subplots(len(figure_slots), 1 + len(names), figsize=(2 * (1 + len(names)), 2 * len(figure_slots)), squeeze=False)
    for row, slot in enumerate(figure_slots):
        show_image(axes[row, 0], images[row], "input" if row == 0 else None)
        for col, name in enumerate(names):
            show_image(axes[row, col + 1], results[name]["influence"]["cells"][block_index, slot], cmap=seq,
                       title=name if row == 0 else None)
    fig.suptitle(f"Fig. 1 analogue: mean scan influence of each cell on the other cells (block {block}, t={args.attn_t})", fontsize=10)
    save(fig, output, "fig1_influence_maps.png")


def plot_fig2(plt, output, results, args):
    names = list(results)
    t_index = args.timesteps.index(args.norm_t)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        result = results[name]
        n = result["tokens"]
        for block, color in zip(args.norm_blocks, BLOCK_COLORS):
            mean = np.nanmean(result["stats"]["norm"][t_index, block], axis=0)
            ax.plot(np.arange(len(mean)), mean, color=color, label=f"block {block}")
        if n:
            ax.axvspan(-.5, n - .5, color=NEUTRAL, zorder=0, label="class tokens")
        ax.set(title=name, xlabel="token index", ylabel="token norm")
        ax.legend(fontsize=8)
    fig.suptitle(f"Fig. 2 analogue: token norms at t={args.norm_t}, mean over images", fontsize=10)
    save(fig, output, "fig2_token_norms.png")


def plot_fig3(plt, output, results, args):
    fig, axes = plt.subplots(1, len(args.patch_t), figsize=(4.2 * len(args.patch_t), 3.2), squeeze=False)
    for ax, t in zip(axes[0], args.patch_t):
        t_index = args.timesteps.index(t)
        for (name, result), color in zip(results.items(), MODEL_COLORS):
            n = result["tokens"]
            mean = np.nanmean(result["stats"]["norm"][t_index, args.patch_block, :, n:], axis=0)
            ax.plot(np.arange(len(mean)), mean, color=color, label=name)
        ax.set(title=f"t = {t}", xlabel="image token index", ylabel="token norm")
        ax.legend(fontsize=8)
    fig.suptitle(f"Fig. 3 analogue: image-token norms after block {args.patch_block}", fontsize=10)
    save(fig, output, "fig3_patch_norms.png")


def plot_fig4a(plt, output, tables, args):
    from matplotlib.colors import TwoSlopeNorm
    _, div = colormaps()
    grid = [args.timesteps[i] for i in tv_grid(args)]
    arms = list(tables)
    fig, axes = plt.subplots(len(arms), 2, figsize=(11, 3.6 * len(arms)), squeeze=False)
    for row, arm in enumerate(arms):
        for col, (key, label) in enumerate((("raw", "TV ratio"), ("scale_free", "TV / mean patch norm, ratio"))):
            ax, values = axes[row, col], tables[arm][key]
            spread = float(np.abs(values - 1).max()) or 1e-3
            image = ax.imshow(values[::-1], aspect="auto", cmap=div, norm=TwoSlopeNorm(1, 1 - spread, 1 + spread))
            ax.set_xticks(range(values.shape[1]), [str(i) for i in range(values.shape[1] - 1)] + ["avg"])
            ax.set_yticks(range(len(grid)), [f"{t:.1f}" for t in grid[::-1]])
            ax.grid(False)
            for y, line in enumerate(values[::-1]):
                for x, value in enumerate(line):
                    ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
            ax.set(title=f"{arm} / baseline: {label}", xlabel="block", ylabel="t")
            fig.colorbar(image, ax=ax, shrink=.85)
    fig.suptitle("Fig. 4a analogue: blue = smoother than the baseline. Tokens enter at block 4; earlier blocks differ only by training.", fontsize=10)
    save(fig, output, "fig4a_tv_ratio.png")


def plot_fig4b(plt, output, results, images, args):
    names = list(results)
    for block_index, block in enumerate(args.pca_blocks):
        rows = images.shape[0]
        fig, axes = plt.subplots(rows, 1 + len(names), figsize=(2 * (1 + len(names)), 2 * rows), squeeze=False)
        for row in range(rows):
            show_image(axes[row, 0], images[row], "input" if row == 0 else None)
            for col, name in enumerate(names):
                show_image(axes[row, col + 1], results[name]["pca"][row, block_index], name if row == 0 else None)
        fig.suptitle(f"Fig. 4b analogue: PCA of image tokens, block {block}, t={args.pca_t}. Colours are not comparable across panels.", fontsize=9)
        save(fig, output, f"fig4b_pca_block{block}.png")


def plot_fig5(plt, output, results, args):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    seq, _ = colormaps()
    names = list(results)
    for t in args.probe_t:
        fig, axes = plt.subplots(len(names), 2, figsize=(12, 2.8 * len(names)), squeeze=False)
        for row, name in enumerate(names):
            result = results[name]
            n = result["tokens"]
            for col, (mode, label) in enumerate((("null", "null label: image evidence only"), ("cond", "true label: class is given to the model"))):
                ax, probe = axes[row, col], result["probes"][f"t{t}_{mode}"]
                scale = Normalize(float(probe["norm"].min()), float(probe["norm"].max()) or 1)
                ax.bar(range(len(probe["accuracy"])), probe["accuracy"], width=.8, color=seq(scale(probe["norm"])))
                ax.axhline(1 / result["num_classes"], color=MUTED, linewidth=.8, linestyle=":")
                if n:
                    ax.axvline(n - .5, color=INK, linewidth=.8)
                ax.set(ylim=(0, 1), title=f"{name}: {label}", xlabel="token index (class tokens first)" if n else "image token index", ylabel="accuracy")
                fig.colorbar(ScalarMappable(scale, seq), ax=ax, label="token norm", shrink=.85)
        fig.suptitle(f"Fig. 5 analogue: per-token linear probes after block {args.probe_block}, t={t}. Dotted line is chance.", fontsize=10)
        save(fig, output, f"fig5_probing_t{t}.png")


def plot_fig6(plt, output, results, images, args):
    seq, _ = colormaps()
    block_index = args.attn_blocks.index(args.attn_blocks[-1])
    t = args.probe_t[0]
    for name, result in results.items():
        n = result["tokens"]
        if not n:
            continue
        rows = min(3, images.shape[0])
        probe = result["probes"][f"t{t}_null"]
        fig, axes = plt.subplots(2 * rows, 1 + n, figsize=(1.7 * (1 + n), 3.4 * rows), squeeze=False)
        for row in range(rows):
            slot = result["figure_positions"][row]
            for offset, (key, label) in enumerate((("read", "reads"), ("write", "writes"))):
                maps = result["influence"][key][block_index, slot]
                top = float(maps.max()) or 1
                axis_row = 2 * row + offset
                show_image(axes[axis_row, 0], images[row] if offset == 0 else np.ones_like(images[row]), label)
                for token in range(n):
                    title = None
                    if axis_row == 0:
                        title = f"tok {token}\nacc {probe['accuracy'][token]:.2f}\nnorm {probe['norm'][token]:.3g}"
                    show_image(axes[axis_row, token + 1], maps[token], title, cmap=seq, vmin=0, vmax=top)
        fig.suptitle(f"Fig. 6 analogue, {name}: scan influence, block {args.attn_blocks[-1]}, t={args.attn_t}. "
                     f"An empty 'reads' map means the token sees no image cell.", fontsize=9)
        save(fig, output, f"fig6_token_maps_{name}.png")


def plot_fig7(plt, output, results, args):
    arms = {k: v for k, v in results.items() if v["tokens"]}
    if not arms:
        return
    t_index = args.timesteps.index(args.norm_t)
    t = args.probe_t[0]
    fig, axes = plt.subplots(len(arms), 2, figsize=(10, 3 * len(arms)), squeeze=False)
    for row, (name, result) in enumerate(arms.items()):
        n = result["tokens"]
        ax = axes[row, 0]
        x = np.arange(n)
        for block, color in zip(args.norm_blocks, BLOCK_COLORS):
            mean = np.nanmean(result["stats"]["norm"][t_index, block], axis=0)
            ax.plot(x, mean[:n], color=color, marker="o", markersize=4, label=f"block {block} tokens")
            ax.axhline(float(np.median(mean[n:])), color=color, linewidth=.8, linestyle=":", label=f"block {block} median image token")
        ax.set(title=f"{name}: (a) class-token norms, t={args.norm_t}", xlabel="class token", ylabel="token norm", xticks=x)
        ax.legend(fontsize=7)
        ax = axes[row, 1]
        null, cond = result["probes"][f"t{t}_null"], result["probes"][f"t{t}_cond"]
        ax.bar(x, null["accuracy"][:n], width=.6, color=MODEL_COLORS[list(results).index(name)], label="null label")
        ax.plot(x, cond["accuracy"][:n], linestyle="", marker="D", markersize=6, color=INK, label="true label", clip_on=False)
        for token in x:
            if null["spread"][token] < 1e-5 * null["norm"][token]:   # float noise of the kernel only
                ax.text(token, .03, "no image\ninput", ha="center", fontsize=6.5, color=MUTED)
        ax.axhline(1 / result["num_classes"], color=MUTED, linewidth=.8, linestyle=":", label="chance")
        ax.set(ylim=(0, 1), title=f"{name}: (b) probes on class tokens, block {args.probe_block}, t={t}",
               xlabel="class token", ylabel="accuracy", xticks=x)
        ax.legend(fontsize=7)
    fig.suptitle("Fig. 7 analogue: do in-context class tokens behave like registers?", fontsize=10)
    save(fig, output, "fig7_incontext_tokens.png")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", action="append", required=True,
                        metavar="NAME_CONFIG_CHECKPOINT", help="NAME CONFIG CHECKPOINT|auto[:TEXT] [KEY=VALUE ...]")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True, help="New or empty directory")
    parser.add_argument("--search-root", default="/kaggle/input")
    parser.add_argument("--ema", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--attn-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stats-per-class", type=int, default=5, help="Val images per class for norms and TV")
    parser.add_argument("--probe-train-per-class", type=int, default=50)
    parser.add_argument("--probe-val-per-class", type=int, default=50)
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
    parser.add_argument("--attn-images", type=int, default=64)
    parser.add_argument("--figure-images", type=int, default=6)
    args = parser.parse_args()
    for spec in args.model:
        if len(spec) < 3:
            parser.error("--model needs NAME CONFIG CHECKPOINT")
    if len(args.model) > len(MODEL_COLORS):
        parser.error(f"At most {len(MODEL_COLORS)} models keep distinguishable colours")
    for t in [args.norm_t, *args.patch_t]:
        if t not in args.timesteps:
            parser.error(f"t={t} must be listed in --timesteps")
    return parser, args


def main():
    parser, args = parse_args()
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
        if cfg["experiment"]["model"] != "vmamba":
            parser.error(f"{name}: this diagnostic is validated for model: vmamba only")
        net = build_model(cfg)
        if net.in_context_len and args.probe_block < net.in_context_start:
            parser.error(f"{name}: --probe-block is before block {net.in_context_start}, where the tokens enter")
        if checkpoint == "auto" or checkpoint.startswith("auto:"):
            checkpoint = discover_checkpoint(net, args.search_root, checkpoint.partition(":")[2])
        elif (Path(checkpoint) / "data.pkl").is_file():
            checkpoint = str(repack_checkpoint(checkpoint))
        elif not Path(checkpoint).is_file():
            parser.error(f"{name}: checkpoint not found: {checkpoint}")
        log(f"{name}: {checkpoint}")
        built.append((name, cfg, net, checkpoint))

    first = built[0][1]
    size, num_classes = first["model"]["input_size"], first["model"]["num_classes"]
    if any((c["model"]["input_size"], c["model"]["num_classes"], c["model"]["depth"]) !=
           (size, num_classes, first["model"]["depth"]) for _, c, _, _ in built):
        parser.error("All models must share input_size, num_classes and depth")

    log("Loading images")
    val_records, classes = validation_records(args.data_dir, num_classes, args.probe_val_per_class, args.seed)
    val = load_tensors(val_records, size, args.seed)
    train = load_tensors(train_records(args.data_dir, classes, args.probe_train_per_class, args.seed), size, args.seed)
    stats_indices = torch.tensor([i for i in range(len(val_records)) if i % args.probe_val_per_class < args.stats_per_class])
    attn_positions = np.linspace(0, len(stats_indices) - 1, args.attn_images).round().astype(int)
    attn_indices = stats_indices[attn_positions]
    figure_positions = np.linspace(0, len(attn_indices) - 1, args.figure_images).round().astype(int).tolist()
    figure_indices = attn_indices[figure_positions]
    display_images = ((val[0][figure_indices] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    results = {}
    for name, cfg, net, checkpoint in built:
        log(f"\n=== {name} ===")
        info = load_weights(net, checkpoint, args.ema)
        log(f"  checkpoint epoch {info.get('epoch')}, EMA{args.ema}")
        net = net.to(device).eval()
        noise_scale = cfg["diffusion"]["noise_scale"]
        log("  norms and total variation")
        stats = measure_statistics(net, val, stats_indices, args, noise_scale, device)
        log("  PCA maps")
        pca = measure_pca(net, val, figure_indices, args, noise_scale, device)
        probes = measure_probes(net, train, val, args, noise_scale, device, num_classes, log)
        log("  scan influence")
        influence, influence_error = measure_influence(net, val, attn_indices, args, noise_scale, device)
        results[name] = dict(stats=stats, pca=pca, probes=probes, influence=influence, influence_error=influence_error,
                             tokens=net.in_context_len, start=net.in_context_start, layout=net.in_context_layout,
                             checkpoint=checkpoint, info=info, num_classes=num_classes,
                             figure_positions=figure_positions)
        net.to("cpu")
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
        arrays = dict(stats, pca=pca, **{f"influence_{k}": v for k, v in influence.items()},
                      **{f"probe_{p}_{k}": v for p, entry in probes.items() for k, v in entry.items()})
        np.savez_compressed(output / f"metrics_{name}.npz", **arrays)

    epochs = {name: result["info"].get("epoch") for name, result in results.items()}
    if len(set(epochs.values())) > 1:
        log(f"WARNING: checkpoints come from different epochs, so ratios mix training length: {epochs}")
    reference = built[0][0]
    tables, intervals = ratio_tables(results, reference, args)
    np.savez_compressed(output / "tv_ratio_tables.npz",
                        **{f"{arm}_{key}": value for arm, entry in tables.items() for key, value in entry.items()})
    source_root = Path(__file__).resolve().parents[1]
    summary = dict(paper="arXiv 2605.16147, Figures 1-7", arguments=vars(args), reference=reference,
                   images_input_sha256=hashlib.sha256(val[0][stats_indices].numpy().tobytes()).hexdigest(),
                   source_sha256={p: hashlib.sha256((source_root / p).read_bytes()).hexdigest()
                                  for p in ("src/internals.py", "scripts/measure_internals.py", "src/models/vmamba.py")},
                   ratios_vs_reference=intervals,
                   models={name: model_summary(name, result, args) for name, result in results.items()},
                   caveats=["SS2D has no attention: Figures 1 and 6 use the exact scan decomposition, an analogue, not attention",
                            "Blocks are zero-based; block 5 follows the paper's figure labels",
                            "Each model is one training run; differences include training-run variation",
                            "Token-specific statements hold for these two token layouts and block choices only",
                            "Not generation quality: no sampling or FID"])
    (output / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2, allow_nan=False), encoding="utf-8")

    plt = setup_style()
    plot_fig1(plt, output, results, display_images, args)
    plot_fig2(plt, output, results, args)
    plot_fig3(plt, output, results, args)
    plot_fig4a(plt, output, tables, args)
    plot_fig4b(plt, output, results, display_images, args)
    plot_fig5(plt, output, results, args)
    plot_fig6(plt, output, results, display_images, args)
    plot_fig7(plt, output, results, args)
    log(f"\nSaved {output.resolve()}")


if __name__ == "__main__":
    main()
