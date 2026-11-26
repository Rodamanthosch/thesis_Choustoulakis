"""Measure trained JiM block redundancy on paired validation denoising inputs.

Run with --help. No training, pruning of checkpoint files, or FID is performed.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.redundancy import image_noise, measure_batch, removal_variants, summarize


def validation_records(root, num_classes, images_per_class, seed):
    """Tiny-ImageNet flat/prepared val, or ImageFolder train/val. No downloads."""
    root = Path(root)
    train = root / "train"
    if train.is_dir():
        classes = sorted(p.name for p in train.iterdir() if p.is_dir())
    elif (root / "wnids.txt").is_file():
        classes = sorted((root / "wnids.txt").read_text().split())
    else:
        raise ValueError("Need train class folders or wnids.txt to verify the training label mapping")
    if len(classes) != num_classes:
        raise ValueError(f"Dataset has {len(classes)} classes, config has {num_classes}")
    groups = {name: [] for name in classes}
    annotations = root / "val" / "val_annotations.txt"
    if annotations.is_file():
        for line in annotations.read_text().splitlines():
            filename, label = line.split()[:2]
            candidates = [root / "val" / "images" / filename,
                          root / "val" / label / filename,
                          root / "val" / label / "images" / filename]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None or label not in groups:
                raise ValueError(f"Missing validation image or unknown class: {line}")
            groups[label].append(path)
    else:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for label in classes:
            groups[label] = [p for p in (root / "val" / label).rglob("*")
                             if p.is_file() and p.suffix.lower() in extensions]
    rng, records = random.Random(seed), []
    for index, label in enumerate(classes):
        paths = sorted(groups[label])
        if len(paths) < images_per_class:
            raise ValueError(f"{label}: need {images_per_class} val images, found {len(paths)}")
        for path in sorted(rng.sample(paths, images_per_class)):
            records.append((str(path), index, f"{label}/{path.name}"))
    if len({r[2] for r in records}) != len(records):
        raise ValueError("Duplicate image IDs in validation selection")
    return records, classes


class ValidationImages(torch.utils.data.Dataset):
    def __init__(self, records, size, seed):
        import torchvision.transforms as T
        self.records, self.seed = records, seed
        self.transform = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        path, label, image_id = self.records[index]
        with Image.open(path) as im:
            x = self.transform(im.convert("RGB")) * 2 - 1
        return x, label, image_noise(x.shape, image_id, self.seed)


def load_weights(net, checkpoint, ema):
    """Strict model + EMA loading; never silently evaluate random/missing EMA."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["model"], strict=True)
    if ema:
        state = payload.get(f"ema{ema}")
        names = {"net." + name for name, _ in net.named_parameters()}
        if state is None or set(state) != names:
            raise ValueError(f"EMA{ema} is absent or its parameter keys do not match this model")
        with torch.no_grad():
            for name, p in net.named_parameters():
                value = state["net." + name]
                if value.shape != p.shape:
                    raise ValueError(f"EMA shape mismatch for {name}")
                p.copy_(value)
    return {k: payload.get(k) for k in ("epoch", "global_step", "best_loss")}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output, rows, ranking, variants, timesteps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [name for name in variants if name != "baseline"]
    for mode in ("conditional", "cfg"):
        lookup = {(r["variant"], r["t"]): r["delta_v_mse"] for r in rows if r["mode"] == mode}
        values = np.array([[lookup[name, t] for t in timesteps] for name in names])
        bound = max(float(np.abs(values).max()), 1e-12)
        fig, ax = plt.subplots(figsize=(10, max(5, len(names) * .23)))
        im = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.set_xticks(range(len(timesteps)), [str(t) for t in timesteps])
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set(xlabel="t (0 = noise; 1 = clean)",
               title=f"{mode}: paired change in velocity MSE (positive = worse)")
        fig.colorbar(im, ax=ax, label="Ablated loss - full-model loss")
        fig.tight_layout()
        fig.savefig(output / f"timestep_heatmap_{mode}.png", dpi=160)
        plt.close(fig)
    selected = sorted((r for r in ranking if r["mode"] == "conditional" and r["removed"]),
                      key=lambda r: (r["removed"], r["mean_delta_v_mse"]))
    fig, ax = plt.subplots(figsize=(10, max(5, len(selected) * .23)))
    ax.barh(range(len(selected)), [r["mean_delta_v_mse"] for r in selected],
            xerr=[1.96 * (r["paired_image_sem"] or 0) for r in selected], color="steelblue")
    ax.set_yticks(range(len(selected)), [r["variant"] for r in selected], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(xlabel="Uniform-t mean delta velocity MSE; error bars = 1.96 paired image SEM",
           title="Screening ranking by removal budget; not generated-image quality")
    fig.tight_layout()
    fig.savefig(output / "ranking_conditional.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True, help="Root containing train/ and val/")
    parser.add_argument("--out-dir", required=True, help="New or empty output directory")
    parser.add_argument("--ema", type=int, choices=(0, 1, 2), default=1, help="0 = raw weights")
    parser.add_argument("--images-per-class", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timesteps", type=float, nargs="+", default=[.05, .2, .4, .6, .8, .9, .95, .98])
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use CUDA bf16; default fp32/no TF32")
    parser.add_argument("--profile", action="store_true", help="Also estimate each ablation's GMACs using fvcore")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.images_per_class < 1 or args.batch_size < 1:
        parser.error("images-per-class and batch-size must be positive")
    if len(set(args.timesteps)) != len(args.timesteps) or any(not 0 <= t <= 1 for t in args.timesteps):
        parser.error("timesteps must be unique and between 0 and 1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; JiM needs the Mamba CUDA environment")
    if args.amp and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("--amp needs a CUDA GPU with bf16 support; otherwise use fp32")
    for path in (args.config, args.checkpoint):
        if not Path(path).is_file():
            parser.error(f"File not found: {path}")
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("out-dir must be empty; choose a new directory to preserve previous results")

    from scripts.evaluate import build_model
    from src.utils import load_config
    cfg = load_config(args.config)
    if cfg["experiment"]["model"] != "vmamba":
        parser.error("This diagnostic is validated for JiM (experiment.model: vmamba)")
    m, diffusion, guidance = cfg["model"], cfg["diffusion"], cfg["cfg"]
    if diffusion["t_eps"] <= 0:
        parser.error("diffusion.t_eps must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    variants = removal_variants(m["depth"], args.window_sizes)
    records, classes = validation_records(args.data_dir, m["num_classes"], args.images_per_class, args.seed)
    net = build_model(cfg)
    checkpoint_info = load_weights(net, args.checkpoint, args.ema)
    net = net.to(device).eval()
    total_params = sum(p.numel() for p in net.parameters())
    block_params = [sum(p.numel() for p in block.parameters()) for block in net.blocks]
    complexity = None
    if args.profile:
        from src.flops_counter import count_complexity, _classify_unsupported
        complexity = count_complexity(net, m["input_size"], m["in_channels"], m["num_classes"], device)
        worrying = _classify_unsupported(complexity["unsupported_ops"])[2]
        if worrying or not any("SelectiveScan" in key for key in complexity["macs_by_op"]):
            raise RuntimeError(f"Unreliable FLOP trace: missing scan or unsupported expensive ops: {worrying}")

    dataset = ValidationImages(records, m["input_size"], args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=0, pin_memory=device.type == "cuda")
    shape = (len(variants), 2, len(args.timesteps), len(records))
    errors, drift = np.empty(shape, np.float64), np.empty(shape, np.float64)
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    source_files = ["src/models/vmamba.py", "src/primitives.py", "src/redundancy.py",
                    "scripts/evaluate.py", "scripts/measure_redundancy.py"]
    protocol = dict(config=cfg, arguments=vars(args), checkpoint_info=checkpoint_info,
                    classes=classes, image_ids=[r[2] for r in records], variants=variants,
                    torch_version=torch.__version__,
                    device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                    preprocessing="RGB Resize(input_size) CenterCrop ToTensor*2-1; no clipping",
                    source_sha256={p: hashlib.sha256((source_root/p).read_bytes()).hexdigest() for p in source_files},
                    caveats=["Teacher-forced one-step validation, not FID or sampled trajectories",
                             "Ranking averages a fixed t grid uniformly, not the training t distribution",
                             "Paired SEM is across images, not independent training seeds",
                             "Selected candidates need independent validation and generation evaluation",
                             "Exportable parameter count is an estimate; original weights remain resident"])
    try:
        protocol["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        protocol["git_commit"] = None
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"{len(records)} images, {len(variants)} variants, {len(args.timesteps)} timesteps, EMA{args.ema}", flush=True)
    digest, offset = hashlib.sha256(), 0
    from tqdm import tqdm
    for images, labels, noise in tqdm(loader, desc="Paired block ablations"):
        for x, e in zip(images, noise):
            digest.update(x.numpy().tobytes())
            digest.update(e.numpy().tobytes())
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            batch_errors, batch_drift = measure_batch(
                net, images.to(device), labels.to(device), noise.to(device), args.timesteps, variants,
                num_classes=m["num_classes"], noise_scale=diffusion["noise_scale"],
                t_eps=diffusion["t_eps"], cfg_scale=guidance["cfg_scale"],
                cfg_interval=guidance["cfg_interval"])
        stop = offset + len(images)
        errors[..., offset:stop], drift[..., offset:stop] = batch_errors, batch_drift
        offset = stop
    rows, ranking = summarize(errors, drift, variants, args.timesteps, diffusion["t_eps"])
    for row in ranking:
        row["retained_blocks"] = m["depth"] - row["removed"]
        row["estimated_export_params_M"] = (total_params - sum(block_params[i] for i in row["skipped_blocks"])) / 1e6
        if complexity is not None:
            saved = sum(complexity["macs_by_module"][f"blocks.{i}"] for i in row["skipped_blocks"])
            row["estimated_forward_gmacs"] = (complexity["macs_total"] - saved) / 1e9
            row["estimated_compute_reduction"] = saved / complexity["macs_total"]
    ranking.sort(key=lambda r: (r["mode"], r["removed"], r["mean_delta_v_mse"]))
    write_csv(output / "per_timestep.csv", rows)
    write_csv(output / "ranking.csv", ranking)
    np.savez_compressed(output / "paired_errors.npz", x_mse=errors, x_drift_mse=drift,
                        variants=np.array(list(variants)), modes=np.array(["conditional", "cfg"]),
                        timesteps=args.timesteps, image_ids=np.array([r[2] for r in records]))
    summary = dict(protocol=protocol, input_sha256=digest.hexdigest(), metrics=rows,
                   ranking=ranking, complexity=complexity)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if not args.no_plots:
        plot_results(output, rows, ranking, variants, args.timesteps)
    for mode in ("conditional", "cfg"):
        print(f"\n{mode}: best three candidates PER removal budget (zero-based block indices)")
        for budget in sorted(set(len(v) for v in variants.values()) - {0}):
            candidates = [r for r in ranking if r["mode"] == mode and r["removed"] == budget][:3]
            for row in candidates:
                print(f"  {row['variant']:24s} delta_v={row['mean_delta_v_mse']:.6g} "
                      f"SEM={row['paired_image_sem']} worst_t={row['worst_t_delta_v_mse']:.6g}")
    print(f"\nSaved {output.resolve()}. Small loss changes identify candidates, not proven FID preservation.")


if __name__ == "__main__":
    main()
