"""
scripts/run_experiment.py
=========================
Run any JiT experiment from a YAML config.

Usage:
    python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml

    # Override any config value from command line:
    python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml \\
        training.epochs=50 cfg.cfg_scale=2.5 cfg.label_drop_prob=0.1

    # Resume from checkpoint:
    python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml \\
        checkpoint.resume_from=experiments/cifar10/jit-s-baseline/checkpoint-last.pt
"""

import argparse
import os
import sys
import yaml
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import Denoiser
from src.utils import (
    load_config, save_checkpoint, load_checkpoint,
    paper_peak_lr, make_lr_fn, MetricLogger,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_nested(d, key_path, value):
    """Set d[a][b][c] = value given key_path='a.b.c'."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d[k]
    # Try to cast to right type
    original = d.get(keys[-1])
    if isinstance(original, bool):
        value = value.lower() in ("true", "1", "yes")
    elif isinstance(original, int):
        value = int(value)
    elif isinstance(original, float):
        value = float(value)
    elif isinstance(original, list):
        value = yaml.safe_load(value)
    d[keys[-1]] = value


def build_model(cfg):
    arch = cfg["experiment"]["model"]
    m = cfg["model"]

    if arch == "jit":
        from src.models.jit import JiT
        return JiT(
            input_size    = m["input_size"],
            patch_size    = m["patch_size"],
            in_channels   = m["in_channels"],
            hidden_size   = m["hidden_size"],
            depth         = m["depth"],
            num_heads     = m["num_heads"],
            mlp_ratio     = m["mlp_ratio"],
            attn_drop     = m["attn_drop"],
            proj_drop     = m["proj_drop"],
            num_classes   = m["num_classes"],
            bottleneck_dim= m["bottleneck_dim"],
            in_context_len  = m.get("in_context_len", 0),
            in_context_start= m.get("in_context_start", 0),
        )
    elif arch == "vim":
        from src.models.vim import JiTViM
        return JiTViM(
            input_size    = m["input_size"],
            patch_size    = m["patch_size"],
            in_channels   = m["in_channels"],
            hidden_size   = m["hidden_size"],
            depth         = m["depth"],
            num_heads     = m["num_heads"],
            mlp_ratio     = m["mlp_ratio"],
            attn_drop     = m["attn_drop"],
            proj_drop     = m["proj_drop"],
            num_classes   = m["num_classes"],
            bottleneck_dim= m["bottleneck_dim"],
            d_state       = m.get("d_state", 16),
            d_conv        = m.get("d_conv", 4),
            expand        = m.get("expand", 1),
        )
    elif arch == "vmamba":
        from src.models.vmamba import JiTVMamba
        return JiTVMamba(
            input_size    = m["input_size"],
            patch_size    = m["patch_size"],
            in_channels   = m["in_channels"],
            hidden_size   = m["hidden_size"],
            depth         = m["depth"],
            num_heads     = m["num_heads"],
            mlp_ratio     = m["mlp_ratio"],
            attn_drop     = m["attn_drop"],
            proj_drop     = m["proj_drop"],
            num_classes   = m["num_classes"],
            bottleneck_dim= m["bottleneck_dim"],
            d_state       = m.get("d_state", 16),
            d_conv        = m.get("d_conv", 3),
            expand        = m.get("expand", 1),
            K             = m.get("K", 4),
        )
    else:
        raise ValueError(f"Unknown model: {arch}. Choose jit | vim | vmamba")


def build_dataset(cfg):
    dataset = cfg["experiment"]["dataset"]
    img_size = cfg["model"]["input_size"]

    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if dataset == "cifar10":
        return torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset == "imagenet":
        return torchvision.datasets.ImageFolder(
            root="./data/imagenet/train",
            transform=T.Compose([
                T.Resize(img_size),
                T.CenterCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*",
                        help="key.subkey=value overrides, e.g. training.epochs=50")
    args = parser.parse_args()

    # Load config and apply CLI overrides
    cfg = load_config(args.config)
    for override in args.overrides:
        key, value = override.split("=", 1)
        set_nested(cfg, key, value)
        print(f"  Override: {key} = {value}")

    t_cfg  = cfg["training"]
    d_cfg  = cfg["diffusion"]
    c_cfg  = cfg["cfg"]
    s_cfg  = cfg["sampling"]
    ck_cfg = cfg["checkpoint"]

    torch.manual_seed(cfg["experiment"].get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Experiment : {cfg['experiment']['name']}")
    print(f"  Model      : {cfg['experiment']['model']}")
    print(f"  Dataset    : {cfg['experiment']['dataset']}")
    print(f"  Device     : {device}")
    print(f"  CFG        : scale={c_cfg['cfg_scale']}, drop_p={c_cfg['label_drop_prob']}")
    print(f"{'='*60}\n")

    # Build model + denoiser
    net = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {n_params/1e6:.2f} M")

    peak_lr = paper_peak_lr(t_cfg["batch_size"], t_cfg["blr"])

    denoiser = Denoiser(
        net                 = net,
        img_size            = cfg["model"]["input_size"],
        num_classes         = cfg["model"]["num_classes"],
        P_mean              = d_cfg["P_mean"],
        P_std               = d_cfg["P_std"],
        t_eps               = d_cfg["t_eps"],
        noise_scale         = d_cfg["noise_scale"],
        ema_decay1          = t_cfg["ema_decay1"],
        ema_decay2          = t_cfg["ema_decay2"],
        sampling_method     = s_cfg["method"],
        num_sampling_steps  = s_cfg["steps"],
        label_drop_prob     = c_cfg["label_drop_prob"],
        cfg_scale           = c_cfg["cfg_scale"],
        cfg_interval        = tuple(c_cfg["cfg_interval"]),
    ).to(device)
    denoiser.init_ema()

    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr           = peak_lr,
        betas        = tuple(t_cfg["betas"]),
        weight_decay = t_cfg["weight_decay"],
    )

    # Dataset + loader
    train_ds = build_dataset(cfg)
    loader = DataLoader(
        train_ds,
        batch_size  = t_cfg["batch_size"],
        shuffle     = True,
        num_workers = 2,
        pin_memory  = True,
        drop_last   = True,
    )

    steps_per_epoch = len(loader)
    warmup_steps    = t_cfg["warmup_epochs"] * steps_per_epoch
    lr_at_step      = make_lr_fn(peak_lr, warmup_steps)
    scaler          = torch.amp.GradScaler("cuda") if (device == "cuda" and t_cfg.get("amp", True)) else None

    # Output dir
    out_dir = ck_cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Resume
    start_epoch = 1
    global_step = 0
    losses      = []
    best_loss   = float("inf")

    resume = ck_cfg.get("resume_from")
    if resume and os.path.exists(resume):
        payload     = load_checkpoint(resume, denoiser, optimizer=optimizer, map_location=device)
        start_epoch = payload["epoch"] + 1
        global_step = payload["global_step"]
        losses      = payload["losses"]
        best_loss   = payload.get("best_loss", float("inf"))
        print(f"Resumed from {resume} (epoch {start_epoch}, step {global_step})\n")

    logger = MetricLogger()

    # Training loop
    for epoch in range(start_epoch, t_cfg["epochs"] + 1):
        denoiser.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{t_cfg['epochs']}", leave=False)

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            for pg in optimizer.param_groups:
                pg["lr"] = lr_at_step(global_step)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = denoiser(images, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = denoiser(images, labels)
                loss.backward()
                optimizer.step()

            denoiser.update_ema()
            global_step += 1
            epoch_loss  += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        avg = epoch_loss / len(loader)
        losses.append(avg)
        logger.log(epoch, loss=avg, lr=optimizer.param_groups[0]["lr"])
        msg = f"Epoch {epoch:3d}/{t_cfg['epochs']} | v-loss {avg:.4f}"

        # Save last
        if epoch % ck_cfg["save_last_freq"] == 0 or epoch == t_cfg["epochs"]:
            save_checkpoint(f"{out_dir}/checkpoint-last.pt",
                            epoch, global_step, denoiser, optimizer, losses, best_loss)
            msg += "  [last]"

        # Save archive
        if epoch % ck_cfg["save_archive_freq"] == 0:
            save_checkpoint(f"{out_dir}/checkpoint-ep{epoch:03d}.pt",
                            epoch, global_step, denoiser, optimizer, losses, best_loss)
            msg += f"  [archive ep{epoch:03d}]"

        # Save best
        if avg < best_loss:
            best_loss = avg
            save_checkpoint(f"{out_dir}/checkpoint-best.pt",
                            epoch, global_step, denoiser, optimizer, losses, best_loss)
            msg += f"  [best ↓ {best_loss:.4f}]"

        print(msg)

    logger.save(f"{out_dir}/metrics.json")
    print(f"\n✅ Training complete. Results → {out_dir}/")


if __name__ == "__main__":
    main()
