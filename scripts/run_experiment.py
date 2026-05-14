"""
scripts/run_experiment.py
=========================
Run any JiT experiment from a YAML config.
Works on single-GPU (Kaggle T4) and multi-GPU (torchrun, 8xH100) automatically.

Usage:
    # Single GPU (Kaggle)
    python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml

    # Multi-GPU, e.g. 8x H100 (Google Cloud)
    torchrun --nproc_per_node=8 scripts/run_experiment.py \\
        --config configs/imagenet/jit-s-imagenet.yaml

    # Override any value:
    python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml \\
        training.epochs=50 cfg.cfg_scale=2.5

    # Resume:
    python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml \\
        checkpoint.resume_from=experiments/cifar10/jit-s-baseline/checkpoint-last.pt
"""

import argparse, os, sys, yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import Denoiser
from src.utils import (
    load_config, save_checkpoint, load_checkpoint,
    paper_peak_lr, make_lr_fn, MetricLogger,
)


# ── Distributed ───────────────────────────────────────────────────────────────

def setup_distributed():
    """Auto-detect torchrun launch. Returns (rank, local_rank, world_size, device)."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return rank, local_rank, world_size, device

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Config ────────────────────────────────────────────────────────────────────

def set_nested(d, key_path, value):
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d[k]
    original = d.get(keys[-1])
    if isinstance(original, bool):   value = value.lower() in ("true", "1", "yes")
    elif isinstance(original, int):  value = int(value)
    elif isinstance(original, float):value = float(value)
    elif isinstance(original, list): value = yaml.safe_load(value)
    d[keys[-1]] = value


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(cfg):
    arch = cfg["experiment"]["model"]
    m    = cfg["model"]
    if arch == "jit":
        from src.models.jit import JiT
        return JiT(
            input_size=m["input_size"], patch_size=m["patch_size"],
            in_channels=m["in_channels"], hidden_size=m["hidden_size"],
            depth=m["depth"], num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"],
            attn_drop=m["attn_drop"], proj_drop=m["proj_drop"],
            num_classes=m["num_classes"], bottleneck_dim=m["bottleneck_dim"],
            in_context_len=m.get("in_context_len", 0),
            in_context_start=m.get("in_context_start", 0),
        )
    elif arch == "vim":
        from src.models.vim import JiTViM
        return JiTViM(
            input_size=m["input_size"], patch_size=m["patch_size"],
            in_channels=m["in_channels"], hidden_size=m["hidden_size"],
            depth=m["depth"], num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"],
            attn_drop=m["attn_drop"], proj_drop=m["proj_drop"],
            num_classes=m["num_classes"], bottleneck_dim=m["bottleneck_dim"],
            d_state=m.get("d_state", 16), d_conv=m.get("d_conv", 4),
            expand=m.get("expand", 1),
        )
    elif arch == "vmamba":
        from src.models.vmamba import JiTVMamba
        return JiTVMamba(
            input_size=m["input_size"], patch_size=m["patch_size"],
            in_channels=m["in_channels"], hidden_size=m["hidden_size"],
            depth=m["depth"], num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"],
            attn_drop=m["attn_drop"], proj_drop=m["proj_drop"],
            num_classes=m["num_classes"], bottleneck_dim=m["bottleneck_dim"],
            d_state=m.get("d_state", 16), d_conv=m.get("d_conv", 3),
            expand=m.get("expand", 1), K=m.get("K", 4),
        )
    else:
        raise ValueError(f"Unknown model: {arch}. Choose jit | vim | vmamba")


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset(cfg):
    dataset  = cfg["experiment"]["dataset"]
    img_size = cfg["model"]["input_size"]
    data_dir = cfg["experiment"].get("data_dir", "./data")  # configurable path
    norm     = T.Normalize([0.5]*3, [0.5]*3)
    if dataset == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True,
            transform=T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), norm]),
        )
    elif dataset == "imagenet":
        return torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "train"),
            transform=T.Compose([
                T.Resize(img_size), T.CenterCrop(img_size),
                T.RandomHorizontalFlip(), T.ToTensor(), norm,
            ]),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    rank, local_rank, world_size, device = setup_distributed()
    main_proc = (rank == 0)

    cfg = load_config(args.config)
    for ov in args.overrides:
        k, v = ov.split("=", 1)
        set_nested(cfg, k, v)
        if main_proc: print(f"  Override: {k} = {v}")

    t_cfg  = cfg["training"]
    d_cfg  = cfg["diffusion"]
    c_cfg  = cfg["cfg"]
    s_cfg  = cfg["sampling"]
    ck_cfg = cfg["checkpoint"]

    torch.manual_seed(cfg["experiment"].get("seed", 42) + rank)

    if main_proc:
        print(f"\n{'='*60}")
        print(f"  Experiment : {cfg['experiment']['name']}")
        print(f"  Model      : {cfg['experiment']['model']}")
        print(f"  Dataset    : {cfg['experiment']['dataset']}")
        print(f"  GPUs       : {world_size}")
        print(f"  Device     : {device}")
        print(f"  CFG        : scale={c_cfg['cfg_scale']}, drop_p={c_cfg['label_drop_prob']}")
        print(f"{'='*60}\n")

    # ── Model ──────────────────────────────────────────────────────────────────
    net = build_model(cfg).to(device)
    if main_proc:
        print(f"Parameters: {sum(p.numel() for p in net.parameters())/1e6:.2f} M")

    # Wrap in DDP for multi-GPU
    if world_size > 1:
        net = DDP(net, device_ids=[local_rank])

    raw_net = net.module if world_size > 1 else net
    peak_lr = paper_peak_lr(t_cfg["batch_size"], t_cfg["blr"])

    # Denoiser holds EMA of the raw (non-DDP) net
    denoiser = Denoiser(
        net=raw_net, img_size=cfg["model"]["input_size"],
        num_classes=cfg["model"]["num_classes"],
        P_mean=d_cfg["P_mean"], P_std=d_cfg["P_std"],
        t_eps=d_cfg["t_eps"], noise_scale=d_cfg["noise_scale"],
        ema_decay1=t_cfg["ema_decay1"], ema_decay2=t_cfg["ema_decay2"],
        sampling_method=s_cfg["method"], num_sampling_steps=s_cfg["steps"],
        label_drop_prob=c_cfg["label_drop_prob"], cfg_scale=c_cfg["cfg_scale"],
        cfg_interval=tuple(c_cfg["cfg_interval"]),
    )
    denoiser.init_ema()

    # Optimizer on DDP net so gradients sync across GPUs
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=peak_lr,
        betas=tuple(t_cfg["betas"]), weight_decay=t_cfg["weight_decay"],
    )

    # ── Data ───────────────────────────────────────────────────────────────────
    train_ds = build_dataset(cfg)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size,
                                   rank=rank, shuffle=True) if world_size > 1 else None
    loader   = DataLoader(
        train_ds,
        batch_size  = t_cfg["batch_size"] // world_size,  # per-GPU batch size
        shuffle     = (sampler is None),
        sampler     = sampler,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
    )

    steps_per_epoch = len(loader)
    warmup_steps    = t_cfg["warmup_epochs"] * steps_per_epoch
    lr_at_step      = make_lr_fn(peak_lr, warmup_steps)
    use_amp         = "cuda" in device and t_cfg.get("amp", True)
    scaler          = torch.amp.GradScaler("cuda") if use_amp else None

    out_dir = ck_cfg["output_dir"]
    if main_proc:
        os.makedirs(out_dir, exist_ok=True)

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch, global_step, losses, best_loss = 1, 0, [], float("inf")
    resume = ck_cfg.get("resume_from")
    if resume and os.path.exists(resume):
        payload     = load_checkpoint(resume, denoiser, optimizer=optimizer, map_location=device)
        start_epoch = payload["epoch"] + 1
        global_step = payload["global_step"]
        losses      = payload["losses"]
        best_loss   = payload.get("best_loss", float("inf"))
        if main_proc:
            print(f"Resumed from {resume} (epoch {start_epoch}, step {global_step})\n")

    logger = MetricLogger()

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, t_cfg["epochs"] + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)  # different shuffle per epoch in DDP

        net.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{t_cfg['epochs']}",
                    leave=False, disable=not main_proc)

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

            # EMA only on main process (tracks raw_net weights)
            if main_proc:
                denoiser.update_ema()

            global_step += 1
            epoch_loss  += loss.item()
            if main_proc:
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # ── Logging + checkpointing (main process only) ──────────────────────
        if main_proc:
            avg = epoch_loss / len(loader)
            losses.append(avg)
            logger.log(epoch, loss=avg, lr=optimizer.param_groups[0]["lr"])
            msg = f"Epoch {epoch:3d}/{t_cfg['epochs']} | v-loss {avg:.4f}"

            if epoch % ck_cfg["save_last_freq"] == 0 or epoch == t_cfg["epochs"]:
                save_checkpoint(f"{out_dir}/checkpoint-last.pt",
                                epoch, global_step, denoiser, optimizer, losses, best_loss)
                msg += "  [last]"

            if epoch % ck_cfg["save_archive_freq"] == 0:
                save_checkpoint(f"{out_dir}/checkpoint-ep{epoch:03d}.pt",
                                epoch, global_step, denoiser, optimizer, losses, best_loss)
                msg += f"  [archive ep{epoch:03d}]"

            if avg < best_loss:
                best_loss = avg
                save_checkpoint(f"{out_dir}/checkpoint-best.pt",
                                epoch, global_step, denoiser, optimizer, losses, best_loss)
                msg += f"  [best ↓ {best_loss:.4f}]"

            print(msg)

    if main_proc:
        logger.save(f"{out_dir}/metrics.json")
        print(f"\n✅ Training complete. Results → {out_dir}/")

    cleanup()


if __name__ == "__main__":
    main()
