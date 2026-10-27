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

PATCHES:
  BUG #4   — autocast dtype is bf16 (matches LTH14/JiT engine_jit.py)
             and GradScaler is removed (bf16 doesn't need it).
  BUG #15  — num_workers bumped from 4 to 12 (matches LTH14/JiT main_jit.py).
  JiT-OPT  — LTH14/JiT optimizer + loop details: add_weight_decay param
             groups (no decay on biases / norms; identical maths at
             weight_decay 0; old single-group checkpoints still resume, see
             src/utils.py _regroup_optimizer_state), and training stops on a
             non-finite loss before the update, as engine_jit.py does.
  DDP-SYNC — multi-GPU runs never averaged gradients: DDP wrapped the bare
             net, but training called the Denoiser, which held the unwrapped
             net, so DDP's forward (which arms the all-reduce) never ran and
             every GPU trained its own model. Now, as in main_jit.py, DDP
             wraps the whole Denoiser and training calls the wrapper. One
             process (plain `python`, no torchrun): no wrapper, unchanged.
  RNG-RESUME — the seed is set once at start-up and checkpoints held no RNG
             state, so every resumed job replayed the run's first epochs
             exactly (data order, flips, t, noise, label drops). Checkpoints
             now carry every rank's RNG state at the epoch boundary and a
             resume restores it: a resumed run continues exactly like an
             uninterrupted one. Also: checkpoints are written via .tmp +
             rename (a kill mid-write keeps the previous file), best_loss is
             updated before saving (checkpoint-last carries it), metrics.json
             is written every epoch and resumed, a missing resume_from warns
             loudly, and training.num_workers (default 12) sets the loader
             workers.
  GRAD-ACCUM — training.accum_iter (default 1 = the loop above, unchanged):
             training.batch_size stays the EFFECTIVE global batch (so the LR
             rule and the EMA keep their meaning); each GPU loads micro-batches
             of batch_size // (world_size * accum_iter), the loss is divided by
             accum_iter, and the optimizer / LR / EMA / global_step advance once
             per accum_iter micro-batches (DDP all-reduces only on the last,
             via no_sync). The net has no BatchNorm and the loss is a batch
             mean, so with equal micro-batches (drop_last) the update equals
             the full-batch one up to float rounding. A trailing remainder of
             < accum_iter micro-batches per epoch is skipped.
"""

import argparse, contextlib, math, os, sys, yaml
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
    paper_peak_lr, make_lr_fn, MetricLogger, CenterCropArr, add_weight_decay,
    gather_rng_states, restore_rng_states,
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


def wrap_ddp(denoiser, world_size, device_ids=None):
    """LTH14/JiT main_jit.py: DDP wraps the whole Denoiser and training calls
    the wrapper. DDP only all-reduces gradients for a forward that goes
    through its own forward(); calling a module it wraps skips the sync.
    One process: no wrapper, the Denoiser itself."""
    if world_size == 1:
        return denoiser
    return DDP(denoiser, device_ids=device_ids)


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
            in_context_len=m.get("in_context_len", 0),
            in_context_start=m.get("in_context_start", 0),
            cond_init=m.get("cond_init", "none"),
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
            in_context_len=m.get("in_context_len", 0),
            in_context_start=m.get("in_context_start", 0),
            in_context_content=m.get("in_context_content", "time_class"),
            in_context_layout=m.get("in_context_layout", "prefix"),
            state_init=m.get("state_init", "none"),
            ssc=m.get("ssc", "none"),
            adaln_cond=m.get("adaln_cond", "full"),
            ssc_z_mlp=m.get("ssc_z_mlp", False),
            ffn=m.get("ffn", "swiglu"),
            norm=m.get("norm", "rmsnorm"),
            derf_alpha_init=m.get("derf_alpha_init", 0.5),
        )
    elif arch == "spatial_mamba":
        from src.models.spatial_mamba import JiTSpatialMamba
        return JiTSpatialMamba(
            input_size=m["input_size"], patch_size=m["patch_size"],
            in_channels=m["in_channels"], hidden_size=m["hidden_size"],
            depth=m["depth"], num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"],
            attn_drop=m["attn_drop"], proj_drop=m["proj_drop"],
            num_classes=m["num_classes"], bottleneck_dim=m["bottleneck_dim"],
            d_state=m.get("d_state", 1), d_conv=m.get("d_conv", 3),
            expand=m.get("expand", 1), gate=m.get("gate", True),
            sasf_dilations=tuple(m.get("sasf_dilations", (1, 3, 5))),
            lpu=m.get("lpu", "none"),
            sasf_state_kernels=m.get("sasf_state_kernels", "shared"),
            grad_checkpoint=m.get("grad_checkpoint", False),
            ffn=m.get("ffn", "swiglu"),
            norm=m.get("norm", "rmsnorm"),
            derf_alpha_init=m.get("derf_alpha_init", 0.5),
            in_context_len=m.get("in_context_len", 0),
            in_context_start=m.get("in_context_start", 0),
            in_context_content=m.get("in_context_content", "class"),
            in_context_positions=m.get("in_context_positions", None),
        )
    elif arch == "vmamba2":
        from src.models.vmamba2 import JiTVMamba2
        return JiTVMamba2(
            input_size=m["input_size"], patch_size=m["patch_size"],
            in_channels=m["in_channels"], hidden_size=m["hidden_size"],
            depth=m["depth"], num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"],
            attn_drop=m["attn_drop"], proj_drop=m["proj_drop"],
            num_classes=m["num_classes"], bottleneck_dim=m["bottleneck_dim"],
            d_state=m.get("d_state", 16), d_conv=m.get("d_conv", 3),
            expand=m.get("expand", 1), K=m.get("K", 4),
            nheads=m.get("nheads", None),
            chunk_size=m.get("chunk_size", 64),
            force_fp32=m.get("force_fp32", False),
            ffn=m.get("ffn", "swiglu"),
            norm=m.get("norm", "rmsnorm"),
            derf_alpha_init=m.get("derf_alpha_init", 0.5),
        )
    elif arch == "vmamba_sasf":
        from src.models.vmamba_sasf import JiTVMambaSASF
        return JiTVMambaSASF(
            input_size=m["input_size"], patch_size=m["patch_size"],
            in_channels=m["in_channels"], hidden_size=m["hidden_size"],
            depth=m["depth"], num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"],
            attn_drop=m["attn_drop"], proj_drop=m["proj_drop"],
            num_classes=m["num_classes"], bottleneck_dim=m["bottleneck_dim"],
            d_state=m.get("d_state", 1), d_conv=m.get("d_conv", 3),
            expand=m.get("expand", 1), K=m.get("K", 4),
            sasf_dilations=tuple(m.get("sasf_dilations", (1, 3, 5))),
            ffn=m.get("ffn", "swiglu"),
            norm=m.get("norm", "rmsnorm"),
            derf_alpha_init=m.get("derf_alpha_init", 0.5),
        )
    else:
        raise ValueError(f"Unknown model: {arch}. Choose jit | vim | vmamba | spatial_mamba | vmamba2 | vmamba_sasf")


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
        # LTH14/JiT main_jit.py: ADM center crop (BOX halving + BICUBIC), flip.
        # ToTensor + Normalize(0.5, 0.5) is bit-identical to JiT's x/255*2-1.
        return torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "train"),
            transform=T.Compose([
                CenterCropArr(img_size),
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

    seed = cfg["experiment"].get("seed", 42)
    torch.manual_seed(seed + rank)

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

    peak_lr = paper_peak_lr(t_cfg["batch_size"], t_cfg["blr"])

    # Denoiser holds the net, its EMA and the flow maths
    denoiser = Denoiser(
        net=net, img_size=cfg["model"]["input_size"],
        num_classes=cfg["model"]["num_classes"],
        P_mean=d_cfg["P_mean"], P_std=d_cfg["P_std"],
        t_eps=d_cfg["t_eps"], noise_scale=d_cfg["noise_scale"],
        ema_decay1=t_cfg["ema_decay1"], ema_decay2=t_cfg["ema_decay2"],
        sampling_method=s_cfg["method"], num_sampling_steps=s_cfg["steps"],
        label_drop_prob=c_cfg["label_drop_prob"], cfg_scale=c_cfg["cfg_scale"],
        cfg_interval=tuple(c_cfg["cfg_interval"]),
    )
    # Multi-GPU: DDP around the whole Denoiser (DDP-SYNC above). Its
    # constructor copies rank 0's weights to every rank, so the EMA starts
    # after it, from the synchronised weights.
    model = wrap_ddp(denoiser, world_size, [local_rank])
    denoiser.init_ema()

    # LTH14/JiT main_jit.py: no weight decay on biases / norms / 1-D params,
    # frozen params (pos_embed) left out.
    param_groups = add_weight_decay(net, t_cfg["weight_decay"])
    optimizer = torch.optim.AdamW(
        param_groups, lr=peak_lr, betas=tuple(t_cfg["betas"]),
    )

    # ── Data ───────────────────────────────────────────────────────────────────
    accum_iter = int(t_cfg.get("accum_iter", 1))
    assert accum_iter >= 1 and t_cfg["batch_size"] % (world_size * accum_iter) == 0, (
        f"training.batch_size {t_cfg['batch_size']} must divide into "
        f"{world_size} GPU(s) x accum_iter {accum_iter} equal micro-batches")
    micro_bs = t_cfg["batch_size"] // (world_size * accum_iter)
    if main_proc and accum_iter > 1:
        print(f"Grad accumulation: {accum_iter} x {micro_bs} per GPU x {world_size} GPU(s) "
              f"= effective batch {t_cfg['batch_size']}")

    train_ds = build_dataset(cfg)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size,
                                   rank=rank, shuffle=True) if world_size > 1 else None
    loader   = DataLoader(
        train_ds,
        batch_size  = micro_bs,  # per-GPU micro-batch (= batch_size // world_size at accum_iter 1)
        shuffle     = (sampler is None),
        sampler     = sampler,
        # ─── BUG #15 FIX: num_workers=12 matches LTH14/JiT main_jit.py ───
        # training.num_workers overrides it (match the CPU cores you get).
        num_workers = t_cfg.get("num_workers", 12),
        # ────────────────────────────────────────────────────────────────
        pin_memory  = True,
        drop_last   = True,
    )

    # Optimizer steps (not micro-batches): warmup, global_step and the EMA
    # all count these.
    steps_per_epoch = len(loader) // accum_iter
    assert steps_per_epoch > 0, "fewer micro-batches per epoch than accum_iter"
    warmup_steps   = t_cfg["warmup_epochs"] * steps_per_epoch
    lr_at_step      = make_lr_fn(peak_lr, warmup_steps)
    use_amp         = "cuda" in device and t_cfg.get("amp", True)

    out_dir = ck_cfg["output_dir"]
    if main_proc:
        os.makedirs(out_dir, exist_ok=True)

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch, global_step, losses, best_loss = 1, 0, [], float("inf")
    logger = MetricLogger()
    resume = ck_cfg.get("resume_from")
    if resume and not os.path.exists(resume) and main_proc:
        print(f"\n⚠ WARNING: checkpoint.resume_from = {resume} does not exist -> "
              f"TRAINING FROM SCRATCH.\n  Expected only for the first job of a chain; "
              f"otherwise check the path before checkpoint-last.pt is overwritten.\n")
    if resume and os.path.exists(resume):
        payload     = load_checkpoint(resume, denoiser, optimizer=optimizer, map_location=device)
        start_epoch = payload["epoch"] + 1
        global_step = payload["global_step"]
        losses      = payload["losses"]
        best_loss   = payload.get("best_loss", float("inf"))
        logger.history = list(payload.get("metrics", []))
        # LAST thing before the loop: continue the random stream exactly
        # where the checkpoint left it (see RNG-RESUME).
        exact = restore_rng_states(payload.get("rng"), rank, world_size, seed, start_epoch)
        if main_proc:
            print(f"Resumed from {resume} (epoch {start_epoch}, step {global_step}); "
                  f"RNG {'restored exactly' if exact else 'not in checkpoint -> reseeded from (seed, rank, epoch)'}\n")

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, t_cfg["epochs"] + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)  # different shuffle per epoch in DDP

        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{t_cfg['epochs']}",
                    leave=False, disable=not main_proc)

        n_micro = steps_per_epoch * accum_iter  # drops the < accum_iter remainder
        for it, (images, labels) in enumerate(pbar):
            if it == n_micro:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            first_micro = it % accum_iter == 0
            last_micro  = it % accum_iter == accum_iter - 1

            if first_micro:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_at_step(global_step)
                optimizer.zero_grad(set_to_none=True)

            # DDP: skip the gradient all-reduce on all but the last micro-batch.
            sync = contextlib.nullcontext() if last_micro or world_size == 1 \
                else model.no_sync()
            with sync:
                # ─── BUG #4 FIX: bf16 autocast (no GradScaler — bf16 doesn't need it) ──
                # Matches LTH14/JiT engine_jit.py line 33-35 exactly.
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(images, labels)
                else:
                    loss = model(images, labels)

                # LTH14/JiT engine_jit.py: stop on a non-finite loss BEFORE the
                # update, so no NaN step is taken and no broken checkpoint saved.
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                (loss / accum_iter if accum_iter > 1 else loss).backward()
            epoch_loss += loss_value

            if not last_micro:
                continue
            optimizer.step()
            # ──────────────────────────────────────────────────────────────────────

            # EMA only on main process (tracks the unwrapped Denoiser's weights)
            if main_proc:
                denoiser.update_ema()

            global_step += 1
            if main_proc:
                pbar.set_postfix(loss=f"{loss_value:.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # RNG state at the epoch boundary, from every rank (a collective under
        # DDP, so outside main_proc). Nothing below draws random numbers before
        # the next epoch's DataLoader iterator, so a resume from here continues
        # exactly like the uninterrupted run.
        rng_states = gather_rng_states(world_size)

        # ── Logging + checkpointing (main process only) ──────────────────────
        if main_proc:
            avg = epoch_loss / n_micro
            losses.append(avg)
            # best_loss is updated BEFORE any save, so checkpoint-last carries
            # it: a resumed run must not overwrite checkpoint-best with a worse
            # epoch.
            is_best = avg < best_loss
            if is_best:
                best_loss = avg
            logger.log(epoch, loss=avg, lr=optimizer.param_groups[0]["lr"])
            logger.save(f"{out_dir}/metrics.json", quiet=True)   # every epoch
            msg = f"Epoch {epoch:3d}/{t_cfg['epochs']} | v-loss {avg:.4f}"
            ckpt = dict(rng_states=rng_states, metrics=logger.history)

            if epoch % ck_cfg["save_last_freq"] == 0 or epoch == t_cfg["epochs"]:
                save_checkpoint(f"{out_dir}/checkpoint-last.pt",
                                epoch, global_step, denoiser, optimizer, losses, best_loss, **ckpt)
                msg += "  [last]"

            if epoch % ck_cfg["save_archive_freq"] == 0:
                save_checkpoint(f"{out_dir}/checkpoint-ep{epoch:03d}.pt",
                                epoch, global_step, denoiser, optimizer, losses, best_loss, **ckpt)
                msg += f"  [archive ep{epoch:03d}]"

            if is_best:
                save_checkpoint(f"{out_dir}/checkpoint-best.pt",
                                epoch, global_step, denoiser, optimizer, losses, best_loss, **ckpt)
                msg += f"  [best ↓ {best_loss:.4f}]"

            print(msg)

    if main_proc:
        logger.save(f"{out_dir}/metrics.json")
        print(f"\n✅ Training complete. Results → {out_dir}/")

    cleanup()


if __name__ == "__main__":
    main()
