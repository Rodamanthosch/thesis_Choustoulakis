"""
scripts/evaluate.py
===================
Evaluate a trained checkpoint: FID, IS, MACs, params, throughput.

Usage:
    python scripts/evaluate.py \\
        --config  configs/cifar10/jit-s-baseline.yaml \\
        --checkpoint experiments/cifar10/jit-s-baseline/checkpoint-best.pt \\
        --n_samples 10000 \\
        --ema 1

    # Use 50000 samples for official FID (slower):
    python scripts/evaluate.py \\
        --config  configs/cifar10/jit-s-baseline.yaml \\
        --checkpoint experiments/cifar10/jit-s-baseline/checkpoint-best.pt \\
        --n_samples 50000
"""

import argparse, os, sys, json
import torch
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import Denoiser
from src.utils import load_config, load_checkpoint, evaluate_fid_is, measure_throughput


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
        raise ValueError(f"Unknown model: {arch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,  help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pt checkpoint")
    parser.add_argument("--n_samples",  type=int, default=10000, help="Samples for FID/IS (50000 for official)")
    parser.add_argument("--batch_size", type=int, default=200,   help="Generation batch size")
    parser.add_argument("--ema",        type=int, default=1,     choices=[1, 2], help="Which EMA to use (1 or 2)")
    parser.add_argument("--out_dir",    default=None, help="Where to save results (default: next to checkpoint)")
    parser.add_argument("--skip_fid",   action="store_true", help="Skip FID/IS (only run complexity)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m_cfg  = cfg["model"]
    s_cfg  = cfg["sampling"]
    c_cfg  = cfg["cfg"]

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Evaluating  : {cfg['experiment']['name']}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  EMA copy    : {args.ema}")
    print(f"  Samples     : {args.n_samples}")
    print(f"  Device      : {device}")
    print(f"{'='*60}\n")

    # ── Build model + load checkpoint ────────────────────────────────────────
    net = build_model(cfg).to(device)
    denoiser = Denoiser(
        net=net, img_size=m_cfg["input_size"], num_classes=m_cfg["num_classes"],
        P_mean=cfg["diffusion"]["P_mean"], P_std=cfg["diffusion"]["P_std"],
        t_eps=cfg["diffusion"]["t_eps"], noise_scale=cfg["diffusion"]["noise_scale"],
        sampling_method=s_cfg["method"], num_sampling_steps=s_cfg["steps"],
        cfg_scale=c_cfg["cfg_scale"], cfg_interval=tuple(c_cfg["cfg_interval"]),
    )
    denoiser.init_ema()
    load_checkpoint(args.checkpoint, denoiser, map_location=device)
    print(f"✅ Checkpoint loaded\n")

    results = {}

    # ── 1. Complexity (MACs, params, throughput) ──────────────────────────────
    print("── Complexity ──────────────────────────────────────────────")
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"  Parameters : {n_params:.2f} M")
    results["params_M"] = round(n_params, 2)

    try:
        from thop import profile
        dummy_x = torch.randn(1, 3, m_cfg["input_size"], m_cfg["input_size"], device=device)
        dummy_t = torch.rand(1, device=device)
        dummy_y = torch.randint(0, m_cfg["num_classes"], (1,), device=device)
        macs, _ = profile(net, inputs=(dummy_x, dummy_t, dummy_y), verbose=False)
        gflops  = (macs * 2) / 1e9
        print(f"  GFLOPs     : {gflops:.4f}")
        results["gflops"] = round(gflops, 4)
    except Exception as e:
        print(f"  GFLOPs     : skipped ({e})")

    if device == "cuda":
        tput = measure_throughput(net,
            input_shape=(128, 3, m_cfg["input_size"], m_cfg["input_size"]),
            device=device, num_classes=m_cfg["num_classes"])
        print(f"  Throughput : {tput:.0f} img/s  (batch=128, single forward)")
        results["throughput_img_per_sec"] = round(tput, 1)

    if args.skip_fid:
        print("\nSkipping FID/IS (--skip_fid set)")
        _save_results(results, out_dir)
        return

    # ── 2. Generate samples ───────────────────────────────────────────────────
    print(f"\n── Generating {args.n_samples} samples ─────────────────────────────")
    gen_dir  = os.path.join(out_dir, "generated")
    real_dir = os.path.join(out_dir, "real")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    denoiser.eval()
    count = 0
    with torch.no_grad(), denoiser.swap_ema(args.ema):
        pbar = tqdm(total=args.n_samples, desc="Generating")
        while count < args.n_samples:
            bs = min(args.batch_size, args.n_samples - count)
            y  = torch.randint(0, m_cfg["num_classes"], (bs,), device=device)
            imgs = denoiser.generate(y)
            imgs = ((imgs.clamp(-1, 1) + 1) / 2).cpu()
            for j in range(bs):
                save_image(imgs[j], f"{gen_dir}/{count+j:05d}.png")
            count += bs
            pbar.update(bs)
        pbar.close()
    print(f"✅ {count} images saved to {gen_dir}\n")

    # ── 3. Save real images ───────────────────────────────────────────────────
    print(f"── Saving {args.n_samples} real images ────────────────────────────")
    dataset = cfg["experiment"]["dataset"]
    data_dir = cfg["experiment"].get("data_dir", "./data")

    if dataset == "cifar10":
        real_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=T.ToTensor()
        )
    elif dataset == "imagenet":
        real_ds = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, "val"),
            transform=T.Compose([
                T.Resize(m_cfg["input_size"]),
                T.CenterCrop(m_cfg["input_size"]),
                T.ToTensor(),
            ])
        )

    written = 0
    for img, _ in real_ds:
        if written >= args.n_samples:
            break
        save_image(img, f"{real_dir}/{written:05d}.png")
        written += 1
    print(f"✅ {written} real images saved to {real_dir}\n")

    # ── 4. FID + IS ───────────────────────────────────────────────────────────
    print("── FID / IS ────────────────────────────────────────────────")
    metrics = evaluate_fid_is(gen_dir, real_dir, device=device, n_samples=args.n_samples)
    k = args.n_samples // 1000
    print(f"  FID-{k}K  : {metrics['fid']:.2f}")
    print(f"  IS-{k}K   : {metrics['is_mean']:.2f} ± {metrics['is_std']:.2f}")
    results.update({
        f"fid_{k}k":    round(metrics["fid"], 2),
        f"is_mean_{k}k": round(metrics["is_mean"], 2),
        f"is_std_{k}k":  round(metrics["is_std"], 2),
    })

    _save_results(results, out_dir)


def _save_results(results, out_dir):
    path = os.path.join(out_dir, "eval_results.json")
    with open(path, "w") as f:
        import json
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  Results saved → {path}")
    print(f"  Summary:")
    for k, v in results.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
