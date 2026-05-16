"""
scripts/profile_model.py
========================
Build a model from a YAML config (with optional overrides) and print its
parameters and GFLOPs. No checkpoint, no FID.

USAGE
-----
    # As-is
    python scripts/profile_model.py --config configs/cifar10/jit-s-baseline.yaml

    # Override any hyperparameter (dot-notation, same as run_experiment.py):
    python scripts/profile_model.py --config configs/cifar10/jit-s-baseline.yaml \
        model.depth=24 model.hidden_size=512
"""

import argparse, os, sys, yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils import load_config
from src.flops_counter import count_complexity, print_report


def set_nested(d, key_path, value):
    """Apply 'key.subkey=value' override to a nested dict."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d[k]
    original = d.get(keys[-1])
    if   isinstance(original, bool):  value = value.lower() in ("true", "1", "yes")
    elif isinstance(original, int):   value = int(value)
    elif isinstance(original, float): value = float(value)
    elif isinstance(original, list):  value = yaml.safe_load(value)
    d[keys[-1]] = value


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
    raise ValueError(f"Unknown model: {arch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*",
                        help="key.path=value pairs to override the config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for ov in args.overrides:
        k, v = ov.split("=", 1)
        set_nested(cfg, k, v)
        print(f"  Override: {k} = {v}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model(cfg).to(device)
    m_cfg = cfg["model"]

    report = count_complexity(
        net,
        img_size=m_cfg["input_size"],
        in_channels=m_cfg["in_channels"],
        num_classes=m_cfg["num_classes"],
        device=device,
    )
    print_report(report, model_name=cfg["experiment"]["name"])


if __name__ == "__main__":
    main()
