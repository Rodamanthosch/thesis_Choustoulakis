# JiT Research — Diffusion with Just image Transformers

Reproduction and extension of **"Back to Basics: Let Denoising Generative Models Denoise"** ([arXiv 2511.13720](https://arxiv.org/abs/2511.13720)), comparing three mixer architectures for pixel-space diffusion:

| Model | Mixer | Key trait |
|---|---|---|
| **JiT-S** | Self-attention + 2D RoPE | Attention baseline |
| **JiT-S2-ViM** | BiMamba (Vision Mamba) | Bidirectional SSM |
| **JiT-S2-VMamba** | SS2D (4-direction CrossScan) | 2D-aware SSM |

---

## Results

| Model | Dataset | FID ↓ | Params | Notes |
|---|---|---|---|---|
| JiT-S | CIFAR-10 | — | 32.6M | baseline |
| JiT-S2-ViM | CIFAR-10 | — | ~33M | BiMamba mixer |
| JiT-S2-VMamba | CIFAR-10 | — | ~33M | SS2D mixer |
| JiT-B/16 | ImageNet 256×256 | 3.66 | 131M | paper result, 600 ep |
| JiT-L/16 | ImageNet 256×256 | 2.36 | 459M | paper result, 600 ep |

---

## Project Structure

```
jit-research/
├── notebooks/cifar10/          ← original experiment notebooks
│   ├── jit-cifar10.ipynb           attention model
│   ├── jit-vim-cifar10.ipynb       ViM model
│   └── jit-vmamba-cifar10.ipynb    VMamba model
├── src/                        ← shared Python code
│   ├── primitives.py               RMSNorm, embedders, pos-embed, SwiGLU
│   ├── train.py                    Denoiser (flow-matching + EMA + CFG)
│   ├── utils.py                    checkpointing, LR, FID/IS evaluation
│   └── models/
│       ├── jit.py                  JiT-S attention model
│       ├── vim.py                  JiT-S2-ViM (BiMamba)
│       └── vmamba.py               JiT-S2-VMamba (SS2D)
├── configs/                    ← YAML hyperparameter configs
│   ├── cifar10/
│   │   ├── jit-s-baseline.yaml
│   │   ├── jit-s2-vim-baseline.yaml
│   │   └── jit-s2-vmamba-baseline.yaml
│   └── imagenet/
│       └── jit-s-imagenet.yaml
├── scripts/
│   └── run_experiment.py       ← single entry point for all experiments
├── experiments/                ← saved checkpoints and metrics (gitignored)
├── data/                       ← datasets (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Setup

### Attention model (JiT-S) — standard install
```bash
git clone https://github.com/Rodamanthosch/thesis_Choustoulakis.git
cd thesis_Choustoulakis
pip install -r requirements.txt
```

### ViM and VMamba models — Kaggle / Mamba setup

Mamba requires prebuilt CUDA wheels. Run this **once** before any Mamba experiment, then **restart the runtime**:

```python
# === RUN ONCE, THEN RESTART RUNTIME ===

# 1) Pin torch to 2.5.1 (required for prebuilt mamba-ssm wheels)
!pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 2) Download prebuilt CUDA kernel wheels (no compilation needed)
import os
WHEELS = [
    ("https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/"
     "causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"),
    ("https://github.com/state-spaces/mamba/releases/download/v2.2.4/"
     "mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"),
]
for url in WHEELS:
    fname = url.split("/")[-1]
    if not os.path.exists(f"/kaggle/working/{fname}"):
        os.system(f"wget -q {url}")

!pip install -q /kaggle/working/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
!pip install -q /kaggle/working/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
!pip install -q pytorch-fid thop

# 3) Patch mamba-ssm utils/generation.py (fixes a removed transformers class)
import glob
for path in glob.glob("/usr/local/lib/python*/dist-packages/mamba_ssm/utils/generation.py"):
    with open(path) as f: src = f.read()
    new = src.replace(
        "from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer",
        "from transformers.generation import GenerateDecoderOnlyOutput, TextStreamer",
    ).replace(
        "output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput",
        "output_cls = GenerateDecoderOnlyOutput",
    )
    if new != src:
        with open(path, "w") as f: f.write(new)
        print(f"✅ Patched {path}")

# >>> RESTART RUNTIME NOW, then run your experiment <<<
```

> **Note:** `causal-conv1d` is only required for the ViM model. VMamba only needs `mamba-ssm`.
> The JiT-S attention model needs neither.

---

## Running Experiments

All experiments go through one script:
```bash
python scripts/run_experiment.py --config <path-to-config>
```

### CIFAR-10 experiments (single GPU, ~4h per 100 epochs on T4)

**Attention baseline (JiT-S):**
```bash
python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml
```

**Vision Mamba (JiT-S2-ViM):**
```bash
python scripts/run_experiment.py --config configs/cifar10/jit-s2-vim-baseline.yaml
```

**VMamba (JiT-S2-VMamba):**
```bash
python scripts/run_experiment.py --config configs/cifar10/jit-s2-vmamba-baseline.yaml
```

### Reproduce JiT-B from the paper (ImageNet 256×256, multi-GPU)

This matches Table 9 of the paper exactly:
```bash
python scripts/run_experiment.py --config configs/imagenet/jit-s-imagenet.yaml \
    model.hidden_size=768 \
    model.num_heads=12 \
    model.depth=12 \
    model.patch_size=16 \
    model.num_classes=1000 \
    model.in_context_len=32 \
    model.in_context_start=4 \
    training.epochs=600 \
    training.batch_size=1024 \
    cfg.label_drop_prob=0.1 \
    cfg.cfg_scale=2.5 \
    cfg.cfg_interval="[0.1, 1.0]"
```
> **Note:** Requires 8× GPU and ImageNet at `data/imagenet/train/`.

---

## Tuning Hyperparameters

### Option A — Edit a config file
```bash
cp configs/cifar10/jit-s-baseline.yaml configs/cifar10/my-experiment.yaml
# edit the yaml, then:
python scripts/run_experiment.py --config configs/cifar10/my-experiment.yaml
```

### Option B — Override from command line
```bash
python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml \
    training.epochs=50 \
    model.depth=6 \
    training.batch_size=256
```

### Enable CFG
CFG requires two steps: train with label dropout, then sample with guidance scale.
```bash
# Training with CFG enabled:
python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml \
    cfg.label_drop_prob=0.1 \
    cfg.cfg_scale=2.5 \
    cfg.cfg_interval="[0.1, 1.0]"
```
> You must train from scratch with `label_drop_prob=0.1` — you cannot add CFG to a model trained without it.

### Enable in-context class tokens (attention model only)
```bash
python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml \
    model.in_context_len=32 \
    model.in_context_start=4
```

### Resume a stopped run
```bash
python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml \
    checkpoint.resume_from=experiments/cifar10/jit-s-baseline/checkpoint-last.pt
```

---

## All Hyperparameters

| Section | Key | Default | Description |
|---|---|---|---|
| `model` | `hidden_size` | 384 | Transformer width |
| `model` | `depth` | 12 | Number of blocks |
| `model` | `num_heads` | 6 | Attention heads (JiT only) |
| `model` | `patch_size` | 2 | Patch size → (img/p)² tokens |
| `model` | `bottleneck_dim` | 128 | Patch embed bottleneck |
| `model` | `in_context_len` | 0 | In-context tokens (0=off) |
| `model` | `in_context_start` | 0 | Block to start prepending |
| `model` | `d_state` | 16 | SSM state size (Mamba only) |
| `model` | `d_conv` | 4/3 | SSM conv size (Mamba only) |
| `model` | `expand` | 1 | SSM expand ratio (Mamba only) |
| `model` | `K` | 4 | CrossScan directions (VMamba only) |
| `training` | `epochs` | 200 | Total epochs |
| `training` | `batch_size` | 128 | Batch size |
| `training` | `blr` | 5e-5 | Base LR (scaled × batch/256) |
| `training` | `warmup_epochs` | 5 | Linear warmup |
| `training` | `ema_decay1` | 0.9999 | EMA decay (fast) |
| `training` | `ema_decay2` | 0.9996 | EMA decay (slow) |
| `diffusion` | `P_mean` | -0.8 | Time sampler mean |
| `diffusion` | `P_std` | 0.8 | Time sampler std |
| `diffusion` | `t_eps` | 0.05 | Denominator clamp |
| `diffusion` | `noise_scale` | 1.0 | Noise magnitude |
| `cfg` | `label_drop_prob` | 0.0 | CFG label dropout (0=no CFG) |
| `cfg` | `cfg_scale` | 1.0 | CFG guidance strength |
| `cfg` | `cfg_interval` | [0,1] | Timestep range for CFG |
| `sampling` | `method` | heun | ODE solver: heun or euler |
| `sampling` | `steps` | 50 | ODE steps |

---

## References

- [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720) — JiT paper
- [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166)
- [Vision Mamba](https://arxiv.org/abs/2401.09417)
