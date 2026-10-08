# From JiT to JiM — Selective State-Space Models for Pixel-Space Diffusion

JiT ([Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720)) is a plain Vision Transformer that predicts **clean images directly in pixel space**, with no tokenizer, no pre-training and no extra loss. This repository reproduces JiT and replaces its self-attention mixer with **VMamba's four-direction SS2D** selective scan, giving **JiM**: the same prediction task and residual structure, with fewer parameters and 15–19% fewer forward GFLOPs. It also studies **how the timestep and class should enter the recurrent backbone** (scan-state initialisation, in-context prefix tokens, and direct modulation of the selective parameters).

| Model | `experiment.model` | Mixer | Role |
|---|---|---|---|
| **JiT-S** | `jit` | Self-attention + 2D RoPE | attention baseline (the paper's model) |
| **JiM** (`JiT-S2-VMamba`) | `vmamba` | SS2D: 4-direction CrossScan → selective scans → CrossMerge | main model |
| JiT-S2-ViM | `vim` | Bidirectional Mamba (Vision Mamba) | secondary SSM baseline |

JiM keeps everything else from JiT: the bottleneck patch embedding, fixed 2-D sin-cos positions, RMSNorm, adaLN-Zero conditioning of both residual branches, the SwiGLU FFN and the output head. Only the token mixer changes.

---

## Results

### ImageNet 256×256 — S scale

12 blocks of width 384, bottleneck 128, 16×16 patches (256 image tokens), identical training recipe. JiT additionally uses its 32 in-context class tokens; JiM uses none.

| Model | In-context conditioning | FID @100 ep | FID @400 ep | Params | GFLOPs |
|---|---|---|---|---|---|
| JiT-S (attention) | 32 class tokens | ≈77 | ≈56 | 33.4M | 6.7 |
| **JiM-S** (SS2D) | none | ≈90 | ≈64 | 31.7M | 5.7 (−14.7%) |

At 400 epochs JiT keeps an ≈8-point FID lead, but its curve has largely plateaued near 56, while JiM is still improving by roughly 2–3 FID per additional 100 epochs.

### Complexity at larger scales (ImageNet 256×256, patch 16)

JiT configurations follow the paper (Table 9), including its 32 in-context class tokens; JiM uses none.

| Scale | JiT params | JiT GFLOPs | JiM params | JiM GFLOPs | GFLOPs reduction |
|---|---|---|---|---|---|
| B | 131.32M | 25.22 | 122.59M | 20.97 | 16.9% |
| L | 459.14M | 87.94 | 426.47M | 72.59 | 17.5% |
| H | 952.84M | 181.95 | 882.67M | 149.19 | 18.0% |
| G | 2007.56M | 383.74 | 1855.21M | 311.16 | 18.9% |

Paper reference (600 epochs): JiT-B/16 FID 3.66, JiT-L/16 FID 2.36.

### Conditioning JiM — Tiny-ImageNet 64×64

Patch 8 (8×8 = 64 tokens), S/12 backbone, 100 epochs. FID-10K / IS-10K from 10,000 samples (50 per class) against the 10,000-image validation set. All rows keep full adaLN-Zero.

| Variant | FID-10K ↓ | IS-10K ↑ | Params | GFLOPs |
|---|---|---|---|---|
| JiM baseline | 91.73 | 6.19 ± 0.13 | 31.03M | 1.423 |
| State-init (DiMSUM) | 92.19 | 6.42 ± 0.18 | 32.80M | 1.427 |
| Class prefix, m = 1 | 89.20 | **6.45** ± 0.10 | 31.03M | 1.437 |
| Class prefix, m = 16 | 89.51 | 6.39 ± 0.11 | 31.04M | 1.656 |
| Class prefix, m = 32 | 89.44 | 6.33 ± 0.17 | 31.04M | 1.889 |
| Time–class prefix, m = 2 | 89.25 | 6.44 ± 0.12 | 31.03M | 1.452 |
| Time–class prefix, m = 4 | **88.92** | 6.43 ± 0.14 | 31.03M | 1.481 |
| SSC-static | 92.63 | 6.07 ± 0.17 | 31.03M | 1.423 |
| SSC-bc | 94.37 | 5.88 ± 0.13 | 31.62M | 1.423 |

Can SSC **replace** adaLN-Zero instead of supplementing it?

| Variant | FID-10K ↓ | IS-10K ↑ | Params |
|---|---|---|---|
| JiM baseline, full adaLN | 91.73 | 6.19 ± 0.13 | 31.03M |
| SSC-bc, adaLN kept for the FFN only (`adaln_cond: mlp`) | 100.74 | 5.51 ± 0.07 | 26.61M |
| SSC-bc, no adaLN (`adaln_cond: none`) | 111.94 | 4.53 ± 0.11 | 21.01M |

**Takeaways.** Every in-context prefix improves on the baseline (−2.2 to −2.8 FID), and the short ones (a single class token, or a `[t, y]` / `[t, y, y, t]` unit) do so at almost no GFLOP cost; longer prefixes add compute without improving FID. Scan-state initialisation and SSC do not improve FID, and SSC alone cannot recover adaLN's contribution. The next step is the ImageNet run with a short class prefix.

---

## Conditioning JiM

JiM conditions both residual branches through adaLN-Zero, so the scan already depends on `(t, y)` through its modulated input. SS2D exposes three additional entry points for the condition `c = e_t(t) + e_y(y)`:

- **Initial state** — every directional scan starts from `h₋₁ = W_u c` instead of zero (DiMSUM), recomputed in each block.
- **Input sequence** — condition tokens are read by every scan before the image tokens (DiM / JiT in-context), created once at block `in_context_start` and carried through the remaining blocks.
- **Selective parameters** — learned offsets on the scan's `B` and `C` (semi-separable conditioning, SSC, after DiM-2, with Mamba-3's bias design).

Every route is selected purely by config flags under `model:`. The arms are **mutually exclusive** (asserted in `JiTVMamba.__init__`), and the defaults reproduce the adaLN-Zero baseline byte-for-byte.

| Arm | Flags under `model:` | Tiny-ImageNet configs |
|---|---|---|
| baseline (adaLN-Zero only) | all defaults | `jit-s2-vmamba-baseline.yaml` |
| state-init | `state_init: dimsum` \| `learned` | `jit-s2-vmamba-stateinit-{dimsum,learned}.yaml` |
| SSC | `ssc: static` \| `bc` \| `abc` | `jit-s2-vmamba-ssc-{static,bc,abc}.yaml` |
| in-context prefix | `in_context_len: K`, `in_context_content: class` \| `time_class`, `in_context_start: 4` | `jit-s2-vmamba-incontext-{class,timeclass}.yaml` |
| in-context row | the above **plus** `in_context_layout: row` | `jit-s2-vmamba-incontext-row-{class,timeclass}.yaml` |

- `static` adds learned, ones-initialised offsets `B + b₀`, `C + c₀`; `bc` adds zero-initialised condition terms `B + b₀ + W_B c`, `C + c₀ + W_C c`; `abc` also gates the decay per direction.
- `time_class` prefixes use `[t, y]` (K = 2) or `[t, y, y, t]` (K = 4); `class` repeats the class token K times. Each token gets a learnable positional slot.

**`adaln_cond`** is a modifier of the SSC arm (not an arm of its own): it selects which residual branches adaLN still carries `(t, y)` into.

| `adaln_cond` | mixer branch | FFN branch + output head | S/12 params (ssc-bc) |
|---|---|---|---|
| `full` (default) | adaLN-Zero | adaLN-Zero | 31.62M |
| `mlp` | zero-init static bias | adaLN-Zero | 26.61M |
| `none` | zero-init static bias | zero-init static bias | 21.01M |

Anything but `full` requires `ssc != none` and forbids the in-context prefix. `ssc_z_mlp: true` (only with `mlp` / `none`) gives SSC DiM-2's dedicated `z = MLP(c)`; any surviving adaLN keeps consuming the raw `c`. Configs: `jit-s2-vmamba-ssc-{bc,abc}-ffnadaln.yaml` (`adaln_cond: mlp`, `ssc_z_mlp: true`).

### In-context prefix vs row layout

`in_context_layout` decides *where* the condition tokens sit inside every SS2D scan: `prefix` stacks them all at the head (DiM), `row` interleaves one content unit at the head of every row (scan directions 0/1) / column (directions 2/3), so no image token is more than `W` scan steps from a condition token. Under `row`, `in_context_len` must be a multiple of the grid size, and `time_class` takes exactly `2 × grid_size` (a `[t,y]` unit per row).

```bash
# one [t,y] unit per row/column — 16 tokens, scan L = 64 + 16
python scripts/run_experiment.py \
    --config configs/tiny_imagenet/jit-s2-vmamba-incontext-row-timeclass.yaml

# one class token per row/column — 8 tokens, scan L = 64 + 8
python scripts/run_experiment.py \
    --config configs/tiny_imagenet/jit-s2-vmamba-incontext-row-class.yaml
```

---

## Project Structure

```
thesis_Choustoulakis/
├── src/
│   ├── primitives.py           RMSNorm, embedders, sin-cos pos-embed, patch embed, SwiGLU, FinalLayer
│   ├── train.py                Denoiser: x-prediction, v-loss, two EMAs, CFG sampling
│   ├── utils.py                config, checkpointing, LR schedule, FID/IS, throughput
│   ├── flops_counter.py        fvcore MACs/params + hooks for selective scan, SDPA, causal conv
│   └── models/
│       ├── jit.py              JiT (attention)
│       ├── vmamba.py           JiM = JiT-S2-VMamba (SS2D) + every conditioning arm
│       └── vim.py              JiT-S2-ViM (bidirectional Mamba)
├── configs/
│   ├── tiny_imagenet/          JiM baseline and every conditioning arm (64×64, patch 8)
│   ├── imagenet/               JiT / ViM / JiM at S and B scale (256×256, patch 16)
│   └── cifar10/                original CIFAR-10 configs
├── scripts/
│   ├── run_experiment.py       single training entry point (single GPU or torchrun DDP)
│   ├── evaluate.py             FID / IS / GMACs / params / throughput of a checkpoint
│   ├── run_imagenet.sh         one-shot multi-GPU ImageNet launcher
│   └── check_*_cuda.py         GPU checks run at the start of a new arm's first session
├── tests/                      CPU equivalence tests (reference scan, no GPU needed)
├── notebooks/
│   ├── tiny_imagenet/          Kaggle notebooks: one per run, plus the complexity notebook
│   └── cifar10/                original experiment notebooks
├── experiments/                checkpoints and metrics (gitignored)
└── requirements.txt
```

`build_model` is duplicated in `scripts/run_experiment.py` and `scripts/evaluate.py`; both must be updated whenever a model flag is added.

---

## Setup

### JiT (attention) — standard install, no extra deps
```bash
git clone https://github.com/Rodamanthosch/thesis_Choustoulakis.git
cd thesis_Choustoulakis
pip install -r requirements.txt
pip install fvcore torch-fidelity        # GMACs counting + FID/IS
```

### JiM and ViM — Mamba CUDA kernels required

Run this **once** before any Mamba experiment, then **restart the runtime**. The cell picks torch and the prebuilt wheels from the Python version, so it works on both of Kaggle's images:

| Python | torch / torchvision | wheels |
|---|---|---|
| 3.12 | 2.5.1 / 0.20.1 | `cu12torch2.5`, `cp312` |
| 3.13 | 2.6.0 / 0.21.0 | `cu12torch2.6`, `cp313` |

```python
import os, sys

PY = f"cp{sys.version_info.major}{sys.version_info.minor}"
if sys.version_info >= (3, 13):
    TORCH, TVISION, TAUDIO, TTAG = "2.6.0", "0.21.0", "2.6.0", "torch2.6"
else:
    TORCH, TVISION, TAUDIO, TTAG = "2.5.1", "0.20.1", "2.5.1", "torch2.5"
print("python", sys.version.split()[0], "->", PY, "| torch", TORCH)

# 1) Pin torch
!pip install -q torch=={TORCH} torchvision=={TVISION} torchaudio=={TAUDIO} \
    --index-url https://download.pytorch.org/whl/cu124

# 2) Prebuilt CUDA kernel wheels (same releases for both Python versions)
CAUSAL = f"causal_conv1d-1.5.0.post8+cu12{TTAG}cxx11abiFALSE-{PY}-{PY}-linux_x86_64.whl"
MAMBA  = f"mamba_ssm-2.2.4+cu12{TTAG}cxx11abiFALSE-{PY}-{PY}-linux_x86_64.whl"

os.system(f"wget -q https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/{CAUSAL} -O /kaggle/working/{CAUSAL}")
os.system(f"wget -q https://github.com/state-spaces/mamba/releases/download/v2.2.4/{MAMBA} -O /kaggle/working/{MAMBA}")
for w in (CAUSAL, MAMBA):
    p = f"/kaggle/working/{w}"
    assert os.path.exists(p) and os.path.getsize(p) > 0, f"wheel download failed: {w}"

!pip install -q /kaggle/working/{CAUSAL}
!pip install -q /kaggle/working/{MAMBA}
!pip install -q fvcore torch-fidelity

# 3) Patch mamba-ssm (fixes a removed transformers class)
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

# >>> RESTART RUNTIME NOW <<<
```

> `causal-conv1d` is required for ViM only. JiM only needs `mamba-ssm`. JiT needs neither. Outside Kaggle, replace `/kaggle/working` with any writable directory.

---

## Quick Start on Kaggle

```python
# Clone repo and install
!git clone https://github.com/Rodamanthosch/thesis_Choustoulakis.git
%cd thesis_Choustoulakis
!pip install -q pyyaml tqdm pytorch-fid
```

**Smoke test — 3 epochs to confirm everything works:**
```python
!python scripts/run_experiment.py \
    --config configs/cifar10/jit-s-baseline.yaml \
    training.epochs=3 \
    checkpoint.save_last_freq=1 \
    checkpoint.save_archive_freq=999 \
    checkpoint.output_dir=/kaggle/working/smoke-test
```

Expected output:
```
Epoch   1/3 | v-loss 2.XXXX  [last]
Epoch   2/3 | v-loss 2.XXXX  [last]
Epoch   3/3 | v-loss 2.XXXX  [last]
✅ Training complete.
```

**Full experiment:**
```python
!python scripts/run_experiment.py \
    --config configs/cifar10/jit-s-baseline.yaml \
    checkpoint.output_dir=/kaggle/working/jit-s-cifar10
```

---

## Running Experiments

All experiments go through one script that works on **single GPU and multi-GPU** automatically:

```bash
python scripts/run_experiment.py --config <path-to-config>
```

### Tiny-ImageNet (single GPU, Kaggle)

The conditioning study runs on Tiny-ImageNet-200 at 64×64. `experiment.dataset: imagenet` reads an `ImageFolder` layout, so the downloaded archive must be restructured to `train/<class>/*.JPEG` and `val/<class>/*.JPEG` (the notebooks do this automatically from `val_annotations.txt`).

```bash
python scripts/run_experiment.py \
    --config configs/tiny_imagenet/jit-s2-vmamba-baseline.yaml \
    experiment.data_dir=/path/to/tiny-imagenet-200
```

Each run has a ready Kaggle notebook in `notebooks/tiny_imagenet/` with the same layout: environment → clone → verify (CPU tests + GPU check, first session only) → download → settings (written into the cloned config so training and evaluation build the same model) → train with checkpoint resume across 12-hour sessions → evaluate.

### CIFAR-10 (single GPU, ~4h per 100 epochs on T4)

```bash
# Attention baseline
python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml

# Vision Mamba
python scripts/run_experiment.py --config configs/cifar10/jit-s2-vim-baseline.yaml

# JiM (VMamba)
python scripts/run_experiment.py --config configs/cifar10/jit-s2-vmamba-baseline.yaml
```

### ImageNet — multi-GPU (8× H100, Google Cloud)

```bash
torchrun --nproc_per_node=8 scripts/run_experiment.py \
    --config configs/imagenet/jit-s-imagenet.yaml \
    experiment.data_dir=/path/to/imagenet
```

or the one-shot launcher (environment, dependencies and `torchrun`; edit `DATA_DIR` at the top first):

```bash
bash scripts/run_imagenet.sh <jit|vim|vmamba> [N_GPUS] [EPOCHS]
```

### Reproduce JiT-B from the paper (Table 9)

```bash
torchrun --nproc_per_node=8 scripts/run_experiment.py \
    --config configs/imagenet/jit-s-imagenet.yaml \
    model.hidden_size=768 \
    model.num_heads=12 \
    model.patch_size=16 \
    model.num_classes=1000 \
    model.in_context_len=32 \
    model.in_context_start=4 \
    training.epochs=600 \
    training.batch_size=1024 \
    cfg.label_drop_prob=0.1 \
    cfg.cfg_scale=2.5 \
    cfg.cfg_interval="[0.1, 1.0]" \
    experiment.data_dir=/path/to/imagenet
```

---

## Tuning Hyperparameters

### Option A — Edit a config file (for named experiments)
```bash
cp configs/cifar10/jit-s-baseline.yaml configs/cifar10/my-experiment.yaml
# edit the yaml, then:
python scripts/run_experiment.py --config configs/cifar10/my-experiment.yaml
```

### Option B — Override from command line (for quick tests)
```bash
python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml \
    training.epochs=50 \
    model.depth=6 \
    training.batch_size=256
```

Command-line overrides reach training only: `evaluate.py` builds the model straight from the config file, so put model changes in the YAML.

### Enable CFG
> You must train with `label_drop_prob=0.1` from the start — you cannot add CFG to a model trained without it.

```bash
python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml \
    cfg.label_drop_prob=0.1 \
    cfg.cfg_scale=2.5 \
    cfg.cfg_interval="[0.1, 1.0]"
```

### Enable in-context class tokens (attention model)
```bash
python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml \
    model.in_context_len=32 \
    model.in_context_start=4
```

For JiM's conditioning arms (in-context prefix/row, state-init, SSC), see [Conditioning JiM](#conditioning-jim).

### Change dataset path
```bash
# Kaggle default (CIFAR-10 downloads automatically)
experiment.data_dir=./data

# Google Cloud / custom path
experiment.data_dir=/gcs/my-bucket/imagenet
```

### Resume a stopped run
```bash
python scripts/run_experiment.py --config configs/cifar10/jit-s-baseline.yaml \
    checkpoint.resume_from=experiments/cifar10/jit-s-baseline/checkpoint-last.pt
```

---

## Evaluation (FID, IS, Complexity)

After training, run the eval script to get FID, IS, GMACs, params, and throughput:

```bash
python scripts/evaluate.py \
    --config configs/cifar10/jit-s-baseline.yaml \
    --checkpoint experiments/cifar10/jit-s-baseline/checkpoint-best.pt \
    --n_samples 10000 \
    --ema 1
```

Output:
```
── Complexity ──────────────────────────────────────────────
  Parameters : 32.64 M
  GFLOPs     : X.XXXX
  Throughput : XXXX img/s

── FID / IS ────────────────────────────────────────────────
  FID-10K  : X.XX
  IS-10K   : X.XX ± X.XX

Results saved → experiments/cifar10/jit-s-baseline/eval/eval_results.json
```

| Flag | Default | Description |
|---|---|---|
| `--config` | required | Same config used for training |
| `--checkpoint` | required | Path to `.pt` checkpoint |
| `--n_samples` | 10000 | Samples for FID/IS — use 50000 for official |
| `--batch_size` | 200 | Generation batch size |
| `--ema` | 1 | Which EMA copy to use (1 or 2) |
| `--cfg_scale` | from config | Override the CFG scale (for CFG sweeps) |
| `--cfg_interval` | from config | Override the CFG interval, e.g. `--cfg_interval 0.1 1.0` |
| `--ode_steps` | from config | Override the number of sampling steps |
| `--seed` | 42 | Seed for reproducible generation |
| `--skip_fid` | false | Only run complexity, skip FID/IS |
| `--out_dir` | next to checkpoint | Where to save results |

Complexity comes from `src/flops_counter.py` (fvcore, with hooks for the selective scan, SDPA and causal conv that fvcore cannot trace). Its "GFLOPs" are **MACs**, the convention JiT / DiT / VMamba tables use; strict FLOPs are 2× that.

**Tiny-ImageNet protocol** (the conditioning results above): 10,000 samples, 50 per class, against the 10,000-image validation set; FID and IS from `torch-fidelity`; EMA 1; CFG 2.5 active for 0.1 < t < 1.0; 50 Heun steps.

```python
!python scripts/evaluate.py \
    --config configs/tiny_imagenet/jit-s2-vmamba-baseline.yaml \
    --checkpoint /kaggle/working/<run>/checkpoint-last.pt \
    --n_samples 10000 --batch_size 200 --ema 1 \
    --cfg_scale 2.5 --cfg_interval 0.1 1.0
```

---

## All Hyperparameters

| Section | Key | Default | Description |
|---|---|---|---|
| `experiment` | `model` | `jit` | `jit` \| `vim` \| `vmamba` |
| `experiment` | `dataset` | `cifar10` | `cifar10` \| `imagenet` (any `ImageFolder`, incl. Tiny-ImageNet) |
| `experiment` | `data_dir` | `./data` | Path to dataset root |
| `model` | `hidden_size` | 384 | Transformer width |
| `model` | `depth` | 12 | Number of blocks |
| `model` | `num_heads` | 6 | Attention heads (JiT only) |
| `model` | `patch_size` | 2 | Patch size → (img/p)² tokens |
| `model` | `bottleneck_dim` | 128 | Patch embed bottleneck |
| `model` | `mlp_ratio` | 4.0 | FFN width before SwiGLU's 2/3 rule: h = int(mlp_ratio · D · 2/3) |
| `model` | `in_context_len` | 0 | In-context tokens (0 = off) |
| `model` | `in_context_start` | 0 | Block at which the tokens are prepended |
| `model` | `in_context_content` | `time_class` | `time_class` \| `class` (JiM only) |
| `model` | `in_context_layout` | `prefix` | `prefix` \| `row` — all at the scan head, or one content unit per row/column (JiM only) |
| `model` | `state_init` | `none` | `none` \| `dimsum` \| `learned` — condition-dependent scan state (JiM only) |
| `model` | `ssc` | `none` | `none` \| `static` \| `bc` \| `abc` — semi-separable conditioning of B/C (JiM only) |
| `model` | `adaln_cond` | `full` | `full` \| `mlp` \| `none` — branches adaLN still conditions; non-`full` needs `ssc` (JiM only) |
| `model` | `ssc_z_mlp` | `false` | DiM-2's `z = MLP(c)` for the SSC path; only with `adaln_cond` ≠ `full` (JiM only) |
| `model` | `cond_init` | `none` | `none` \| `conv_state` — DiMSUM conditional conv state (ViM only) |
| `model` | `d_state` | 16 | SSM state size (Mamba only) |
| `model` | `d_conv` | 4/3 | SSM conv size (ViM 4, JiM 3) |
| `model` | `expand` | 1 | SSM expand ratio (Mamba only) |
| `model` | `K` | 4 | CrossScan directions (JiM only) |
| `training` | `epochs` | 200 | Total epochs |
| `training` | `batch_size` | 128 | Total batch (split across GPUs automatically) |
| `training` | `blr` | 5e-5 | Base LR — scaled × batch/256 automatically |
| `training` | `warmup_epochs` | 5 | Linear warmup |
| `training` | `ema_decay1` | 0.9999 | EMA decay (fast copy) |
| `training` | `ema_decay2` | 0.9996 | EMA decay (slow copy) |
| `training` | `amp` | `true` | Mixed precision (set false on CPU) |
| `diffusion` | `P_mean` | -0.8 | Time sampler mean |
| `diffusion` | `P_std` | 0.8 | Time sampler std |
| `diffusion` | `t_eps` | 0.05 | Denominator clamp |
| `diffusion` | `noise_scale` | 1.0 | Noise magnitude |
| `cfg` | `label_drop_prob` | 0.0 | CFG label dropout (0 = no CFG) |
| `cfg` | `cfg_scale` | 1.0 | CFG guidance strength at sampling |
| `cfg` | `cfg_interval` | [0,1] | Timestep range for CFG |
| `sampling` | `method` | `heun` | ODE solver: `heun` or `euler` |
| `sampling` | `steps` | 50 | ODE steps |
| `checkpoint` | `save_last_freq` | 5 | Overwrite last checkpoint every N epochs |
| `checkpoint` | `save_archive_freq` | 25 | Keep numbered checkpoint every N epochs |
| `checkpoint` | `resume_from` | `null` | Path to checkpoint to resume from |
| `checkpoint` | `output_dir` | — | Where to save checkpoints and metrics |

---

## Tests

CPU tests stub `mamba_ssm` with a reference selective scan, so they run without a GPU. Run them from the repo root **with `PYTHONPATH=.`** — without it, `from src.models.vmamba import ...` fails and the tests misreport it as a missing `mamba_ssm`.

```bash
PYTHONPATH=. python tests/test_stateinit_equivalence.py   # state-init == exact h₋₁ injection
PYTHONPATH=. python tests/test_ssc_equivalence.py         # SSC B/C offsets and the exact A-gate identity
PYTHONPATH=. python tests/test_ssc_static_cpu.py          # ssc=static
PYTHONPATH=. python tests/test_incontext_row_cpu.py       # row layout; "off" settings bit-identical to HEAD
PYTHONPATH=. python tests/test_adaln_mode_cpu.py          # adaln_cond = full | mlp | none
```

`tests/test_stateinit_model.py` and `tests/test_ssc_model.py` need the real `mamba_ssm`. The GPU checks (`scripts/check_stateinit_cuda.py`, `scripts/check_incontext_row_cuda.py`) run at the start of an arm's first Kaggle session.

Every new arm keeps its "off" setting byte-identical to the previous model and proves it in a CPU test (state_dict transplant from `git show HEAD:` and `max|diff| == 0`).

---

## References

- [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720) — JiT
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) — DiT, adaLN-Zero
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166) — SS2D
- [Vision Mamba](https://arxiv.org/abs/2401.09417) — ViM
- [DiMSUM: Diffusion Mamba — A Scalable and Unified Spatial-Frequency Method for Image Generation](https://arxiv.org/abs/2411.04168) — state initialisation
- [DiM: Diffusion Mamba for Efficient High-Resolution Image Synthesis](https://arxiv.org/abs/2405.14224) — in-context prefix
- [Scalable Diffusion Models with State Space Backbone](https://arxiv.org/abs/2402.05608) — DiS
- [All are Worth Words: A ViT Backbone for Diffusion Models](https://arxiv.org/abs/2209.12152) — U-ViT
- [DiM-2: Exploiting Structured State Space Duality in Mamba-2 for Unified Image and Video Diffusion](https://doi.org/10.5281/zenodo.18689888) — semi-separable conditioning
- [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569) — B/C bias design
