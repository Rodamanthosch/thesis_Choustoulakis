# JiT Research — Diffusion with Just image Transformers

Experiments comparing **JiT-S**, **JiT-S2-VMamba**, and **JiT-S2-ViM** diffusion models on CIFAR-10 and ImageNet, using x-prediction with v-loss following the flow-matching framework.

---

## Results

| Model | Dataset | FID ↓ | Params | Notes |
|---|---|---|---|---|
| JiT-S | CIFAR-10 | — | 32.6M | baseline |
| JiT-S2-VMamba | CIFAR-10 | — | — | VMamba SSM variant |
| JiT-S2-ViM | CIFAR-10 | — | — | Vision Mamba variant |
| JiT-S | ImageNet 256×256 | — | — | planned |

---

## Project Structure

```
jit-research/
├── notebooks/          ← experiment notebooks (one per model/dataset)
│   ├── cifar10/
│   └── imagenet/
├── src/                ← shared Python code
│   ├── models/         ← model architectures
│   ├── train.py
│   ├── sample.py
│   └── utils.py
├── configs/            ← YAML hyperparameter configs
│   ├── cifar10/
│   └── imagenet/
├── experiments/        ← saved metrics and samples (not committed to git)
├── scripts/            ← training launch scripts
├── data/               ← datasets (not committed to git)
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/jit-research.git
cd jit-research

pip install -r requirements.txt
```

### Datasets

**CIFAR-10** — downloaded automatically by torchvision.

**ImageNet** — download manually from [image-net.org](https://image-net.org) and place under `data/imagenet/`.

---

## Running Experiments

### From a notebook
Open any notebook under `notebooks/cifar10/` or `notebooks/imagenet/` and run all cells.

### From the command line

```bash
# CIFAR-10 (single GPU)
bash scripts/train_cifar10.sh configs/cifar10/jit-s-baseline.yaml

# ImageNet (8 GPUs)
bash scripts/train_imagenet.sh configs/imagenet/jit-s-imagenet.yaml 8
```

---

## Hyperparameter Tuning

All hyperparameters live in `configs/`. To run a new experiment, **copy an existing config and change what you need**:

```bash
cp configs/cifar10/jit-s-baseline.yaml configs/cifar10/jit-s-lr1e-4.yaml
# edit lr in the new file, then:
bash scripts/train_cifar10.sh configs/cifar10/jit-s-lr1e-4.yaml
```

For a full sweep, see `configs/cifar10/jit-s-hparam-sweep.yaml` (wandb Bayesian sweep format).

---

## Key Hyperparameters

| Parameter | CIFAR-10 | ImageNet |
|---|---|---|
| Patch size | 8 | 16 |
| Hidden dim | 384 | 384 |
| Depth | 12 | 12 |
| In-context tokens | 32 | 32 |
| Batch size | 128 | 1024 |
| Learning rate | 2e-4 | 2e-4 |
| Epochs | 200 | 600 |
| ODE steps (sampling) | 50 | 50 |

---

## References

- [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720) — JiT paper
- [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166)
- [Vision Mamba](https://arxiv.org/abs/2401.09417)
