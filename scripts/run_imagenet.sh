#!/usr/bin/env bash
# =============================================================================
# run_imagenet.sh — one-shot large-scale ImageNet training (JiT / ViM / VMamba)
#
# Sets up the environment + dependencies, then launches the training command
# on a single node with N GPUs (DDP via torchrun — NOT DataParallel).
#
# Usage:
#   bash scripts/run_imagenet.sh <jit|vim|vmamba> [N_GPUS] [EPOCHS]
# Examples:
#   bash scripts/run_imagenet.sh jit              # JiT-S baseline, 8 GPUs, 100 ep
#   bash scripts/run_imagenet.sh vim 8 100        # ViM baseline
#   bash scripts/run_imagenet.sh vmamba 8 2       # VMamba, 2-epoch smoke test
#
# All three configs are BASELINE ("γραμμικό", adaLN only): no in-context, no
# conv-state conditioning.
# =============================================================================
set -euo pipefail

MODEL=${1:-jit}        # jit | vim | vmamba
N_GPUS=${2:-8}
EPOCHS=${3:-100}

# ---- EDIT THESE FOR YOUR CLOUD MACHINE -------------------------------------
DATA_DIR=/path/to/imagenet                                   # folder containing train/ and val/ (class subdirs)
REPO_URL=https://github.com/Rodamanthosch/thesis_Choustoulakis.git
WORKDIR=$HOME/thesis_Choustoulakis
OUT_ROOT=$HOME/experiments/imagenet
# ----------------------------------------------------------------------------

case "$MODEL" in
  jit)    CONFIG=configs/imagenet/jit-s-imagenet.yaml ;;
  vim)    CONFIG=configs/imagenet/jit-s2-vim-imagenet.yaml ;;
  vmamba) CONFIG=configs/imagenet/jit-s2-vmamba-imagenet.yaml ;;
  *) echo "Usage: bash scripts/run_imagenet.sh <jit|vim|vmamba> [N_GPUS] [EPOCHS]"; exit 1 ;;
esac

echo "=============================================================="
echo "  model=$MODEL  gpus=$N_GPUS  epochs=$EPOCHS"
echo "  config=$CONFIG"
echo "  data=$DATA_DIR  out=$OUT_ROOT/$MODEL"
echo "=============================================================="

# 1) repo --------------------------------------------------------------------
if [ ! -d "$WORKDIR/.git" ]; then
  git clone -b main "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

# 2) dependencies (idempotent) -----------------------------------------------
# Skip if the Mamba kernels already import. Source-build is portable (builds
# against the torch already on the VM). ~10-20 min the first time; needs nvcc.
if ! python -c "import mamba_ssm, causal_conv1d" 2>/dev/null; then
  echo ">>> installing dependencies (first run only)"
  # If the build fails on a torch-version mismatch, pin a known-good torch:
  #   pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
  CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install -q --no-build-isolation causal-conv1d
  MAMBA_FORCE_BUILD=TRUE         pip install -q --no-build-isolation mamba-ssm
  pip install -q fvcore pyyaml torch-fidelity
  # fix a stale transformers import inside mamba_ssm (if present)
  python - <<'PY'
import glob
pats = glob.glob("/usr/local/lib/python*/dist-packages/mamba_ssm/utils/generation.py")
pats += glob.glob("**/site-packages/mamba_ssm/utils/generation.py", recursive=True)
for p in set(pats):
    s = open(p).read()
    n = (s.replace("from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer",
                   "from transformers.generation import GenerateDecoderOnlyOutput, TextStreamer")
           .replace("output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput",
                    "output_cls = GenerateDecoderOnlyOutput"))
    if n != s:
        open(p, "w").write(n); print("patched", p)
PY
fi

# 3) train (DDP, single node, N GPUs) ----------------------------------------
torchrun --standalone --nproc_per_node="$N_GPUS" scripts/run_experiment.py \
    --config "$CONFIG" \
    experiment.data_dir="$DATA_DIR" \
    training.epochs="$EPOCHS" \
    checkpoint.output_dir="$OUT_ROOT/$MODEL"

echo "==== done: checkpoints in $OUT_ROOT/$MODEL ===="
