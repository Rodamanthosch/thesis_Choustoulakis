#!/bin/bash
# scripts/train_imagenet.sh
# Usage: bash scripts/train_imagenet.sh configs/imagenet/jit-s-imagenet.yaml

set -e

CONFIG=${1:-configs/imagenet/jit-s-imagenet.yaml}
N_GPUS=${2:-8}

echo "=========================================="
echo "  Training on ImageNet"
echo "  Config:  $CONFIG"
echo "  GPUs:    $N_GPUS"
echo "=========================================="

torchrun --nproc_per_node=$N_GPUS -m src.train --config "$CONFIG"
