#!/bin/bash
# scripts/train_cifar10.sh
# Usage: bash scripts/train_cifar10.sh configs/cifar10/jit-s-baseline.yaml

set -e

CONFIG=${1:-configs/cifar10/jit-s-baseline.yaml}

echo "=========================================="
echo "  Training on CIFAR-10"
echo "  Config: $CONFIG"
echo "=========================================="

python -m src.train --config "$CONFIG"
