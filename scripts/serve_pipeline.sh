#!/usr/bin/env bash
# System-1 gate + cascade service on cuda:1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-$REPO_ROOT/config/pipeline.yaml}"

# The gate runs on the 4060 while the 9B engine holds the 5060 Ti. PCI_BUS_ID
# ordering puts the 4060 at index 1; without the order flag CUDA reports the 5060 Ti
# there instead. config/pipeline.yaml documents why they cannot share one card.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/hf}"
export USE_TF=0

exec "$REPO_ROOT/.venv-laya/bin/python" -m system1.server --config "$CONFIG"
