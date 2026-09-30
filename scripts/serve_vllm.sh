#!/usr/bin/env bash
# System-2 engine: Qwen3.5-9B int4 on the RTX 5060 Ti.
#
# Flag choices are justified in PLAN.md §4 Phase 1. The short version:
#   --language-model-only          drops the 0.85 GiB vision encoder; the VRAM buys
#                                  KV cache instead and the gate is text-only anyway
#   --max-model-len 16384          native is 262144, unreachable when the engine and
#                                  the gate share one 16 GB card
#   --gpu-memory-utilization 0.80  leaves the Laya process its ~900 MiB on the same
#                                  card; see the budget table in config/pipeline.yaml
#   --max-cudagraph-capture-size  the Qwen3.5 recipe documents a causal_conv1d_update
#                                  assert when capture size exceeds the mamba cache;
#                                  set it explicitly instead of rediscovering the bug
#   MTP speculative config         the model ships MTP weights; this is the TPOT lever
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${QWEN_REPO:-nicklas373/Qwen3.5-9B-AWQ}"
SERVED_NAME="${QWEN_SERVED_NAME:-qwen35-9b-int4}"
PORT="${QWEN_PORT:-8000}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
GPU_UTIL="${QWEN_GPU_UTIL:-0.85}"
SPEC_TOKENS="${QWEN_SPEC_TOKENS:-0}"
MAX_SEQS="${QWEN_MAX_SEQS:-8}"
# Only the 5060 Ti is visible here. PCI_BUS_ID ordering puts it at index 0.
# Selecting by PCI bus id in CUDA_VISIBLE_DEVICES does not work on this machine
# ("0000:07:00.0" silently yields the 5060 Ti), so use the index under a pinned
# order and let system1.gate._assert_device refuse to start on the wrong card.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/hf}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# The gate runs on this same card in its own process; these are host CPU threads.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"

ARGS=(
  --served-model-name "$SERVED_NAME"
  --host 127.0.0.1
  --port "$PORT"
  --language-model-only
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-num-seqs "$MAX_SEQS"
  --max-cudagraph-capture-size 128
  --reasoning-parser qwen3
)

# MTP speculative decoding loads the draft layer unquantized: a 1.9 GiB contiguous
# allocation on top of the 10.1 GiB of AWQ weights. With the gate sharing the card
# that does not fit at util 0.80, so it is opt-in via QWEN_SPEC_TOKENS.
if [ "$SPEC_TOKENS" -gt 0 ]; then
  ARGS+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${SPEC_TOKENS}}")
fi

exec "$REPO_ROOT/.venv-vllm/bin/vllm" serve "$MODEL" "${ARGS[@]}" "$@"


