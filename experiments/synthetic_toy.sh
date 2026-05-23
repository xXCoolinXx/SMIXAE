#!/usr/bin/env bash
# Canonical synthetic manifold benchmark sweep.
# Trains SMIXAE at n_experts=42 (ground-truth manifold count) across a range
# of k_experts budgets and emits results/ artifacts.
#
# Usage:
#   bash experiments/synthetic_toy.sh              # full sweep
#   bash experiments/synthetic_toy.sh --smoke      # fast smoke-test (2 configs)

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/synthetic/${TIMESTAMP}"

K_EXPERTS_LIST="2,4,6,8,12,16"
TRAINING_SAMPLES=50000000   # 50M  (~25 epochs × 2M dataset; ~49K steps at batch 1024)
EVAL_SAMPLES=200000

if [[ "${1:-}" == "--smoke" ]]; then
  K_EXPERTS_LIST="4,8"
  TRAINING_SAMPLES=100000
  EVAL_SAMPLES=5000
  OUTPUT_DIR="results/synthetic/_smoke"
  echo "[smoke] fast run: k_experts=${K_EXPERTS_LIST}, training_samples=${TRAINING_SAMPLES}"
fi

echo "Output: ${OUTPUT_DIR}"

uv run smixae synthetic \
  --d-in 128 \
  --l0 4 \
  --eval-samples "${EVAL_SAMPLES}" \
  --n-experts 42 \
  --d-expert 16 \
  --d-bottleneck 3 \
  --k-experts-list "${K_EXPERTS_LIST}" \
  --training-samples "${TRAINING_SAMPLES}" \
  --batch-size 8192 \
  --lr 3e-3 \
  --output-dir "${OUTPUT_DIR}" \
  --seed 0

echo "Done.  Artifacts in ${OUTPUT_DIR}"
