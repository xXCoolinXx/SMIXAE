#!/usr/bin/env bash
# Run core evaluation on all trained experiments and their baselines.
#
# Each call evaluates a single SAE and accumulates results in
# results/core_eval_results.json (model → layer → SAE name → metrics).
#
# Usage:
#   bash experiments/core_eval.sh
#   bash experiments/core_eval.sh --verbose

set -euo pipefail

OUTPUT_JSON="results/core_eval_results.json"
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verbose) EXTRA_FLAGS="--verbose"; shift 1 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

eval_sae() {
    echo "=== $1 ==="
    smixae core "${@:2}" --output-json "${OUTPUT_JSON}" ${EXTRA_FLAGS}
}

# ── Gemma 2 9B, Layer 11 ──────────────────────────────────────────────────── #
eval_sae "SMIXAE 9B L11" \
    "results/gemma_2_9b_l11/model" \
    --base-model-name google/gemma-2-9b \
    --hook-point model.layers.11 \
    --display-name SMIXAE

eval_sae "GemmaScope 9B L11" \
    --hf-release gemma-scope-9b-pt-res \
    --hf-sae-id layer_11/width_16k/average_l0_131 \
    --base-model-name google/gemma-2-9b \
    --hook-point model.layers.11 \
    --display-name "GemmaScope 9B 16k (L0≈131)"

# ── Gemma 2 9B, Layer 20 ──────────────────────────────────────────────────── #
eval_sae "SMIXAE 9B L20" \
    "results/gemma_2_9b_l20/model" \
    --base-model-name google/gemma-2-9b \
    --hook-point model.layers.20 \
    --display-name SMIXAE

eval_sae "GemmaScope 9B L20" \
    --hf-release gemma-scope-9b-pt-res \
    --hf-sae-id layer_20/width_16k/average_l0_131 \
    --base-model-name google/gemma-2-9b \
    --hook-point model.layers.20 \
    --display-name "GemmaScope 9B 16k (L0≈131)"

# ── Gemma 2 2B, Layer 12 ──────────────────────────────────────────────────── #
eval_sae "SMIXAE 2B L12" \
    "results/gemma_2_2b_l12/model" \
    --base-model-name google/gemma-2-2b \
    --hook-point model.layers.12 \
    --display-name SMIXAE

eval_sae "GemmaScope 2B L12" \
    --hf-release gemma-scope-2b-pt-res \
    --hf-sae-id layer_12/width_16k/average_l0_176 \
    --base-model-name google/gemma-2-2b \
    --hook-point model.layers.12 \
    --display-name "GemmaScope 2B 16k (L0≈176)"
