#!/usr/bin/env bash
# Experiment: Gemma 2-9B, layer 11, 4096 experts, d_bottleneck=3
# Run via the generic PBS wrapper, or directly: bash experiments/gemma_2_9b_l11.sh
set -euo pipefail

EXPERIMENT_NAME="gemma_2_9b_l11"
RESULTS_DIR="results/${EXPERIMENT_NAME}"
MODEL="google/gemma-2-9b"
HOOK="model.layers.11"
DATASETS_CONFIG="datasets/probing/dataset_config.json"

# --------------------------------------------------------------------------- #
# 1. Train                                                                      #
# --------------------------------------------------------------------------- #
smixae train \
    --model-name "${MODEL}" \
    --hook-name "${HOOK}" \
    --training-tokens 500000000 \
    --n-experts 4096 \
    --d-in 3584 \
    --d-expert 8 \
    --k-experts 128 \
    --output-path "${RESULTS_DIR}/model" \
    --checkpoint-path "${RESULTS_DIR}/checkpoints"

# --------------------------------------------------------------------------- #
# 2. Probe all datasets                                                         #
# Final model is at the fixed output_path — no glob needed.                    #
# --------------------------------------------------------------------------- #
smixae probe all-datasets \
    --checkpoint-path "${RESULTS_DIR}/model" \
    --base-model-name "${MODEL}" \
    --hook-point "${HOOK}" \
    --datasets-config "${DATASETS_CONFIG}" \
    --output-dir "${RESULTS_DIR}/probe"

# --------------------------------------------------------------------------- #
# 3. Newline-position manifold analysis                                         #
# --------------------------------------------------------------------------- #
smixae newline main \
    --smixae-path "${RESULTS_DIR}/model" \
    --model-name "${MODEL}" \
    --hook-name "${HOOK}" \
    --output-path "${RESULTS_DIR}/newline"

# --------------------------------------------------------------------------- #
# 4. Steering                                                                   #
# --------------------------------------------------------------------------- #
smixae steer main \
    --checkpoint-path "${RESULTS_DIR}/model" \
    --base-model-name "${MODEL}" \
    --hook-point "${HOOK}" \
    --hours-dataset "datasets/probing/hours.csv" \
    --output-dir "${RESULTS_DIR}/steer"
