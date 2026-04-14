#!/usr/bin/env bash
# Generic SMIXAE experiment runner.
#
# Required flags:
#   --experiment-name NAME   Label used for results/ subdirectory
#   --model           MODEL  HuggingFace model name
#   --hook            HOOK   Hook point (e.g. model.layers.11)
#   --d-in            D_IN   Residual stream dimension for the model
#
# Optional flags (defaults shown):
#   --training-tokens    500000000
#   --n-experts          4096
#   --d-expert           8
#   --k-experts          128
#   --datasets-config    datasets/probing/dataset_config.json
#   --hours-dataset      datasets/probing/hours.csv
#   --tokenized-dataset  (unset — streams raw dataset if omitted)
#
# Usage:
#   bash experiments/run.sh \
#       --experiment-name gemma_2_9b_l11 \
#       --model google/gemma-2-9b \
#       --hook model.layers.11 \
#       --d-in 3584

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────── #
TRAINING_TOKENS=500000000
N_EXPERTS=2048
D_EXPERT=16
K_EXPERTS=64
DATASETS_CONFIG="datasets/probing/dataset_config.json"
HOURS_DATASET="datasets/probing/hours.csv"
TOKENIZED_DATASET=""
USE_AFFINE_SMIXAE=false
SWEEP_LAYERS=false
LAYER_START=0
LAYER_END=-1
STEPS="train,probe,newline"

# ── Parse flags ──────────────────────────────────────────────────────────── #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --model)           MODEL="$2";           shift 2 ;;
        --hook)            HOOK="$2";            shift 2 ;;
        --d-in)            D_IN="$2";            shift 2 ;;
        --training-tokens) TRAINING_TOKENS="$2"; shift 2 ;;
        --n-experts)       N_EXPERTS="$2";       shift 2 ;;
        --d-expert)        D_EXPERT="$2";        shift 2 ;;
        --k-experts)       K_EXPERTS="$2";       shift 2 ;;
        --datasets-config)    DATASETS_CONFIG="$2";    shift 2 ;;
        --hours-dataset)      HOURS_DATASET="$2";      shift 2 ;;
        --tokenized-dataset)  TOKENIZED_DATASET="$2";  shift 2 ;;
        --use-affine-smixae) USE_AFFINE_SMIXAE="$2";   shift 2 ;;
        --sweep-layers)      SWEEP_LAYERS="$2";        shift 2 ;;
        --layer-start)       LAYER_START="$2";          shift 2 ;;
        --layer-end)         LAYER_END="$2";            shift 2 ;;
        --steps)             STEPS="$2";              shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ── Validate required flags ──────────────────────────────────────────────── #
: "${EXPERIMENT_NAME:?--experiment-name is required}"
: "${MODEL:?--model is required}"
: "${HOOK:?--hook is required}"
: "${D_IN:?--d-in is required}"

RESULTS_DIR="results/${EXPERIMENT_NAME}"
RESULTS_JSON="results/results.json"

_has_step() { [[ ",${STEPS}," == *",${1},"* ]]; }

# --------------------------------------------------------------------------- #
# 1. Train                                                                      #
# --------------------------------------------------------------------------- #
if _has_step train; then
    if [[ -n "${TOKENIZED_DATASET}" ]]; then
        DATASET_FLAGS="--dataset-path ${TOKENIZED_DATASET} --is-dataset-tokenized --no-streaming"
    else
        DATASET_FLAGS=""
    fi

    smixae train \
        --model-name "${MODEL}" \
        --hook-name "${HOOK}" \
        --training-tokens "${TRAINING_TOKENS}" \
        --n-experts "${N_EXPERTS}" \
        --d-in "${D_IN}" \
        --d-expert "${D_EXPERT}" \
        --k-experts "${K_EXPERTS}" \
        --output-path "${RESULTS_DIR}/model" \
        --checkpoint-path "${RESULTS_DIR}/checkpoints" \
        --use-affine-smixae "${USE_AFFINE_SMIXAE}" \
        ${DATASET_FLAGS}
fi

# --------------------------------------------------------------------------- #
# 2. Probe all datasets                                                         #
# Final model is at the fixed output_path — no glob needed.                    #
# --------------------------------------------------------------------------- #
if _has_step probe; then
    smixae probe all-datasets \
        --checkpoint-path "${RESULTS_DIR}/model" \
        --base-model-name "${MODEL}" \
        --hook-point "${HOOK}" \
        --datasets-config "${DATASETS_CONFIG}" \
        --output-dir "${RESULTS_DIR}/probe" \
        --min-active-fraction 0.75 \
        --results-json "${RESULTS_JSON}"
fi

# --------------------------------------------------------------------------- #
# 3. Newline-position manifold analysis                                         #
# --------------------------------------------------------------------------- #
if _has_step newline; then
    smixae newline main \
        --smixae-path "${RESULTS_DIR}/model" \
        --model-name "${MODEL}" \
        --hook-name "${HOOK}" \
        --output-path "${RESULTS_DIR}/newline_150" \
        --line-length 150 \
        --results-json "${RESULTS_JSON}"

    smixae newline main \
        --smixae-path "${RESULTS_DIR}/model" \
        --model-name "${MODEL}" \
        --hook-name "${HOOK}" \
        --output-path "${RESULTS_DIR}/newline_80" \
        --line-length 80 \
        --results-json "${RESULTS_JSON}"
fi

# --------------------------------------------------------------------------- #
# 4. Steering                                                                    #
# Experts are auto-selected from cyc_24h hypothesis in results.json.            #
# Use --sweep-layers true to sweep across layers below the trained layer.       #
# --------------------------------------------------------------------------- #
if _has_step steer; then
    STEER_FLAGS=(
        --checkpoint-path "${RESULTS_DIR}/model"
        --base-model-name "${MODEL}"
        --hook-point "${HOOK}"
        --hours-dataset "${HOURS_DATASET}"
        --run-name "${EXPERIMENT_NAME}"
        --results-json "${RESULTS_JSON}"
        --output-dir "${RESULTS_DIR}/steer"
    )
    if [[ "${SWEEP_LAYERS}" == "true" ]]; then
        STEER_FLAGS+=(
            --sweep-layers
            --layer-start "${LAYER_START}"
            --layer-end "${LAYER_END}"
        )
    fi
    smixae steer main "${STEER_FLAGS[@]}"
fi
