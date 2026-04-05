#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "$0")/run.sh" \
    --experiment-name   gemma_2_2b_l12 \
    --model             google/gemma-2-2b \
    --hook              model.layers.12 \
    --d-in              2304 \
    --tokenized-dataset datasets/tokenized/pile-uncopyrighted-gemma2
