#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "$0")/run.sh" \
    --experiment-name   gemma_2_9b_l11_affine \
    --model             google/gemma-2-9b \
    --hook              model.layers.11 \
    --d-in              3584 \
    --tokenized-dataset datasets/tokenized/pile-uncopyrighted-gemma2 \
    --use-affine-smixae true
