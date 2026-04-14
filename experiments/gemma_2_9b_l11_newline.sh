#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "$0")/run.sh" \
    --experiment-name   gemma_2_9b_l11 \
    --model             google/gemma-2-9b \
    --hook              model.layers.11 \
    --d-in              3584 \
    --tokenized-dataset datasets/tokenized/pile-uncopyrighted-gemma2 \
    --steps steer \
    --sweep-layers \
    --layer-start 5 \
    --layer-end 10
    
