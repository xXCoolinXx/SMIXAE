#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "$0")/run.sh" \
    --experiment-name gemma_2_9b_l20 \
    --model         google/gemma-2-9b \
    --hook          model.layers.20 \
    --d-in          3584
