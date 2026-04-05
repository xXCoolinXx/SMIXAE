#!/usr/bin/env bash
set -euo pipefail

bash "$(dirname "$0")/run.sh" \
    --experiment-name   llama_31_8b_newline \
    --model             meta-llama/Llama-3.1-8B \
    --hook              model.layers.5 \
    --d-in              4096 \
    --tokenized-dataset datasets/tokenized/pile-uncopyrighted-llama3
