#!/usr/bin/env bash
set -euo pipefail

smixae pretokenize pretokenize \
    --model-name  meta-llama/Meta-Llama-3-8B \
    --output-path datasets/tokenized/pile-uncopyrighted-llama3
