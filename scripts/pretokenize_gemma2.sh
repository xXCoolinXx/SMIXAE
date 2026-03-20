#!/usr/bin/env bash
set -euo pipefail

smixae pretokenize pretokenize \
    --model-name  google/gemma-2-9b \
    --output-path datasets/tokenized/pile-uncopyrighted-gemma2
