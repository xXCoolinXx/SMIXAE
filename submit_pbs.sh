#!/bin/bash
# Submit a job script to PBS.
# Usage: ./submit_pbs.sh [qsub options] <script.sh>
#
# Examples:
#   ./submit_pbs.sh -N gemma_l11 -l select=1:ncpus=8:ngpus=1:mem=16gb experiments/gemma_2_9b_l11_newline.sh
#   ./submit_pbs.sh -N gemma_l20 -l select=1:ncpus=8:ngpus=1:mem=16gb experiments/gemma_2_9b_l20_general.sh
#   ./submit_pbs.sh -q large -N pretok_gemma2 -l select=1:ncpus=8:mem=32gb scripts/pretokenize_gemma2.sh
#   ./submit_pbs.sh -q large -N pretok_llama3 -l select=1:ncpus=8:mem=32gb scripts/pretokenize_llama3.sh

if [ "$#" -lt 1 ]; then
    echo "Usage: ./submit_pbs.sh [qsub options] <script.sh>"
    exit 1
fi

SCRIPT="${@: -1}"      # last argument is the shell script
QSUB_ARGS="${@:1:$#-1}"  # all other arguments are passed to qsub

qsub $QSUB_ARGS -v "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES,SCRIPT=$SCRIPT,PYTHON=${PYTHON:-python3}" "$(dirname "${BASH_SOURCE[0]}")/pbs_runner.pbs"
