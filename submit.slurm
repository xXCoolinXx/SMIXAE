#!/bin/bash
# Generic SLURM submission wrapper for SMIXAE experiments.
#
# Usage:
#   sbatch experiments/submit.slurm <experiment_script> [extra args...]
#
# Examples:
#   sbatch submit.slurm experiments/gemma_2_9b_l11_newline.sh
#   sbatch submit.slurm experiments/gemma_2_9b_l20_general.sh
#
#   Override SLURM resources at submission time:
#   sbatch --time=20:00:00 --mem=32G experiments/submit.slurm experiments/gemma_2_9b_l11_newline.sh
#
# The experiment script must be a path relative to the repo root (SLURM_SUBMIT_DIR).

# --- SLURM Directives ---

#SBATCH --job-name=smixae
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-80:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm/slurm_%j.log
#SBATCH --error=logs/slurm/slurm_%j.err

set -euo pipefail

# ── Validate argument ─────────────────────────────────────────────────────── #
if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch experiments/submit.slurm <experiment_script> [extra args...]" >&2
    exit 1
fi

EXPERIMENT_SCRIPT="$1"
shift  # remaining args (if any) are forwarded to the experiment script

# ── Environment setup ────────────────────────────────────────────────────── #
echo "Job started on $(hostname) at $(date)"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "Running experiment: ${EXPERIMENT_SCRIPT}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm

module add compilers/gcc/11.1.0

export CC=gcc
export CXX=g++

export LD_LIBRARY_PATH=/share/apps/gcc/11.1.0/lib64:$LD_LIBRARY_PATH

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

export HF_HOME=/scratch/ctfrancel/.cache/huggingface
export WANDB_API_KEY=$(cat $SLURM_SUBMIT_DIR/.wandb_key) 
export HF_TOKEN=$(cat /home/ctfrancel/.cache/huggingface/token)

# ── Run experiment ───────────────────────────────────────────────────────── #
LOGFILE="${SLURM_SUBMIT_DIR}/logs/debug_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"

{
    if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
        nvidia-smi
    fi

    bash "${EXPERIMENT_SCRIPT}" "$@"
} 2>&1 | tee "${LOGFILE}"

echo "Job finished at $(date)"
