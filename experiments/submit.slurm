#!/bin/bash
# Generic SLURM submission wrapper for SMIXAE experiments.
#
# Usage:
#   sbatch experiments/submit.slurm <experiment_script> [extra args...]
#
# Examples:
#   sbatch experiments/submit.slurm experiments/gemma_2_9b_l11_newline.sh
#   sbatch experiments/submit.slurm experiments/gemma_2_9b_l20_general.sh
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
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=64G
#SBATCH --time=15:00:00
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

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
module add compilers/gcc/11.1.0

export HF_HOME=/scratch/Collin/.cache/huggingface
export WANDB_API_KEY=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

# ── Run experiment ───────────────────────────────────────────────────────── #
LOGFILE="${SLURM_SUBMIT_DIR}/logs/slurm/debug_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S).log"

{
    nvidia-smi

    bash "${EXPERIMENT_SCRIPT}" "$@"
} 2>&1 | tee "${LOGFILE}"

echo "Job finished at $(date)"
