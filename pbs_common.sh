# Common HPC environment setup.
# Source this file from job scripts: source "$REPO_ROOT/pbs_common.sh"

module load cuda/12.6.0

# REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# source "$REPO_DIR/.venv/bin/activate"

# : "${PYTHON:=python3}"

export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "stubs" | tr '\n' ':')
export HF_HOME=/scratch/Collin/.cache/huggingface