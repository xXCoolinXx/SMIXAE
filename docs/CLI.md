# SMIXAE CLI Reference

All commands run through the `smixae` CLI (installed as an editable package via `pip install -e .` or `uv sync`).

```
smixae
├── train                        # Train a SMIXAE (all config options exposed as flags)
├── pretokenize                  # Tokenize a HuggingFace dataset and save for SAELens training
├── generate-probing-data
│   └── generate                 # Generate all probing datasets → datasets/probing/
├── generate-steering-data
│   └── generate                 # Generate steering prompt datasets → datasets/steering/
├── probe
│   ├── single                   # Analyze one labeled dataset against a checkpoint
│   └── all-datasets             # Batch over a JSON config of datasets
├── newline
│   └── main                     # Newline-position manifold analysis
├── steer
│   └── main                     # Steering experiments (coordinate substitution)
├── core                         # Evaluate a single SAE (local checkpoint or HuggingFace) on core metrics
└── latex
    ├── save-server              # Start local HTTP figure-collection server (port 7788)
    ├── figures                  # Assemble camera-ready PNGs into LaTeX figure files
    └── tables                   # Generate all LaTeX tables (probing + newline + core eval) from results.json
```

---

## Usage Examples

```bash
# Install the package in editable mode
pip install -e .

# Generate probing datasets
smixae generate-probing-data generate

# Generate steering datasets
smixae generate-steering-data generate

# Probe a single dataset
smixae probe single \
    --checkpoint-path <path/to/checkpoint> \
    --base-model-name google/gemma-2-9b \
    --hook-point model.layers.11 \
    --dataframe-path datasets/probing/weekdays.csv \
    --label-column Label

# Batch probe all datasets
smixae probe all-datasets \
    --checkpoint-path results/my_run/model \
    --base-model-name google/gemma-2-9b \
    --hook-point model.layers.11 \
    --datasets-config datasets/probing/dataset_config.json \
    --output-dir results/my_run/probe

# Newline-position analysis
smixae newline main \
    --smixae-path results/my_run/model \
    --model-name google/gemma-2-9b \
    --hook-name model.layers.11 \
    --output-path results/my_run/newline

# Steering experiments
smixae steer main \
    --checkpoint-path results/my_run/model \
    --base-model-name google/gemma-2-9b \
    --hook-point model.layers.11 \
    --output-dir results/my_run/steer

# Core SAE evaluation: local SMIXAE checkpoint
smixae core results/my_run/model \
    --base-model-name google/gemma-2-9b \
    --hook-point model.layers.11 \
    --display-name SMIXAE

# Core SAE evaluation: GemmaScope baseline from HuggingFace
smixae core \
    --hf-release gemma-scope-9b-pt-res \
    --hf-sae-id layer_11/width_16k/average_l0_118 \
    --base-model-name google/gemma-2-9b \
    --hook-point model.layers.11 \
    --display-name "GemmaScope 9B 16k (L0=118)"

# Batch over all experiments + GemmaScope baselines
bash experiments/core_eval.sh
```

---

## `smixae_run.py`

Thin shim that calls the CLI with hardcoded Gemma 2-9B defaults — use it via PBS or `python smixae_run.py` for quick invocation without arguments.
