# SMIXAE — Claude Code Guide

## What is SMIXAE?

**Sparse MIXture of Autoencoders (SMIXAE)** is an interpretability architecture for large language models. It is an alternative to Sparse Autoencoders (SAEs) designed to model **nonlinear features** — general manifolds — rather than assuming features lie along independent linear directions.

A standard SAE decomposes a model's residual stream into a sparse sum of linear directions. SMIXAE instead assigns activations to a sparse set of **experts**, each of which has a low-dimensional **bottleneck** (currently 3D for visualization). This bottleneck allows each expert to learn an arbitrary geometry: rings, spirals, helices, clusters, or any other manifold structure.

**Core novelty**: SAEs assume feature independence along directions. SMIXAE explicitly relaxes this assumption, allowing each expert's bottleneck to capture nonlinear relationships in the data.

---

## Repository Structure

```
SMIXAE/
├── src/
│   ├── smixae/
│   │   ├── __init__.py             # Public exports + SAELens architecture registration
│   │   └── smixae.py               # Core architecture: SMIXAE model + training classes
│   ├── analysis/
│   │   ├── utils.py                 # Shared infrastructure: load_llm, load_sae, collect_hook_activations, encode_sae_batched, Expert
│   │   ├── generate_probing_data.py # Synthetic probing dataset generation (outputs to datasets/probing/)
│   │   ├── generate_steering_data.py # Steering prompt dataset generation (outputs to datasets/steering/)
│   │   ├── categorize_all.py        # Expert probing pipeline: load checkpoint, evaluate experts, produce HTML visualizations
│   │   ├── anthropic_newline.py     # Newline-position manifold analysis
│   │   └── steer.py                 # Steering experiments: coordinate substitution via top Fisher expert
│   └── cli/
│       ├── cli.py                   # Centralized CLI entry point (smixae command)
│       └── train.py                 # smixae train subcommand — all training options as CLI flags
├── datasets/
│   ├── probing/                     # Labeled datasets for probing experiments (populated by generate_probing_data.py)
│   └── steering/                    # Steering prompt datasets (populated by generate_steering_data.py)
├── experiments/                     # Self-contained experiment scripts (one per run configuration)
│   └── gemma_2_9b_l11.sh            # Gemma 2-9B layer 11: train → probe → newline
├── results/                         # All outputs, created at runtime (not committed)
│   └── {experiment_name}/
│       ├── model/                   # Final inference-ready SAE (fixed path, used by analysis scripts)
│       ├── checkpoints/             # Intermediate training checkpoints (run_id subdir, for resume only)
│       ├── probe/                   # categorize_all HTML outputs
│       └── newline/                 # anthropic_newline outputs
├── smixae_run.py                    # Thin shim for PBS / direct invocation — calls smixae train with hardcoded Gemma 2-9B defaults
├── run_sae.pbs                      # HPC PBS job submission script
└── pyproject.toml                   # Dependencies + CLI entry point
```

---

## Architecture Overview

### SMIXAE Encoding Pipeline

Given a residual stream activation `x` of shape `(batch, d_model)` (e.g. 3584 for Gemma 2-9B):

1. **Linear projection + bias**: `x @ W_enc + b_enc` → shape `(batch, n_experts * d_expert)`
2. **LeakyReLU** (slope 1e-4): avoids dead neurons in the expert space
3. **Reshape**: `(batch, n_experts, d_expert)` — one activation vector per expert
4. **Bottleneck projection**: `einsum(bne, ned → bnd)` with `W_bottleneck` → `(batch, n_experts, d_bottleneck)`
5. **BatchTop-k masking**: only the k experts with highest activation norm are kept active (others zeroed)
6. **Threshold mask**: additional gate based on learned threshold (inference only)

### Decoding Pipeline

1. `W_latent_dec` projects bottleneck back to expert space: `(batch, n_experts, d_bottleneck)` → `(batch, n_experts, d_expert)`
2. `W_dec` projects expert activations back to residual stream: flatten → `(batch, d_model)`

### Key Hyperparameters (current experiment)

| Parameter       | Value | Meaning                                      |
|-----------------|-------|----------------------------------------------|
| `n_experts`     | 4096  | Total number of experts                      |
| `d_expert`      | 8     | Dimensionality of each expert's activation   |
| `d_bottleneck`  | 3     | Bottleneck dimension (3D for visualization)  |
| `k_experts`     | 128   | Experts active per forward pass (top-k)      |
| `d_model`       | 3584  | Gemma 2-9B residual stream dimension         |

`d_bottleneck=3` is chosen for direct 3D visualization. Future work will explore higher dimensions with a minimum-dimensionality penalty.

### Dead Expert Recovery

Experts that haven't fired in `dead_after_n_passes` (default 500) passes are considered dead. An auxiliary loss reconstructs the residual from dead experts using their top-k activations, encouraging them to participate.

---

## Key Classes and Files

### `src/smixae/__init__.py`

Exports the four public classes and registers the SMIXAE architecture with SAELens under the name `"smixae"`. Importing `smixae` (e.g. at the top of a script) is sufficient to register the architecture — no manual registration needed.

### `src/smixae/smixae.py`

- **`SMIXAEConfig`**: Inference-only config dataclass. Parameters: `n_experts`, `d_expert`, `d_bottleneck`, `rescale_acts_by_decoder_norm`.
- **`SMIXAE`**: Inference model. Inherits from `SAE[SMIXAEConfig]`. Implements `encode()` and `decode()`.
- **`SMIXAETrainingConfig`**: Training config. Adds `k_experts`, `aux_loss_coefficient`, `threshold_lr`, `dead_after_n_passes`.
- **`SMIXAETraining`**: Training model. Implements `training_forward_pass()` with MSE + auxiliary loss, `update_threshold()`, and dead expert tracking.
- **`smixae_encode()`**: Shared encoding logic used by both inference and training classes.

### `smixae_run.py`

Thin shim for PBS / direct invocation. Injects hardcoded Gemma 2-9B defaults into `sys.argv` and calls `app()` from `src/cli/cli.py`. For full control use `smixae train` directly.

### `src/cli/train.py`

Typer app exposing all `LanguageModelSAERunnerConfig` and `SMIXAETrainingConfig` options as CLI flags, grouped by category (Model, Data, Training, SAE Architecture, Logging). Importing `smixae` at the top registers the architecture with SAELens as a side effect.

### `src/analysis/utils.py`

Shared infrastructure used by all analysis scripts:
- **`load_llm()`** / **`load_sae()`**: model and checkpoint loading
- **`collect_hook_activations()`**: HuggingFace `register_forward_hook` pattern; returns one `(batch, seq, d_model)` CPU tensor per batch
- **`encode_sae_batched()`**: batched SAE encoding; returns `(N, n_experts, d_bottleneck)` float32 tensor
- **`collect_activations()`**: end-to-end data loading + tokenization + LLM activation collection; returns `(activations, str_tokens, labels, label_names, last_token_positions, n_classes)`
- **`get_sae_activations()`**: encodes LLM activations through SMIXAE, builds and returns a list of `Expert` objects filtered by activity threshold
- **`Expert`**: holds per-expert bottleneck activations and labels; implements `evaluate_fisher()`, `evaluate_manifold()`, `get_plot()`, `get_mean_plot()`
- **`_strip_prefix()`**: strips `NN_` ordering prefixes from label display strings

### `src/analysis/categorize_all.py`

The primary analysis script. Loads a trained SMIXAE checkpoint and a labeled dataset, then:
1. Collects LLM activations at the hook point
2. Runs SMIXAE encoding, filters experts by activity
3. Scores experts by Fisher discriminant ratio or manifold continuity
4. Plots top-N experts as interactive 3D Plotly scatters in an HTML file

Exposed via CLI as the `probe` subcommand group.

### `src/analysis/generate_probing_data.py`

Generates template-based synthetic datasets for probing. Each dataset has sentences labeled with a concept (weekdays, hours, temperatures, months, etc.). Output: CSV with `Sentence` and `Label` columns, written directly to `datasets/probing/`.

### `src/analysis/anthropic_newline.py`

Analyzes how experts encode distance-since-newline (a continuous position signal). Scores experts using linear and periodic (Fourier) regression on the bottleneck.

### `src/analysis/steer.py`

Causal intervention script using **coordinate substitution**: for a target expert, subtracts its current contribution to the residual stream and adds the decoded mean bottleneck for a target class. Two tasks:

- **Task 1 (current time)**: `"Right now it is {src_hour}. What time is it?"` — steers the perceived current time.
- **Task 2 (elapsed time)**: `"Right now it is {curr_hour}. How much time has it been since {start_hour}?"` — steers the current-time representation by `+target_delta_hours`, changing the perceived elapsed time.

Expert discovery reuses `collect_activations()` + `get_sae_activations()` from `utils.py`. Output: `steering_results.csv` with columns `task`, `expert_id`, `src_hour`, `tgt_hour`, `start_hour`, `prompt`, `baseline_output`, `steered_output`. Exposed via CLI as `smixae steer main`.

---

## How to Write New Code

### New Analysis Scripts

Analysis scripts are **standalone** — they do not inherit from a base class. Follow the pattern in `categorize_all.py`:

1. Load the LLM with `load_llm()` (uses `AutoModelForCausalLM` / SAELens internals — **not TransformerLens**)
2. Load the SMIXAE checkpoint with `load_sae()` (returns an `SMIXAE` instance)
3. Collect activations at a hook point using `collect_activations()` or a similar streaming loop
4. Run `model.encode(activations)` to get bottleneck activations per expert
5. Filter/score experts, then visualize or save results

### Scoring New Geometric Properties

To add a new interpretability metric on experts:
- Add a method to the `Expert` class in `categorize_all.py`, or write a standalone scoring function
- The expert's bottleneck activations (shape: `n_active_samples × d_bottleneck`) are the primary input to any geometric analysis
- For regression-based metrics, see `anthropic_newline.py` as a reference pattern

### Adding New Probing Datasets

Add a new generator function in `generate_probing_data.py` following the existing pattern:
- Define templates (some with `{name}`, all with a concept placeholder)
- Define concept values
- Return a de-duplicated, shuffled DataFrame with `Sentence` and `Label` columns
- Output to `datasets/probing/<dataset_name>.csv`

---

## What to Avoid

### Do NOT use TransformerLens

**Always use SAELens / HuggingFace `transformers` for model loading and activation collection.** TransformerLens is significantly slower and is not needed — `categorize_all.py` and `smixae_run.py` both demonstrate the correct approach.

All scripts now use HuggingFace for model loading. No TransformerLens usage remains in the codebase.

### Do NOT assume features are linear

The whole point of SMIXAE is that features can be nonlinear manifolds. Don't apply analysis methods that assume linearity (e.g. PCA for interpreting bottleneck structure) without also checking for nonlinear structure.

---

## Training Workflow

```bash
# Run a full experiment (train → probe → newline):
bash experiments/gemma_2_9b_l11.sh

# Or via the generic PBS wrapper:
qsub run_sae.pbs

# Train only, via CLI (full control):
smixae train \
    --model-name google/gemma-2-9b \
    --hook-name model.layers.11 \
    --training-tokens 500000000 \
    --n-experts 4096 \
    --d-in 3584 \
    --d-expert 8 \
    --k-experts 128 \
    --output-path results/my_run/model \
    --checkpoint-path results/my_run/checkpoints

# Thin shim with hardcoded defaults (called by run_sae.pbs):
python smixae_run.py
```

Training logs to W&B. Intermediate checkpoints are saved under `--checkpoint-path` with a `{run_id}/{step}/` subdirectory appended (non-deterministic path, useful for resuming). The final inference-ready model is saved to `--output-path` as a flat directory — no subdirs — making it directly referenceable by downstream analysis scripts.

The `smixae train` command exposes all `LanguageModelSAERunnerConfig` and `SMIXAETrainingConfig` options. Run `smixae train --help` to see all flags grouped by category (Model, Data, Training, SAE Architecture, Logging, etc.). Run-specific args (`--model-name`, `--hook-name`, `--training-tokens`, `--n-experts`, `--d-in`, `--d-expert`, `--k-experts`) are required; all others have sensible defaults. A `--factor` multiplier scales batch size, LR, warm-up, and dead-expert window together.

### Experiment Scripts

Each file in `experiments/` is a self-contained bash script for one run configuration. It chains training → probing → newline analysis, with all outputs consolidated under `results/{experiment_name}/`:

```
results/{experiment_name}/
├── model/          # Final SAE (output_path) — used by all downstream scripts
├── checkpoints/    # Intermediate checkpoints (for resume only)
├── probe/          # categorize_all HTML outputs
└── newline/        # anthropic_newline outputs
```

Add a new experiment by copying an existing script and adjusting the variables at the top.

---

## CLI

All commands run through the `smixae` CLI (installed as an editable package via `pip install -e .` or `uv sync`).

```
smixae
├── train                        # Train a SMIXAE (all config options exposed as flags)
├── generate-probing-data
│   └── generate                 # Generate all probing datasets → datasets/probing/
├── generate-steering-data
│   └── generate                 # Generate steering prompt datasets → datasets/steering/
├── probe
│   ├── single                   # Analyze one labeled dataset against a checkpoint
│   └── all-datasets             # Batch over a JSON config of datasets
├── newline
│   └── main                     # Newline-position manifold analysis
└── steer
    └── main                     # Steering experiments (coordinate substitution)
```

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
```

`smixae_run.py` is a thin shim that calls the CLI with hardcoded Gemma 2-9B defaults — use it via PBS or `python smixae_run.py` for quick invocation without arguments.

---

## Known TODOs

- [ ] Refine Task 2 (elapsed-time steering) — the exact intervention point (current vs start time token) and evaluation metric are still being determined
- [ ] Explore `d_bottleneck > 3` with a minimum-dimensionality penalty