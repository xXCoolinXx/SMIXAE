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
│   │   ├── generate_data.py         # (TODO: rename → generate_probing_data.py) Synthetic probing dataset generation
│   │   ├── categorize_all.py        # Expert probing pipeline: load checkpoint, evaluate experts, produce HTML visualizations
│   │   └── anthropic_newline.py     # Newline-position manifold analysis (TODO: migrate off TransformerLens)
│   └── cli/
│       └── cli.py                   # Centralized CLI entry point (smixae command)
├── datasets/
│   ├── probing/                     # Labeled datasets for probing experiments (populated by generate_data.py)
│   └── steering/                    # Steering datasets (not yet implemented)
├── smixae_run.py                    # Training entry point (standalone, not part of the CLI)
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

Training entry point. Configures the SAELens training loop with SMIXAE. Streams `monology/pile-uncopyrighted` via HuggingFace, hooks into `model.layers.11` of Gemma 2-9B, logs to W&B.

### `src/analysis/categorize_all.py`

The primary analysis script. Loads a trained SMIXAE checkpoint and a labeled dataset, then:
1. Collects LLM activations at the hook point
2. Runs SMIXAE encoding, filters experts by activity
3. Scores experts by Fisher discriminant ratio or manifold continuity
4. Plots top-N experts as interactive 3D Plotly scatters in an HTML file

Exposed via CLI as the `probe` subcommand group.

### `src/analysis/generate_data.py`

Generates template-based synthetic datasets for probing. Each dataset has sentences labeled with a concept (weekdays, hours, temperatures, months, etc.). Output: CSV with `Sentence` and `Label` columns.

**TODO**: Rename to `generate_probing_data.py` and update to write directly to `datasets/probing/`.

### `src/analysis/anthropic_newline.py`

Analyzes how experts encode distance-since-newline (a continuous position signal). Scores experts using linear and periodic (Fourier) regression on the bottleneck.

**TODO**: Migrate off TransformerLens (see "What to Avoid" below).

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

Add a new generator function in `generate_data.py` following the existing pattern:
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
# On HPC (PBS):
qsub run_sae.pbs

# Locally:
python smixae_run.py
```

Training logs to W&B. Checkpoints are saved at intervals (3 checkpoints by default). Checkpoints are not included in this repo and must be pointed to manually in analysis scripts.

---

## CLI

All analysis commands run through the `smixae` CLI (installed as an editable package via `pip install -e .` or `uv sync`).

```
smixae
├── generate-probing-data        # Generate all probing datasets → datasets/probing/
├── probe
│   ├── single                   # Analyze one labeled dataset against a checkpoint
│   └── all-datasets             # Batch over a JSON config of datasets
└── newline
    └── main                     # Newline-position manifold analysis
```

```bash
# Install the package in editable mode
pip install -e .

# Generate probing datasets
smixae generate-probing-data

# Probe a single dataset
smixae probe single \
    --checkpoint-path <path/to/checkpoint> \
    --base-model-name google/gemma-2-9b \
    --hook-point model.layers.11 \
    --dataframe-path datasets/probing/weekdays.csv \
    --label-column Label

# Batch probe all datasets
smixae probe all-datasets --config datasets.json --checkpoint-path <path>

# Newline-position analysis
smixae newline main --smixae-path <path/to/checkpoint>
```

`smixae_run.py` is intentionally excluded from the CLI — it is a one-off training script run directly or via PBS.

---

## Known TODOs

- [ ] Rename `generate_data.py` → `generate_probing_data.py`
- [ ] Update `generate_probing_data.py` to write output directly to `datasets/probing/`
- [x] Migrate `anthropic_newline.py` off TransformerLens to HuggingFace pattern
- [ ] Implement steering experiments (`datasets/steering/`)
- [ ] Explore `d_bottleneck > 3` with a minimum-dimensionality penalty