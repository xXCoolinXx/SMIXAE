# SMIXAE — Claude Code Guide

## What is SMIXAE?

**Sparse MIXture of Autoencoders (SMIXAE)** is an interpretability architecture for large language models. It is an alternative to Sparse Autoencoders (SAEs) designed to model **nonlinear features** — general manifolds — rather than assuming features lie along independent linear directions.

A standard SAE decomposes a model's residual stream into a sparse sum of linear directions. SMIXAE instead assigns activations to a sparse set of **experts**, each of which has a low-dimensional **bottleneck** (currently 3D for visualization). This bottleneck allows each expert to learn an arbitrary geometry: rings, spirals, helices, clusters, or any other manifold structure.

**Core novelty**: SAEs assume feature independence along directions. SMIXAE explicitly relaxes this assumption, allowing each expert's bottleneck to capture nonlinear relationships in the data.

---

## Documentation Policy

After every major change — new class, new metric, new CLI command, changed default, changed algorithm — update both `CLAUDE.md` and the relevant file(s) under `docs/` to reflect the new state. Do not leave docs describing removed or replaced behavior.

---

## Repository Structure

```
SMIXAE/
├── src/
│   ├── smixae/
│   │   ├── __init__.py             # Public exports + SAELens architecture registration
│   │   ├── smixae.py               # Core architecture: SMIXAE model + training classes
│   │   └── affine_smixae.py        # AffineSMIXAE variant with W_directions routing (NOT actively used/maintained)
│   ├── analysis/
│   │   ├── utils.py                 # Shared infrastructure: load_llm, load_sae, collect_hook_activations, Expert, ExpertFilterConfig, DatasetConfig
│   │   ├── generate_probing_data.py # Synthetic probing dataset generation (outputs to datasets/probing/)
│   │   ├── generate_steering_data.py # Steering prompt dataset generation (outputs to datasets/steering/)
│   │   ├── categorize_all.py        # Expert probing pipeline: load checkpoint, evaluate experts, produce HTML visualizations
│   │   ├── anthropic_newline.py     # Newline-position manifold analysis
│   │   ├── steer.py                 # Steering experiments: coordinate substitution via top Fisher expert
│   │   ├── scatter3d.py             # Flexible 3-D Plotly scatter with per-class means, labels, and colorbar
│   │   ├── _html_save.py            # Client-side JS injected into experts.html for figure capture via save server
│   │   └── pretokenize.py           # Converts HuggingFace datasets to SAELens tokenized format
│   ├── latex/
│   │   ├── save_server.py           # Local HTTP server (port 7788) for interactive figure collection
│   │   ├── camera_ready.py          # Camera-ready LaTeX figure assembly with PIL legends
│   │   └── tables.py                # LaTeX table generation from results.json
│   └── cli/
│       ├── cli.py                   # Centralized CLI entry point (smixae command)
│       └── train.py                 # smixae train subcommand — all training options as CLI flags
├── docs/
│   ├── ARCHITECTURE.md              # Encoding/decoding pipelines, tensor shapes, dead expert recovery
│   ├── ANALYSIS.md                  # Probing workflow, scoring metrics, dataset format
│   ├── TRAINING.md                  # Training procedure, hyperparameters, checkpointing
│   ├── STEERING.md                  # Steering experiments, activation patching
│   └── LATEX.md                     # LaTeX/figure export toolkit reference
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

| Parameter       | Value | Notes                                                     |
|-----------------|-------|-----------------------------------------------------------|
| `n_experts`     | 2048  | Used in current paper experiments; 4096 is possible but increases memory |
| `d_expert`      | 8     | Expert capacity — scales parameters vs. expressivity      |
| `d_bottleneck`  | 3     | Chosen for 3-D visualization; tuning requires dimensionality regularization (not in this repo) |
| `k_experts`     | 128   | Active experts per token; 64–128 is a good range          |
| `d_model`       | 3584  | Gemma 2-9B residual stream dimension                      |

`d_bottleneck=3` is chosen for direct 3-D visualization. Increasing it is possible but requires a minimum-dimensionality penalty (not currently implemented) to prevent the bottleneck from collapsing to fewer effective dimensions.

### Dead Expert Recovery

Experts that haven't fired in `dead_after_n_passes` (default 1000) passes are considered dead. An auxiliary loss reconstructs the residual from dead experts using their top-k activations, encouraging them to participate.

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
- **`ExpertFilterConfig`**: controls expert selection — `active_threshold` (float, default `1e-5`), `min_active_fraction` (float, default `0.10`), `max_points` (int, default `1000`)
- **`Expert`**: holds per-expert bottleneck activations and labels; implements `evaluate_fisher()`, `evaluate_manifold()`, `evaluate_regression()`, `get_plot()`, `get_mean_plot()`, `sort_key()`
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

Causal intervention script using **full-sequence activation patching**: at every token position where the target expert fires, subtracts its current decoded contribution and adds the decoded target class mean. Inactive positions are untouched.

- **Current-time steering**: `"You glance at the clock and find it is {src_hour}. Your friend {name} asks you for the time, and you respond, saying it is"` — steers the perceived current time.

Expert selection reads `results.json` and picks the top experts by `cyc_24h` regression hypothesis R² (the hypothesis that best matches circular hour-of-day structure). A probing pass still runs to compute per-class bottleneck means via `collect_activations()` + `get_sae_activations()`. Output: `summary.csv` with per-expert accuracy, plus `scores/` directory with per-prompt detail. Exposed via CLI as `smixae steer main`.

### `src/analysis/scatter3d.py`

Flexible 3-D Plotly scatter utility used by `categorize_all.py` and `anthropic_newline.py`. Key public API:

- **`plot_3d_scatter(xyz, labels, ...)`**: top-level entry point — builds a complete figure with per-class means, optional label annotations, and optional colorbar.
- **`build_color_map(classes, colorscale)`**: maps class labels to `"rgb(...)"` strings.
- **`add_scatter_trace` / `add_mean_trace` / `add_label_annotations` / `add_colorbar_trace`**: composable building blocks; each returns the modified figure for chaining.

Colorscale formats: `None`/`"auto"` (HSV rainbow), any Plotly named scale (e.g. `"Viridis"`), `list[color]`, or `dict[label, color]`.

### `src/analysis/pretokenize.py`

Wraps SAELens `PretokenizeRunner` with a token-count cap so you can produce a fixed-size tokenized dataset from a streaming HuggingFace source. Exposed via CLI as `smixae pretokenize`. Key classes:

- **`LimitedPretokenizeRunnerConfig`**: extends `PretokenizeRunnerConfig` with `n_tokens` field.
- **`LimitedPretokenizeRunner`**: streams, materialises, tokenizes, and saves exactly `n_tokens` tokens (or fewer if the source is exhausted).

### `src/smixae/affine_smixae.py`

> **Not actively used or maintained.** Kept for historical reference only.

A SMIXAE variant that replaces bottleneck-norm routing with cosine-similarity routing via a learned `W_directions` parameter of shape `(n_experts, d_in)`. Each expert has an associated direction in input space; routing selects experts whose directions align with the current input. See `affine_smixae_encode()` for the full forward pass.

---

## Analysis Metrics

Four scoring metrics: **Fisher** (class separability in bottleneck space), **adjusted Fisher** (penalises experts active on only a subset of classes), **KNN continuity** (bottleneck proximity → LLM-space coherence, tends to surface linear directions), and **regression probing** (`Expert.evaluate_regression()`, CV mean ± std, modes: linear/ridge/logistic/multinomial). Newline-distance metrics (`decode_r2`, `encode_linear_r2`, `encode_periodic_r2`, `periodic_gain`) are produced by `anthropic_newline.py`. `sort_key()` accepts `"fisher"`, `"adjusted_fisher"`, `"continuity"`, or `"regression"`.

See [docs/ANALYSIS.md](docs/ANALYSIS.md) for full definitions, interpretation guides, the dataset config schema, and the `results.json` schema.

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
- Add a method to the `Expert` class in `src/analysis/utils.py`, or write a standalone scoring function
- The expert's bottleneck activations (shape: `n_active_samples × d_bottleneck`) are the primary input to any geometric analysis
- For regression-based metrics, see `anthropic_newline.py` as a reference pattern

### Adding New Probing Datasets

Add a new generator function in `generate_probing_data.py` following the existing pattern:
- Define templates (some with `{name}`, all with a concept placeholder)
- Define concept values
- Return a de-duplicated, shuffled DataFrame with `Sentence` and `Label` columns
- Output to `datasets/probing/<dataset_name>.csv`

---

## Running Python

Always use `uv run python` (or `uv run <script>`) instead of bare `python` or `python3`. The system Python is 3.6 and lacks modern type annotations; `uv` activates the correct project virtualenv.

```bash
# Correct
uv run python -c "from analysis.utils import DatasetConfig; ..."
uv run smixae probe all-datasets --help

# Wrong — uses Python 3.6, will fail on type annotation syntax
python my_script.py
```

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
    --n-experts 2048 \
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

`--use-affine-smixae` switches to `AffineSMIXAETraining` instead of `SMIXAETraining`. **Not actively used or maintained** — only use this flag for historical reproduction experiments.

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
└── latex
    ├── save-server              # Start local HTTP figure-collection server (port 7788)
    ├── figures                  # Assemble camera-ready PNGs into LaTeX figure files
    └── tables                   # Generate four LaTeX tables (probing + newline, summary + appendix) from results.json
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

## Figure Export and LaTeX Toolkit

See [docs/LATEX.md](docs/LATEX.md) for full details. Summary:

The browser-side capture and LaTeX assembly pipeline works as follows:

1. **Run the save server**: `smixae latex save-server` — starts a local HTTP server on port 7788. On HPC, SSH port-forward: `ssh -L 7788:localhost:7788 <host>`.
2. **Open an `experts.html`** in a browser — the embedded JS (from `src/analysis/_html_save.py`) polls the server every 3s and shows a queue badge when the server is available.
3. **Queue figures**: click the save button on any expert panel — the JS POSTs a 2200×1700px Plotly PNG to the server.
4. **Review and save**: visit `http://127.0.0.1:7788/` to inspect the gallery, remove unwanted figures, and batch-save all to disk (auto-crops white borders).
5. **Assemble LaTeX**: `smixae latex figures` — reads saved PNGs, generates PIL legends, and produces a `.tex` file with `\includegraphics` layout.
6. **Generate tables**: `smixae latex tables [--output-dir results/]` — reads `results.json` and writes four `.tex` files to `--output-dir`: `table_probing.tex` (summary with `\pm` CV std), `table_newline.tex` (summary), `table_probing_appendix.tex` (all 10 experts per model/hypothesis), `table_newline_appendix.tex` (all 10 experts per model/line-length). Requires `booktabs`, `multirow` packages.

---

## Known TODOs

- [ ] Explore `d_bottleneck > 3` with a minimum-dimensionality penalty
- [ ] **LaTeX table fixes**: Color bar in regenerated figures is too small and unreadable. Need a shared colorbar utility used by both `scatter3d.py` and `camera_ready.py`. Generated expert descriptions also need to be more prosaic.
- [ ] **SAEBench evaluation**: Test SMIXAE on SAEBench core, benchmarked against comparable Gemma Scope models. Requires monkey-patching SAEBench (upstream is not well-structured for custom architectures).