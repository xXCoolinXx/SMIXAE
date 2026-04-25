# SMIXAE — Agent Guide

## What is SMIXAE?

**Sparse MIXture of Autoencoders (SMIXAE)** is an interpretability architecture for large language models. It is an alternative to Sparse Autoencoders (SAEs) designed to model **nonlinear features** — general manifolds — rather than assuming features lie along independent linear directions.

A standard SAE decomposes a model's residual stream into a sparse sum of linear directions. SMIXAE instead assigns activations to a sparse set of **experts**, each of which has a low-dimensional **bottleneck** (currently 3D for visualization). This bottleneck allows each expert to learn an arbitrary geometry: rings, spirals, helices, clusters, or any other manifold structure.

**Core novelty**: SAEs assume feature independence along directions. SMIXAE explicitly relaxes this assumption, allowing each expert's bottleneck to capture nonlinear relationships in the data.

---

## Documentation Policy

After every major change — new class, new metric, new CLI command, changed default, changed algorithm — update both `AGENTS.md` and the relevant file(s) under `docs/` to reflect the new state. Do not leave docs describing removed or replaced behavior. `AGENTS.md` should be reserved mainly for meta-level documentation - style and where to find other documentation. Major techniques and tools should have their own entire documentation.

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
│   │   ├── categorize_all.py        # Expert probing pipeline: load checkpoint, evaluate experts, write index.json + experts/ task dirs
│   │   ├── anthropic_newline.py     # Newline-position manifold analysis
│   │   ├── steer.py                 # Steering experiments: coordinate substitution via top Fisher expert
│   │   ├── scatter3d.py             # Flexible 3-D Plotly scatter with per-class means, labels, and colorbar
│   │   ├── colors.py                # Shared colour/colorbar/legend backend (Plotly traces + PIL PNGs)
│   │   ├── core_eval.py             # Core SAE eval metrics (SAEBench reimplemented, HuggingFace, no TransformerLens)
│   │   └── pretokenize.py           # Converts HuggingFace datasets to SAELens tokenized format
│   ├── latex/
│   │   ├── save_server.py           # Local HTTP server (port 7788) for interactive figure collection
│   │   ├── static_export.py         # Static HTML export: self-contained pages for web hosting (smixae latex export-static)
│   │   ├── camera_ready.py          # Camera-ready LaTeX figure assembly with PIL legends
│   │   ├── tables.py                # LaTeX table generation from results.json + core_eval_results.json
│   │   └── viewer/                  # Static SPA viewer (index.html, viewer.js, scatter.js, save_client.js, viewer.css)
│   └── cli/
│       ├── cli.py                   # Centralized CLI entry point (smixae command)
│       ├── train.py                 # smixae train subcommand — all training options as CLI flags
│       └── core_eval.py             # smixae core subcommand — evaluate a single SAE on core metrics
├── docs/
│   ├── ARCHITECTURE.md              # Encoding/decoding pipelines, tensor shapes, dead expert recovery
│   ├── ANALYSIS.md                  # Probing workflow, scoring metrics, dataset format, extending analysis
│   ├── TRAINING.md                  # Training procedure, hyperparameters, checkpointing
│   ├── STEERING.md                  # Steering experiments, activation patching
│   ├── LATEX.md                     # LaTeX/figure export toolkit reference
│   ├── CORE_EVAL.md                 # Core SAE evaluation metrics, SMIXAE adapter, GemmaScope baselines
│   ├── CLI.md                       # Full CLI command tree and usage examples
│   └── REFERENCE.md                 # Per-file code reference: classes, functions, and module descriptions
├── datasets/
│   ├── probing/                     # Labeled datasets for probing experiments (populated by generate_probing_data.py)
│   │   ├── dataset_config.json      # Probing dataset configs — read by categorize_all.py only
│   │   └── newline_config.json      # Newline color info — read by camera_ready.py only
│   └── steering/                    # Steering prompt datasets (populated by generate_steering_data.py)
├── experiments/                     # Experiment scripts
│   ├── run.sh                       # Generic runner: --model/--hook/--experiment-name/--steps → train/probe/newline/saebench/steer
│   ├── core_eval.sh                 # Core eval for all experiments + GemmaScope baselines
│   ├── gemma_2_2b_l12.sh            # Gemma 2-2B layer 12: thin wrapper over run.sh
│   ├── gemma_2_9b_l11_newline.sh    # Gemma 2-9B layer 11: probe + newline only
│   └── gemma_2_9b_l20_general.sh    # Gemma 2-9B layer 20: thin wrapper over run.sh
├── results/                         # All outputs, created at runtime (not committed)
│   ├── results.json                 # Probing + newline metrics (written by probe/newline steps)
│   ├── core_eval_results.json       # Core eval metrics (written by smixae core)
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

## Documentation Index

| Doc | What it covers |
|-----|----------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Encoding/decoding pipelines with tensor shapes, bottleneck rationale, dead expert recovery, threshold semantics |
| [ANALYSIS.md](docs/ANALYSIS.md) | Probing workflow, scoring metrics (Fisher, continuity, regression), dataset config schema, results.json schema, how to extend analysis |
| [TRAINING.md](docs/TRAINING.md) | Training command, hyperparameter guide, W&B monitoring, resuming, experiment scripts |
| [STEERING.md](docs/STEERING.md) | Activation patching mechanism, expert selection, cross-layer sweeps |
| [LATEX.md](docs/LATEX.md) | Browser→server→LaTeX pipeline, figure naming convention, table generation |
| [CORE_EVAL.md](docs/CORE_EVAL.md) | Core eval metrics (L0, MSE, CE score), SMIXAE adapter, GemmaScope baselines, results JSON structure |
| [CLI.md](docs/CLI.md) | Full CLI command tree and usage examples |
| [REFERENCE.md](docs/REFERENCE.md) | Per-file code reference — classes, functions, and module descriptions |

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

If the virtual environment is not initialized, run 
```bash
source .venv/bin/activate
```

---

## Linting

After making code changes, run `uv run ruff check` and fix any errors before handing the task back. The ruff config (`pyproject.toml`) selects `E`, `F`, `W`, `I`, `D` with the Google docstring convention; `ANEW_Processing/`, `experiments/`, and `smixae_run.py` are exempted from `D` rules via per-file-ignores.

Notes on common fixes:
- `F841` (unused variable): comment the assignment out with `# ` rather than deleting it.
- `D417` (missing arg description): add the missing parameter to the existing `Args:` block — don't replace the docstring.
- `D301` (backslash in docstring): prefix the docstring with `r`.

---

## What to Avoid

### Do NOT use TransformerLens

**Always use SAELens / HuggingFace `transformers` for model loading and activation collection.** TransformerLens is significantly slower and is not needed — `categorize_all.py` and `smixae_run.py` both demonstrate the correct approach.

All scripts now use HuggingFace for model loading. No TransformerLens usage remains in the codebase.

### Do NOT assume features are linear

The whole point of SMIXAE is that features can be nonlinear manifolds. Don't apply analysis methods that assume linearity (e.g. PCA for interpreting bottleneck structure) without also checking for nonlinear structure.

### Do NOT mix probing and newline configs

`dataset_config.json` is for the probing pipeline (`categorize_all.py`) only — every entry must have a real `dataframe_path` pointing to a local CSV. `newline_config.json` is for `camera_ready.py` figure assembly only — entries use `"task"` directly and provide color scheme info for newline figures. Do not add newline/pile-uncopyrighted entries to `dataset_config.json` — it will break the probing pipeline with file-not-found or wrong-column errors.

---

## Known TODOs

- [ ] Explore `d_bottleneck > 3` with a minimum-dimensionality penalty
- [x] Add the rank of the experts chosen for visualization on the probing task to avoid cherry picking claims