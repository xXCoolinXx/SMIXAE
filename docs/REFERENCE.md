# SMIXAE Code Reference

Per-file descriptions of source modules and their public API. For architecture details see [ARCHITECTURE.md](ARCHITECTURE.md); for analysis workflow see [ANALYSIS.md](ANALYSIS.md).

---

## `src/smixae/__init__.py`

Exports the four public classes and registers the SMIXAE architecture with SAELens under the name `"smixae"`. Importing `smixae` (e.g. at the top of a script) is sufficient to register the architecture — no manual registration needed.

---

## `src/smixae/smixae.py`

- **`SMIXAEConfig`**: Inference-only config dataclass. Parameters: `n_experts`, `d_expert`, `d_bottleneck`, `rescale_acts_by_decoder_norm`.
- **`SMIXAE`**: Inference model. Inherits from `SAE[SMIXAEConfig]`. Implements `encode()` and `decode()`.
- **`SMIXAETrainingConfig`**: Training config. Adds `k_experts`, `aux_loss_coefficient`, `threshold_lr`, `dead_after_n_passes`.
- **`SMIXAETraining`**: Training model. Implements `training_forward_pass()` with MSE + auxiliary loss, `update_threshold()`, and dead expert tracking.
- **`smixae_encode()`**: Shared encoding logic used by both inference and training classes.

---

## `src/smixae/affine_smixae.py`

> **Not actively used or maintained.** Kept for historical reference only.

A SMIXAE variant that replaces bottleneck-norm routing with cosine-similarity routing via a learned `W_directions` parameter of shape `(n_experts, d_in)`. Each expert has an associated direction in input space; routing selects experts whose directions align with the current input. See `affine_smixae_encode()` for the full forward pass.

---

## `src/cli/cli.py`

Centralized CLI entry point. Registers all subcommand groups (`train`, `probe`, `newline`, `steer`, `core`, `latex`, etc.).

---

## `src/cli/train.py`

Typer app exposing all `LanguageModelSAERunnerConfig` and `SMIXAETrainingConfig` options as CLI flags, grouped by category (Model, Data, Training, SAE Architecture, Logging). Importing `smixae` at the top registers the architecture with SAELens as a side effect.

---

## `src/cli/core_eval.py`

Exposes `smixae core` with a single positional `checkpoint_path` argument (mutually exclusive with `--hf-release` + `--hf-sae-id`). Always writes to `results/core_eval_results.json` unless `--output-json` is overridden, keyed by `base_model_name` / derived `layer_{N}` / `display-name`.

---

## `src/analysis/utils.py`

Shared infrastructure used by all analysis scripts:
- **`load_llm()`** / **`load_sae()`**: model and checkpoint loading
- **`collect_hook_activations()`**: HuggingFace `register_forward_hook` pattern; returns one `(batch, seq, d_model)` CPU tensor per batch
- **`encode_sae_batched()`**: batched SAE encoding; returns `(N, n_experts, d_bottleneck)` float32 tensor
- **`collect_activations()`**: end-to-end data loading + tokenization + LLM activation collection; returns `(activations, str_tokens, labels, label_names, last_token_positions, n_classes)`
- **`get_sae_activations()`**: encodes LLM activations through SMIXAE, builds and returns a list of `Expert` objects filtered by activity threshold
- **`ExpertFilterConfig`**: controls expert selection — `active_threshold` (float, default `1e-5`), `min_active_fraction` (float, default `0.10`), `max_points` (int, default `1000`)
- **`Expert`**: holds per-expert bottleneck activations and labels; implements `evaluate_fisher()`, `evaluate_manifold()`, `evaluate_regression()`, `get_plot()`, `get_mean_plot()`, `sort_key()`
- **`_strip_prefix()`**: strips `NN_` ordering prefixes from label display strings

---

## `src/analysis/categorize_all.py`

The primary analysis script. Loads a trained SMIXAE checkpoint and a labeled dataset, then:
1. Collects LLM activations at the hook point
2. Runs SMIXAE encoding, filters experts by activity
3. Scores experts by Fisher (labelled) or KNN continuity (unlabelled); for `sort_by="auto"` + unlabelled, randomly samples experts with >75% of `max_points` active tokens instead of ranking by continuity
4. Plots top-N experts as interactive 3D Plotly scatters in an HTML file

Exposed via CLI as the `probe` subcommand group.

**Internal structure notes:**
- `_make_plot_entry` is a **nested function** inside `run_pipeline`, so it has closure access to `use_random_sample`, `batch`, `run_cfg`, `cfg`, etc. — these do not need to be passed as arguments.
- The `all-datasets` command's unlabeled pass uses `base_run_cfg` directly (inheriting `sort_by="auto"`). It previously hardcoded `sort_by="continuity"`, which bypassed random sampling — do not reintroduce that override.
- For unlabeled flat entries (no regression hypotheses), you must pass an `expert_meta` dict with `hyp_name`, `hyp_score`, and `score_type` to `_make_plot_entry` so the browser save button generates a filename parseable by `camera_ready.py`. Without it, the filename is short-form and silently skipped. See [docs/LATEX.md](LATEX.md) for the full filename convention.

---

## `src/analysis/generate_probing_data.py`

Generates template-based synthetic datasets for probing. Each dataset has sentences labeled with a concept (weekdays, hours, temperatures, months, etc.). Output: CSV with `Sentence` and `Label` columns, written directly to `datasets/probing/`.

---

## `src/analysis/anthropic_newline.py`

Analyzes how experts encode distance-since-newline (a continuous position signal). Scores experts using linear and periodic (Fourier) regression on the bottleneck.

---

## `src/analysis/steer.py`

Causal intervention script using **full-sequence activation patching**: at every token position where the target expert fires, subtracts its current decoded contribution and adds the decoded target class mean. Inactive positions are untouched.

- **Current-time steering**: `"You glance at the clock and find it is {src_hour}. Your friend {name} asks you for the time, and you respond, saying it is"` — steers the perceived current time.

Expert selection reads `results.json` and picks the top experts by `cyc_24h` regression hypothesis R² (the hypothesis that best matches circular hour-of-day structure). A probing pass still runs to compute per-class bottleneck means via `collect_activations()` + `get_sae_activations()`. Output: `summary.csv` with per-expert accuracy, plus `scores/` directory with per-prompt detail. Exposed via CLI as `smixae steer main`.

---

## `src/analysis/scatter3d.py`

Flexible 3-D Plotly scatter utility used by `categorize_all.py` and `anthropic_newline.py`. Key public API:

- **`plot_3d_scatter(xyz, labels, ...)`**: top-level entry point — builds a complete figure with per-class means, optional label annotations, and optional colorbar.
- **`build_color_map(classes, colorscale)`**: maps class labels to `"rgb(...)"` strings.
- **`add_scatter_trace` / `add_mean_trace` / `add_label_annotations` / `add_colorbar_trace`**: composable building blocks; each returns the modified figure for chaining.

Colorscale formats: `None`/`"auto"` (HSV rainbow), any Plotly named scale (e.g. `"Viridis"`), `list[color]`, or `dict[label, color]`.

---

## `src/analysis/pretokenize.py`

Wraps SAELens `PretokenizeRunner` with a token-count cap so you can produce a fixed-size tokenized dataset from a streaming HuggingFace source. Exposed via CLI as `smixae pretokenize`. Key classes:

- **`LimitedPretokenizeRunnerConfig`**: extends `PretokenizeRunnerConfig` with `n_tokens` field.
- **`LimitedPretokenizeRunner`**: streams, materialises, tokenizes, and saves exactly `n_tokens` tokens (or fewer if the source is exhausted).

---

## `src/latex/save_server.py`

Local HTTP server (port 7788) for interactive figure collection. See [LATEX.md](LATEX.md) for full details.

---

## `src/latex/camera_ready.py`

Camera-ready LaTeX figure assembly with PIL legends. See [LATEX.md](LATEX.md) for full details.

---

## `src/latex/tables.py`

LaTeX table generation from `results.json` + `core_eval_results.json`. See [LATEX.md](LATEX.md) for full details.

---

## `smixae_run.py`

Thin shim for PBS / direct invocation. Injects hardcoded Gemma 2-9B defaults into `sys.argv` and calls `app()` from `src/cli/cli.py`. For full control use `smixae train` directly.
