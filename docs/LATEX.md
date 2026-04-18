# SMIXAE LaTeX and Figure Export Toolkit

The `src/latex/` package provides a browser→server→LaTeX pipeline for assembling camera-ready figures and tables for publication.

---

## Overview

```
experts.html (browser)
    │  JS polls localhost:7788/ping
    │  POSTs 2200×1700px Plotly PNG on user click
    ▼
smixae latex save-server  (src/latex/save_server.py)
    │  Gallery at http://127.0.0.1:7788/
    │  Review, remove, batch-save to disk (PIL auto-crops white borders)
    ▼
smixae latex figures      (src/latex/camera_ready.py)
    │  Reads saved PNGs, generates PIL legends, writes .tex layout
    ▼
smixae latex tables       (src/latex/tables.py)
    │  Reads results.json, writes probing and newline LaTeX tables
    ▼
LaTeX document
```

---

## CLI Reference

All commands are under the `smixae latex` subcommand group.

```bash
# 1. Start the save server
smixae latex save-server [--output-dir PATH] [--port INT] [--results-dir PATH]

# 2. Assemble camera-ready figures (writes .tex, camera_ready/, legends/ into <output-dir>/paper/)
smixae latex figures --camera-ready-dir PATH --output-dir PATH [--results-json PATH] [OPTIONS]

# 3. Generate tables (writes 4 .tex files into <output-dir>/paper/)
smixae latex tables [--output-dir results/]
```

Both `figures` and `tables` use the same `--output-dir` convention: it is the
**parent** of the final `paper/` folder. Everything a LaTeX document needs ends
up under `<output-dir>/paper/`. Run `smixae latex <subcommand> --help` for the
full flag reference.

---

## `src/latex/save_server.py` — Figure Collection Server

A minimal HTTP server (`BaseHTTPRequestHandler`) that receives Plotly-rendered PNGs from the browser and queues them for review.

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Gallery HTML — thumbnail grid with remove and save-all controls |
| `/ping` | GET | Liveness check used by the browser JS |
| `/queue` | POST | Accept a base64-encoded PNG with a filename; add to queue |
| `/thumbnail/<id>` | GET | Serve a queued PNG as a thumbnail |
| `/save-all` | POST | Write all queued PNGs to `--output-dir`, applying PIL white-border crop |

### HPC usage

The server binds to `localhost`. On a remote HPC node, forward the port in your SSH session:

```bash
ssh -L 7788:localhost:7788 <cluster-hostname>
```

Then open `http://127.0.0.1:7788/` in your local browser.

---

## `src/analysis/_html_save.py` — Browser-Side JS

The string constant `SAVE_CLIENT_JS` is injected into every `experts.html` produced by `categorize_all.py`. It:

- Polls `localhost:7788/ping` every 3s to detect save-server availability
- Shows a floating badge displaying the current queue count when the server is reachable
- Exposes `_queueFigure(divId, filename)`: uses `Plotly.toImage` to snapshot the 3D scatter at 2200×1700px (2× scale), hides legend/title/margins for a clean export, then POSTs the PNG to `/queue`
- Falls back to a browser download if the server is unreachable

---

## `src/latex/camera_ready.py` — LaTeX Figure Assembly

Reads saved PNGs and produces a `.tex` file with `\includegraphics` layout.

### PNG filename convention

The actual delimiter is `__` (double underscore). The full format:

```
{experiment_id}__{task}__{E}{expert_id}__{hyp_name}__{score_type}__{score}__{figure_type}.png
```

Example: `gemma_2_9b_l11__weekdays__E1234__cyc_7d__r2__0.8541__scatter.png`

`camera_ready.py` uses `_FILENAME_RE` (a compiled regex) to parse this. Files that don't match are **silently skipped** with a `[skip]` log line — check stderr if figures are missing.

**Short-form filenames (no `hyp_name`/`score`) are not parseable.** These occur when a flat (non-regression) expert entry has no `expert_meta` populated. In `_build_flat_html` in `utils.py`, the save-button filename falls back to `{experiment_id}__{task}__E{expert_id}__{figure_type}.png`, which the regex does not match. To fix: pass an `expert_meta` dict with `hyp_name`, `hyp_score`, and `score_type` when calling `_make_plot_entry` for unlabeled/flat entries.

**How `task` is determined:** `DatasetConfig.output_subdir` → `subdir` in `run_pipeline` → `dataset_title` string → JS `DATASET_TITLE` (lowercased, spaces → `_`) → embedded in the PNG filename → parsed by `camera_ready.py`. `_canonical_task()` strips the `_—_expert_analysis` suffix, so `"continuity_—_expert_analysis"` → canonical task `"continuity"`. This determines which `.tex` file a PNG is routed to.

The assembler groups figures by canonical task and hypothesis for layout.

### Legend generation

Legends are rendered by the shared backend in `src/analysis/colors.py` (PIL, not Plotly/Kaleido):
- Discrete legends: colored swatches + label text, one per class
- Continuous legends: vertical colorbar gradient using a named Plotly colorscale

The `continuous_color` field in `dataset_config.json` picks which one to render, and is also honoured by the interactive `experts.html` figures — a dataset with `continuous_color: false` and a named `color_scale` (e.g. `hours.csv` + `phase`) renders as a discrete swatch legend both on the paper and in the browser.

For named scales, `sample_named_scale_discrete` samples at `(i + 1) / (n + 1)` positions (instead of `0, …, 1`) so circular scales don't collide at the endpoints — the first and last class are guaranteed visually distinct. `plot_3d_scatter`'s colorbar widget inherits the same clipping via `build_clipped_colorscale`.

Colorscale resolution goes through `plotly.colors` — any named Plotly scale (e.g. `"Plasma"`, `"Phase"`) is accepted.

### LaTeX output

The generated `.tex` uses:
- `\includegraphics` with fixed-height `minipage` blocks for consistent sizing
- `\multirow` for task-level row labels
- `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`)

Required LaTeX packages: `graphicx`, `booktabs`, `multirow`, `caption`.

### Caption format

Captions are generated automatically by `_caption_text()`. The format is:

```
{Model Display Name}. \textbf{Task}: (a) Expert {id}, {Hypothesis} ({score_label}\,=\,{value}). (b) ...
```

Example: `Gemma 2 9B, Layer 11. \textbf{Hours}: (a) Expert 541, 24-Hour Ring ($R^2$\,=\,0.876). (b) Expert 1521, 12-Hour Ring ($R^2$\,=\,0.734).`

Score labels: `$R^2$`, `Accuracy`, `Score`, `$\Delta R^2_{\mathrm{per}}$`, `Cont.`

Each caption includes a description of the plotted points:
- **Probe figures**: "Larger points denote class mean activations in the bottleneck space; smaller points are individual token activations."
- **Newline figures**: "Points represent individual token activations in the bottleneck space, colored by distance since the last newline."

Generates four table files from `results.json` into `<output-dir>/paper/` (default `results/paper/`):

| File | Contents |
|------|----------|
| `table_probing.tex` | Models × Tasks × Hypotheses, top-1 and top-5μ scores with `$\pm$` CV std |
| `table_newline.tex` | Gemma 2 9B periodic gain summary broken down by line length |
| `table_probing_appendix.tex` | Per-model detail: all 10 experts per hypothesis with score `$\pm$` std |
| `table_newline_appendix.tex` | Per-model detail: all 10 experts per line length with periodic gain |

The `$\pm$` values in the probing tables are cross-validation standard deviations from `Expert.evaluate_regression()`. The top-5μ `$\pm$` is the mean of individual expert CV stds across the top-5. See [docs/ANALYSIS.md — results.json Schema](ANALYSIS.md) for the underlying data format.

Required LaTeX packages: `booktabs`, `multirow`.

---

## Known Issues (TODOs)

- Generated expert descriptions could be further refined for more natural paper prose beyond the current "Expert N, Hypothesis (Score=Value)" format.
