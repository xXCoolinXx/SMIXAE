# SMIXAE LaTeX and Figure Export Toolkit

The `src/latex/` package provides a browser→server→LaTeX pipeline for assembling camera-ready figures and tables for publication.

---

## Overview

```
smixae probe  (src/analysis/categorize_all.py)
    │  Writes index.json + experts/ task data directories
    ▼
smixae latex save-server  (src/latex/save_server.py)
    │  Browse tasks at http://127.0.0.1:7788/
    │  Click task → /viewer/index.html?task=<rel>
    │  JS in viewer polls localhost:7788/ping
    │  POSTs 2200×1700px Plotly PNG on user click
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

# 1a. Export visualizations as self-contained static HTML (no server needed)
smixae latex export-static --results-dir PATH --output-dir PATH

# 2. Assemble camera-ready figures (writes .tex, camera_ready/, legends/ into <output-dir>/paper/)
smixae latex figures --camera-ready-dir PATH --output-dir PATH [--results-json PATH] [--results-dir PATH] [OPTIONS]

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

### Endpoints (Legacy + New)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Gallery HTML — thumbnail grid with remove and save-all controls |
| `/ping` | GET | Liveness check used by the browser JS |
| `/queue` | POST | Accept a base64-encoded PNG with a filename; add to queue |
| `/thumbnail/<id>` | GET | Serve a queued PNG as a thumbnail |
| `/save-all` | POST | Write all queued PNGs to `--output-dir`, applying PIL white-border crop |
| `/tasks` | GET | List all probing tasks under `--results-dir` |
| `/task?p=<rel>` | GET | Return task's `index.json` |
| `/expert?p=<rel>&id=<id>` | GET | Return per-expert JSON metadata |
| `/expert-tensors?p=<rel>&id=<id>` | GET | Return tensors as raw binary (application/octet-stream) |
| `/colorscale?name=<name>&n=<n>` | GET | Return sampled colors from Plotly scale |
| `/view?p=<rel>` | GET | Redirect to interactive viewer |
| `/viewer/<asset>` | GET | Serve static viewer assets (index.html, viewer.js, etc.) |

### Interactive Viewer

The save server serves an interactive **browser-based viewer** at `/viewer/`:

```
src/latex/viewer/
├── index.html     # Viewer shell with task navigation
├── viewer.css    # Viewer styles
├── viewer.js     # Main controller (task loading, tab strip, rendering)
├── scatter.js    # Plotly 3D scatter trace builder
└── save_client.js # PNG queue logic (same as legacy)
```

**Workflow:**
1. Browse tasks via `/tasks` or the gallery nav
2. Click a task to open `/viewer/index.html?task=<rel>`
3. The viewer loads `index.json` and builds a tab strip from `experts_by_view`
4. Click an expert tab to fetch `/expert?p=...&id=...` + `/expert-tensors?p=...&id=...`
5. Render the 3D scatter via Plotly client-side
6. Click "Save scatter" to queue a PNG (same filename convention as legacy)

The PNG queue and filename convention are **unchanged** — `camera_ready.py` works identically.

### HPC usage

The server binds to `localhost`. On a remote HPC node, forward the port in your SSH session:

```bash
ssh -L 7788:localhost:7788 <cluster-hostname>
```

Then open `http://127.0.0.1:7788/` in your local browser.

---

## Viewer Assets

The browser-side JavaScript is now served as static assets from `src/latex/viewer/` (previously was embedded in `experts.html` via `_html_save.py`):

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

The `continuous_color` field in `dataset_config.json` picks which one to render, and is also honoured by the interactive viewer figures — a dataset with `continuous_color: false` and a named `color_scale` (e.g. `hours.csv` + `phase`) renders as a discrete swatch legend both on the paper and in the browser.

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
{Model Display Name}. \textbf{Task}: (a) Expert {id}, rank~{rank}, {Hypothesis} ({score_label}\,=\,{value}). (b) ...
```

Example: `Gemma 2 9B, Layer 11. \textbf{Hours}: (a) Expert 541, rank~1, 24-Hour Ring ($R^2$\,=\,0.876). (b) Expert 1521, rank~3, 12-Hour Ring ($R^2$\,=\,0.734).`

The `rank~N` field is the expert's rank among all experts for that hypothesis (1 = top expert). It is populated by passing `--results-dir` to `smixae latex figures`, which scans the results directory for probe `index.json` files. If `--results-dir` is omitted or an expert is not found in the index, the rank is silently omitted from that entry's caption text.

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

---

## `src/latex/static_export.py` — Static HTML Export

Generates self-contained HTML pages for each probing task, suitable for hosting on a static web server (GitHub Pages, Netlify, S3, etc.) or opening directly as local files — no running server required.

### How it works

The command walks the same `--results-dir` tree as `save-server` and, for each task directory:

1. Reads `index.json` and all per-expert JSON files.
2. Loads each expert's `.pth` tensors, serializes them to the same binary layout as the `/expert-tensors` endpoint, and base64-encodes the result.
3. Pre-computes all colorscales (using Python's `plotly` via `analysis.colors`) that the viewer would normally fetch from the server.
4. Inlines everything into `window._STATIC_DATA` in the HTML.
5. Injects a **`fetch` mock** (a small JS IIFE) before the viewer scripts load. The mock intercepts all five server API calls (`/task`, `/expert`, `/expert-tensors`, `/colorscale`, `/ping`) and returns responses built from the embedded data, so `viewer.js` and `scatter.js` run unchanged.
6. Replaces `save_client.js` with a lightweight download-only variant — "Save scatter" / "Save means" become direct browser downloads instead of queuing to a server.

Output per invocation:
```
<output-dir>/
├── index.html                           # Task browser / landing page
├── {exp}__{task}.html                   # One self-contained page per task
└── ...
```

### Usage

```bash
smixae latex export-static \
    --results-dir results/ \
    --output-dir output/web/
```

### Size note

Each page embeds tensor data as base64 (~33 % overhead). A typical task with 10 experts × 5 000 points produces about 1–2 MB of HTML, plus Plotly.js loaded at runtime from CDN (~3 MB). Very large tasks (many experts or many points per expert) will produce proportionally larger files.

---

## Known Issues (TODOs)

- Generated expert descriptions could be further refined for more natural paper prose beyond the current "Expert N, Hypothesis (Score=Value)" format.
