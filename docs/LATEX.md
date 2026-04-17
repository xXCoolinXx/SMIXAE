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

# 2. Assemble camera-ready figures
smixae latex figures [--input-dir PATH] [--output PATH] [OPTIONS]

# 3. Generate tables (writes 4 .tex files to --output-dir)
smixae latex tables [--output-dir results/]
```

Run `smixae latex <subcommand> --help` for the full flag reference.

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

Saved figures must follow:

```
{experiment_id}_{task}_{expert_id}_{hypothesis}_{score}.png
```

The assembler parses this to group figures by task and hypothesis for layout purposes.

### Legend generation

Legends are rendered with PIL (not Plotly/Kaleido), using:
- Discrete legends: colored swatches + label text, one per class
- Continuous legends: vertical colorbar gradient using a named Plotly colorscale

Colorscale resolution goes through `plotly.colors` — any named Plotly scale (e.g. `"Plasma"`, `"Phase"`) is accepted.

### LaTeX output

The generated `.tex` uses:
- `\includegraphics` with fixed-height `minipage` blocks for consistent sizing
- `\multirow` for task-level row labels
- `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`)

Required LaTeX packages: `graphicx`, `booktabs`, `multirow`, `caption`.

---

## `src/latex/tables.py` — LaTeX Table Generation

Generates four table files from `results.json` into `--output-dir` (default `results/`):

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

- The color bar in regenerated figures is too small and unreadable at paper scale. A shared colorbar utility is needed that is used consistently by both `scatter3d.py` and `camera_ready.py`.
- Generated figure descriptions/captions are mechanical/templated — they should be more prosaic for paper inclusion.
