r"""Static HTML export of expert visualizations.

Produces self-contained HTML pages for each probing task found under
``--results-dir``, plus a top-level ``index.html``.  Every page embeds all
task-index JSON, per-expert JSON metadata, and binary tensor data inline, then
mocks the save-server HTTP API so the unmodified ``viewer.js`` and
``scatter.js`` work without any running server.

Pages work on any static host (GitHub Pages, Netlify, S3, etc.) and when
opened directly from the filesystem as ``file://`` URLs.

CLI usage::

    smixae latex export-static \\
        --results-dir results/ \\
        --output-dir output/web/
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import typer

app = typer.Typer()

_VIEWER_DIR = Path(__file__).parent / "viewer"

# ── Inline JS: fetch mock injected before viewer scripts ─────────────────────

_FETCH_MOCK_JS = r"""
(function () {
  var _orig = window.fetch.bind(window);
  window.fetch = function (url, opts) {
    var urlStr = String(url);
    var u;
    try { u = new URL(urlStr, window.location.href); } catch (_) { return _orig(url, opts); }
    var path = u.pathname;
    var qs = {};
    u.searchParams.forEach(function (v, k) { qs[k] = v; });
    var sd = window._STATIC_DATA;

    function jsonResp(obj) {
      return Promise.resolve(
        new Response(JSON.stringify(obj), { headers: { 'Content-Type': 'application/json' } })
      );
    }
    function binaryResp(b64) {
      var bin = atob(b64);
      var buf = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      return Promise.resolve(
        new Response(buf.buffer, { headers: { 'Content-Type': 'application/octet-stream' } })
      );
    }

    if (path === '/task') { return jsonResp(sd.taskIndex); }

    if (path === '/expert') {
      var e = sd.experts[qs.id];
      if (!e) return Promise.resolve(new Response('{}', { status: 404 }));
      return jsonResp(e.meta);
    }

    if (path === '/expert-tensors') {
      var e2 = sd.experts[qs.id];
      if (!e2) return Promise.resolve(new Response('', { status: 404 }));
      return binaryResp(e2.tensors);
    }

    if (path === '/colorscale') {
      var csKey = (qs.name || '') + '__' + (qs.n || '7') + '__' + (qs.skip_endpoints || 'true');
      return jsonResp({ colors: sd.colorscales[csKey] || [] });
    }

    return _orig(url, opts);
  };
})();
"""

# ── Inline JS: download-only save client (replaces save_client.js) ──────────

_STATIC_SAVE_JS = r"""
/**
 * Static-export save client.
 * saveFigurePNG always triggers a direct browser download — no queue server.
 */

function showToast(msg) {
  var t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(function () { t.classList.remove('show'); }, 3000);
}

async function saveFigurePNG(divId, suffix) {
  var gd = document.getElementById(divId);
  if (!gd || !gd.data) return;

  var SCALE = 2;
  var divRect = gd.getBoundingClientRect();
  var captureW = Math.round(divRect.width)  || 1100;
  var captureH = Math.round(divRect.height) || 850;

  // Snapshot annotation positions (scatter only)
  var annSnapshots = [];
  if (divId === 'active-scatter') {
    var wrapper = document.getElementById('scatter-wrapper');
    var overlay = document.getElementById('annotation-overlay');
    if (wrapper && overlay) {
      var wr = wrapper.getBoundingClientRect();
      overlay.querySelectorAll('.annotation-label').forEach(function (ann) {
        var textEl = ann.querySelector('.annotation-text');
        var text = textEl ? textEl.textContent.trim() : '';
        if (!text) return;
        var ar = ann.getBoundingClientRect();
        annSnapshots.push({
          x: (ar.left - wr.left) * SCALE, y: (ar.top - wr.top) * SCALE,
          w: ar.width * SCALE,            h: ar.height * SCALE,
          text: text,
        });
      });
    }
  }

  var fl = gd._fullLayout || {};
  var cleanData = gd.data.map(function (t) {
    var tc = Object.assign({}, t);
    tc.showlegend = false;
    if (tc.marker) tc.marker = Object.assign({}, tc.marker, { showscale: false });
    return tc;
  });
  var captureLayout = Object.assign({}, gd.layout || {}, {
    showlegend: false,
    title: { text: '' },
    margin: fl.margin,
    scene: Object.assign({}, (gd.layout || {}).scene || {}, {
      camera: (fl.scene || {}).camera,
    }),
  });

  var imgData = await Plotly.toImage(
    { data: cleanData, layout: captureLayout },
    { format: 'png', width: captureW, height: captureH, scale: SCALE }
  );

  var base64;
  if (annSnapshots.length > 0) {
    var IW = captureW * SCALE, IH = captureH * SCALE;
    var canvas = document.createElement('canvas');
    canvas.width = IW; canvas.height = IH;
    var ctx = canvas.getContext('2d');
    var img = new Image();
    await new Promise(function (res) { img.onload = res; img.src = imgData; });
    ctx.drawImage(img, 0, 0);
    var handleH = 16 * SCALE;
    annSnapshots.forEach(function (s) {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(s.x, s.y, s.w, s.h);
      ctx.strokeStyle = '#333333';
      ctx.lineWidth = 2 * SCALE;
      ctx.strokeRect(s.x, s.y, s.w, s.h);
      var fs = Math.round(14 * SCALE);
      ctx.fillStyle = '#000000';
      ctx.font = fs + 'px sans-serif';
      ctx.textBaseline = 'middle';
      ctx.fillText(s.text, s.x + 9 * SCALE, s.y + handleH + (s.h - handleH) / 2);
    });
    base64 = canvas.toDataURL('image/png').split(',')[1];
  } else {
    base64 = imgData.split(',')[1];
  }

  var m = window._currentExpertMeta;
  if (!m) return;
  var scoreLabel = m.score_type || 'score';
  var score = m.hyp_score != null ? m.hyp_score.toFixed(4) : 'NA';
  var filename = [
    window._experimentId || '',
    window._datasetName  || '',
    'E' + m.expert_id,
    m.hyp_name   || '',
    scoreLabel,
    score,
    suffix,
  ].join('__') + '.png';

  var link = document.createElement('a');
  link.href = 'data:image/png;base64,' + base64;
  link.download = filename;
  link.click();
  showToast('Downloaded: ' + filename);
}
"""


# ── Tensor helpers ─────────────────────────────────────────────────────────────


def _process_expert(task_dir: Path, expert_id: int) -> tuple[str, list[int]]:
    """Load one expert's .pth and return (base64_binary, unique_label_ids).

    The binary format matches ``/expert-tensors``:
    points (float32 n*3) | labels (int32 n) | continuity (float32 n)
    """
    import torch

    tensors = torch.load(
        task_dir / "experts" / f"E{expert_id}.pth",
        map_location="cpu",
        weights_only=True,
    )
    pts = tensors["points"].contiguous()
    parts: list[bytes] = [pts.numpy().astype("float32").tobytes()]

    unique_labels: list[int] = []
    if "labels" in tensors and tensors["labels"] is not None:
        lbl = tensors["labels"].contiguous()
        parts.append(lbl.numpy().astype("int32").tobytes())
        unique_labels = sorted({int(x) for x in lbl.tolist()})

    if "continuity" in tensors and tensors["continuity"] is not None:
        parts.append(tensors["continuity"].contiguous().numpy().astype("float32").tobytes())

    b64 = base64.b64encode(b"".join(parts)).decode("ascii")
    return b64, unique_labels


# ── Colorscale pre-computation ────────────────────────────────────────────────


def _compute_colorscales(
    task_index: dict[str, Any],
    expert_label_sets: dict[int, list[int]],
) -> dict[str, list[str]]:
    """Return all colorscale responses the viewer will request for this task.

    Keys match the mock's lookup: ``"{name}__{n}__{skip_endpoints}"``.

    Args:
        task_index: Parsed index.json for the task.
        expert_label_sets: Mapping from expert_id to its unique label ids.

    Returns:
        Dict of colorscale key → list of ``'rgb(...)'`` strings.
    """
    from analysis.colors import sample_named_scale_discrete

    colorscales: dict[str, list[str]] = {}
    color_spec = task_index.get("color", {})
    scale_name: str = color_spec.get("scale") or "Plasma"
    mode: str = color_spec.get("mode", "discrete")
    skip_ep: bool = color_spec.get("skip_endpoints", True)
    has_color_map: bool = bool(color_spec.get("color_map"))

    # Smooth 64-sample array — always needed (continuous path + means colorbar).
    arr_key = f"{scale_name}__64__false"
    if arr_key not in colorscales:
        colorscales[arr_key] = sample_named_scale_discrete(scale_name, 64, skip_endpoints=False)

    # Per-expert discrete swatches — only when mode=discrete and no explicit map.
    if mode == "discrete" and not has_color_map:
        for labels in expert_label_sets.values():
            n = len(labels) if labels else 7
            key = f"{scale_name}__{n}__{str(skip_ep).lower()}"
            if key not in colorscales:
                colorscales[key] = sample_named_scale_discrete(scale_name, n, skip_endpoints=skip_ep)

    return colorscales


# ── HTML builders ──────────────────────────────────────────────────────────────


def _build_task_html(
    task_index: dict[str, Any],
    expert_meta: dict[int, dict[str, Any]],
    expert_tensors_b64: dict[int, str],
    colorscales: dict[str, list[str]],
) -> str:
    """Return a fully self-contained HTML string for one probing task."""
    title = task_index.get("title") or task_index.get("dataset_name") or "Expert Viewer"

    viewer_css = (_VIEWER_DIR / "viewer.css").read_text(encoding="utf-8")
    scatter_js = (_VIEWER_DIR / "scatter.js").read_text(encoding="utf-8")
    viewer_js = (_VIEWER_DIR / "viewer.js").read_text(encoding="utf-8")

    static_data: dict[str, Any] = {
        "taskIndex": task_index,
        "experts": {
            str(eid): {"meta": meta, "tensors": expert_tensors_b64[eid]}
            for eid, meta in expert_meta.items()
            if eid in expert_tensors_b64
        },
        "colorscales": colorscales,
    }
    static_data_json = json.dumps(static_data, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>SMIXAE Expert Viewer — {title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>{viewer_css}</style>
</head>
<body>
  <header>
    <h1 id="task-title">SMIXAE Expert Viewer</h1>
    <span id="queue-badge" style="display:none">
      <a id="badge-link" href="#">0 queued</a>
    </span>
  </header>

  <div id="layout">
    <nav id="views-nav">
      <div id="view-list"></div>
    </nav>
    <main id="main">
      <div id="tab-strip"></div>
      <div id="expert-header"></div>
      <div id="save-bar" style="display:none">
        <button class="save-btn" onclick="saveFigurePNG('active-scatter','scatter')">Download scatter</button>
        <button class="save-btn" id="save-means-btn" style="display:none"
                onclick="saveFigurePNG('active-mean','means')">Download means</button>
        <button class="save-btn" id="add-label-btn" onclick="createAnnotation()">Add Label</button>
        <button class="save-btn" id="lock-btn" onclick="toggleLockView()">Lock View</button>
      </div>
      <div id="reg-scores-container"></div>
      <div id="scatter-wrapper">
        <div class="plot-box" id="active-scatter"></div>
        <div id="plot-lock-mask"></div>
        <div id="annotation-overlay"></div>
      </div>
      <div class="plot-box" id="active-mean" style="display:none"></div>
    </main>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    window._STATIC_DATA = {static_data_json};
    history.replaceState(null, '', window.location.pathname + '?task=__static__');
  </script>
  <script>{_FETCH_MOCK_JS}</script>
  <script>{scatter_js}</script>
  <script>{_STATIC_SAVE_JS}</script>
  <script>{viewer_js}</script>
</body>
</html>"""


def _build_index_html(tasks: list[dict[str, Any]]) -> str:
    """Return a styled top-level index listing all exported task pages."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        exp = t.get("experiment_id") or "Unknown"
        groups.setdefault(exp, []).append(t)

    rows_html = ""
    for exp in sorted(groups):
        rows_html += f'<div class="exp-group">{exp}</div>\n'
        for item in sorted(groups[exp], key=lambda x: x.get("dataset_name", "")):
            fn = item["filename"]
            label = item.get("title") or item.get("dataset_name") or fn
            rows_html += f'    <a class="task-link" href="{fn}">{label}</a>\n'

    n = len(tasks)
    plural = "s" if n != 1 else ""
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>SMIXAE Expert Visualizations</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f5f5f5; color: #333; }}
    header {{ background: #1a1a2e; color: #eee; padding: 0 24px; height: 56px;
              display: flex; align-items: center; }}
    header h1 {{ font-size: 20px; font-weight: 600; }}
    .content {{ max-width: 800px; margin: 40px auto; padding: 0 24px; }}
    p.subtitle {{ color: #555; margin-bottom: 24px; font-size: 14px; }}
    .exp-group {{ font-size: 11px; font-weight: 700; color: #999;
                  letter-spacing: .06em; text-transform: uppercase;
                  margin: 20px 0 6px; }}
    .task-link {{ display: block; padding: 10px 14px; background: #fff;
                  border: 1px solid #e0e0e0; border-radius: 6px; color: #333;
                  text-decoration: none; margin-bottom: 6px; font-size: 14px; }}
    .task-link:hover {{ background: #eef2ff; border-color: #4c6ef5; color: #4c6ef5; }}
  </style>
</head>
<body>
  <header><h1>SMIXAE Expert Visualizations</h1></header>
  <div class="content">
    <p class="subtitle">{n} visualization{plural} exported. Click a task to open its interactive 3-D viewer.</p>
{rows_html}  </div>
</body>
</html>"""


# ── CLI command ────────────────────────────────────────────────────────────────


@app.command()
def export_static(
    results_dir: Path = typer.Option(..., help="Root results directory to scan for probing tasks (same as --results-dir for save-server)."),
    output_dir: Path = typer.Option(..., help="Directory to write static HTML files."),
) -> None:
    """Export expert visualizations as self-contained static HTML pages.

    Generates one HTML file per probing task found under ``--results-dir``,
    plus a top-level ``index.html``.  All task metadata, expert data, and
    tensor payloads are embedded inline — no web server is required.  The
    pages can be served from any static host (GitHub Pages, Netlify, S3) or
    opened directly as local files.

    Each page uses the same interactive viewer as ``save-server`` (Plotly 3-D
    scatter, rotation, annotations).  The save buttons trigger browser
    downloads instead of queuing to a server.
    """
    from analysis.probing_io import iter_task_dirs, read_expert_meta, read_task_index

    output_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = list(iter_task_dirs(results_dir))
    if not task_dirs:
        typer.echo(f"No probing tasks found under {results_dir}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(task_dirs)} task(s) under {results_dir}")

    exported: list[dict[str, Any]] = []

    for task_dir in task_dirs:
        rel = str(task_dir.relative_to(results_dir))
        typer.echo(f"  Exporting {rel} …")

        try:
            task_index = read_task_index(task_dir)
        except (OSError, json.JSONDecodeError) as exc:
            typer.echo(f"    [skip] Cannot read index.json: {exc}")
            continue

        # Collect all expert IDs referenced in the task index (deduplicated).
        all_eids: list[int] = []
        seen: set[int] = set()
        for rankings in task_index.get("experts_by_view", {}).values():
            for r in rankings:
                eid = r.get("expert_id")
                if eid is not None and eid not in seen:
                    all_eids.append(eid)
                    seen.add(eid)

        if not all_eids:
            typer.echo("    [skip] No experts in index.json")
            continue

        # Load per-expert metadata and tensors.
        expert_meta: dict[int, dict[str, Any]] = {}
        expert_tensors_b64: dict[int, str] = {}
        expert_label_sets: dict[int, list[int]] = {}

        for eid in all_eids:
            try:
                expert_meta[eid] = read_expert_meta(task_dir, eid)
            except (OSError, json.JSONDecodeError) as exc:
                typer.echo(f"    [warn] Cannot read E{eid}.json: {exc}")
                continue
            try:
                b64, labels = _process_expert(task_dir, eid)
                expert_tensors_b64[eid] = b64
                expert_label_sets[eid] = labels
            except Exception as exc:
                typer.echo(f"    [warn] Cannot read E{eid}.pth: {exc}")

        if not expert_meta:
            typer.echo("    [skip] No experts loaded successfully")
            continue

        # Pre-compute all colorscales Python would serve.
        try:
            colorscales = _compute_colorscales(task_index, expert_label_sets)
        except Exception as exc:
            typer.echo(f"    [warn] Colorscale computation failed: {exc}")
            colorscales = {}

        # Output filename: rel-path components joined with '__'.
        filename = rel.replace("/", "__").replace("\\", "__") + ".html"

        html = _build_task_html(task_index, expert_meta, expert_tensors_b64, colorscales)
        (output_dir / filename).write_text(html, encoding="utf-8")

        size_kb = (output_dir / filename).stat().st_size // 1024
        typer.echo(f"    → {filename}  ({size_kb} KB)")

        exported.append(
            {
                "filename": filename,
                "rel": rel,
                "experiment_id": task_index.get("experiment_id", ""),
                "dataset_name": task_index.get("dataset_name", ""),
                "title": task_index.get("title", rel),
            }
        )

    if not exported:
        typer.echo("No tasks exported.")
        raise typer.Exit(1)

    index_html = _build_index_html(exported)
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")

    typer.echo(f"\nExported {len(exported)} task(s) → {output_dir.resolve()}")
    typer.echo(f"Open  {output_dir.resolve() / 'index.html'}  to browse.")
