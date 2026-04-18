"""Shared utilities for SMIXAE analysis scripts."""

from __future__ import annotations

import dataclasses
import gc
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.offline as pyo
import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from plotly.graph_objects import Figure
from sae_lens import SAE
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from analysis._html_save import SAVE_CLIENT_JS
from analysis.scatter3d import plot_3d_scatter
from smixae import SMIXAE

# ── Tabbed HTML builder ───────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script>__PLOTLYJS__</script>
  __SAVEJS__
  <style>
    body {{ font-family: sans-serif; margin: 8px; }}
    .tab-strip {{ display:flex; flex-wrap:wrap; gap:4px; margin-bottom:8px; }}
    .tab-btn {{ padding:4px 10px; cursor:pointer; border:1px solid #aaa;
                border-radius:3px; background:#f0f0f0; font-size:13px; }}
    .tab-btn.active {{ background:#333; color:#fff; }}
    .tab-pane {{ display:none; flex-direction:column; gap:8px; }}
    .tab-pane.active {{ display:flex; }}
    .plot-box {{ width:100%; height:650px; }}
    .save-bar {{ margin:2px 0 4px; display:flex; gap:6px; }}
    .save-btn {{ padding:3px 10px; cursor:pointer; border:1px solid #888;
                 border-radius:3px; background:#f8f8f8; font-size:12px; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <div class="tab-strip">{tab_buttons}</div>
  {tab_panes}
  <script>
    const FIGURES = {{{figures_json}}};
    const META = {meta_json};
    const EXPERIMENT_ID = {experiment_id_js};
    const DATASET_TITLE = {dataset_title_js};
    const rendered = new Set();
    function savePNG(divId, key, suffix) {{
      const m = META[key];
      const filename = m && m.hyp_name
        ? [EXPERIMENT_ID, DATASET_TITLE,
           'E' + m.expert_id, m.hyp_name,
           m.score_type, m.hyp_score != null ? m.hyp_score.toFixed(4) : 'NA', suffix].join('__')
        : [EXPERIMENT_ID, DATASET_TITLE,
           'E' + (m ? m.expert_id : key), suffix].join('__');
      _queueFigure(divId, filename);
    }}
    function renderTab(idx) {{
      document.querySelectorAll('.tab-pane').forEach((pane, i) => {{
        if (i !== idx) return;
        pane.querySelectorAll('.plot-box').forEach(box => {{
          if (!rendered.has(box.id)) {{
            Plotly.newPlot(box.id, FIGURES[box.id].data, FIGURES[box.id].layout, {{responsive: true}});
            rendered.add(box.id);
          }} else {{
            Plotly.Plots.resize(box);
          }}
        }});
      }});
    }}
    function switchTab(idx) {{
      document.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
      document.querySelectorAll('.tab-pane').forEach((p, i) => p.classList.toggle('active', i === idx));
      renderTab(idx);
    }}
    renderTab(0);
  </script>
</body>
</html>"""


def build_dataset_html(
    expert_entries: list[tuple[str, Figure, Figure | None] | tuple[str, Figure, Figure | None, dict[str, float]]],
    dataset_title: str,
    per_hypothesis_entries: "dict[str, tuple[str, list[tuple[str, Figure, Figure | None, dict[str, float]]]]] | None" = None,
    experiment_id: str = "",
) -> str:
    """Build a self-contained HTML page containing Plotly figures for experts.

    Two rendering modes:

    **Per-hypothesis mode** (when ``per_hypothesis_entries`` is provided):
        Renders one row-section per regression hypothesis, each containing a
        horizontal tab strip of the top-10 experts for that hypothesis.  The
        page scrolls vertically through the hypothesis rows.

    **Flat mode** (fallback):
        Renders all experts as a single horizontal tab strip (original behaviour).

    Figures are embedded as JSON and rendered lazily (only when the tab is first
    selected).  The Plotly.js bundle is inlined so the file is fully standalone.

    Args:
        expert_entries: Flat list of ``(tab_label, scatter_fig, mean_fig[, reg_scores])``
            tuples used in flat mode.  Pass an empty list when using
            ``per_hypothesis_entries``.
        dataset_title: String shown as the page ``<h2>`` heading and ``<title>``.
        per_hypothesis_entries: Optional dict mapping hypothesis name →
            ``(description, [(tab_label, scatter_fig, mean_fig, reg_scores), ...])``.
            When provided, the per-hypothesis row layout is used instead of
            the flat tab strip.
        experiment_id: Identifier embedded into the page for figure-capture
            filenames; safe to leave empty.

    Returns:
        A complete UTF-8 HTML document as a string.
    """
    if per_hypothesis_entries:
        return _build_per_hypothesis_html(per_hypothesis_entries, dataset_title, experiment_id)
    return _build_flat_html(expert_entries, dataset_title, experiment_id)


def _reg_scores_table(reg_scores: dict[str, float]) -> str:
    """Return an HTML ``<details>`` block listing hypothesis scores, or empty string."""
    if not reg_scores:
        return ""
    sorted_scores = sorted(reg_scores.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else float("-inf")))
    rows_html = "".join(
        f"<tr><td style='padding:2px 8px;border:1px solid #ccc'>{n}</td>"
        f"<td style='padding:2px 8px;border:1px solid #ccc'>{'%.4f' % v if v == v else 'nan'}</td></tr>"
        for n, v in sorted_scores
    )
    return (
        "<details style='margin-top:6px;font-size:12px'>"
        "<summary style='cursor:pointer'>Regression scores</summary>"
        "<table style='border-collapse:collapse;margin-top:4px'>"
        "<thead><tr>"
        "<th style='padding:2px 8px;border:1px solid #ccc'>Hypothesis</th>"
        "<th style='padding:2px 8px;border:1px solid #ccc'>Score</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody></table></details>"
    )


def _build_flat_html(
    expert_entries: list[tuple[str, Figure, Figure | None] | tuple[str, Figure, Figure | None, dict[str, float]] | tuple[str, Figure, Figure | None, dict[str, float], dict]],
    dataset_title: str,
    experiment_id: str = "",
) -> str:
    import json as _json
    tab_buttons: list[str] = []
    tab_panes: list[str] = []
    figures_json_parts: list[str] = []
    meta_entries: dict[str, dict] = {}

    for idx, entry in enumerate(expert_entries):
        tab_label, scatter_fig, mean_fig = entry[0], entry[1], entry[2]
        reg_scores: dict[str, float] = entry[3] if len(entry) > 3 else {}  # type: ignore[misc]
        expert_meta: dict = entry[4] if len(entry) > 4 else {}  # type: ignore[misc]

        scatter_id = f"scatter_{idx}"
        mean_id = f"mean_{idx}"
        key = str(idx)
        has_mean = mean_fig is not None

        meta_entries[key] = {
            "expert_id":   expert_meta.get("expert_id", "?"),
            "hyp_name":    expert_meta.get("hyp_name", ""),
            "hyp_score":   expert_meta.get("hyp_score"),
            "score_type":  expert_meta.get("score_type", "score"),
            "has_mean":    has_mean,
            "reg_scores":  {k: v for k, v in reg_scores.items() if v == v} if reg_scores else {},
        }

        active_cls = " active" if idx == 0 else ""

        tab_buttons.append(f'<button class="tab-btn{active_cls}" onclick="switchTab({idx})">{tab_label}</button>')

        save_means_html = ""
        if has_mean:
            save_means_html = (
                f'<button class="save-btn" '
                f"onclick=\"savePNG('{mean_id}','{key}','means')\">Save means</button>"
            )
            figures_json_parts.append(f'"{mean_id}": {pio.to_json(mean_fig, engine="json")}')

        save_bar = (
            f'<div class="save-bar">'
            f'<button class="save-btn" '
            f"onclick=\"savePNG('{scatter_id}','{key}','scatter')\">Save scatter</button>"
            f"{save_means_html}"
            f"</div>"
        )
        plot_divs = f'{_reg_scores_table(reg_scores)}\n    {save_bar}\n    <div class="plot-box" id="{scatter_id}"></div>'
        if has_mean:
            plot_divs += f'\n    <div class="plot-box" id="{mean_id}"></div>'

        tab_panes.append(f'<div class="tab-pane{active_cls}">\n    {plot_divs}\n  </div>')
        figures_json_parts.append(f'"{scatter_id}": {pio.to_json(scatter_fig, engine="json")}')

    html = _HTML_TEMPLATE.format(
        title=dataset_title,
        tab_buttons="\n    ".join(tab_buttons),
        tab_panes="\n  ".join(tab_panes),
        figures_json=",\n    ".join(figures_json_parts),
        meta_json=_json.dumps(meta_entries),
        experiment_id_js=_json.dumps(experiment_id or ""),
        dataset_title_js=_json.dumps(dataset_title.lower().replace(" ", "_")),
    )
    return (
        html
        .replace("__PLOTLYJS__", pyo.get_plotlyjs(), 1)
        .replace("__SAVEJS__", SAVE_CLIENT_JS, 1)
    )


def _build_per_hypothesis_html(
    per_hypothesis_entries: "dict[str, tuple[str, list]]",
    dataset_title: str,
    experiment_id: str = "",
) -> str:
    """All hypothesis tab-rows at top; single shared plot area below.

    Entry format per expert: ``(button_label, scatter_fig, mean_fig, reg_scores[, expert_meta])``.
    ``expert_meta`` is an optional 5th element dict with keys ``expert_id``,
    ``hyp_name``, ``hyp_score``, ``fisher_score``, ``n_points``.
    """
    import json as _json

    figures_json_parts: list[str] = []
    meta_entries: dict[str, dict] = {}
    hyp_rows_html: list[str] = []
    first_hyp: str | None = None

    for hyp_name, (hyp_desc, entries) in per_hypothesis_entries.items():
        if first_hyp is None:
            first_hyp = hyp_name

        btn_parts: list[str] = []
        for rank, entry in enumerate(entries):
            button_label: str = entry[0]
            scatter_fig: Figure = entry[1]
            mean_fig: "Figure | None" = entry[2]
            reg_scores: dict = entry[3] if len(entry) > 3 else {}  # type: ignore[misc]
            expert_meta: dict = entry[4] if len(entry) > 4 else {}  # type: ignore[misc]

            key = f"{hyp_name}_{rank}"
            figures_json_parts.append(f'"scatter_{key}": {pio.to_json(scatter_fig, engine="json")}')
            has_mean = mean_fig is not None
            if has_mean:
                figures_json_parts.append(f'"mean_{key}": {pio.to_json(mean_fig, engine="json")}')

            # Strip NaN from reg_scores so json.dumps doesn't choke
            clean_scores = {k: v for k, v in reg_scores.items() if v == v}

            meta_entries[key] = {
                "expert_id":   expert_meta.get("expert_id", "?"),
                "hyp_name":    expert_meta.get("hyp_name", hyp_name),
                "hyp_score":   expert_meta.get("hyp_score"),
                "fisher_score": expert_meta.get("fisher_score"),
                "n_points":    expert_meta.get("n_points"),
                "score_type":  expert_meta.get("score_type", "score"),
                "has_mean":    has_mean,
                "reg_scores":  clean_scores,
            }

            is_first = (rank == 0 and hyp_name == first_hyp)
            active_cls = " active" if is_first else ""
            btn_parts.append(
                f'<button class="tab-btn{active_cls}" '
                f'data-hyp="{hyp_name}" data-rank="{rank}" '
                f'onclick="showExpert(\'{hyp_name}\',{rank})">{button_label}</button>'
            )

        hyp_rows_html.append(
            f'<div class="hyp-row">'
            f'<span class="hyp-label"><b>{hyp_name}</b> — {hyp_desc}</span>'
            f'<div class="tab-strip">{"".join(btn_parts)}</div>'
            f'</div>'
        )

    figures_json = ",\n    ".join(figures_json_parts)
    meta_json = _json.dumps(meta_entries)
    rows_html = "\n  ".join(hyp_rows_html)
    experiment_id_js = _json.dumps(experiment_id or "")
    dataset_title_js = _json.dumps(dataset_title.lower().replace(" ", "_"))

    html = f"""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{dataset_title}</title>
  <script>__PLOTLYJS__</script>
  __SAVEJS__
  <style>
    body {{ font-family: sans-serif; margin: 8px; }}
    #hyp-rows {{ margin-bottom: 0; }}
    .hyp-row {{ display:flex; align-items:flex-start; gap:10px;
                padding:5px 0; border-bottom:1px solid #e0e0e0; }}
    .hyp-label {{ min-width:200px; max-width:200px; font-size:13px;
                  color:#444; padding-top:3px; line-height:1.4; }}
    .tab-strip {{ display:flex; flex-wrap:wrap; gap:3px; flex:1; }}
    .tab-btn {{ padding:3px 8px; cursor:pointer; border:1px solid #bbb;
                border-radius:3px; background:#f4f4f4; font-size:12px;
                white-space:nowrap; }}
    .tab-btn.active {{ background:#333; color:#fff; border-color:#333; }}
    #plot-area {{ margin-top:16px; padding-top:10px; border-top:2px solid #888; }}
    #expert-header {{ font-size:14px; color:#222; margin-bottom:6px;
                      padding:4px 0; }}
    .plot-box {{ width:100%; height:650px; }}
    #save-bar {{ margin:2px 0 6px; display:none; gap:6px; }}
    .save-btn {{ padding:3px 10px; cursor:pointer; border:1px solid #888;
                 border-radius:3px; background:#f8f8f8; font-size:12px; }}
  </style>
</head>
<body>
  <h2>{dataset_title}</h2>
  <div id="hyp-rows">
  {rows_html}
  </div>
  <div id="plot-area">
    <div id="expert-header"></div>
    <div id="save-bar" style="display:none">
      <button class="save-btn" onclick="saveFigurePNG('active-scatter','scatter')">Save scatter</button>
      <button class="save-btn" id="save-means-btn" style="display:none"
              onclick="saveFigurePNG('active-mean','means')">Save means</button>
    </div>
    <div id="reg-scores-container"></div>
    <div class="plot-box" id="active-scatter"></div>
    <div class="plot-box" id="active-mean" style="display:none"></div>
  </div>
  <script>
    const FIGURES = {{{figures_json}}};
    const META = {meta_json};
    const EXPERIMENT_ID = {experiment_id_js};
    const DATASET_TITLE = {dataset_title_js};
    let currentKey = null;

    function saveFigurePNG(divId, suffix) {{
      const m = META[currentKey];
      if (!m) return;
      const scoreLabel = m.score_type || 'score';
      const score = m.hyp_score != null ? m.hyp_score.toFixed(4) : 'NA';
      const filename = [EXPERIMENT_ID, DATASET_TITLE,
                        'E' + m.expert_id, m.hyp_name,
                        scoreLabel, score, suffix].join('__');
      _queueFigure(divId, filename);
    }}

    function buildRegTable(scores) {{
      if (!scores || !Object.keys(scores).length) return '';
      const rows = Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .map(([k, v]) =>
          '<tr>' +
          '<td style="padding:2px 8px;border:1px solid #ccc">' + k + '</td>' +
          '<td style="padding:2px 8px;border:1px solid #ccc">' +
            (isNaN(v) ? 'nan' : v.toFixed(4)) + '</td></tr>'
        ).join('');
      return '<details style="margin-bottom:8px;font-size:12px">' +
             '<summary style="cursor:pointer">All regression scores</summary>' +
             '<table style="border-collapse:collapse;margin-top:4px">' +
             '<thead><tr>' +
             '<th style="padding:2px 8px;border:1px solid #ccc">Hypothesis</th>' +
             '<th style="padding:2px 8px;border:1px solid #ccc">Score</th>' +
             '</tr></thead><tbody>' + rows + '</tbody></table></details>';
    }}

    function showExpert(hyp, rank) {{
      currentKey = hyp + '_' + rank;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      const btn = document.querySelector('[data-hyp="' + hyp + '"][data-rank="' + rank + '"]');
      if (btn) btn.classList.add('active');

      const key = currentKey;
      const m = META[key];
      if (!m) return;

      const parts = ['<b>Expert ' + m.expert_id + '</b>'];
      if (m.hyp_score != null) parts.push(m.hyp_name + ' = ' + m.hyp_score.toFixed(3));
      if (m.fisher_score != null) parts.push('Fisher = ' + m.fisher_score.toFixed(2));
      if (m.n_points != null) parts.push(m.n_points + ' points');
      document.getElementById('expert-header').innerHTML = parts.join(' &nbsp;|&nbsp; ');
      document.getElementById('reg-scores-container').innerHTML = buildRegTable(m.reg_scores);

      document.getElementById('save-bar').style.display = 'flex';
      document.getElementById('save-means-btn').style.display =
        (m.has_mean) ? 'inline-block' : 'none';

      Plotly.newPlot('active-scatter',
        FIGURES['scatter_' + key].data,
        FIGURES['scatter_' + key].layout,
        {{responsive: true}});

      const meanBox = document.getElementById('active-mean');
      if (m.has_mean && FIGURES['mean_' + key]) {{
        meanBox.style.display = 'block';
        Plotly.newPlot('active-mean',
          FIGURES['mean_' + key].data,
          FIGURES['mean_' + key].layout,
          {{responsive: true}});
      }} else {{
        meanBox.style.display = 'none';
      }}
    }}

    showExpert('{first_hyp}', 0);
  </script>
</body>
</html>"""

    return (
        html
        .replace("__PLOTLYJS__", pyo.get_plotlyjs(), 1)
        .replace("__SAVEJS__", SAVE_CLIENT_JS, 1)
    )


# ── GPU memory ────────────────────────────────────────────────────────────────


def gpu_mem_mb() -> str:
    """Return a formatted string of current and peak GPU memory usage in MB, or empty string if no CUDA."""
    if not torch.cuda.is_available():
        return ""
    cur = torch.cuda.memory_allocated() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    return f"[GPU {cur:.0f}/{peak:.0f} MB]"


def flush_gpu():
    """Run Python GC and clear the CUDA memory cache and peak memory stats."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ── Model / SAE loading ───────────────────────────────────────────────────────


def load_llm(
    model_name: str,
    device: str,
    dtype: torch.dtype | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load an AutoModelForCausalLM and its tokenizer.

    Sets ``pad_token = eos_token`` when no pad token is defined.
    Padding side is left to the caller to configure.
    """
    if dtype is None:
        dtype = torch.float32
    logger.info(f"Loading {model_name}  {gpu_mem_mb()}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device).eval()
    logger.info(f"Loaded {model_name}  {gpu_mem_mb()}")
    return model, tokenizer


def load_sae(checkpoint_path: str, device: str) -> SMIXAE:
    """Load a SMIXAE checkpoint from disk."""
    logger.info(f"Loading SAE from {checkpoint_path}  {gpu_mem_mb()}")
    sae = SAE.load_from_disk(path=checkpoint_path, device=device)
    logger.info(f"SAE threshold: {sae.threshold}  {gpu_mem_mb()}")
    return sae  # type: ignore


# ── Activation collection ─────────────────────────────────────────────────────


def collect_hook_activations(
    model: PreTrainedModel,
    hook_name: str,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: str | torch.device,
    desc: str = "Collecting activations",
) -> list[torch.Tensor]:
    """Register a forward hook at ``hook_name`` and collect hidden states over batches.

    Args:
        model:     HuggingFace causal LM.
        hook_name: Dotted submodule path, e.g. ``"model.layers.20"``.
        batches:   Iterable of ``(input_ids, attention_mask)`` tensors on any device;
                   moved to ``device`` internally.
        device:    Target device for inference.
        desc:      tqdm description string.

    Returns:
        One ``(batch_size, seq_len, d_model)`` CPU tensor per input batch.
    """
    _store: dict[str, torch.Tensor] = {}

    def _hook(_module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        _store["h"] = h.detach().cpu()

    handle = model.get_submodule(hook_name).register_forward_hook(_hook)
    results: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for ids, mask in tqdm(batches, desc=desc):
                ids = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                model(input_ids=ids, attention_mask=mask)
                results.append(_store["h"])
                del ids, mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        handle.remove()
    return results


# ── SAE encoding ──────────────────────────────────────────────────────────────


def encode_sae_batched(
    sae: SMIXAE,
    hiddens: torch.Tensor,
    batch_size: int = 4096,
    desc: str = "SAE encode",
    return_latent_l0: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, float]:
    """Encode a flat ``(N, d_model)`` tensor through the SAE in batches.

    Args:
        sae:              SMIXAE model (on any device).
        hiddens:          ``(N, d_model)`` CPU tensor of LLM hidden states.
        batch_size:       Number of tokens per SAE forward pass.
        desc:             tqdm description string.
        return_latent_l0: If True, also return the mean number of pre-bottleneck
                          latent dimensions (per token) that are strictly > 0.

    Returns:
        ``(N, n_experts, d_bottleneck)`` CPU float32 tensor, or a tuple of that
        tensor and the mean latent L0 (float) when ``return_latent_l0=True``.
    """
    device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype
    chunks: list[torch.Tensor] = []
    l0_total: float = 0.0
    n_tokens: int = 0
    sae.eval()
    with torch.no_grad():
        for b in tqdm(hiddens.split(batch_size), desc=desc):
            b = b.to(device=device, dtype=sae_dtype)
            if return_latent_l0:
                bottleneck, h_latent = sae.encode_with_latents(b)
                chunks.append(bottleneck.float().cpu())
                l0_total += float((h_latent > 0).sum(dim=-1).float().sum().item())
                n_tokens += b.shape[0]
            else:
                chunks.append(sae.encode(b).float().cpu())
            del b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    result = torch.cat(chunks, dim=0)
    if return_latent_l0:
        return result, l0_total / n_tokens if n_tokens > 0 else 0.0
    return result


# ── Expert ────────────────────────────────────────────────────────────────────


def _strip_prefix(label: str) -> str:
    """Strip a leading ordering prefix of the form '01_' from a display label."""
    return re.sub(r"^\d+_", "", label)


def extract_layer_from_hook(hook_name: str) -> int | None:
    """Extract the first integer layer index from a hook name string.

    Args:
        hook_name: Hook point name such as ``"model.layers.11"`` or ``"blocks.5.hook_resid_post"``.

    Returns:
        The first integer found after a ``.`` separator, or ``None`` if no integer is present.
    """
    m = re.search(r"\.(\d+)(?:\.|$)", hook_name)
    return int(m.group(1)) if m else None


@dataclasses.dataclass
class DatasetConfig:
    """Per-dataset configuration for data loading and visualization.

    Single source of truth for data-source options, pipeline per-dataset overrides,
    and visualization settings.  Used by :func:`collect_activations`,
    :func:`run_pipeline`, and :meth:`Expert.get_plot` / :meth:`Expert.get_mean_plot`.

    Attributes:
        dataframe_path:     Path to a local CSV/Parquet/JSONL file.
        text_column:        Column containing input text.
        label_column:       Column containing class labels, or ``None`` for unlabelled.
        dataset_name:       HuggingFace dataset name for streaming (mutually exclusive
                            with ``dataframe_path``).
        bucket_column:      Continuous column to discretise into Fisher-scoring bins.
        n_buckets:          Number of equal-width bins for ``bucket_column``.
        output_subdir:      Override for the output sub-directory name; defaults to
                            the stem of ``dataframe_path``.
        n_input_samples:    Per-dataset override for the number of texts to sample.
        max_points:         Per-dataset cap on scatter points per expert (0 = no cap).
        regression_hypotheses: List of hypothesis specs for regression probing.
        color_scale:        Named Plotly colorscale (e.g. ``"Plasma"``, ``"Phase"``).
        continuous_color:   Colour points by label ordinal rather than by class.
        color_map:          Explicit ``{label: CSS_color}`` mapping.
        show_labels:        Annotate class-mean markers with text labels.
    """

    # ── Data source ──────────────────────────────────────────────────────
    dataframe_path: str = ""
    text_column: str = "Sentence"
    label_column: str | None = "Label"
    dataset_name: str | None = None
    bucket_column: str | None = None
    n_buckets: int = 10

    # ── Pipeline overrides ───────────────────────────────────────────────
    output_subdir: str | None = None
    n_input_samples: int | None = None
    max_points: int | None = None
    regression_hypotheses: list[dict] | None = None

    # ── Visualization ────────────────────────────────────────────────────
    color_scale: str | None = None
    continuous_color: bool = False
    color_map: dict[str, str] | None = None
    hypothesis_color_overrides: dict[str, dict[str, str]] | None = None
    show_labels: bool = False
    colorbar_title: str = ""
    colorbar_tick_increment: int | None = None

    @property
    def effective_continuous_color(self) -> bool:
        """True when ``color_scale`` is set or ``continuous_color`` is explicitly ``True``."""
        return self.continuous_color or self.color_scale is not None

    @property
    def effective_color_scale(self) -> str:
        """Resolved colorscale name, defaulting to ``"Plasma"``."""
        return self.color_scale if self.color_scale is not None else "Plasma"


@dataclasses.dataclass
class ExpertFilterConfig:
    """Expert-selection thresholds for :func:`get_sae_activations`.

    Attributes:
        active_threshold:     Minimum L2 norm for an expert activation to count as active.
        min_active_fraction:  Drop experts active on fewer than this fraction of tokens.
        max_points:           Randomly downsample experts exceeding this count (0 = no cap).
    """

    active_threshold: float = 1e-5
    min_active_fraction: float = 0.10
    max_points: int = 1000


@dataclasses.dataclass
class ActivationBatch:
    """Outputs of :func:`collect_activations` bundled into a single object.

    Attributes:
        activations:          ``(B, S, d_model)`` CPU tensor of LLM hidden states.
        str_tokens:           Tokenizer string tokens per sequence.
        labels:               ``(B, S)`` long tensor of class ids for display, or ``None``.
        label_names:          ``{id: label_str}`` mapping, or ``None``.
        last_token_positions: ``(B,)`` long tensor — last non-pad token index per sequence.
        n_classes:            Number of unique labels (or Fisher buckets if no display labels).
        regression_targets:   ``(B, n_targets)`` float32 tensor, or ``None``.
        fisher_labels:        ``(B,)`` long tensor of bucket IDs for Fisher scoring, or ``None``.
    """

    activations: "torch.Tensor"
    str_tokens: "list[list[str]]"
    labels: "torch.Tensor | None"
    label_names: "dict[int, str] | None"
    last_token_positions: "torch.Tensor"
    n_classes: int
    regression_targets: "torch.Tensor | None"
    fisher_labels: "torch.Tensor | None"

    @property
    def is_labelled(self) -> bool:
        """True when the batch carries display or Fisher-only labels."""
        return self.labels is not None or self.fisher_labels is not None



def _resolve_colorscale(
    color_map: dict[str, str] | None,
    label_names: dict[int, str],
    int_labels: "np.ndarray",
) -> Any:
    """Convert a ``{display_str: CSS_color}`` map to ``{int_id: color}`` for scatter3d.

    Returns ``None`` when ``color_map`` is ``None`` or when not all active label ids
    have a matching entry (falls back to the caller's colorscale).
    """
    if not color_map:
        return None
    candidate = {
        i: color_map[_strip_prefix(label_names[i])]
        for i in label_names
        if _strip_prefix(label_names[i]) in color_map
    }
    return candidate if set(int_labels.tolist()) <= candidate.keys() else None


def _build_label_mapping(raw_labels: list[str]) -> tuple["torch.Tensor", dict[int, str], int]:
    """Build a label tensor and id→name mapping from raw string labels.

    Labels are sorted numerically when all values parse as numbers, otherwise
    alphabetically.  Returns ``(ids_tensor, {id: label_str}, n_classes)``.
    """
    unique_strs = list({lbl for lbl in raw_labels})
    try:
        unique_sorted = sorted(unique_strs, key=lambda x: float(x))
    except ValueError:
        unique_sorted = sorted(unique_strs)
    label_to_id = {lbl: i for i, lbl in enumerate(unique_sorted)}
    names = {i: lbl for lbl, i in label_to_id.items()}
    ids = torch.tensor([label_to_id[lbl] for lbl in raw_labels], dtype=torch.long)
    return ids, names, len(unique_sorted)


class Expert:
    """Container for one SMIXAE expert's bottleneck activations and analysis results.

    Stores the subset of token positions where the expert was active (norm above
    threshold), together with optional ground-truth labels, sequence positions, and
    LLM activations for continuity scoring.  Provides methods to score the expert
    (:meth:`evaluate_fisher`, :meth:`evaluate_manifold`) and to generate interactive
    Plotly figures (:meth:`get_plot`, :meth:`get_mean_plot`).

    Attributes:
        expert_activations: Active bottleneck activations, shape ``(n_active, d_bottleneck)``.
        llm_activations: LLM residual-stream activations for the same positions,
            shape ``(n_active, d_model)``.  ``None`` if not collected.
        labels: Integer class labels for each active token, or ``None`` if unlabelled.
        seq_positions: Per-sample last-token positions (used for last-token-only mode).
        n_classes: Total number of label classes in the dataset (for coverage scoring).
        expert_id: Global expert index.
        active_indices: List of ``[batch_idx, seq_idx]`` pairs pointing back to the
            original token positions.
        mean_latent_l0: Mean pre-bottleneck L0 sparsity for this expert, or ``None``.
        local_continuity_scores: Per-token continuity scores (set by
            :meth:`evaluate_manifold`), shape ``(n_active,)``, or ``None``.
        fisher_score: Multivariate Fisher discriminant ratio (set by
            :meth:`evaluate_fisher`), or ``None``.
    """

    def __init__(
        self,
        active_mask: torch.Tensor,
        expert_id: int,
        llm_activations: torch.Tensor | None,
        expert_activations: torch.Tensor,
        labels: torch.Tensor | None = None,
        seq_positions: torch.Tensor | None = None,
        n_classes: int = 0,
        mean_latent_l0: float | None = None,
        regression_targets: torch.Tensor | None = None,
        fisher_labels: torch.Tensor | None = None,
    ):
        self.expert_activations = expert_activations[active_mask].float().cpu()
        self.llm_activations = llm_activations[active_mask].float().cpu() if llm_activations is not None else None
        self.labels = labels[active_mask].long().cpu() if labels is not None else None
        # fisher_labels: separate label tensor used only for Fisher scoring (e.g. bucket IDs).
        # When None, Fisher falls back to self.labels.
        _fl_mask = active_mask[:, 0] if active_mask.dim() == 2 else active_mask
        self.fisher_labels = fisher_labels[_fl_mask].long().cpu() if fisher_labels is not None else None
        self.seq_positions = seq_positions
        self.n_classes = n_classes

        self.expert_id = expert_id
        self.active_indices = active_mask.nonzero().cpu().tolist()
        self.mean_latent_l0 = mean_latent_l0

        # Regression targets: (B, S, n_targets) → masked to (n_active, n_targets)
        if regression_targets is not None:
            rt_exp = regression_targets.unsqueeze(1).expand(-1, active_mask.shape[1], -1)
            self.regression_targets: torch.Tensor | None = rt_exp[active_mask].float().cpu()
        else:
            self.regression_targets = None

        # Metrics
        self.local_continuity_scores: torch.Tensor | None = None
        self.fisher_score: float | None = None
        self.regression_scores: dict[str, float] = {}
        self.regression_scores_std: dict[str, float] = {}
        self.best_regression_score: float | None = None
        self.best_regression_score_std: float | None = None
        self.best_regression_name: str | None = None

    # ── accessors ─────────────────────────────────────────────────────
    @property
    def mean_continuity(self) -> float | None:
        """Mean local manifold-continuity score across all active samples, or ``None`` if not evaluated."""
        if self.local_continuity_scores is None:
            return None
        return float(self.local_continuity_scores.mean().item())

    @property
    def _effective_fisher_labels(self) -> torch.Tensor | None:
        """Labels to use for Fisher scoring: fisher_labels if set, else display labels."""
        return self.fisher_labels if self.fisher_labels is not None else self.labels

    @property
    def n_unique_labels(self) -> int | None:
        """Number of distinct label values seen in active samples (uses Fisher labels), or ``None``."""
        lbl = self._effective_fisher_labels
        if lbl is None:
            return None
        return int(lbl.unique().numel())

    @property
    def adjusted_fisher_score(self) -> float | None:
        """Fisher score rescaled by the fraction of dataset classes actually present in active samples."""
        if self.fisher_score is None or self.n_unique_labels is None or self.n_classes == 0:
            return None
        return self.fisher_score * (self.n_unique_labels / self.n_classes)

    def sort_key(self, sort_by: str) -> float:
        """Return the numeric value of the requested metric, or ``-inf`` if not yet computed.

        Args:
            sort_by: One of ``"fisher"``, ``"adjusted_fisher"``, ``"continuity"``,
                or ``"regression"`` (best score across all regression hypotheses).

        Returns:
            The metric value as a float, or ``float("-inf")`` if the metric is ``None``.

        Raises:
            ValueError: If ``sort_by`` is not a recognised metric name.
        """
        mapping: dict[str, float | None] = {
            "fisher": self.fisher_score,
            "adjusted_fisher": self.adjusted_fisher_score,
            "continuity": self.mean_continuity,
            "regression": self.best_regression_score,
        }
        if sort_by not in mapping:
            raise ValueError(f"Unknown sort_by: {sort_by}. Options: {', '.join(mapping.keys())}")
        v = mapping[sort_by]
        return v if v is not None else float("-inf")

    # ── density-aware subsampling ──────────────────────────────────────
    def density_subsample(self, max_points: int, k: int = 12) -> None:
        """Thin activations to *max_points* via k-NN density rejection.

        Keeps points in sparse regions (low local density) with high
        probability and aggressively thins dense clusters, producing a
        spatially stratified sample across the learned manifold.
        """
        X = self.expert_activations
        n = X.shape[0]
        if n <= max_points:
            return

        eps = 1e-12
        D = torch.cdist(X, X)
        r_k = D.topk(k + 1, largest=False).values[:, -1]
        del D

        inv_rho = r_k + eps  # proportional to 1/density
        p = (max_points * inv_rho / inv_rho.sum()).clamp(max=1.0)

        keep_mask = torch.rand(n) < p
        n_kept = int(keep_mask.sum().item())

        if n_kept > max_points:
            kept_idx = keep_mask.nonzero(as_tuple=True)[0]
            trim = kept_idx[torch.randperm(n_kept)[:max_points]]
            keep_mask = torch.zeros(n, dtype=torch.bool)
            keep_mask[trim] = True
        elif n_kept < max_points:
            rejected = (~keep_mask).nonzero(as_tuple=True)[0]
            n_need = max_points - n_kept
            if n_need <= rejected.numel():
                pad = rejected[torch.randperm(rejected.numel())[:n_need]]
            else:
                pad = rejected
            keep_mask[pad] = True

        self._apply_mask(keep_mask)

    def _apply_mask(self, mask: torch.Tensor) -> None:
        """Apply a boolean mask to all stored per-sample tensors."""
        self.expert_activations = self.expert_activations[mask]
        if self.llm_activations is not None:
            self.llm_activations = self.llm_activations[mask]
        if self.labels is not None:
            self.labels = self.labels[mask]
        if self.fisher_labels is not None:
            self.fisher_labels = self.fisher_labels[mask]
        if self.regression_targets is not None:
            self.regression_targets = self.regression_targets[mask]
        kept_indices = [self.active_indices[i] for i in mask.nonzero(as_tuple=True)[0].tolist()]
        self.active_indices = kept_indices
        if self.local_continuity_scores is not None:
            self.local_continuity_scores = self.local_continuity_scores[mask]

    # ── continuity (unlabelled) ───────────────────────────────────────
    def evaluate_manifold(self, k_neighbors: int = 10, device: str = "cuda") -> torch.Tensor:
        """Score expert continuity by comparing bottleneck neighbourhoods to LLM space.

        For each token, finds the ``k_neighbors`` nearest neighbours in bottleneck
        space, then computes the mean cosine similarity between the token's LLM
        activation and those neighbours' LLM activations.  High continuity means that
        tokens close in the bottleneck also tend to be close in the LLM residual stream,
        implying the expert captures a coherent linguistic feature.

        Results are stored in ``self.local_continuity_scores``.

        Args:
            k_neighbors: Number of nearest bottleneck neighbours to consider.
            device: Device to move tensors to for distance computation.

        Returns:
            Continuity scores, shape ``(n_active,)``, on CPU.

        Raises:
            ValueError: If ``self.llm_activations`` is ``None``.
        """
        if self.llm_activations is None:
            raise ValueError("evaluate_manifold requires llm_activations, which was not provided.")
        expert_acts_gpu = self.expert_activations.to(device)
        llm_acts_gpu = self.llm_activations.to(device)

        dists = torch.cdist(expert_acts_gpu, expert_acts_gpu)
        _, indices = torch.topk(dists, k=k_neighbors + 1, largest=False)
        indices = indices[:, 1:]
        del dists

        llm_normed = F.normalize(llm_acts_gpu, p=2, dim=-1)
        llm_neighbors = llm_normed[indices]
        center = llm_normed.unsqueeze(1)
        sims = (center * llm_neighbors).sum(dim=-1)
        self.local_continuity_scores = sims.mean(dim=-1).cpu()

        del expert_acts_gpu, llm_acts_gpu, llm_normed, llm_neighbors, center, sims
        torch.cuda.empty_cache()

        return self.local_continuity_scores

    # ── multivariate fisher score (labelled) ──────────────────────────
    def evaluate_fisher(self) -> float:
        """Multivariate Fisher discriminant ratio: tr(S_W^{-1} S_B).

        Measures how well the 3D bottleneck separates classes.
        Works for clusters, ordered clusters, and rings — anything
        where different labels occupy different regions of the space.
        """
        y = self._effective_fisher_labels
        if y is None:
            self.fisher_score = 0.0
            return 0.0

        X = self.expert_activations  # (N, D)
        _, D = X.shape

        overall_mean = X.mean(dim=0)  # (D,)

        S_W = torch.zeros(D, D)
        S_B = torch.zeros(D, D)

        classes = y.unique()

        for c in classes:
            mask = y == c
            X_c = X[mask]
            n_c = X_c.shape[0]

            if n_c < 2:
                continue

            mean_c = X_c.mean(dim=0)

            # Within-class scatter
            diff_w = X_c - mean_c  # (n_c, D)
            S_W += diff_w.T @ diff_w

            # Between-class scatter
            diff_b = (mean_c - overall_mean).unsqueeze(1)  # (D, 1)
            S_B += n_c * (diff_b @ diff_b.T)

        # Regularise S_W for numerical stability
        S_W += 1e-6 * torch.eye(D)

        try:
            S_W_inv = torch.linalg.inv(S_W)
            self.fisher_score = float(torch.trace(S_W_inv @ S_B).item())
        except torch.linalg.LinAlgError:
            # Fallback: ratio of traces
            tr_w = torch.trace(S_W).item()
            tr_b = torch.trace(S_B).item()
            self.fisher_score = tr_b / tr_w if tr_w > 1e-10 else 0.0

        return self.fisher_score

    # ── regression probing (labelled) ─────────────────────────────────
    def evaluate_regression(self, hypotheses: list[dict]) -> dict[str, float]:
        """Fit sklearn regression probes along each hypothesis and return CV scores.

        Each hypothesis is a dict with keys:
            - ``name`` (str): identifier used as the score key.
            - ``target_indices`` (list[int]): column indices into
              ``self.regression_targets`` forming Y.
            - ``regression_type`` (str): ``"linear"``, ``"logistic"``,
              or ``"multinomial"``.

        Bottleneck activations are scalar-normalised before fitting by dividing
        all samples by the mean L2 norm of the expert (preserves geometry).
        Cross-validation uses 5 folds; multinomial uses :class:`StratifiedKFold`
        with the fold count clamped to the minimum class count.

        Scores stored in ``self.regression_scores``; the hypothesis with the
        highest finite score is stored in ``self.best_regression_name`` /
        ``self.best_regression_score``.

        Args:
            hypotheses: List of hypothesis dicts as described above.

        Returns:
            ``{name: score}`` dict (NaN for any hypothesis that failed).
        """
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
        from sklearn.metrics import make_scorer, r2_score
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if self.regression_targets is None or not hypotheses:
            return {}

        X = self.expert_activations.numpy()  # (n_active, d_bottleneck)
        # No global pre-scaling: StandardScaler inside each Pipeline fold handles
        # normalization on the training split only, preventing any data leakage.
        reg_targets = self.regression_targets.numpy()

        def _classification_cv(y_int: "np.ndarray") -> "StratifiedKFold | None":
            """Return a StratifiedKFold with splits clamped to the minimum class count."""
            classes, counts = np.unique(y_int, return_counts=True)
            if len(classes) < 2:
                return None
            n_splits = max(2, min(5, int(counts.min())))
            return StratifiedKFold(n_splits=n_splits)

        def _fit_one(hyp: dict) -> tuple[str, float, float]:
            name = hyp["name"]
            idxs: list[int] = hyp["target_indices"]
            reg_type: str = hyp["regression_type"]

            Y = reg_targets[:, idxs]
            if Y.shape[1] == 1:
                Y = Y.ravel()

            try:
                if reg_type in ("linear", "ridge"):
                    estimator = Ridge() if reg_type == "ridge" else LinearRegression()
                    pipe = Pipeline([("sc", StandardScaler()), ("reg", estimator)])
                    scorer = make_scorer(r2_score, multioutput="uniform_average")
                    cv_s = cross_val_score(pipe, X, Y, cv=KFold(n_splits=5), scoring=scorer)
                    return name, float(np.mean(cv_s)), float(np.std(cv_s))

                elif reg_type == "logistic":
                    y_int = Y.astype(int)
                    cv = _classification_cv(y_int)
                    if cv is None:
                        return name, float("nan"), float("nan")
                    pipe = Pipeline([
                        ("sc", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                    ])
                    cv_s = cross_val_score(pipe, X, y_int, cv=cv, scoring="balanced_accuracy")
                    return name, float(np.mean(cv_s)), float(np.std(cv_s))

                elif reg_type == "multinomial":
                    y_int = Y.astype(int)
                    cv = _classification_cv(y_int)
                    if cv is None:
                        return name, float("nan"), float("nan")
                    # solver='lbfgs' handles multi-class natively; multi_class param
                    # was removed in sklearn 1.7.
                    pipe = Pipeline([
                        ("sc", StandardScaler()),
                        ("clf", LogisticRegression(solver="lbfgs", max_iter=1000)),
                    ])
                    cv_s = cross_val_score(pipe, X, y_int, cv=cv, scoring="f1_macro")
                    return name, float(np.mean(cv_s)), float(np.std(cv_s))

                else:
                    return name, float("nan"), float("nan")

            except Exception:
                return name, float("nan"), float("nan")

        # Run hypotheses sequentially; sklearn's cross_val_score already
        # parallelises folds internally, so no additional nesting is needed.
        results: list[tuple[str, float, float]] = [_fit_one(hyp) for hyp in hypotheses]

        scores:     dict[str, float] = {name: mean for name, mean, _   in results}
        scores_std: dict[str, float] = {name: std  for name, _,    std in results}
        self.regression_scores     = scores
        self.regression_scores_std = scores_std
        valid = {k: v for k, v in scores.items() if np.isfinite(v)}
        if valid:
            self.best_regression_name      = max(valid, key=valid.__getitem__)
            self.best_regression_score     = valid[self.best_regression_name]
            self.best_regression_score_std = scores_std.get(self.best_regression_name)
        return scores

    # ── context windows ───────────────────────────────────────────────
    def get_context_windows(self, str_tokens: list[list[str]], context_window: int = 10) -> list[str]:
        """Build HTML-formatted context window strings for each active token.

        For each active token, extracts a surrounding window of ``context_window``
        tokens on each side, bolds the target token with ``<b>[token]</b>``, and
        replaces newlines with ``<br>`` for HTML display.

        Args:
            str_tokens: Nested list of string tokens, indexed by
                ``[batch_idx][seq_idx]``, as returned by the tokenizer.
            context_window: Number of tokens to include on each side of the target.

        Returns:
            List of HTML strings, one per active token, in the same order as
            ``self.active_indices``.
        """
        contexts = []
        for batch_idx, seq_idx in self.active_indices:
            seq = str_tokens[batch_idx]
            if self.seq_positions is not None:
                seq_idx = int(self.seq_positions[batch_idx].item())
            start = max(0, seq_idx - context_window)
            end = min(len(seq), seq_idx + context_window + 1)
            window = list(seq[start:end])
            target_rel_idx = seq_idx - start
            window[target_rel_idx] = f"<b>[{window[target_rel_idx]}]</b>"
            contexts.append("".join(window).replace("\n", "<br>"))
        return contexts

    # ── plotting ──────────────────────────────────────────────────────
    def _make_title(self) -> str:
        """Build a plot title string summarising this expert's key metrics."""
        parts = [f"Expert {self.expert_id}  (n={self.expert_activations.shape[0]}"]
        if self.mean_latent_l0 is not None:
            parts.append(f"L0={self.mean_latent_l0:.1f}")
        if self.n_unique_labels is not None:
            parts.append(f"labels={self.n_unique_labels}")
        if self.fisher_score is not None:
            parts.append(f"fisher={self.fisher_score:.3f}")
        if self.adjusted_fisher_score is not None:
            parts.append(f"adj_fisher={self.adjusted_fisher_score:.3f}")
        if self.mean_continuity is not None:
            parts.append(f"cont={self.mean_continuity:.3f}")
        if self.best_regression_name is not None and self.best_regression_score is not None:
            parts.append(f"best_reg={self.best_regression_name}({self.best_regression_score:.3f})")
        return ", ".join(parts) + ")"

    def get_plot(
        self,
        str_tokens: list[list[str]],
        cfg: DatasetConfig = DatasetConfig(),
        label_names: "dict[int, str] | None" = None,
        *,
        hypothesis_name: str | None = None,
        k_neighbors: int = 10,
        context_window: int = 10,
        device: str = "cuda",
        scatter_size: float = 1,
        unlabeled_color: str = "continuity",
    ) -> Figure:
        """Generate an interactive 3D scatter of this expert's bottleneck activations.

        Lazily evaluates manifold continuity if it hasn't been computed yet.
        For labelled data, colours points by class (using ``cfg.color_map`` when
        provided, else ``cfg.color_scale``). ``cfg.continuous_color`` selects
        between a discrete swatch legend and a Plotly colorbar widget.
        For unlabelled data, colours points by continuity score or Euclidean distance
        from the origin, controlled by ``unlabeled_color``.

        Args:
            str_tokens: Nested list of string tokens for building hover context windows.
            cfg: Dataset visualization settings (colorscale, color map, labels flag).
            label_names: Map from integer label id to display string.
            hypothesis_name: Optional hypothesis key used to look up a
                per-hypothesis override in ``cfg.hypothesis_color_overrides``.
            k_neighbors: Neighbourhood size for lazy continuity evaluation.
            context_window: Tokens on each side of the target in hover text.
            device: Device for continuity computation.
            scatter_size: Marker size for scatter points (default 1 for labeled, 5 for unlabeled).
            unlabeled_color: ``"continuity"`` (default) or ``"distance"`` (Euclidean from origin).

        Returns:
            A Plotly :class:`Figure` with a single 3D scatter trace.
        """
        # Lazy-evaluate continuity
        if self.local_continuity_scores is None:
            self.evaluate_manifold(k_neighbors=k_neighbors, device=device)

        contexts = self.get_context_windows(str_tokens, context_window=context_window)
        pts = self.expert_activations.numpy()

        if self.labels is not None and label_names is not None:
            int_labels = self.labels.numpy()
            lnames = {k: _strip_prefix(v) for k, v in label_names.items()}
            effective_color_map = (
                cfg.hypothesis_color_overrides.get(hypothesis_name)
                if hypothesis_name and cfg.hypothesis_color_overrides
                else cfg.color_map
            )
            cscale = _resolve_colorscale(effective_color_map, label_names, int_labels)
            fig = plot_3d_scatter(
                pts, int_labels,
                label_names=lnames,
                colorscale=cscale if cscale is not None else cfg.color_scale,
                continuous_color=cfg.continuous_color,
                connect_means=False,
                show_labels=cfg.show_labels,
                title=self._make_title(),
            )
            # Inject per-token context windows into the scatter trace hover
            fig.data[0].update(hovertext=contexts, hoverinfo="text")
        else:
            # Unlabeled: color by continuity score or Euclidean distance from origin
            if unlabeled_color == "distance":
                color_arr = np.linalg.norm(pts, axis=1).astype(np.float32)
                colorbar_label = "Distance from origin"
            else:
                color_arr = (
                    self.local_continuity_scores.numpy()
                    if self.local_continuity_scores is not None
                    else np.zeros(pts.shape[0], dtype=np.float32)
                )
                colorbar_label = "Continuity"
            fig = plot_3d_scatter(
                pts, color_arr,
                colorscale="Viridis",
                colorbar_title=colorbar_label,
                scatter_alpha=1.0,
                scatter_size=scatter_size,
                title=self._make_title(),
            )
            fig.data[0].update(hovertext=contexts, hoverinfo="text")

        return fig

    def get_mean_plot(
        self,
        cfg: DatasetConfig = DatasetConfig(),
        label_names: "dict[int, str] | None" = None,
        hypothesis_name: str | None = None,
    ) -> "Figure | None":
        """Generate a 3D scatter showing only per-class mean bottleneck activations.

        Uses the same colour/scale options as :meth:`get_plot` via ``cfg``, but
        ``scatter_alpha=0.0`` so individual token points are hidden.  Returns ``None``
        when labels or label names are unavailable.

        Args:
            cfg: Dataset visualization settings.
            label_names: Map from integer label id to display string.
            hypothesis_name: Optional hypothesis key used to look up a
                per-hypothesis override in ``cfg.hypothesis_color_overrides``.

        Returns:
            A Plotly :class:`Figure` or ``None`` if unlabelled.
        """
        if self.labels is None or label_names is None:
            return None

        pts = self.expert_activations.numpy()
        int_labels = self.labels.numpy()
        lnames = {k: _strip_prefix(v) for k, v in label_names.items()}
        effective_color_map = (
            cfg.hypothesis_color_overrides.get(hypothesis_name)
            if hypothesis_name and cfg.hypothesis_color_overrides
            else cfg.color_map
        )
        cscale = _resolve_colorscale(effective_color_map, label_names, int_labels)
        return plot_3d_scatter(
            pts, int_labels,
            label_names=lnames,
            colorscale=cscale if cscale is not None else cfg.color_scale,
            continuous_color=cfg.continuous_color,
            scatter_alpha=0.0,
            connect_means=False,
            show_labels=cfg.show_labels,
            title=self._make_title() + " [class means]",
        )


# ── Shared activation pipeline ────────────────────────────────────────────────


def _collect_target_cols(cfg: DatasetConfig) -> list[str]:
    """Return a deduplicated ordered list of regression target column names from ``cfg``."""
    if not cfg.regression_hypotheses:
        return []
    seen: set[str] = set()
    cols: list[str] = []
    for hyp in cfg.regression_hypotheses:
        for col in hyp.get("target_columns", []):
            if col not in seen:
                cols.append(col)
                seen.add(col)
    return cols


def _load_texts_and_labels(
    cfg: DatasetConfig,
    n_input_samples: int,
) -> tuple[list[str], list[str] | None, list[int] | None, list[list[float]] | None]:
    """Load raw texts and labels from a local file or a streaming HuggingFace dataset.

    Returns:
        texts:                  List of input strings.
        raw_labels:             String labels for display, or ``None`` if no label column.
        raw_fisher_labels:      Integer bucket IDs for Fisher scoring, or ``None``.
        raw_regression_targets: Nested list of float regression targets, or ``None``.
    """
    texts: list[str] = []
    raw_labels: list[str] | None = [] if cfg.label_column is not None else None
    raw_fisher_labels: list[int] | None = None
    raw_regression_targets: list[list[float]] | None = None

    target_cols = _collect_target_cols(cfg)

    if cfg.dataframe_path:
        print(f"Loading data from {cfg.dataframe_path}")
        ext = Path(cfg.dataframe_path).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(cfg.dataframe_path)
        elif ext in (".parquet", ".pq"):
            df = pd.read_parquet(cfg.dataframe_path)
        elif ext in (".json", ".jsonl"):
            df = pd.read_json(cfg.dataframe_path, lines=(ext == ".jsonl"))
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        texts = df[cfg.text_column].tolist()[:n_input_samples]
        if cfg.label_column is not None:
            raw_labels = [str(v) for v in df[cfg.label_column].tolist()[:n_input_samples]]
        if cfg.bucket_column is not None:
            bucket_series = pd.cut(df[cfg.bucket_column].iloc[:n_input_samples], bins=cfg.n_buckets, labels=False)
            raw_fisher_labels = [int(b) if not pd.isna(b) else 0 for b in bucket_series]
            print(f"Bucketed '{cfg.bucket_column}' into {cfg.n_buckets} bins for Fisher scoring only.")
        if target_cols:
            reg_df = df[target_cols].iloc[:n_input_samples].fillna(0.0)
            raw_regression_targets = reg_df.values.tolist()
    elif cfg.dataset_name is not None:
        print(f"Streaming {cfg.dataset_name}")
        dataset = load_dataset(cfg.dataset_name, streaming=True, split="train")
        for i, sample in enumerate(dataset):
            if i == n_input_samples:
                break
            texts.append(sample[cfg.text_column])
            if raw_labels is not None:
                raw_labels.append(str(sample[cfg.label_column]))  # type: ignore[index]
    else:
        raise ValueError("DatasetConfig must set either dataframe_path or dataset_name.")

    return texts, raw_labels, raw_fisher_labels, raw_regression_targets


def collect_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    hook_name: str,
    max_length: int,
    n_input_samples: int,
    device: str,
    llm_batch_size: int,
    cfg: DatasetConfig = DatasetConfig(),
) -> ActivationBatch:
    """Load texts + labels, tokenize, and collect LLM residual-stream activations.

    Data source, label, and visualization options are bundled in ``cfg`` (a
    :class:`DatasetConfig`).  When ``cfg.bucket_column`` is provided, that continuous
    column is discretised into ``cfg.n_buckets`` equal-width bins used **only** for
    Fisher scoring.

    Returns an :class:`ActivationBatch` with all tensors on CPU.
    """
    texts, raw_labels, raw_fisher_labels, raw_regression_targets = _load_texts_and_labels(cfg, n_input_samples)

    labels_tensor: torch.Tensor | None = None
    label_names: dict[int, str] | None = None
    n_classes: int = 0
    fisher_labels_tensor: torch.Tensor | None = None

    if raw_labels is not None and len(raw_labels) > 0:
        label_ids, label_names, n_classes = _build_label_mapping(raw_labels)
        unique_display = [label_names[i] for i in range(n_classes)]
        print(
            f"Found {n_classes} unique labels (numerically sorted): "
            f"{', '.join(unique_display[:10])}{'…' if n_classes > 10 else ''}"
        )
    else:
        label_ids = None

    if raw_fisher_labels is not None:
        # Fisher labels are already integer bucket IDs; keep n_classes for Fisher as n_buckets.
        fisher_labels_tensor = torch.tensor(raw_fisher_labels, dtype=torch.long)
        if label_ids is None:
            # No display labels — Fisher n_classes = number of non-empty buckets
            n_classes = len(set(raw_fisher_labels))
            print(f"Fisher-only mode: {n_classes} non-empty buckets (no display labels)")
        print(f"Fisher bucket labels: min={min(raw_fisher_labels)} max={max(raw_fisher_labels)}")

    print("Collecting activations...")
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    tokenized = enc["input_ids"]  # (B, S)
    attention_mask = enc["attention_mask"]  # (B, S)
    B, S = tokenized.shape

    if label_ids is not None:
        labels_tensor = label_ids[:B].unsqueeze(1).expand(B, S).clone()

    pad_token_id = tokenizer.pad_token_id
    non_pad_mask = tokenized != pad_token_id
    col_indices = torch.arange(S).unsqueeze(0).expand(B, S)
    last_token_positions = (col_indices.masked_fill(~non_pad_mask, -1).max(dim=1).values).clamp(min=0)

    print(
        f"Last non-pad positions — min: {last_token_positions.min().item()}, "
        f"max: {last_token_positions.max().item()}, "
        f"mean: {last_token_positions.float().mean().item():.1f} "
        f"(seq length {S})"
    )

    batches = (
        (tokenized[i : i + llm_batch_size], attention_mask[i : i + llm_batch_size]) for i in range(0, B, llm_batch_size)
    )
    all_acts = collect_hook_activations(model, hook_name, batches, device)
    activations = torch.cat(all_acts, dim=0)
    del all_acts
    str_tokens: list[list[str]] = [list(tokenizer.convert_ids_to_tokens(tokenized[i].tolist()) or []) for i in range(B)]

    regression_targets_tensor: torch.Tensor | None = None
    if raw_regression_targets is not None:
        regression_targets_tensor = torch.tensor(raw_regression_targets[:B], dtype=torch.float32)

    return ActivationBatch(
        activations=activations,
        str_tokens=str_tokens,
        labels=labels_tensor,
        label_names=label_names,
        last_token_positions=last_token_positions,
        n_classes=n_classes,
        regression_targets=regression_targets_tensor,
        fisher_labels=fisher_labels_tensor,
    )


def get_sae_activations(
    sae: SMIXAE,
    device: str,
    batch: ActivationBatch,
    sae_batch_size: int,
    filter_cfg: ExpertFilterConfig = ExpertFilterConfig(),
) -> list[Expert]:
    """Encode LLM activations through SMIXAE and return a list of active ``Expert`` objects.

    Args:
        sae:         Trained SMIXAE model.
        device:      Device for SAE inference.
        batch:       :class:`ActivationBatch` from :func:`collect_activations`.
        sae_batch_size: Tokens per SAE forward pass.
        filter_cfg:  Thresholds controlling which experts are retained.

    Returns:
        List of ``Expert`` objects for all experts that meet the activity threshold.
    """
    activations = batch.activations
    labels = batch.labels
    last_token_positions = batch.last_token_positions
    last_token_only = batch.is_labelled
    B, S_full, D = activations.shape

    seq_positions: torch.Tensor | None = None
    if last_token_only:
        seq_positions = last_token_positions
        gather_idx = last_token_positions.unsqueeze(1).unsqueeze(2).expand(B, 1, D)
        activations = activations.gather(1, gather_idx)
        if labels is not None:
            label_idx = last_token_positions.unsqueeze(1)
            labels = labels.gather(1, label_idx)
        S = 1
        print(
            f"last_token_only=True → gathered last non-pad token per sequence ({B * S_full} → {B} tokens through SAE)"
        )
    else:
        S = S_full

    # When last_token_only=True, S=1 so B*S = B (number of samples); otherwise all tokens.
    n_total = B * S
    min_points_abs = max(1, int(filter_cfg.min_active_fraction * n_total))
    activations_flat = activations.reshape(n_total, D)

    sae_activations_cat, mean_latent_l0 = encode_sae_batched(sae, activations_flat, sae_batch_size, return_latent_l0=True)  # type: ignore[misc]
    sae_activations_cat = sae_activations_cat.view(B, S, sae_activations_cat.shape[1], sae_activations_cat.shape[2])
    print(f"Mean latent L0 (pre-bottleneck dims > 0 per token): {mean_latent_l0:.2f}")
    print(f"min_active_fraction={filter_cfg.min_active_fraction:.2%} → min_points={min_points_abs} / {n_total} tokens")

    experts: list[Expert] = []
    n_experts = sae_activations_cat.shape[-2]

    for i in tqdm(range(n_experts), desc="Building experts"):
        expert_pts = sae_activations_cat[..., i, :]
        active_mask = torch.norm(expert_pts, p=2, dim=-1) > filter_cfg.active_threshold
        n_active = int(active_mask.sum().item())

        if n_active < min_points_abs:
            continue
        if filter_cfg.max_points and n_active > filter_cfg.max_points:
            active_indices = active_mask.nonzero(as_tuple=False)
            perm = torch.randperm(n_active)[:filter_cfg.max_points]
            chosen = active_indices[perm]
            sampled_mask = torch.zeros_like(active_mask, dtype=torch.bool)
            sampled_mask[chosen[:, 0], chosen[:, 1]] = True
            active_mask = sampled_mask

        expert = Expert(
            active_mask,
            i,
            activations,
            expert_pts,
            labels=labels,
            regression_targets=batch.regression_targets,
            seq_positions=seq_positions,
            n_classes=batch.n_classes,
            mean_latent_l0=mean_latent_l0,
            fisher_labels=batch.fisher_labels,
        )
        experts.append(expert)

    del sae_activations_cat
    return experts


def update_results_json(
    path: str | Path,
    run_name: str,
    model_name: str,
    hook_name: str,
    section: str,
    key: str,
    data: dict,
) -> None:
    """Read-modify-write the combined results JSON.

    Structure::

        {
          "<run_name>": {
            "model_name": "...",
            "hook_name": "...",
            "<section>": {
              "<key>": { ...data... }
            }
          }
        }

    Missing keys at any level are created; existing ones are overwritten.
    The file is created if it does not exist.
    """
    import json as _json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            try:
                existing = _json.load(f)
            except _json.JSONDecodeError:
                existing = {}

    run_entry = existing.setdefault(run_name, {})
    run_entry["model_name"] = model_name
    run_entry["hook_name"] = hook_name
    run_entry.setdefault(section, {})[key] = data

    with open(path, "w", encoding="utf-8") as f:
        _json.dump(existing, f, indent=2)
