"""Shared utilities for SMIXAE analysis scripts."""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
import plotly.io as pio
import plotly.offline as pyo
from plotly.graph_objects import Figure
from sae_lens import SAE
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from smixae import SMIXAE
from analysis.scatter3d import plot_3d_scatter

# ── Tabbed HTML builder ───────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script>__PLOTLYJS__</script>
  <style>
    body {{ font-family: sans-serif; margin: 8px; }}
    .tab-strip {{ display:flex; flex-wrap:wrap; gap:4px; margin-bottom:8px; }}
    .tab-btn {{ padding:4px 10px; cursor:pointer; border:1px solid #aaa;
                border-radius:3px; background:#f0f0f0; font-size:13px; }}
    .tab-btn.active {{ background:#333; color:#fff; }}
    .tab-pane {{ display:none; flex-direction:column; gap:8px; }}
    .tab-pane.active {{ display:flex; }}
    .plot-box {{ width:100%; height:650px; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <div class="tab-strip">{tab_buttons}</div>
  {tab_panes}
  <script>
    const FIGURES = {{{figures_json}}};
    const rendered = new Set();
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
    expert_entries: list[tuple[str, Figure, Figure | None]],
    dataset_title: str,
) -> str:
    tab_buttons: list[str] = []
    tab_panes: list[str] = []
    figures_json_parts: list[str] = []

    for idx, (tab_label, scatter_fig, mean_fig) in enumerate(expert_entries):
        scatter_id = f"scatter_{idx}"
        active_cls = " active" if idx == 0 else ""

        tab_buttons.append(f'<button class="tab-btn{active_cls}" onclick="switchTab({idx})">{tab_label}</button>')

        plot_divs = f'<div class="plot-box" id="{scatter_id}"></div>'
        if mean_fig is not None:
            mean_id = f"mean_{idx}"
            plot_divs += f'\n    <div class="plot-box" id="{mean_id}"></div>'
            figures_json_parts.append(f'"{mean_id}": {pio.to_json(mean_fig, engine="json")}')

        tab_panes.append(f'<div class="tab-pane{active_cls}">\n    {plot_divs}\n  </div>')
        figures_json_parts.append(f'"{scatter_id}": {pio.to_json(scatter_fig, engine="json")}')

    html = _HTML_TEMPLATE.format(
        title=dataset_title,
        tab_buttons="\n    ".join(tab_buttons),
        tab_panes="\n  ".join(tab_panes),
        figures_json=",\n    ".join(figures_json_parts),
    )
    return html.replace("__PLOTLYJS__", pyo.get_plotlyjs(), 1)


# ── GPU memory ────────────────────────────────────────────────────────────────


def gpu_mem_mb() -> str:
    if not torch.cuda.is_available():
        return ""
    cur = torch.cuda.memory_allocated() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    return f"[GPU {cur:.0f}/{peak:.0f} MB]"


def flush_gpu():
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


class Expert:
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
    ):
        self.expert_activations = expert_activations[active_mask].float().cpu()
        self.llm_activations = llm_activations[active_mask].float().cpu() if llm_activations is not None else None
        self.labels = labels[active_mask].long().cpu() if labels is not None else None
        self.seq_positions = seq_positions
        self.n_classes = n_classes

        self.expert_id = expert_id
        self.active_indices = active_mask.nonzero().cpu().tolist()
        self.mean_latent_l0 = mean_latent_l0

        # Metrics
        self.local_continuity_scores: torch.Tensor | None = None
        self.fisher_score: float | None = None

    # ── accessors ─────────────────────────────────────────────────────
    @property
    def mean_continuity(self) -> float | None:
        if self.local_continuity_scores is None:
            return None
        return float(self.local_continuity_scores.mean().item())

    @property
    def n_unique_labels(self) -> int | None:
        if self.labels is None:
            return None
        return int(self.labels.unique().numel())

    @property
    def adjusted_fisher_score(self) -> float | None:
        if self.fisher_score is None or self.n_unique_labels is None or self.n_classes == 0:
            return None
        return self.fisher_score * (self.n_unique_labels / self.n_classes)

    def sort_key(self, sort_by: str) -> float:
        mapping: dict[str, float | None] = {
            "fisher": self.fisher_score,
            "adjusted_fisher": self.adjusted_fisher_score,
            "continuity": self.mean_continuity,
        }
        if sort_by not in mapping:
            raise ValueError(f"Unknown sort_by: {sort_by}. Options: {', '.join(mapping.keys())}")
        v = mapping[sort_by]
        return v if v is not None else float("-inf")

    # ── continuity (unlabelled) ───────────────────────────────────────
    def evaluate_manifold(self, k_neighbors: int = 10, device: str = "cuda") -> torch.Tensor:
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
        """
        Multivariate Fisher discriminant ratio: tr(S_W^{-1} S_B)

        Measures how well the 3D bottleneck separates classes.
        Works for clusters, ordered clusters, and rings — anything
        where different labels occupy different regions of the space.
        """
        if self.labels is None:
            self.fisher_score = 0.0
            return 0.0

        X = self.expert_activations  # (N, D)
        y = self.labels  # (N,)
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

    # ── context windows ───────────────────────────────────────────────
    def get_context_windows(self, str_tokens: list[list[str]], context_window: int = 10) -> list[str]:
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
        return ", ".join(parts) + ")"

    def get_plot(
        self,
        str_tokens: list[list[str]],
        k_neighbors: int = 10,
        context_window: int = 10,
        device: str = "cuda",
        label_names: dict[int, str] | None = None,
        continuous_color: bool = False,
        color_scale: str = "Plasma",
        color_map: dict[str, str] | None = None,
        show_labels: bool | None = None,
        show_colorbar: bool | None = None,
        connect_means: bool | None = None,
    ) -> Figure:
        # Lazy-evaluate continuity
        if self.local_continuity_scores is None:
            self.evaluate_manifold(k_neighbors=k_neighbors, device=device)

        contexts = self.get_context_windows(str_tokens, context_window=context_window)
        pts = self.expert_activations.numpy()

        if self.labels is not None and label_names is not None:
            int_labels = self.labels.numpy()
            lnames = {k: _strip_prefix(v) for k, v in label_names.items()}

            # Convert color_map {display_str: color} → {int_id: color} for scatter3d
            cscale: Any = None
            if color_map:
                candidate = {
                    i: color_map[_strip_prefix(label_names[i])]
                    for i in label_names
                    if _strip_prefix(label_names[i]) in color_map
                }
                if len(candidate) == len(set(int_labels.tolist())):
                    cscale = candidate

            _show_labels = show_labels if show_labels is not None else (not continuous_color)
            _show_colorbar = show_colorbar if show_colorbar is not None else continuous_color
            _connect_means = connect_means if connect_means is not None else continuous_color
            fig = plot_3d_scatter(
                pts, int_labels,
                label_names=lnames,
                colorscale=color_scale if continuous_color else cscale,
                show_labels=_show_labels,
                show_colorbar=_show_colorbar,
                connect_means=_connect_means,
                title=self._make_title(),
            )
            # Inject per-token context windows into the scatter trace hover
            fig.data[0].update(hovertext=contexts, hoverinfo="text")
        else:
            # Unlabeled: color by per-point continuity score via float-label path
            cont_arr = (
                self.local_continuity_scores.numpy()
                if self.local_continuity_scores is not None
                else np.zeros(pts.shape[0], dtype=np.float32)
            )
            fig = plot_3d_scatter(
                pts, cont_arr.astype(np.float32),
                colorscale="Viridis",
                colorbar_title="Continuity",
                scatter_alpha=0.8,
                title=self._make_title(),
            )
            fig.data[0].update(hovertext=contexts, hoverinfo="text")

        return fig

    def get_mean_plot(
        self,
        label_names: dict[int, str] | None = None,
        color_scale: str = "Plasma",
        continuous_color: bool = False,
        color_map: dict[str, str] | None = None,
        show_labels: bool | None = None,
        show_colorbar: bool | None = None,
        connect_means: bool | None = None,
    ) -> Figure | None:
        if self.labels is None or label_names is None:
            return None

        pts = self.expert_activations.numpy()
        int_labels = self.labels.numpy()
        lnames = {k: _strip_prefix(v) for k, v in label_names.items()}

        # Convert color_map {display_str: color} → {int_id: color} for scatter3d
        cscale: Any = None
        if color_map:
            candidate = {
                i: color_map[_strip_prefix(label_names[i])]
                for i in label_names
                if _strip_prefix(label_names[i]) in color_map
            }
            if len(candidate) == len(set(int_labels.tolist())):
                cscale = candidate

        _show_labels = show_labels if show_labels is not None else (not continuous_color)
        _show_colorbar = show_colorbar if show_colorbar is not None else continuous_color
        _connect_means = connect_means if connect_means is not None else continuous_color
        return plot_3d_scatter(
            pts, int_labels,
            label_names=lnames,
            colorscale=color_scale if continuous_color else cscale,
            show_labels=_show_labels,
            show_colorbar=_show_colorbar,
            scatter_alpha=0.0,
            connect_means=_connect_means,
            title=self._make_title() + " [class means]",
        )


# ── Shared activation pipeline ────────────────────────────────────────────────


def collect_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    hook_name: str,
    max_length: int,
    n_input_samples: int,
    device: str,
    llm_batch_size: int,
    dataset_name: str | None = None,
    dataframe_path: str | None = None,
    text_column: str = "text",
    label_column: str | None = None,
) -> tuple[
    torch.Tensor,
    list[list[str]],
    torch.Tensor | None,
    dict[int, str] | None,
    torch.Tensor,
    int,
]:
    """Load texts + labels, tokenize, collect LLM residual-stream activations.

    Returns:
        activations:          ``(B, S, d_model)`` CPU tensor
        str_tokens:           list of token-string lists, one per sequence
        labels_tensor:        ``(B, S)`` long tensor of class ids, or ``None``
        label_names:          ``dict[int, str]`` id→label mapping, or ``None``
        last_token_positions: ``(B,)`` long tensor — last non-pad position per sequence
        n_classes:            number of unique labels (0 if unlabelled)
    """
    texts: list[str] = []
    raw_labels: list[str | int] | None = [] if label_column else None

    if dataframe_path is not None:
        print(f"Loading data from {dataframe_path}")
        ext = Path(dataframe_path).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(dataframe_path)
        elif ext in (".parquet", ".pq"):
            df = pd.read_parquet(dataframe_path)
        elif ext in (".json", ".jsonl"):
            df = pd.read_json(dataframe_path, lines=(ext == ".jsonl"))
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        texts = df[text_column].tolist()[:n_input_samples]
        if raw_labels is not None:
            raw_labels = df[label_column].tolist()[:n_input_samples]  # type: ignore[index]
    elif dataset_name is not None:
        print(f"Streaming {dataset_name}")
        dataset = load_dataset(dataset_name, streaming=True, split="train")
        for i, sample in enumerate(dataset):
            if i == n_input_samples:
                break
            texts.append(sample[text_column])
            if raw_labels is not None:
                raw_labels.append(sample[label_column])  # type: ignore[index]
    else:
        raise ValueError("Provide either dataset_name or dataframe_path.")

    labels_tensor: torch.Tensor | None = None
    label_names: dict[int, str] | None = None
    n_classes: int = 0

    if raw_labels is not None and len(raw_labels) > 0:
        unique = sorted({str(lbl) for lbl in raw_labels})
        n_classes = len(unique)
        label_to_id = {lbl: i for i, lbl in enumerate(unique)}
        label_names = {i: lbl for lbl, i in label_to_id.items()}
        label_ids = torch.tensor([label_to_id[str(lbl)] for lbl in raw_labels], dtype=torch.long)
        print(
            f"Found {n_classes} unique labels (sorted → ordinal ids): "
            f"{', '.join(unique[:10])}{'…' if n_classes > 10 else ''}"
        )
    else:
        label_ids = None

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

    return (
        activations,
        str_tokens,
        labels_tensor,
        label_names,
        last_token_positions,
        n_classes,
    )


def get_sae_activations(
    sae: SMIXAE,
    device: str,
    activations: torch.Tensor,
    sae_batch_size: int,
    active_threshold: float = 1e-5,
    min_points: int = 100,
    max_points: int = 1000,
    labels: torch.Tensor | None = None,
    last_token_only: bool = False,
    last_token_positions: torch.Tensor | None = None,
    n_classes: int = 0,
) -> list[Expert]:
    """Encode LLM activations through SMIXAE and return a list of active ``Expert`` objects.

    Args:
        sae:                  Trained SMIXAE model.
        device:               Device for SAE inference.
        activations:          ``(B, S, d_model)`` CPU tensor from ``collect_activations``.
        sae_batch_size:       Tokens per SAE forward pass.
        active_threshold:     L2 norm threshold to consider an expert active.
        min_points:           Skip experts with fewer active tokens than this.
        max_points:           Randomly downsample experts exceeding this count (0 = no cap).
        labels:               ``(B, S)`` long tensor of class ids, or ``None``.
        last_token_only:      If True, only encode the last non-pad token per sequence.
        last_token_positions: Required when ``last_token_only=True``.
        n_classes:            Total number of label classes (passed through to ``Expert``).

    Returns:
        List of ``Expert`` objects for all experts that meet the activity threshold.
    """
    B, S_full, D = activations.shape

    seq_positions: torch.Tensor | None = None
    if last_token_only:
        if last_token_positions is None:
            raise ValueError("last_token_only=True but no last_token_positions provided.")
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

    activations_flat = activations.reshape(B * S, D)

    sae_activations_cat, mean_latent_l0 = encode_sae_batched(sae, activations_flat, sae_batch_size, return_latent_l0=True)  # type: ignore[misc]
    sae_activations_cat = sae_activations_cat.view(B, S, sae_activations_cat.shape[1], sae_activations_cat.shape[2])
    print(f"Mean latent L0 (pre-bottleneck dims > 0 per token): {mean_latent_l0:.2f}")

    experts: list[Expert] = []
    n_experts = sae_activations_cat.shape[-2]

    for i in tqdm(range(n_experts), desc="Building experts"):
        expert_pts = sae_activations_cat[..., i, :]
        active_mask = torch.norm(expert_pts, p=2, dim=-1) > active_threshold
        n_active = int(active_mask.sum().item())

        if n_active < min_points:
            continue
        if max_points and n_active > max_points:
            active_indices = active_mask.nonzero(as_tuple=False)
            perm = torch.randperm(n_active)[:max_points]
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
            seq_positions=seq_positions,
            n_classes=n_classes,
            mean_latent_l0=mean_latent_l0,
        )
        experts.append(expert)

    del sae_activations_cat
    return experts
