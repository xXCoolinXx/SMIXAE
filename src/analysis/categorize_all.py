import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import plotly.io as pio
import plotly.offline as pyo
import torch
import typer
from datasets import load_dataset
from plotly.graph_objects import Figure
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from smixae import SMIXAE
from analysis.utils import Expert, collect_hook_activations, encode_sae_batched, load_llm, load_sae


# ======================================================================
# Data loading + LLM activation collection
# ======================================================================
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
        raise ValueError("Provide either --dataset-name or --dataframe-path.")

    labels_tensor: torch.Tensor | None = None
    label_names: dict[int, str] | None = None
    n_classes: int = 0

    if raw_labels is not None and len(raw_labels) > 0:
        unique = sorted({str(l) for l in raw_labels})
        n_classes = len(unique)
        label_to_id = {l: i for i, l in enumerate(unique)}
        label_names = {i: l for l, i in label_to_id.items()}
        label_ids = torch.tensor(
            [label_to_id[str(l)] for l in raw_labels], dtype=torch.long
        )
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
    last_token_positions = (
        col_indices.masked_fill(~non_pad_mask, -1).max(dim=1).values
    ).clamp(min=0)

    print(
        f"Last non-pad positions — min: {last_token_positions.min().item()}, "
        f"max: {last_token_positions.max().item()}, "
        f"mean: {last_token_positions.float().mean().item():.1f} "
        f"(seq length {S})"
    )

    batches = (
        (tokenized[i : i + llm_batch_size], attention_mask[i : i + llm_batch_size])
        for i in range(0, B, llm_batch_size)
    )
    all_acts = collect_hook_activations(model, hook_name, batches, device)
    activations = torch.cat(all_acts, dim=0)
    del all_acts
    str_tokens: list[list[str]] = [
        list(tokenizer.convert_ids_to_tokens(tokenized[i].tolist()) or [])
        for i in range(B)
    ]

    return (
        activations,
        str_tokens,
        labels_tensor,
        label_names,
        last_token_positions,
        n_classes,
    )


# ======================================================================
# SAE encoding → Expert construction
# ======================================================================
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
    B, S_full, D = activations.shape

    seq_positions: torch.Tensor | None = None
    if last_token_only:
        if last_token_positions is None:
            raise ValueError(
                "last_token_only=True but no last_token_positions provided."
            )
        seq_positions = last_token_positions
        gather_idx = last_token_positions.unsqueeze(1).unsqueeze(2).expand(B, 1, D)
        activations = activations.gather(1, gather_idx)
        if labels is not None:
            label_idx = last_token_positions.unsqueeze(1)
            labels = labels.gather(1, label_idx)
        S = 1
        print(
            f"last_token_only=True → gathered last non-pad token per sequence "
            f"({B * S_full} → {B} tokens through SAE)"
        )
    else:
        S = S_full

    activations_flat = activations.reshape(B * S, D)

    sae_activations_cat = encode_sae_batched(sae, activations_flat, sae_batch_size)
    sae_activations_cat = sae_activations_cat.view(
        B, S, sae_activations_cat.shape[1], sae_activations_cat.shape[2]
    )

    experts: list[Expert] = []
    n_experts = sae_activations_cat.shape[-2]

    for i in tqdm(range(n_experts)):
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
        )
        experts.append(expert)

    del sae_activations_cat
    return experts


# ======================================================================
# Per-dataset config
# ======================================================================
@dataclasses.dataclass
class DatasetConfig:
    dataframe_path: str
    text_column: str = "Sentence"
    label_column: str | None = "Label"
    color_scale: str | None = None  # None = discrete categorical colors
    continuous_color: bool = False
    output_subdir: str | None = None

    @property
    def effective_continuous_color(self) -> bool:
        """True if a color_scale was specified, unless continuous_color was explicitly False."""
        return self.continuous_color or self.color_scale is not None

    @property
    def effective_color_scale(self) -> str:
        return self.color_scale if self.color_scale is not None else "Plasma"


# ======================================================================
# HTML output helpers
# ======================================================================
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

        tab_buttons.append(
            f'<button class="tab-btn{active_cls}" onclick="switchTab({idx})">'
            f"{tab_label}</button>"
        )

        plot_divs = f'<div class="plot-box" id="{scatter_id}"></div>'
        if mean_fig is not None:
            mean_id = f"mean_{idx}"
            plot_divs += f'\n    <div class="plot-box" id="{mean_id}"></div>'
            figures_json_parts.append(
                f'"{mean_id}": {pio.to_json(mean_fig, engine="json")}'
            )

        tab_panes.append(
            f'<div class="tab-pane{active_cls}">\n    {plot_divs}\n  </div>'
        )
        figures_json_parts.append(
            f'"{scatter_id}": {pio.to_json(scatter_fig, engine="json")}'
        )

    html = _HTML_TEMPLATE.format(
        title=dataset_title,
        tab_buttons="\n    ".join(tab_buttons),
        tab_panes="\n  ".join(tab_panes),
        figures_json=",\n    ".join(figures_json_parts),
    )
    return html.replace("__PLOTLYJS__", pyo.get_plotlyjs(), 1)


# ======================================================================
# Core pipeline (runs on one dataset with pre-loaded model + SAE)
# ======================================================================
def run_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sae: SMIXAE,
    cfg: DatasetConfig,
    final_output_dir: str,
    hook_point: str,
    n_input_samples: int,
    input_sequence_length: int,
    device: str,
    llm_batch_size: int,
    sae_batch_size: int,
    sort_by: str,
    sort_ascending: bool,
    adjusted_fisher: bool,
    k_neighbors: int,
    active_threshold: float,
    min_points: int,
    max_points: int,
    n_interesting_experts_to_plot: int,
    context_window_display: int,
    dataset_name: str | None = None,
) -> None:
    subdir = cfg.output_subdir or Path(cfg.dataframe_path).stem
    output_dir = os.path.join(final_output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Dataset: {cfg.dataframe_path}  →  {output_dir}")
    print(f"{'=' * 60}")

    # ── 1. Collect LLM activations ────────────────────────────────────
    llm_acts, str_tokens, labels, label_names, last_token_positions, n_classes = (
        collect_activations(
            model=model,
            tokenizer=tokenizer,
            hook_name=hook_point,
            max_length=input_sequence_length,
            n_input_samples=n_input_samples,
            device=device,
            llm_batch_size=llm_batch_size,
            dataset_name=dataset_name,
            dataframe_path=cfg.dataframe_path or None,
            text_column=cfg.text_column,
            label_column=cfg.label_column,
        )
    )

    is_labelled = labels is not None

    # ── 2. SAE encoding ───────────────────────────────────────────────
    experts = get_sae_activations(
        sae=sae,
        device=device,
        activations=llm_acts,
        sae_batch_size=sae_batch_size,
        active_threshold=active_threshold,
        min_points=min_points,
        max_points=max_points,
        labels=labels,
        last_token_only=is_labelled,
        last_token_positions=last_token_positions,
        n_classes=n_classes,
    )

    del llm_acts, labels

    if not experts:
        print("No experts fired enough times to exceed the min_points threshold.")
        return

    # ── 3. Evaluate ───────────────────────────────────────────────────
    if is_labelled:
        print(f"Evaluating Fisher score for {len(experts)} experts…")
        for expert in tqdm(experts):
            expert.evaluate_fisher()
    else:
        print(f"Evaluating manifold continuity (k={k_neighbors})…")
        for expert in tqdm(experts):
            expert.evaluate_manifold(k_neighbors=k_neighbors, device=device)

    # ── 4. Sort ───────────────────────────────────────────────────────
    effective_sort_by = sort_by
    if sort_by == "auto":
        effective_sort_by = (
            ("adjusted_fisher" if adjusted_fisher else "fisher")
            if is_labelled
            else "continuity"
        )

    print(
        f"Sorting by {effective_sort_by} "
        f"({'ascending' if sort_ascending else 'descending'})…"
    )
    experts.sort(
        key=lambda e: e.sort_key(effective_sort_by),
        reverse=not sort_ascending,
    )

    # ── 5. Plot ───────────────────────────────────────────────────────
    n_to_plot = min(n_interesting_experts_to_plot, len(experts))
    print(f"\nBuilding HTML for top {n_to_plot} experts…")

    expert_entries: list[tuple[str, Figure, Figure | None]] = []
    for i in range(n_to_plot):
        expert = experts[i]
        score_val = expert.sort_key(effective_sort_by)

        parts = [
            f"Rank {i + 1:02d}",
            f"Expert {expert.expert_id:4d}",
        ]
        if expert.n_unique_labels is not None:
            parts.append(f"Labels: {expert.n_unique_labels}/{n_classes}")
        if expert.fisher_score is not None:
            parts.append(f"Fisher: {expert.fisher_score:.4f}")
        if expert.adjusted_fisher_score is not None:
            parts.append(f"Adj-Fisher: {expert.adjusted_fisher_score:.4f}")
        if expert.mean_continuity is not None:
            parts.append(f"Cont: {expert.mean_continuity:.4f}")
        parts.append(f"Points: {expert.expert_activations.shape[0]}")
        print(" | ".join(parts))

        scatter_fig = expert.get_plot(
            str_tokens=str_tokens,  # type: ignore[arg-type]
            k_neighbors=k_neighbors,
            context_window=context_window_display,
            device=device,
            label_names=label_names,
            continuous_color=cfg.effective_continuous_color,
            color_scale=cfg.effective_color_scale,
        )
        mean_fig = expert.get_mean_plot(
            label_names=label_names,
            color_scale=cfg.effective_color_scale,
            continuous_color=cfg.effective_continuous_color,
        )
        tab_label = (
            f"#{i + 1} E{expert.expert_id} ({effective_sort_by}={score_val:.3f})"
        )
        expert_entries.append((tab_label, scatter_fig, mean_fig))

    html_str = build_dataset_html(expert_entries, f"{subdir} — Expert Analysis")
    output_path = os.path.join(output_dir, "experts.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"Saved: {output_path}")


# ======================================================================
# CLI
# ======================================================================
app = typer.Typer()

# Shared option defaults reused across subcommands
_SHARED_OPTIONS = dict(
    hook_point=typer.Option(..., help="The hook point where your SAE was trained"),
    sort_by=typer.Option(
        "auto",
        help="Metric to sort by: 'fisher', 'adjusted_fisher', 'continuity', or 'auto'.",
    ),
    sort_ascending=typer.Option(False, help="Sort ascending instead of descending."),
    n_input_samples=typer.Option(1000, help="Number of input texts to sample"),
    input_sequence_length=typer.Option(128, help="Input sequence length"),
    context_window_display=typer.Option(
        10, help="Number of surrounding tokens to display"
    ),
    n_interesting_experts_to_plot=typer.Option(
        50, help="Number of top experts to plot"
    ),
    device=typer.Option("cuda", help="Device to load models on"),
    llm_batch_size=typer.Option(16, help="Batch size for base LLM inference"),
    sae_batch_size=typer.Option(2048, help="Batch size for SAE inference"),
    k_neighbors=typer.Option(10, help="Number of neighbors for continuity"),
    active_threshold=typer.Option(
        1e-5, help="L2 norm threshold to consider an expert active"
    ),
    min_points=typer.Option(
        100, help="Minimum active tokens required to evaluate an expert"
    ),
    max_points=typer.Option(1000, help="Max active tokens per expert (0 = no cap)"),
    adjusted_fisher=typer.Option(
        False,
        help="Sort by Fisher × (n_classes_present / n_classes) to penalise low class coverage",
    ),
    output_dir=typer.Option("expert_plots", help="Base directory to save HTML plots"),
)


@app.command()
def single(
    checkpoint_path: str = typer.Option(..., help="Path to your checkpoint"),
    base_model_name: str = typer.Option(..., help="Model to load"),
    hook_point: str = typer.Option(
        ..., help="The hook point where your SAE was trained"
    ),
    # ── data source ──
    dataset_name: str | None = typer.Option(
        None, help="HF dataset to stream (unlabelled)"
    ),
    dataframe_path: str | None = typer.Option(
        None, help="Path to a CSV / Parquet / JSON(L) file"
    ),
    text_column: str = typer.Option("text", help="Column containing text"),
    label_column: str | None = typer.Option(None, help="Column containing labels"),
    # ── sorting ──
    sort_by: str = typer.Option(
        "auto",
        help="Metric to sort by: 'fisher', 'adjusted_fisher', 'continuity', or 'auto'.",
    ),
    sort_ascending: bool = typer.Option(
        False, help="Sort ascending instead of descending."
    ),
    # ── general ──
    n_input_samples: int = typer.Option(1000, help="Number of input texts to sample"),
    input_sequence_length: int = typer.Option(128, help="Input sequence length"),
    context_window_display: int = typer.Option(
        10, help="Number of surrounding tokens to display"
    ),
    n_interesting_experts_to_plot: int = typer.Option(
        50, help="Number of top experts to plot"
    ),
    device: str = typer.Option("cuda", help="Device to load models on"),
    llm_batch_size: int = typer.Option(16, help="Batch size for base LLM inference"),
    sae_batch_size: int = typer.Option(2048, help="Batch size for SAE inference"),
    k_neighbors: int = typer.Option(10, help="Number of neighbors for continuity"),
    active_threshold: float = typer.Option(
        1e-5, help="L2 norm threshold to consider an expert active"
    ),
    min_points: int = typer.Option(
        100, help="Minimum active tokens required to evaluate an expert"
    ),
    max_points: int = typer.Option(
        1000, help="Max active tokens per expert (0 = no cap)"
    ),
    adjusted_fisher: bool = typer.Option(
        False,
        help="Sort by Fisher × (n_classes_present / n_classes) to penalise low class coverage",
    ),
    continuous_color: bool = typer.Option(
        False,
        help="Color points by continuous label ordinal rather than discrete categories",
    ),
    color_scale: str = typer.Option(
        "Plasma", help="Plotly continuous colorscale name (e.g. Plasma, Viridis, RdBu)"
    ),
    output_dir: str = typer.Option(
        "expert_plots", help="Base directory to save the HTML plots"
    ),
):
    if dataset_name is None and dataframe_path is None:
        dataset_name = "monology/pile-uncopyrighted"
        print(f"No data source specified — defaulting to {dataset_name}")

    ckpt_path_obj = Path(checkpoint_path)
    run_hash = ckpt_path_obj.parent.name
    run_step = ckpt_path_obj.name.replace("final_", "")
    run_folder_name = f"{run_hash}_{run_step}"
    final_output_dir = os.path.join(output_dir, run_folder_name)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Plots will be saved to: {final_output_dir}")

    model, tokenizer = load_llm(base_model_name, device)
    tokenizer.padding_side = "right"
    sae = load_sae(checkpoint_path, device)

    cfg = DatasetConfig(
        dataframe_path=dataframe_path or "",
        text_column=text_column,
        label_column=label_column,
        color_scale=color_scale,
        continuous_color=continuous_color,
    )

    run_pipeline(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        cfg=cfg,
        final_output_dir=final_output_dir,
        hook_point=hook_point,
        n_input_samples=n_input_samples,
        input_sequence_length=input_sequence_length,
        device=device,
        llm_batch_size=llm_batch_size,
        sae_batch_size=sae_batch_size,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
        adjusted_fisher=adjusted_fisher,
        k_neighbors=k_neighbors,
        active_threshold=active_threshold,
        min_points=min_points,
        max_points=max_points,
        n_interesting_experts_to_plot=n_interesting_experts_to_plot,
        context_window_display=context_window_display,
        dataset_name=dataset_name,
    )

    del model, sae
    torch.cuda.empty_cache()


@app.command()
def all_datasets(
    checkpoint_path: str = typer.Option(..., help="Path to your checkpoint"),
    base_model_name: str = typer.Option(..., help="Model to load"),
    hook_point: str = typer.Option(
        ..., help="The hook point where your SAE was trained"
    ),
    datasets_config: str = typer.Option(
        ...,
        help=(
            "Path to a JSON file listing datasets. Each entry may have: "
            "dataframe_path (required), text_column, label_column, "
            "color_scale, continuous_color, output_subdir."
        ),
    ),
    # ── sorting ──
    sort_by: str = typer.Option(
        "auto",
        help="Metric to sort by: 'fisher', 'adjusted_fisher', 'continuity', or 'auto'.",
    ),
    sort_ascending: bool = typer.Option(
        False, help="Sort ascending instead of descending."
    ),
    # ── general ──
    n_input_samples: int = typer.Option(1000, help="Number of input texts to sample"),
    input_sequence_length: int = typer.Option(128, help="Input sequence length"),
    context_window_display: int = typer.Option(
        10, help="Number of surrounding tokens to display"
    ),
    n_interesting_experts_to_plot: int = typer.Option(
        50, help="Number of top experts to plot"
    ),
    device: str = typer.Option("cuda", help="Device to load models on"),
    llm_batch_size: int = typer.Option(16, help="Batch size for base LLM inference"),
    sae_batch_size: int = typer.Option(2048, help="Batch size for SAE inference"),
    k_neighbors: int = typer.Option(10, help="Number of neighbors for continuity"),
    active_threshold: float = typer.Option(
        1e-5, help="L2 norm threshold to consider an expert active"
    ),
    min_points: int = typer.Option(
        100, help="Minimum active tokens required to evaluate an expert"
    ),
    max_points: int = typer.Option(
        1000, help="Max active tokens per expert (0 = no cap)"
    ),
    adjusted_fisher: bool = typer.Option(
        False,
        help="Sort by Fisher × (n_classes_present / n_classes) to penalise low class coverage",
    ),
    output_dir: str = typer.Option(
        "expert_plots", help="Base directory to save the HTML plots"
    ),
    continuity_dataset: str = typer.Option(
        "monology/pile-uncopyrighted",
        help="HuggingFace dataset to stream for the unlabelled continuity pass",
    ),
):
    config_dir = Path(datasets_config).resolve().parent
    with open(datasets_config) as f:
        raw_configs = json.load(f)

    for entry in raw_configs:
        p = Path(entry["dataframe_path"])
        if not p.is_absolute():
            entry["dataframe_path"] = str(config_dir / p)

    dataset_cfgs = [DatasetConfig(**entry) for entry in raw_configs]
    print(f"Loaded {len(dataset_cfgs)} dataset configs from {datasets_config}")

    ckpt_path_obj = Path(checkpoint_path)
    run_hash = ckpt_path_obj.parent.name
    run_step = ckpt_path_obj.name.replace("final_", "")
    run_folder_name = f"{run_hash}_{run_step}"
    final_output_dir = os.path.join(output_dir, run_folder_name)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"All plots will be saved under: {final_output_dir}")

    # Load model and SAE once
    model, tokenizer = load_llm(base_model_name, device)
    tokenizer.padding_side = "right"
    sae = load_sae(checkpoint_path, device)

    pipeline_kwargs: dict[str, Any] = dict(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        final_output_dir=final_output_dir,
        hook_point=hook_point,
        n_input_samples=n_input_samples,
        input_sequence_length=input_sequence_length,
        device=device,
        llm_batch_size=llm_batch_size,
        sae_batch_size=sae_batch_size,
        sort_ascending=sort_ascending,
        adjusted_fisher=adjusted_fisher,
        k_neighbors=k_neighbors,
        active_threshold=active_threshold,
        min_points=min_points,
        max_points=max_points,
        n_interesting_experts_to_plot=n_interesting_experts_to_plot,
        context_window_display=context_window_display,
    )

    for cfg in dataset_cfgs:
        run_pipeline(cfg=cfg, sort_by=sort_by, **pipeline_kwargs)  # type: ignore[arg-type]

    # Unlabelled continuity pass
    print(f"\n{'=' * 60}")
    print(f"Continuity pass: streaming from {continuity_dataset}")
    continuity_cfg = DatasetConfig(
        dataframe_path="",
        text_column="text",
        label_column=None,
        output_subdir="continuity",
    )
    run_pipeline(
        cfg=continuity_cfg,
        sort_by="continuity",
        dataset_name=continuity_dataset,
        **pipeline_kwargs,
    )  # type: ignore[arg-type]

    del model, sae
    torch.cuda.empty_cache()
    print("\nAll datasets processed.")


if __name__ == "__main__":
    app()
