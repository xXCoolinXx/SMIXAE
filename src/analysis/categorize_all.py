import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import torch
import typer
from plotly.graph_objects import Figure
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from analysis.utils import build_dataset_html, collect_activations, get_sae_activations, load_llm, load_sae
from smixae import SMIXAE


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
    color_map: dict[str, str] | None = None  # label → CSS color for 1:1 color schemes
    n_input_samples: int | None = None  # overrides CLI --n-input-samples when set
    max_points: int | None = None  # overrides CLI --max-points when set (0 = no cap)
    show_labels: bool = False  # annotate class means with text labels

    @property
    def effective_continuous_color(self) -> bool:
        """True if a color_scale was specified, unless continuous_color was explicitly False."""
        return self.continuous_color or self.color_scale is not None

    @property
    def effective_color_scale(self) -> str:
        return self.color_scale if self.color_scale is not None else "Plasma"


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
    llm_acts, str_tokens, labels, label_names, last_token_positions, n_classes = collect_activations(
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
        effective_sort_by = ("adjusted_fisher" if adjusted_fisher else "fisher") if is_labelled else "continuity"

    print(f"Sorting by {effective_sort_by} ({'ascending' if sort_ascending else 'descending'})…")
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
        if expert.mean_latent_l0 is not None:
            parts.append(f"L0: {expert.mean_latent_l0:.1f}")
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
            color_map=cfg.color_map,
            connect_means=False,
            show_labels=cfg.show_labels,
        )
        mean_fig = expert.get_mean_plot(
            label_names=label_names,
            color_scale=cfg.effective_color_scale,
            continuous_color=cfg.effective_continuous_color,
            color_map=cfg.color_map,
            connect_means=False,
            show_labels=cfg.show_labels,
        )
        l0_str = f" L0={expert.mean_latent_l0:.1f}" if expert.mean_latent_l0 is not None else ""
        tab_label = f"#{i + 1} E{expert.expert_id}{l0_str} ({effective_sort_by}={score_val:.3f})"
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
    context_window_display=typer.Option(10, help="Number of surrounding tokens to display"),
    n_interesting_experts_to_plot=typer.Option(50, help="Number of top experts to plot"),
    device=typer.Option("cuda", help="Device to load models on"),
    llm_batch_size=typer.Option(16, help="Batch size for base LLM inference"),
    sae_batch_size=typer.Option(2048, help="Batch size for SAE inference"),
    k_neighbors=typer.Option(10, help="Number of neighbors for continuity"),
    active_threshold=typer.Option(1e-5, help="L2 norm threshold to consider an expert active"),
    min_points=typer.Option(100, help="Minimum active tokens required to evaluate an expert"),
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
    hook_point: str = typer.Option(..., help="The hook point where your SAE was trained"),
    # ── data source ──
    dataset_name: str | None = typer.Option(None, help="HF dataset to stream (unlabelled)"),
    dataframe_path: str | None = typer.Option(None, help="Path to a CSV / Parquet / JSON(L) file"),
    text_column: str = typer.Option("text", help="Column containing text"),
    label_column: str | None = typer.Option(None, help="Column containing labels"),
    # ── sorting ──
    sort_by: str = typer.Option(
        "auto",
        help="Metric to sort by: 'fisher', 'adjusted_fisher', 'continuity', or 'auto'.",
    ),
    sort_ascending: bool = typer.Option(False, help="Sort ascending instead of descending."),
    # ── general ──
    n_input_samples: int = typer.Option(1000, help="Number of input texts to sample"),
    input_sequence_length: int = typer.Option(128, help="Input sequence length"),
    context_window_display: int = typer.Option(10, help="Number of surrounding tokens to display"),
    n_interesting_experts_to_plot: int = typer.Option(50, help="Number of top experts to plot"),
    device: str = typer.Option("cuda", help="Device to load models on"),
    llm_batch_size: int = typer.Option(16, help="Batch size for base LLM inference"),
    sae_batch_size: int = typer.Option(2048, help="Batch size for SAE inference"),
    k_neighbors: int = typer.Option(10, help="Number of neighbors for continuity"),
    active_threshold: float = typer.Option(1e-5, help="L2 norm threshold to consider an expert active"),
    min_points: int = typer.Option(100, help="Minimum active tokens required to evaluate an expert"),
    max_points: int = typer.Option(1000, help="Max active tokens per expert (0 = no cap)"),
    adjusted_fisher: bool = typer.Option(
        False,
        help="Sort by Fisher × (n_classes_present / n_classes) to penalise low class coverage",
    ),
    continuous_color: bool = typer.Option(
        False,
        help="Color points by continuous label ordinal rather than discrete categories",
    ),
    color_scale: str = typer.Option("Plasma", help="Plotly continuous colorscale name (e.g. Plasma, Viridis, RdBu)"),
    output_dir: str = typer.Option("expert_plots", help="Base directory to save the HTML plots"),
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
    hook_point: str = typer.Option(..., help="The hook point where your SAE was trained"),
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
    sort_ascending: bool = typer.Option(False, help="Sort ascending instead of descending."),
    # ── general ──
    n_input_samples: int = typer.Option(1000, help="Number of input texts to sample"),
    input_sequence_length: int = typer.Option(128, help="Input sequence length"),
    context_window_display: int = typer.Option(10, help="Number of surrounding tokens to display"),
    n_interesting_experts_to_plot: int = typer.Option(50, help="Number of top experts to plot"),
    device: str = typer.Option("cuda", help="Device to load models on"),
    llm_batch_size: int = typer.Option(16, help="Batch size for base LLM inference"),
    sae_batch_size: int = typer.Option(2048, help="Batch size for SAE inference"),
    k_neighbors: int = typer.Option(10, help="Number of neighbors for continuity"),
    active_threshold: float = typer.Option(1e-5, help="L2 norm threshold to consider an expert active"),
    min_points: int = typer.Option(100, help="Minimum active tokens required to evaluate an expert"),
    max_points: int = typer.Option(1000, help="Max active tokens per expert (0 = no cap)"),
    adjusted_fisher: bool = typer.Option(
        False,
        help="Sort by Fisher × (n_classes_present / n_classes) to penalise low class coverage",
    ),
    output_dir: str = typer.Option("expert_plots", help="Base directory to save the HTML plots"),
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
        per_cfg_kwargs = {
            **pipeline_kwargs,
            "n_input_samples": cfg.n_input_samples or n_input_samples,
            "max_points": cfg.max_points if cfg.max_points is not None else max_points,
        }
        run_pipeline(cfg=cfg, sort_by=sort_by, **per_cfg_kwargs)  # type: ignore[arg-type]

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
