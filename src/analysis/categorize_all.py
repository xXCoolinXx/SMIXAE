"""Expert probing pipeline: load a SMIXAE checkpoint, score experts on labeled datasets, and produce HTML visualizations."""

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
    """Per-dataset configuration for the probing pipeline.

    Controls which data file to load, how labels are mapped to colours, and how
    many samples to encode.  Instances are typically constructed from entries in
    ``datasets/probing/dataset_config.json``.

    Attributes:
        dataframe_path: Path to the CSV/Parquet/JSONL file (relative to the config
            directory, or absolute).
        text_column: Column name containing the input text. Defaults to ``"Sentence"``.
        label_column: Column name containing class labels, or ``None`` for unlabelled
            (continuity-only) analysis.
        color_scale: Named Plotly colorscale (e.g. ``"Plasma"``, ``"Phase"``) used when
            ``continuous_color`` is ``True``.  ``None`` uses discrete categorical colours.
        continuous_color: If ``True``, colour points by label ordinal rather than by
            class.  Automatically set to ``True`` when ``color_scale`` is provided.
        output_subdir: Override for the output sub-directory name.  Defaults to the
            stem of ``dataframe_path``.
        color_map: Explicit ``{label: CSS_color}`` mapping for 1:1 colour schemes.
            Takes precedence over ``color_scale`` for discrete plots.
        n_input_samples: Override for the ``--n-input-samples`` CLI flag for this
            dataset only.  ``None`` uses the global CLI value.
        max_points: Override for ``--max-points`` for this dataset.
            ``0`` means no cap; ``None`` uses the global CLI value.
        show_labels: Annotate class-mean markers with text labels in the 3D scatter.
    """

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
    regression_hypotheses: list[dict] | None = None  # list of hypothesis specs for regression probing
    bucket_column: str | None = None  # continuous column to bucket into Fisher labels when label_column is None
    n_buckets: int = 10  # number of equal-width bins for bucket_column

    @property
    def effective_continuous_color(self) -> bool:
        """True if a color_scale was specified, unless continuous_color was explicitly False."""
        return self.continuous_color or self.color_scale is not None

    @property
    def effective_color_scale(self) -> str:
        """Return the configured color scale, falling back to ``"Plasma"``."""
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
    """Run the full analysis pipeline for a single dataset and write an HTML report.

    Five-stage pipeline:

    1. **Collect LLM activations** — tokenise the dataset and register a forward hook
       at ``hook_point`` to capture residual stream activations.
    2. **SAE encoding** — run SMIXAE on the activations in batches; filter experts
       by ``active_threshold`` and ``min_points``.
    3. **Evaluate experts** — score by Fisher discriminant ratio (labelled data) or
       manifold continuity (unlabelled data).
    4. **Sort** — rank experts by the chosen metric (``sort_by``).
    5. **Plot** — generate interactive 3D Plotly scatters for the top-N experts and
       serialise everything into a single self-contained HTML file.

    Output is written to ``{final_output_dir}/{subdir}/experts.html``.

    Args:
        model: Pre-loaded HuggingFace language model.
        tokenizer: Corresponding tokenizer.
        sae: Loaded SMIXAE inference checkpoint.
        cfg: Per-dataset configuration (colours, sample count overrides, etc.).
        final_output_dir: Root directory under which per-dataset sub-directories
            are created.
        hook_point: HuggingFace module path to hook (e.g. ``"model.layers.11"``).
        n_input_samples: Number of texts to encode (overridden by ``cfg.n_input_samples``
            when set).
        input_sequence_length: Max token length for padding/truncation.
        device: Torch device string (e.g. ``"cuda"``).
        llm_batch_size: Batch size for LLM forward passes.
        sae_batch_size: Batch size for SAE encoding.
        sort_by: Metric to rank experts by — one of ``"fisher"``,
            ``"adjusted_fisher"``, ``"continuity"``, or ``"auto"``.
        sort_ascending: If ``True``, rank lowest scores first.
        adjusted_fisher: If ``True``, multiply Fisher score by class coverage fraction.
        k_neighbors: Neighbourhood size for manifold continuity scoring.
        active_threshold: Minimum L2 norm for an expert to be considered active.
        min_points: Minimum number of active tokens required to include an expert.
        max_points: Cap on active tokens per expert (``0`` = no cap).
        n_interesting_experts_to_plot: Number of top-ranked experts to include in the
            HTML report.
        context_window_display: Number of surrounding tokens shown on hover.
        dataset_name: HuggingFace dataset name to stream (used when
            ``cfg.dataframe_path`` is empty, e.g. for the continuity pass).
    """
    subdir = cfg.output_subdir or Path(cfg.dataframe_path).stem
    output_dir = os.path.join(final_output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Dataset: {cfg.dataframe_path}  →  {output_dir}")
    print(f"{'=' * 60}")

    # ── 1. Collect LLM activations ────────────────────────────────────
    # Build the deduplicated list of target columns required by regression hypotheses.
    all_target_cols: list[str] = []
    if cfg.regression_hypotheses:
        seen_cols: set[str] = set()
        for hyp in cfg.regression_hypotheses:
            for col in hyp.get("target_columns", []):
                if col not in seen_cols:
                    all_target_cols.append(col)
                    seen_cols.add(col)

    llm_acts, str_tokens, labels, label_names, last_token_positions, n_classes, reg_targets = collect_activations(
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
        regression_target_columns=all_target_cols if all_target_cols else None,
        bucket_column=cfg.bucket_column,
        n_buckets=cfg.n_buckets,
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
        regression_targets=reg_targets,
    )

    del llm_acts, labels, reg_targets

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

    # ── 4. Sort (Stage 1: Fisher / continuity) ────────────────────────
    effective_sort_by = sort_by
    if sort_by == "auto":
        effective_sort_by = ("adjusted_fisher" if adjusted_fisher else "fisher") if is_labelled else "continuity"

    print(f"Sorting by {effective_sort_by} ({'ascending' if sort_ascending else 'descending'})…")
    experts.sort(
        key=lambda e: e.sort_key(effective_sort_by),
        reverse=not sort_ascending,
    )

    # ── 4b. Regression probing on top-N experts (Stage 2) ────────────
    n_to_plot = min(n_interesting_experts_to_plot, len(experts))
    top_experts = experts[:n_to_plot]

    # Attach column indices to each hypothesis before running
    hypotheses_with_indices: list[dict] = []
    if cfg.regression_hypotheses and all_target_cols:
        col_to_idx = {col: i for i, col in enumerate(all_target_cols)}
        for hyp in cfg.regression_hypotheses:
            h = dict(hyp)
            h["target_indices"] = [col_to_idx[c] for c in hyp["target_columns"] if c in col_to_idx]
            if h["target_indices"]:
                hypotheses_with_indices.append(h)

    if hypotheses_with_indices:
        print(f"\nRunning regression probing ({len(hypotheses_with_indices)} hypotheses) on top {n_to_plot} experts…")
        for expert in tqdm(top_experts, desc="Regression probing"):
            expert.evaluate_regression(hypotheses_with_indices)

        # Stage 2 re-sort: best regression score descending
        top_experts.sort(key=lambda e: e.sort_key("regression"), reverse=True)
        final_sort_label = "regression"
        print("Re-sorted by best regression score.")
    else:
        final_sort_label = effective_sort_by

    # ── 5. Plot ───────────────────────────────────────────────────────
    print(f"\nBuilding HTML for top {n_to_plot} experts…")

    # Console summary
    for i, expert in enumerate(top_experts):
        fisher_val = expert.sort_key(effective_sort_by)
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
        if expert.best_regression_name is not None:
            parts.append(f"Best-Reg: {expert.best_regression_name}={expert.best_regression_score:.4f}")
        parts.append(f"Points: {expert.expert_activations.shape[0]}")
        print(" | ".join(parts))

    # Helper: build one (tab_label, scatter_fig, mean_fig, reg_scores) entry
    def _make_plot_entry(expert, rank: int, score_str: str) -> tuple:
        l0_str = f" L0={expert.mean_latent_l0:.1f}" if expert.mean_latent_l0 is not None else ""
        tab_label = f"#{rank + 1} E{expert.expert_id}{l0_str}{score_str}"
        s_fig = expert.get_plot(
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
        m_fig = expert.get_mean_plot(
            label_names=label_names,
            color_scale=cfg.effective_color_scale,
            continuous_color=cfg.effective_continuous_color,
            color_map=cfg.color_map,
            connect_means=False,
            show_labels=cfg.show_labels,
        )
        return (tab_label, s_fig, m_fig, expert.regression_scores)

    # Build per-hypothesis top-10 rows (when regression hypotheses exist)
    per_hypothesis_entries: dict[str, tuple[str, list]] = {}
    expert_entries: list = []

    if hypotheses_with_indices:
        print("Building per-hypothesis top-10 plots…")
        for hyp in hypotheses_with_indices:
            name = hyp["name"]
            desc = hyp.get("description", name)
            sorted_for_hyp = sorted(
                top_experts,
                key=lambda e, n=name: e.regression_scores.get(n, float("-inf")),
                reverse=True,
            )[:10]
            hyp_entries = []
            for rank, expert in enumerate(sorted_for_hyp):
                hyp_score = expert.regression_scores.get(name, float("nan"))
                fisher_val = expert.fisher_score or 0.0
                score_str = f" {name}={'%.3f' % hyp_score} F={'%.2f' % fisher_val}"
                hyp_entries.append(_make_plot_entry(expert, rank, score_str))
            per_hypothesis_entries[name] = (desc, hyp_entries)
    else:
        # No regression — flat tab strip sorted by Fisher/continuity
        for i, expert in enumerate(top_experts):
            fisher_val = expert.sort_key(effective_sort_by)
            l0_str = f" L0={expert.mean_latent_l0:.1f}" if expert.mean_latent_l0 is not None else ""
            score_str = f" ({effective_sort_by}={fisher_val:.3f})"
            expert_entries.append(_make_plot_entry(expert, i, score_str))

    html_str = build_dataset_html(
        expert_entries,
        f"{subdir} — Expert Analysis",
        per_hypothesis_entries=per_hypothesis_entries or None,
    )
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
    """Probe a single dataset against a SMIXAE checkpoint and write an HTML report.

    Loads the LLM and SAE once, runs :func:`run_pipeline` for the specified dataset,
    and writes interactive 3D scatter plots of the top-N experts to
    ``{output_dir}/{run_hash}_{step}/{dataset_stem}/experts.html``.

    Supports both labelled data (CSV/Parquet/JSONL with a label column, scored by Fisher
    discriminant ratio) and unlabelled streaming (HuggingFace dataset, scored by manifold
    continuity).  If neither ``--dataset-name`` nor ``--dataframe-path`` is given,
    defaults to streaming ``monology/pile-uncopyrighted``.
    """
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
    """Probe all datasets listed in a JSON config file, then run an unlabelled continuity pass.

    Loads the LLM and SAE **once** and iterates over every entry in ``datasets_config``,
    calling :func:`run_pipeline` for each.  After all labelled datasets are processed,
    runs a final continuity pass by streaming ``continuity_dataset`` (unlabelled) and
    scoring experts by manifold continuity.

    The JSON config is a list of objects, each following the :class:`DatasetConfig`
    schema (``dataframe_path`` required; all other fields optional).  Relative paths in
    the config are resolved relative to the config file's parent directory.

    All outputs land under ``{output_dir}/{run_hash}_{step}/``, with one sub-directory
    per dataset (named by ``output_subdir`` or the CSV stem) and a ``continuity/``
    sub-directory for the unlabelled pass.
    """
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
