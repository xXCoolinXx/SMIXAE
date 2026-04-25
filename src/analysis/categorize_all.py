"""Expert probing pipeline: load a SMIXAE checkpoint, score experts on labeled datasets, and produce HTML visualizations."""

import dataclasses
import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from analysis import probing_io
from analysis.utils import _strip_prefix

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from analysis.utils import DatasetConfig
    from smixae import SMIXAE


# ======================================================================
# Pipeline configuration
# ======================================================================
@dataclasses.dataclass
class ProbeRunConfig:
    """Runtime configuration for the probing pipeline.

    Bundles all tuning parameters that control how experts are scored, filtered,
    and visualized.  Pass an instance to :func:`run_pipeline` in place of the
    individual keyword arguments.

    Attributes:
        device:                       Torch device string (e.g. ``"cuda"``).
        hook_point:                   HuggingFace module path to hook.
        n_input_samples:              Number of texts to encode.
        input_sequence_length:        Max token length for padding/truncation.
        llm_batch_size:               Batch size for LLM forward passes.
        sae_batch_size:               Batch size for SAE encoding.
        sort_by:                      Metric to rank experts by.
        sort_ascending:               If ``True``, rank lowest scores first.
        adjusted_fisher:              Multiply Fisher score by class-coverage fraction.
        k_neighbors:                  k-NN neighbourhood size for continuity scoring.
        active_threshold:             Minimum L2 norm for an expert to be active.
        min_active_fraction:          Minimum fraction of tokens an expert must fire on.
        max_points:                   Cap on active tokens per expert (0 = no cap).
        n_interesting_experts_to_plot: Top-N experts to include in the HTML report.
        context_window_display:       Surrounding tokens shown on hover.
        random_sample_n_input_samples: Number of prompts for the unlabelled random-sample path.
        random_sample_min_active_fraction: Minimum fraction of ``random_sample_max_points`` an
            expert must fire on in the random-sample path (e.g. 0.1 → ≥10% of max_points).
        random_sample_max_points: Max points collected per expert in the random-sample path.
    """

    device: str
    hook_point: str
    n_input_samples: int
    input_sequence_length: int
    llm_batch_size: int
    sae_batch_size: int
    sort_by: str = "auto"
    sort_ascending: bool = False
    adjusted_fisher: bool = False
    k_neighbors: int = 10
    active_threshold: float = 1e-5
    min_active_fraction: float = 0.10
    max_points: int = 1000
    n_interesting_experts_to_plot: int = 50
    context_window_display: int = 10
    random_seed: int = 42

    # Random-sample path only (overrides main filter params for unlabelled auto mode)
    random_sample_n_input_samples: int = 30000
    random_sample_min_active_fraction: float = 0.1
    random_sample_max_points: int = 1000


# ======================================================================
# Pipeline helpers
# ======================================================================

_SCORE_TYPE: dict[str, str] = {
    "linear": "r2",
    "ridge": "r2",
    "logistic": "acc",
    "multinomial": "acc",
}


def _regression_score_type(regression_type: str) -> str:
    """Map regression_type to the short label used in camera-ready PNG filenames."""
    return _SCORE_TYPE.get(regression_type, "score")


def _log_expert_summary(experts: list, n_classes: int, sort_metric: str) -> None:
    """Print a ranked console summary table for a list of scored experts."""
    for i, expert in enumerate(experts):
        # fisher_val = expert.sort_key(sort_metric)
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


def _build_dataset_results(
    top_experts: list,
    per_hypothesis_entries: dict,
    effective_sort_by: str,
    hypotheses_with_indices: list[dict],
) -> dict[str, Any]:
    """Build the structured results dict for the shared results JSON.

    Stores an ordered top-10 expert list (with rank, expert_id, and score) for
    both the Fisher/continuity ranking and each regression hypothesis.
    """
    dataset_results: dict[str, Any] = {}

    fisher_sorted = sorted(top_experts, key=lambda e: e.sort_key(effective_sort_by), reverse=True)
    if fisher_sorted and fisher_sorted[0].sort_key(effective_sort_by) > float("-inf"):
        scores_f = [e.sort_key(effective_sort_by) for e in fisher_sorted]
        dataset_results["fisher"] = {
            "sort_by": effective_sort_by,
            "top10_experts": [
                {"rank": i + 1, "expert_id": e.expert_id, "score": float(s)}
                for i, (e, s) in enumerate(zip(fisher_sorted[:10], scores_f[:10]))
            ],
            "top5_mean": float(sum(scores_f[:5]) / min(5, len(scores_f))),
            "top10_mean": float(sum(scores_f[:10]) / min(10, len(scores_f))),
        }

    if per_hypothesis_entries:
        dataset_results["hypotheses"] = {}
        for hyp_name, (hyp_desc, hyp_entries) in per_hypothesis_entries.items():
            metas = [entry[4] for entry in hyp_entries]
            hyp_scores = [m["hyp_score"] for m in metas if m.get("hyp_score") is not None]
            if not hyp_scores:
                continue
            hyp_score_stds = [m.get("hyp_score_std") for m in metas[:len(hyp_scores)]]
            reg_type = next(
                (h.get("regression_type", "unknown") for h in hypotheses_with_indices if h["name"] == hyp_name),
                "unknown",
            )
            valid_stds_top5 = [s for s in hyp_score_stds[:5] if s is not None]
            top5_mean_std = float(sum(valid_stds_top5) / len(valid_stds_top5)) if valid_stds_top5 else None
            dataset_results["hypotheses"][hyp_name] = {
                "description": hyp_desc,
                "regression_type": reg_type,
                "top10_experts": [
                    {
                        "rank": i + 1,
                        "expert_id": m["expert_id"],
                        "score": hyp_scores[i],
                        "score_std": hyp_score_stds[i],
                    }
                    for i, m in enumerate(metas[:10])
                    if i < len(hyp_scores)
                ],
                "top5_mean": float(sum(hyp_scores[:5]) / min(5, len(hyp_scores))),
                "top5_mean_std": top5_mean_std,
                "top10_mean": float(sum(hyp_scores[:10]) / min(10, len(hyp_scores))),
            }

    return dataset_results


# ======================================================================
# Probing-task helpers (build metadata, no visualization code)
# ======================================================================


def _build_color_spec(cfg: "DatasetConfig", batch: Any) -> "probing_io.ColorSpec":
    """Build a :class:`probing_io.ColorSpec` from a DatasetConfig + ActivationBatch."""
    if not batch.is_labelled:
        return probing_io.ColorSpec(
            mode="continuous",
            scale="Viridis",
            continuous_label="Distance from origin",
        )
    scale = cfg.color_scale or "Plasma"
    mapped: dict[str, str] | None = None
    if cfg.color_map:
        mapped = {_strip_prefix(k): v for k, v in cfg.color_map.items()}
    return probing_io.ColorSpec(
        mode="continuous" if cfg.continuous_color else "discrete",
        scale=scale,
        color_map=mapped,
    )


def _build_hypothesis_specs_and_views(
    cfg: "DatasetConfig",
    hypotheses_with_indices: list[dict],
    top_experts: list,
    effective_sort_by: str,
    use_random_sample: bool,
    is_labelled: bool,
) -> tuple[list[probing_io.HypothesisSpec], dict[str, list[probing_io.ExpertRanking]]]:
    """Build hypothesis specs and per-view expert rankings for the task index."""
    overrides = cfg.hypothesis_color_overrides or {}
    hyp_specs: list[probing_io.HypothesisSpec] = []
    for hyp in hypotheses_with_indices:
        name = hyp["name"]
        override = overrides.get(name)
        mapped_override: dict[str, str] | None = None
        if override:
            mapped_override = {_strip_prefix(k): v for k, v in override.items()}
        hyp_specs.append(
            probing_io.HypothesisSpec(
                name=name,
                description=hyp.get("description", name),
                regression_type=hyp.get("regression_type", ""),
                score_type=_regression_score_type(hyp.get("regression_type", "")),
                color_override=mapped_override,
            )
        )

    experts_by_view: dict[str, list[probing_io.ExpertRanking]] = {}
    if hypotheses_with_indices:
        for hyp in hypotheses_with_indices:
            name = hyp["name"]
            sorted_for_hyp = sorted(
                top_experts,
                key=lambda e, n=name: e.regression_scores.get(n, float("-inf")),
                reverse=True,
            )[:10]
            rankings: list[probing_io.ExpertRanking] = []
            for rank, expert in enumerate(sorted_for_hyp, start=1):
                score = expert.regression_scores.get(name)
                std = (
                    expert.regression_scores_std.get(name)
                    if expert.regression_scores_std else None
                )
                rankings.append(
                    probing_io.ExpertRanking(
                        rank=rank,
                        expert_id=int(expert.expert_id),
                        score=float(score) if score is not None and score == score else None,
                        score_std=float(std) if std is not None and std == std else None,
                    )
                )
            experts_by_view[name] = rankings
    else:
        view_name = (
            "random" if (use_random_sample and not is_labelled)
            else ("continuity" if not is_labelled else effective_sort_by)
        )
        rankings = []
        for rank, expert in enumerate(top_experts, start=1):
            score_val = expert.sort_key(effective_sort_by)
            rankings.append(
                probing_io.ExpertRanking(
                    rank=rank,
                    expert_id=int(expert.expert_id),
                    score=float(score_val) if score_val != float("-inf") else None,
                )
            )
        experts_by_view[view_name] = rankings

    return hyp_specs, experts_by_view


def _legacy_per_hypothesis_for_results_json(
    hypotheses_with_indices: list[dict],
    top_experts: list,
) -> dict[str, tuple[str, list]]:
    """Reconstruct the legacy ``per_hypothesis_entries`` for ``_build_dataset_results``."""
    if not hypotheses_with_indices:
        return {}
    out: dict[str, tuple[str, list]] = {}
    for hyp in hypotheses_with_indices:
        name = hyp["name"]
        desc = hyp.get("description", name)
        sorted_for_hyp = sorted(
            top_experts,
            key=lambda e, n=name: e.regression_scores.get(n, float("-inf")),
            reverse=True,
        )[:10]
        entries = []
        for expert in sorted_for_hyp:
            score = expert.regression_scores.get(name, float("nan"))
            std = (
                expert.regression_scores_std.get(name)
                if expert.regression_scores_std else None
            )
            meta = {
                "expert_id": int(expert.expert_id),
                "hyp_score": None if score != score else float(score),
                "hyp_score_std": None if std is None or std != std else float(std),
            }
            entries.append((None, None, None, None, meta))
        out[name] = (desc, entries)
    return out


# ======================================================================
# Core pipeline (runs on one dataset with pre-loaded model + SAE)
# ======================================================================
def run_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sae: SMIXAE,
    cfg: DatasetConfig,
    run_cfg: ProbeRunConfig,
    final_output_dir: str,
    dataset_name: str | None = None,
    results_json_path: str | None = None,
    run_name: str | None = None,
    model_name_for_json: str = "",
) -> None:
    """Run the full analysis pipeline for a single dataset and write an HTML report.

    Five-stage pipeline:

    1. **Collect LLM activations** — tokenise the dataset and register a forward hook
       to capture residual stream activations.
    2. **SAE encoding** — run SMIXAE on the activations in batches; filter experts
       by activity thresholds.
    3. **Evaluate experts** — score by Fisher discriminant ratio (labelled data) or
       manifold continuity (unlabelled data).
    4. **Sort** — rank experts by the chosen metric.
    5. **Plot** — generate interactive 3D Plotly scatters for the top-N experts and
       serialise everything into a single self-contained HTML file.

    Output is written to ``{final_output_dir}/{subdir}/experts.html``.

    Args:
        model: Pre-loaded HuggingFace language model.
        tokenizer: Corresponding tokenizer.
        sae: Loaded SMIXAE inference checkpoint.
        cfg: Per-dataset configuration (colours, sample count overrides, etc.).
        run_cfg: Pipeline tuning parameters (thresholds, batch sizes, sorting, etc.).
        final_output_dir: Root directory under which per-dataset sub-directories are created.
        dataset_name: HuggingFace dataset name to stream (used when ``cfg.dataframe_path``
            is empty, e.g. for the continuity pass).
        results_json_path: Optional path to a ``results.json`` to append per-expert scores to.
        run_name: Optional run identifier used as a top-level key in ``results.json``.
        model_name_for_json: Human-readable model label stored alongside results in ``results.json``.
    """
    from tqdm import tqdm

    from analysis.utils import (
        ExpertFilterConfig,
        _collect_target_cols,
        collect_activations,
        get_sae_activations,
    )

    subdir = cfg.output_subdir or Path(cfg.dataframe_path).stem
    output_dir = os.path.join(final_output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Dataset: {cfg.dataframe_path}  →  {output_dir}")
    print(f"{'=' * 60}")

    # Preliminary check (no label column → unlabeled) to override collection params.
    _will_random_sample = (not cfg.label_column) and run_cfg.sort_by == "auto"

    # ── 1. Collect LLM activations ────────────────────────────────────
    # When a dataset_name override is passed (e.g. continuity streaming pass),
    # merge it into cfg so collect_activations sees the right source.
    effective_cfg = dataclasses.replace(cfg, dataset_name=dataset_name) if dataset_name else cfg

    n_input_samples = (
        run_cfg.random_sample_n_input_samples
        if _will_random_sample
        else run_cfg.n_input_samples
    )

    batch = collect_activations(
        model=model,
        tokenizer=tokenizer,
        hook_name=run_cfg.hook_point,
        max_length=run_cfg.input_sequence_length,
        n_input_samples=n_input_samples,
        device=run_cfg.device,
        llm_batch_size=run_cfg.llm_batch_size,
        cfg=effective_cfg,
    )

    # ── 2. SAE encoding ───────────────────────────────────────────────
    if _will_random_sample:
        # Random-sample path: threshold as a fraction of max_points, not total tokens.
        max_points = run_cfg.random_sample_max_points
        min_active_tokens = max(1, int(run_cfg.random_sample_min_active_fraction * max_points))
        filter_cfg = ExpertFilterConfig(
            active_threshold=run_cfg.active_threshold,
            min_active_tokens=min_active_tokens,
            max_points=max_points,
        )
    else:
        filter_cfg = ExpertFilterConfig(
            active_threshold=run_cfg.active_threshold,
            min_active_fraction=run_cfg.min_active_fraction,
            max_points=run_cfg.max_points,
        )
    experts = get_sae_activations(
        sae=sae,
        device=run_cfg.device,
        batch=batch,
        sae_batch_size=run_cfg.sae_batch_size,
        filter_cfg=filter_cfg,
    )

    del batch.activations

    if not experts:
        print("No experts fired enough times to exceed the min_active_fraction threshold.")
        return

    # ── 3. Evaluate ───────────────────────────────────────────────────
    if batch.is_labelled:
        print(f"Evaluating Fisher score for {len(experts)} experts…")
        for expert in tqdm(experts):
            expert.evaluate_fisher()
    else:
        print(f"Evaluating manifold continuity (k={run_cfg.k_neighbors})…")
        for expert in tqdm(experts):
            expert.evaluate_manifold(k_neighbors=run_cfg.k_neighbors, device=run_cfg.device)

    # ── 4. Sort / Select (Stage 1: Fisher / continuity / random) ─────
    effective_sort_by = run_cfg.sort_by
    if run_cfg.sort_by == "auto":
        effective_sort_by = (
            ("adjusted_fisher" if run_cfg.adjusted_fisher else "fisher")
            if batch.is_labelled else "continuity"
        )

    # For unlabeled data in auto mode, randomly sample rather than ranking by
    # continuity so the visualized experts are representative, not biased toward
    # geometrically "smooth" features.  Continuity is still computed above and
    # stored in each expert for reference.
    use_random_sample = not batch.is_labelled and run_cfg.sort_by == "auto"

    if use_random_sample:
        print(f"Random sampling: {len(experts)} eligible experts (seed={run_cfg.random_seed})…")
        rng = random.Random(run_cfg.random_seed)
        n_to_plot = min(run_cfg.n_interesting_experts_to_plot, len(experts))
        top_experts = rng.sample(experts, n_to_plot)
    else:
        print(f"Sorting by {effective_sort_by} ({'ascending' if run_cfg.sort_ascending else 'descending'})…")
        experts.sort(key=lambda e: e.sort_key(effective_sort_by), reverse=not run_cfg.sort_ascending)
        n_to_plot = min(run_cfg.n_interesting_experts_to_plot, len(experts))
        top_experts = experts[:n_to_plot]

    # ── 4b. Regression probing on top-N experts (Stage 2) ────────────

    # Attach column indices to each hypothesis before running
    all_target_cols = _collect_target_cols(cfg)
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
        top_experts.sort(key=lambda e: e.sort_key("regression"), reverse=True)
        print("Re-sorted by best regression score.")

    # ── 5. Write probing-task artifacts ─────────────────────────────────
    print(f"\nBuilding probing task for top {n_to_plot} experts…")

    _log_expert_summary(top_experts, batch.n_classes, effective_sort_by)

    # Per-expert records via Expert.to_probing_record()
    expert_records: list[probing_io.ExpertRecord] = [
        expert.to_probing_record(batch.str_tokens, run_cfg.context_window_display)
        for expert in top_experts
    ]

    # Task-level colour spec (colour is not per-expert, it's per-task or per-hypothesis).
    color_spec = _build_color_spec(cfg, batch)
    label_names_int = (
        {int(k): _strip_prefix(v) for k, v in batch.label_names.items()}
        if batch.label_names else None
    )

    # Hypothesis specs + per-view rankings.
    hyp_specs, experts_by_view = _build_hypothesis_specs_and_views(
        cfg, hypotheses_with_indices, top_experts, effective_sort_by, use_random_sample, batch.is_labelled,
    )

    task_type: probing_io.TaskType = "labeled_probe" if batch.is_labelled else "unlabeled_probe"
    index = probing_io.TaskIndex(
        task_type=task_type,
        experiment_id=run_name or "",
        dataset_name=subdir,
        title=f"{subdir} — Expert Analysis",
        model_name=model_name_for_json or "",
        hook_name=run_cfg.hook_point,
        d_bottleneck=int(top_experts[0].expert_activations.shape[1]),
        n_experts_total=int(getattr(sae.cfg, "n_experts", 0)) or len(experts),
        color=color_spec,
        label_names=label_names_int,
        hypotheses=hyp_specs,
        experts_by_view=experts_by_view,
        scatter_size=5.0 if not batch.is_labelled else 1.0,
    )

    probing_io.write_probing_task(output_dir, index=index, experts=expert_records)
    print(f"Saved task: {output_dir}")

    # ── Write structured results into shared results JSON ─────────────
    # Reconstruct the legacy per_hypothesis_entries shape that _build_dataset_results expects.
    per_hypothesis_entries = _legacy_per_hypothesis_for_results_json(
        hypotheses_with_indices, top_experts,
    )
    if results_json_path and run_name:
        from analysis.utils import update_results_json
        update_results_json(
            path=results_json_path,
            run_name=run_name,
            model_name=model_name_for_json,
            hook_name=run_cfg.hook_point,
            section="probe",
            key=subdir,
            data=_build_dataset_results(top_experts, per_hypothesis_entries, effective_sort_by, hypotheses_with_indices),
        )
        print(f"Updated results JSON: {results_json_path} [{run_name}/probe/{subdir}]")


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
    min_active_fraction=typer.Option(0.10, help="Minimum fraction of tokens an expert must fire on (0–1, e.g. 0.10 = 10%)"),
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
    min_active_fraction: float = typer.Option(0.10, help="Minimum fraction of tokens an expert must fire on (0–1, e.g. 0.10 = 10%)"),
    max_points: int = typer.Option(1000, help="Max active tokens per expert (0 = no cap)"),
    random_seed: int = typer.Option(42, help="Random seed for expert sampling in unlabeled auto mode"),
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
    # ── random-sample path (unlabelled auto mode) ──
    random_sample_n_input_samples: int = typer.Option(10000, help="Number of prompts for unlabeled random-sample path"),
    random_sample_min_active_fraction: float = typer.Option(0.1, help="Min fraction of random_sample_max_points an expert must fire on (0–1)"),
    random_sample_max_points: int = typer.Option(1000, help="Max points collected per expert in random-sample path"),
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
    import torch

    from analysis.utils import DatasetConfig, load_llm, load_sae

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
    run_cfg = ProbeRunConfig(
        device=device,
        hook_point=hook_point,
        n_input_samples=n_input_samples,
        input_sequence_length=input_sequence_length,
        llm_batch_size=llm_batch_size,
        sae_batch_size=sae_batch_size,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
        adjusted_fisher=adjusted_fisher,
        k_neighbors=k_neighbors,
        active_threshold=active_threshold,
        min_active_fraction=min_active_fraction,
        max_points=max_points,
        n_interesting_experts_to_plot=n_interesting_experts_to_plot,
        context_window_display=context_window_display,
        random_seed=random_seed,
        random_sample_n_input_samples=random_sample_n_input_samples,
        random_sample_min_active_fraction=random_sample_min_active_fraction,
        random_sample_max_points=random_sample_max_points,
    )

    run_pipeline(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        cfg=cfg,
        run_cfg=run_cfg,
        final_output_dir=final_output_dir,
        dataset_name=dataset_name,
        run_name=run_hash,
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
    min_active_fraction: float = typer.Option(0.10, help="Minimum fraction of tokens an expert must fire on (0–1, e.g. 0.10 = 10%)"),
    max_points: int = typer.Option(1000, help="Max active tokens per expert (0 = no cap)"),
    random_seed: int = typer.Option(42, help="Random seed for expert sampling in unlabeled auto mode"),
    adjusted_fisher: bool = typer.Option(
        False,
        help="Sort by Fisher × (n_classes_present / n_classes) to penalise low class coverage",
    ),
    output_dir: str = typer.Option("expert_plots", help="Base directory to save the HTML plots"),
    continuity_dataset: str = typer.Option(
        "monology/pile-uncopyrighted",
        help="HuggingFace dataset to stream for the unlabelled continuity pass",
    ),
    results_json: str = typer.Option(
        "",
        help="Path to the shared results JSON (e.g. results/results.json). "
        "When set, probe results for each dataset are merged into this file keyed by run name. "
        "Leave empty to skip.",
    ),
    # ── random-sample path (unlabelled auto mode) ──
    random_sample_n_input_samples: int = typer.Option(10_000, help="Number of prompts for unlabeled random-sample path"),
    random_sample_min_active_fraction: float = typer.Option(0.1, help="Min fraction of random_sample_max_points an expert must fire on (0–1)"),
    random_sample_max_points: int = typer.Option(1000, help="Max points collected per expert in random-sample path"),
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
    import torch

    from analysis.utils import DatasetConfig, load_llm, load_sae

    config_dir = Path(datasets_config).resolve().parent
    with open(datasets_config) as f:
        raw_configs = json.load(f)

    for entry in raw_configs:
        df_path = entry.get("dataframe_path", "")
        if df_path:
            p = Path(df_path)
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

    # The run name for the shared results JSON is the checkpoint's parent dir name
    # (e.g. "gemma_2_9b_l11" for results/gemma_2_9b_l11/model).
    results_json_path: str | None = results_json.strip() or None

    # Load model and SAE once
    model, tokenizer = load_llm(base_model_name, device)
    tokenizer.padding_side = "right"
    sae = load_sae(checkpoint_path, device)

    base_run_cfg = ProbeRunConfig(
        device=device,
        hook_point=hook_point,
        n_input_samples=n_input_samples,
        input_sequence_length=input_sequence_length,
        llm_batch_size=llm_batch_size,
        sae_batch_size=sae_batch_size,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
        adjusted_fisher=adjusted_fisher,
        k_neighbors=k_neighbors,
        active_threshold=active_threshold,
        min_active_fraction=min_active_fraction,
        max_points=max_points,
        n_interesting_experts_to_plot=n_interesting_experts_to_plot,
        context_window_display=context_window_display,
        random_seed=random_seed,
        random_sample_n_input_samples=random_sample_n_input_samples,
        random_sample_min_active_fraction=random_sample_min_active_fraction,
        random_sample_max_points=random_sample_max_points,
    )

    for cfg in dataset_cfgs:
        # Apply per-dataset overrides for sample count and max_points
        per_cfg_run_cfg = dataclasses.replace(
            base_run_cfg,
            n_input_samples=cfg.n_input_samples or n_input_samples,
            max_points=cfg.max_points if cfg.max_points is not None else max_points,
        )
        run_pipeline(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            cfg=cfg,
            run_cfg=per_cfg_run_cfg,
            final_output_dir=final_output_dir,
            results_json_path=results_json_path,
            run_name=run_hash,
            model_name_for_json=base_model_name,
        )

    # Unlabelled pass (random sample by default; sort_by="continuity" for ranked mode)
    print(f"\n{'=' * 60}")
    print(f"Unlabelled pass: streaming from {continuity_dataset}")
    continuity_cfg = DatasetConfig(
        dataframe_path="",
        text_column="text",
        label_column=None,
        output_subdir="continuity",
    )
    run_pipeline(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        cfg=continuity_cfg,
        run_cfg=base_run_cfg,
        final_output_dir=final_output_dir,
        dataset_name=continuity_dataset,
        results_json_path=results_json_path,
        run_name=run_hash,
        model_name_for_json=base_model_name,
    )

    del model, sae
    torch.cuda.empty_cache()
    print("\nAll datasets processed.")


if __name__ == "__main__":
    app()
