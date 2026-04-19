"""CLI commands for SAEBench core evaluation.

Exposes two subcommands under ``smixae saebench``:

  run-all    Evaluate all experiments listed in results.json against
             their GemmaScope baselines and write saebench_results.json.

  run-single Evaluate one SMIXAE checkpoint (plus optional baseline).
"""

from pathlib import Path

import typer

from analysis.saebench_core import (
    SAEBENCH_RESULTS_PATH,
    SAEBenchCoreConfig,
    evaluate_experiment,
    run_all_from_results_json,
    save_saebench_results,
)

app = typer.Typer()

_RESULTS_JSON = Path("results/results.json")
_CHECKPOINT_BASE = Path("results")
_DEFAULT_DATASET = "Skylion007/openwebtext"


@app.command(name="run-all")
def run_all(
    results_json: Path = typer.Option(_RESULTS_JSON, help="Path to results.json listing all experiments."),
    checkpoint_base_dir: Path = typer.Option(_CHECKPOINT_BASE, help="Root directory containing per-experiment model/ subdirs."),
    output_json: Path = typer.Option(SAEBENCH_RESULTS_PATH, help="Output JSON path for SAEBench results."),
    device: str = typer.Option("cuda", help="Torch device."),
    dtype: str = typer.Option("bfloat16", help="Dtype for LLM activations."),
    n_reconstruction_batches: int = typer.Option(200, help="Batches for CE-loss metrics."),
    n_sparsity_batches: int = typer.Option(2000, help="Batches for L0/MSE/EVR metrics."),
    batch_size: int = typer.Option(16, help="Sequences per batch."),
    context_size: int = typer.Option(128, help="Token context length."),
    dataset: str = typer.Option(_DEFAULT_DATASET, help="HuggingFace dataset for evaluation."),
    force_rerun: bool = typer.Option(False, help="Re-evaluate even if results already exist."),
    verbose: bool = typer.Option(False, help="Show per-batch progress bars."),
) -> None:
    """Evaluate all SMIXAE experiments in results.json and their GemmaScope baselines."""
    cfg = SAEBenchCoreConfig(
        dataset=dataset,
        context_size=context_size,
        n_reconstruction_batches=n_reconstruction_batches,
        n_sparsity_batches=n_sparsity_batches,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )

    run_all_from_results_json(
        results_json_path=results_json,
        checkpoint_base_dir=checkpoint_base_dir,
        cfg=cfg,
        output_path=output_json,
        force_rerun=force_rerun,
        verbose=verbose,
    )
    typer.echo(f"Results written to {output_json}")


@app.command(name="run-single")
def run_single(
    checkpoint_path: Path = typer.Argument(..., help="Path to the SMIXAE model/ directory."),
    base_model_name: str = typer.Option(..., help="HuggingFace model name, e.g. google/gemma-2-9b."),
    hook_point: str = typer.Option(..., help="Hook point in HuggingFace notation, e.g. model.layers.11."),
    output_json: Path = typer.Option(SAEBENCH_RESULTS_PATH, help="Output JSON path."),
    gemmascope_release: str = typer.Option("", help="SAELens release for GemmaScope baseline (empty = skip)."),
    gemmascope_sae_id: str = typer.Option("", help="SAELens sae_id for GemmaScope baseline (empty = use default)."),
    gemmascope_display_name: str = typer.Option("", help="Human-readable name for GemmaScope baseline."),
    device: str = typer.Option("cuda", help="Torch device."),
    dtype: str = typer.Option("bfloat16", help="Dtype for LLM activations."),
    n_reconstruction_batches: int = typer.Option(200, help="Batches for CE-loss metrics."),
    n_sparsity_batches: int = typer.Option(2000, help="Batches for L0/MSE/EVR metrics."),
    batch_size: int = typer.Option(16, help="Sequences per batch."),
    context_size: int = typer.Option(128, help="Token context length."),
    dataset: str = typer.Option(_DEFAULT_DATASET, help="HuggingFace dataset."),
    verbose: bool = typer.Option(False, help="Show per-batch progress bars."),
) -> None:
    """Evaluate a single SMIXAE checkpoint and optionally a GemmaScope baseline."""
    cfg = SAEBenchCoreConfig(
        dataset=dataset,
        context_size=context_size,
        n_reconstruction_batches=n_reconstruction_batches,
        n_sparsity_batches=n_sparsity_batches,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )

    layer = int(hook_point.split(".")[-1])
    layer_key = f"layer_{layer}"

    # Build explicit baseline list if release is provided.
    baseline_sae_ids: dict[str, tuple[str, str]] | None = None
    if gemmascope_release:
        if not gemmascope_sae_id:
            from analysis.saebench_core import GEMMASCOPE_BASELINES
            defaults = GEMMASCOPE_BASELINES.get(base_model_name, {})
            if layer in defaults:
                _, sae_id = defaults[layer]
                gemmascope_sae_id = sae_id
            else:
                typer.echo(f"No default GemmaScope SAE ID for {base_model_name} layer {layer}. "
                           "Provide --gemmascope-sae-id.", err=True)
                raise typer.Exit(1)
        display = gemmascope_display_name or f"GemmaScope 16k ({gemmascope_sae_id})"
        baseline_sae_ids = {display: (gemmascope_release, gemmascope_sae_id)}

    layer_results = evaluate_experiment(
        experiment_name=str(checkpoint_path),
        model_name=base_model_name,
        hook_name=hook_point,
        checkpoint_path=str(checkpoint_path),
        cfg=cfg,
        baseline_sae_ids=baseline_sae_ids,
        verbose=verbose,
    )

    save_saebench_results({base_model_name: {layer_key: layer_results}}, output_json)
    typer.echo(f"Results written to {output_json}")
