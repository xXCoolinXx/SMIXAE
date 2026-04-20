r"""CLI command for core SAE evaluation.

Evaluates a single SAE and writes results to a JSON file.  The file is
organised as ``model → layer → SAE name → metrics`` and entries with the
same name are overwritten on re-run.

Supports two ways to specify the SAE:

  1. **Local checkpoint** (default): pass a path as a positional argument.
  2. **HuggingFace**: pass ``--hf-release`` and ``--hf-sae-id`` to load
     via SAELens ``SAE.from_pretrained`` (e.g. GemmaScope or any other
     SAE hosted on HuggingFace).

Usage examples::

    # Evaluate a local SMIXAE checkpoint
    smixae core results/gemma_2_9b_l11/model \\
        --base-model-name google/gemma-2-9b \\
        --hook-point model.layers.11 \\
        --display-name SMIXAE

    # Evaluate a GemmaScope SAE from HuggingFace
    smixae core \\
        --hf-release gemma-scope-9b-pt-res \\
        --hf-sae-id layer_11/width_16k/average_l0_118 \\
        --base-model-name google/gemma-2-9b \\
        --hook-point model.layers.11 \\
        --display-name "GemmaScope 9B 16k (L0=118)"
"""

from pathlib import Path

import typer

from analysis.core_eval import (
    CORE_EVAL_RESULTS_PATH,
    CoreEvalConfig,
    load_llm_for_eval,
    load_sae_from_hf,
    load_sae_from_path,
    run_single_eval,
    save_core_eval_results,
)

app = typer.Typer()

_DEFAULT_DATASET = "Skylion007/openwebtext"


@app.command()
def core(
    checkpoint_path: Path = typer.Argument(
        None,
        help="Path to a local SAE checkpoint directory.  Mutually exclusive with --hf-release.",
    ),
    hf_release: str = typer.Option(
        "",
        help="HuggingFace release name for SAELens SAE.from_pretrained "
        "(e.g. 'gemma-scope-9b-pt-res').  Requires --hf-sae-id.",
    ),
    hf_sae_id: str = typer.Option(
        "",
        help="SAELens sae_id for HuggingFace download "
        "(e.g. 'layer_11/width_16k/average_l0_118').  Requires --hf-release.",
    ),
    base_model_name: str = typer.Option(
        ...,
        help="HuggingFace model name, e.g. google/gemma-2-9b.",
    ),
    hook_point: str = typer.Option(
        ...,
        help="Hook point in HuggingFace notation, e.g. model.layers.11.",
    ),
    display_name: str = typer.Option(
        "",
        help="Human-readable name for this SAE in the results JSON. "
        "Defaults to the last path component or the hf-sae-id.",
    ),
    output_json: Path = typer.Option(
        CORE_EVAL_RESULTS_PATH,
        help="Output JSON path.",
    ),
    device: str = typer.Option("cuda", help="Torch device."),
    dtype: str = typer.Option("bfloat16", help="Dtype for LLM activations."),
    n_reconstruction_batches: int = typer.Option(200, help="Batches for CE-loss metrics."),
    n_sparsity_batches: int = typer.Option(2000, help="Batches for L0/MSE/EVR metrics."),
    batch_size: int = typer.Option(16, help="Sequences per batch."),
    context_size: int = typer.Option(128, help="Token context length."),
    dataset: str = typer.Option(_DEFAULT_DATASET, help="HuggingFace dataset."),
    verbose: bool = typer.Option(False, help="Show per-batch progress bars."),
) -> None:
    """Evaluate a single SAE on core metrics."""
    cfg = CoreEvalConfig(
        dataset=dataset,
        context_size=context_size,
        n_reconstruction_batches=n_reconstruction_batches,
        n_sparsity_batches=n_sparsity_batches,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )

    from_hf = bool(hf_release)
    from_path = checkpoint_path is not None

    if from_hf == from_path:
        typer.echo(
            "Specify exactly one SAE source: either a checkpoint path as a "
            "positional argument, or --hf-release + --hf-sae-id.",
            err=True,
        )
        raise typer.Exit(1)

    if from_hf and not hf_sae_id:
        typer.echo("--hf-release requires --hf-sae-id.", err=True)
        raise typer.Exit(1)

    if from_hf:
        name = display_name or hf_sae_id
    else:
        if not checkpoint_path.exists():
            typer.echo(f"Checkpoint not found: {checkpoint_path}", err=True)
            raise typer.Exit(1)
        name = display_name or checkpoint_path.name

    layer = int(hook_point.split(".")[-1])
    layer_key = f"layer_{layer}"

    llm, tokenizer = load_llm_for_eval(base_model_name, cfg)

    if from_hf:
        sae = load_sae_from_hf(hf_release, hf_sae_id, cfg.device)
    else:
        sae = load_sae_from_path(str(checkpoint_path), cfg.device)

    metrics = run_single_eval(sae, llm, tokenizer, hook_point, cfg, verbose=verbose)

    save_core_eval_results(
        {base_model_name: {layer_key: {name: metrics}}},
        output_json,
    )
    typer.echo(f"{name}: {metrics}")
    typer.echo(f"Results written to {output_json}")
