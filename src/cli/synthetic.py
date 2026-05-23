"""smixae synthetic — train and evaluate SMIXAE on the manifold toy-model benchmark.

Trains one SMIXAE per k_experts value in a sweep (n_experts fixed at the
ground-truth manifold count of 42), evaluates per-manifold R²(n=1,2,3) and
overall reconstruction quality, and emits a CSV + HTML plot bundle.

Training uses sae_lens.synthetic.train_toy_sae, which handles the optimizer,
lr scheduler, and train loop via SAETrainer.  The data source is a
ManifoldActivationGenerator (signed manifold coordinates, exact L0) combined
with the zoo's FeatureDictionary (ambient projection x = features @ V).
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Annotated

import torch
import typer
from loguru import logger
from sae_lens.saes.sae import SAEMetadata
from sae_lens.synthetic import train_toy_sae

import smixae  # noqa: F401 — registers "smixae" architecture with SAELens
from analysis.synthetic import (
    ManifoldActivationGenerator,
    ManifoldZoo,
    build_manifold_zoo,
    compute_metrics,
    compute_restricted_r2,
    generate_eval_set,
    plot_all_experts_with_originals,
    plot_bottlenecks,
    plot_metrics_vs_k_experts,
)
from smixae.smixae import SMIXAETraining, SMIXAETrainingConfig


def _build_smixae(
    zoo: ManifoldZoo,
    n_experts: int,
    d_expert: int,
    d_bottleneck: int,
    k_experts: int,
    device: str,
    seed: int,
) -> SMIXAETraining:
    """Instantiate a fresh SMIXAETraining with the given architecture config.

    Args:
        zoo: ManifoldZoo (provides d_in).
        n_experts: Total expert count.
        d_expert: Expert latent dimension.
        d_bottleneck: Bottleneck dimension (3 for 3-D visualisation).
        k_experts: Active experts per sample (BatchTopK budget).
        device: Device string.
        seed: Random seed for weight initialisation.

    Returns:
        Freshly initialised SMIXAETraining on device.
    """
    torch.manual_seed(seed)
    cfg = SMIXAETrainingConfig(
        d_in=zoo.d_in,
        d_sae=n_experts * d_expert,   # overwritten by SMIXAETraining.__init__
        n_experts=n_experts,
        d_expert=d_expert,
        d_bottleneck=d_bottleneck,
        k_experts=k_experts,
        normalize_activations="none",
        apply_b_dec_to_input=False,
        device=device,
        dead_after_n_passes=200,
        metadata=SAEMetadata(model_name="synthetic_toy", hook_name="ambient"),
    )
    return SMIXAETraining(cfg).to(device)


def synthetic(
    # Data
    d_in: Annotated[int, typer.Option(help="Ambient dimension.")] = 128,
    l0: Annotated[int, typer.Option(help="Active manifolds per sample.")] = 4,
    eval_samples: Annotated[int, typer.Option(help="Eval set size.")] = 200_000,
    # Architecture (n_experts fixed at ground-truth manifold count)
    n_experts: Annotated[int, typer.Option(help="Expert count (default=42 = ground-truth manifold instances).")] = 42,
    d_expert: Annotated[int, typer.Option(help="Expert latent dimension.")] = 16,
    d_bottleneck: Annotated[int, typer.Option(help="Bottleneck dimension.")] = 3,
    # Sweep
    k_experts_list: Annotated[str, typer.Option(help="Comma-separated k_experts values to sweep.")] = "2,4,6,8,12,16",
    # Training (passed to train_toy_sae)
    training_samples: Annotated[int, typer.Option(help="Total training samples per config.")] = 50_000_000,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 2048,
    lr: Annotated[float, typer.Option(help="Adam learning rate.")] = 3e-3,
    lr_warm_up_steps: Annotated[int, typer.Option(help="LR warmup steps (linear ramp from 0 to lr).")] = 2_000,
    # Output
    output_dir: Annotated[str, typer.Option(help="Output directory for all artifacts.")] = "results/synthetic",
    keep_all_models: Annotated[bool, typer.Option(help="Save every trained model, not just the best.")] = False,
    # Misc
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    device: Annotated[str, typer.Option(help="Device (cuda / cpu).")] = "cuda" if torch.cuda.is_available() else "cpu",
    sigma_bias: Annotated[float, typer.Option(help="Amplitude of per-instance bias atoms (0 = no bias).")] = 3.0,
) -> None:
    """Train SMIXAE on the synthetic manifold benchmark and evaluate manifold recovery.

    Sweeps k_experts with n_experts fixed at the ground-truth instance count (42).
    Outputs results.csv, metrics.html, best_model_bottlenecks.html, and summary.json.
    """
    k_vals = [int(k.strip()) for k in k_experts_list.split(",")]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output dir : {out_path.resolve()}")
    logger.info(f"k_experts sweep: {k_vals}")
    logger.info(f"Device: {device}")

    # ── Build shared zoo and eval set ─────────────────────────────────────────
    logger.info("Building manifold zoo…")
    zoo = build_manifold_zoo(d_in=d_in, seed=seed, device=device, sigma_bias=sigma_bias)
    logger.info(
        f"Zoo: {len(zoo.instances)} instances, {zoo.n_atoms} atoms, d_in={d_in}"
    )

    logger.info(f"Generating eval set ({eval_samples:,} samples, L0={l0})…")
    eval_data = generate_eval_set(
        zoo, n_samples=eval_samples, l0=l0, seed=seed + 1, device=device
    )
    logger.info("Eval set ready.")

    # ── Sweep over k_experts ──────────────────────────────────────────────────
    all_results: list[dict] = []
    best_r2    = -math.inf
    best_k     = k_vals[0]
    best_model: SMIXAETraining | None = None
    best_b_experts: list[int] = []

    for k_experts in k_vals:
        logger.info(f"\n{'='*60}\nTraining  k_experts={k_experts}\n{'='*60}")

        model = _build_smixae(
            zoo=zoo,
            n_experts=n_experts,
            d_expert=d_expert,
            d_bottleneck=d_bottleneck,
            k_experts=k_experts,
            device=device,
            seed=seed,
        )

        # Data source: ManifoldActivationGenerator + zoo.feature_dict
        manifold_ag = ManifoldActivationGenerator(zoo=zoo, l0=l0, device=device)

        t0 = time.time()
        train_toy_sae(
            sae=model,
            feature_dict=zoo.feature_dict,
            activations_generator=manifold_ag,
            training_samples=training_samples,
            batch_size=batch_size,
            lr=lr,
            lr_warm_up_steps=lr_warm_up_steps,
            device=device,
        )
        logger.info(f"Training done in {time.time() - t0:.1f}s")

        logger.info("Computing metrics…")
        r2, b_experts = compute_restricted_r2(
            model, zoo, eval_data, device=device, max_n=3
        )
        metrics = compute_metrics(model, zoo, eval_data, device=device)

        mean_r2_n1 = float(r2[:, 0].mean().item())
        logger.info(
            f"k={k_experts}  mean R²(n=1)={mean_r2_n1:.4f}  "
            f"MSE={metrics['mse']:.5f}  dead={metrics['dead_experts']}"
        )

        result: dict = {
            "k_experts":        k_experts,
            "r2":               r2,
            "best_experts":     b_experts,
            "instances":        zoo.instances,
            "l0_ground_truth":  l0,
            **metrics,
        }
        all_results.append(result)

        # Per-config: all-experts plot with original manifold overlay
        fig_all  = plot_all_experts_with_originals(
            model, zoo, eval_data, b_experts, k_experts=k_experts, device=device
        )
        all_path = out_path / f"all_experts_k{k_experts}.html"
        fig_all.write_html(str(all_path))
        logger.info(f"All-experts plot → {all_path}")

        if keep_all_models:
            model_dir = out_path / "models" / f"k_experts={k_experts}" / "model"
            model.save_inference_model(str(model_dir))
            logger.info(f"Model saved → {model_dir}")

        if mean_r2_n1 > best_r2:
            best_r2       = mean_r2_n1
            best_k        = k_experts
            best_model    = model
            best_b_experts = b_experts

        if not keep_all_models:
            # Drop non-best models to free memory; best_model reference kept
            pass

    # ── Save best model ───────────────────────────────────────────────────────
    assert best_model is not None
    best_model_dir = out_path / "models" / f"k_experts={best_k}_best" / "model"
    best_model.save_inference_model(str(best_model_dir))
    logger.info(f"Best model (k={best_k}) saved → {best_model_dir}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = out_path / "results.csv"
    fieldnames = [
        "k_experts", "n_experts",
        "manifold_type", "variant_idx", "k_i", "d_i",
        "r2_n1", "r2_n2", "r2_n3", "best_expert",
        "mse", "effective_l0", "dead_experts",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            k_exp  = result["k_experts"]
            r2_t   = result["r2"]
            for inst_idx, inst in enumerate(zoo.instances):
                writer.writerow({
                    "k_experts":     k_exp,
                    "n_experts":     n_experts,
                    "manifold_type": inst.type_name,
                    "variant_idx":   inst.variant_idx,
                    "k_i":           inst.k_i,
                    "d_i":           inst.d_i,
                    "r2_n1":         round(float(r2_t[inst_idx, 0].item()), 6),
                    "r2_n2":         round(float(r2_t[inst_idx, 1].item()), 6),
                    "r2_n3":         round(float(r2_t[inst_idx, 2].item()), 6),
                    "best_expert":   result["best_experts"][inst_idx],
                    "mse":           round(result["mse"], 6),
                    "effective_l0":  round(result["effective_l0"], 4),
                    "dead_experts":  result["dead_experts"],
                })
    logger.info(f"CSV → {csv_path}")

    # ── Write summary JSON ────────────────────────────────────────────────────
    summary = {
        "n_experts":          n_experts,
        "d_in":               d_in,
        "l0":                 l0,
        "training_samples":   training_samples,
        "seed":               seed,
        "best_k_experts":     best_k,
        "best_mean_r2_n1":    round(best_r2, 6),
        "configs": [
            {
                "k_experts":    r["k_experts"],
                "mean_r2_n1":   round(float(r["r2"][:, 0].mean().item()), 6),
                "mean_r2_n2":   round(float(r["r2"][:, 1].mean().item()), 6),
                "mean_r2_n3":   round(float(r["r2"][:, 2].mean().item()), 6),
                "mse":          round(r["mse"], 6),
                "effective_l0": round(r["effective_l0"], 4),
                "dead_experts": r["dead_experts"],
            }
            for r in all_results
        ],
    }
    summary_path = out_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Summary JSON → {summary_path}")

    # ── Metric plot ───────────────────────────────────────────────────────────
    fig_metrics  = plot_metrics_vs_k_experts(all_results)
    metrics_path = out_path / "metrics.html"
    fig_metrics.write_html(str(metrics_path))
    logger.info(f"Metrics plot → {metrics_path}")

    # ── Bottleneck plot (best model) ──────────────────────────────────────────
    logger.info(f"Rendering bottleneck figure for best model (k_experts={best_k})…")
    fig_bott  = plot_bottlenecks(best_model, zoo, eval_data, best_b_experts, device=device)
    bott_path = out_path / "best_model_bottlenecks.html"
    fig_bott.write_html(str(bott_path))
    logger.info(f"Bottleneck plot → {bott_path}")

    logger.info(f"\nDone.  Best k_experts={best_k}  mean R²(n=1)={best_r2:.4f}")
    logger.info(f"All artifacts in {out_path.resolve()}")
