"""smixae toy — synthetic manifold toy-model benchmark.

Four subcommands with a clean generate → train → plot workflow:

    smixae toy generate   Build the manifold zoo and eval set; save to toy_data/.
    smixae toy train      Train SMIXAE on a saved dataset; sweep k_experts.
    smixae toy plot       Regenerate HTML plots from saved zoo + results.
    smixae toy pipeline   Run generate (if needed) → train → plot in sequence.

Datasets are keyed by seed + ambient dimension + L0, stored as:

    toy_data/seed{seed}_d{d_in}_l{l0}/
        manifest.json   Scalar hyperparameters and summary.
        zoo.pt          Serialised ManifoldZoo (embedding matrices, bias).
        eval.pt         Serialised EvalData (x, feature_acts, color_param).
        results/        Populated by `toy train`.
            results.csv
            summary.json
            k{k}/model/ Saved SMIXAE inference checkpoints.
        plots/          Populated by `toy plot`.
            metrics.html
            bottlenecks.html
            all_experts_k{k}.html
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
from smixae.smixae import SMIXAETraining, SMIXAETrainingConfig
from toy.metrics import compute_metrics, compute_restricted_r2
from toy.plot import plot_all_experts_with_originals, plot_bottlenecks, plot_metrics_vs_k_experts
from toy.zoo import EvalData, ManifoldActivationGenerator, ManifoldZoo, build_manifold_zoo, generate_eval_set

app = typer.Typer(help="Synthetic manifold toy-model benchmark.")

# ── Shared helpers ─────────────────────────────────────────────────────────────

_DEFAULT_TOY_DATA = "toy_data"


def _dataset_dir(
    base: str,
    seed: int,
    d_in: int,
    l0: int,
    sigma_bias: float = 3.0,
    frac_dense: float = 0.0,
    norm_floor: float = 0.1,
) -> Path:
    return Path(base) / (
        f"seed{seed}_d{d_in}_l{l0}_b{sigma_bias:g}_fd{frac_dense:g}_nf{norm_floor:g}"
    )


def _build_smixae(
    zoo: ManifoldZoo,
    n_experts: int,
    d_expert: int,
    d_bottleneck: int,
    k_experts: int,
    device: str,
    seed: int,
) -> SMIXAETraining:
    """Instantiate a fresh SMIXAETraining.

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
        d_sae=n_experts * d_expert,
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


# ── smixae toy generate ────────────────────────────────────────────────────────

@app.command(name="generate")
def generate(
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    d_in: Annotated[int, typer.Option(help="Ambient dimension.")] = 128,
    l0: Annotated[int, typer.Option(help="Active sparse manifolds per sample.")] = 4,
    sigma_bias: Annotated[float, typer.Option(help="Global bias norm (0 = no bias).")] = 3.0,
    frac_dense: Annotated[float, typer.Option(help="Fraction of instances that are dense (always-active, origin-passing).")] = 0.0,
    norm_floor: Annotated[float, typer.Option(help="Median min active norm for sparse instances.")] = 0.1,
    norm_floor_spread: Annotated[float, typer.Option(help="Lognormal spread of the per-instance norm floor.")] = 0.5,
    eval_samples: Annotated[int, typer.Option(help="Eval set size.")] = 200_000,
    grassmannian_steps: Annotated[int, typer.Option(help="Grassmannian optimisation steps.")] = 500,
    skip_grassmannian: Annotated[bool, typer.Option(help="Skip Grassmannian opt; use random QR (debugging).")] = False,
    device: Annotated[str, typer.Option(help="Device (cuda / cpu).")] = "cuda" if torch.cuda.is_available() else "cpu",
    toy_data_dir: Annotated[str, typer.Option(help="Root directory for datasets.")] = _DEFAULT_TOY_DATA,
    force: Annotated[bool, typer.Option(help="Overwrite existing dataset.")] = False,
) -> None:
    """Build the manifold zoo and eval set; save to toy_data/.

    Skips generation if the dataset directory already exists (use --force to
    overwrite).  The Grassmannian optimisation step (~minutes on CPU) is seeded
    for reproducibility.  Use --skip-grassmannian for a fast random-QR fallback
    during debugging.
    """
    out = _dataset_dir(toy_data_dir, seed, d_in, l0, sigma_bias, frac_dense, norm_floor)

    if out.exists() and not force:
        logger.info(f"Dataset already exists at {out} — skipping (use --force to overwrite).")
        return

    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating dataset → {out.resolve()}")

    # ── Build zoo ──────────────────────────────────────────────────────────────
    logger.info("Building manifold zoo…")
    zoo = build_manifold_zoo(
        d_in=d_in,
        seed=seed,
        device=device,
        sigma_bias=sigma_bias,
        frac_dense=frac_dense,
        norm_floor=norm_floor,
        norm_floor_spread=norm_floor_spread,
        skip_grassmannian=skip_grassmannian,
        grassmannian_steps=grassmannian_steps,
    )
    n_dense = sum(inst.is_dense for inst in zoo.instances)
    logger.info(
        f"Zoo: {len(zoo.instances)} instances ({n_dense} dense / "
        f"{len(zoo.instances) - n_dense} sparse), {zoo.n_atoms} atoms, "
        f"d_in={d_in}, sigma_bias={sigma_bias}, b_global_norm={zoo.b_global.norm():.3f}"
    )

    # ── Generate eval set ──────────────────────────────────────────────────────
    logger.info(f"Generating eval set ({eval_samples:,} samples, L0={l0})…")
    eval_data = generate_eval_set(
        zoo, n_samples=eval_samples, l0=l0, seed=seed + 1, device=device
    )
    logger.info(f"Eval set ready: x shape {eval_data.x.shape}")

    # ── Save ───────────────────────────────────────────────────────────────────
    zoo.save(out / "zoo.pt")
    logger.info(f"Zoo → {out / 'zoo.pt'}")

    eval_data.save(out / "eval.pt")
    logger.info(f"Eval → {out / 'eval.pt'}")

    manifest = {
        "seed": seed,
        "d_in": d_in,
        "l0": l0,
        "sigma_bias": sigma_bias,
        "frac_dense": frac_dense,
        "norm_floor": norm_floor,
        "norm_floor_spread": norm_floor_spread,
        "n_dense": int(sum(inst.is_dense for inst in zoo.instances)),
        "eval_samples": eval_samples,
        "n_instances": len(zoo.instances),
        "n_atoms": zoo.n_atoms,
        "grassmannian_steps": 0 if skip_grassmannian else grassmannian_steps,
        "skip_grassmannian": skip_grassmannian,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"Manifest → {out / 'manifest.json'}")
    logger.info("Generation complete.")


# ── smixae toy train ───────────────────────────────────────────────────────────

@app.command(name="train")
def train(
    seed: Annotated[int, typer.Option(help="Dataset seed (used to locate the dataset).")] = 0,
    d_in: Annotated[int, typer.Option(help="Ambient dimension.")] = 128,
    l0: Annotated[int, typer.Option(help="Active sparse manifolds per sample.")] = 4,
    sigma_bias: Annotated[float, typer.Option(help="Global bias norm (used for dataset path).")] = 3.0,
    frac_dense: Annotated[float, typer.Option(help="Dense fraction (used for dataset path).")] = 0.0,
    norm_floor: Annotated[float, typer.Option(help="Norm floor (used for dataset path).")] = 0.1,
    toy_data_dir: Annotated[str, typer.Option(help="Root directory for datasets.")] = _DEFAULT_TOY_DATA,
    dataset_dir: Annotated[str, typer.Option(help="Explicit dataset path (overrides seed/d_in/l0).")] = "",
    n_experts: Annotated[int, typer.Option(help="Expert count (default 48 = ground-truth instance count).")] = 48,
    d_expert: Annotated[int, typer.Option(help="Expert latent dimension.")] = 16,
    d_bottleneck: Annotated[int, typer.Option(help="Bottleneck dimension.")] = 3,
    k_experts_list: Annotated[str, typer.Option(help="Comma-separated k_experts values.")] = "2,4,6,8,12,16",
    training_samples: Annotated[int, typer.Option(help="Total training samples per config.")] = 50_000_000,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 2048,
    lr: Annotated[float, typer.Option(help="Adam learning rate.")] = 3e-3,
    lr_warm_up_steps: Annotated[int, typer.Option(help="LR warmup steps.")] = 2_000,
    keep_all_models: Annotated[bool, typer.Option(help="Save every trained model, not just the best.")] = True,
    device: Annotated[str, typer.Option(help="Device (cuda / cpu).")] = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Train SMIXAE on a saved dataset; sweep k_experts.

    Loads zoo.pt and eval.pt from the dataset directory, runs one training run
    per k_experts value, and writes results.csv + summary.json + model checkpoints
    to a results/ subdirectory inside the dataset directory.
    """
    ds = Path(dataset_dir) if dataset_dir else _dataset_dir(toy_data_dir, seed, d_in, l0, sigma_bias, frac_dense, norm_floor)
    if not ds.exists():
        logger.error(f"Dataset not found: {ds}  (run `smixae toy generate` first)")
        raise typer.Exit(1)

    k_vals = [int(k.strip()) for k in k_experts_list.split(",")]
    out = ds / "results"
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset : {ds.resolve()}")
    logger.info(f"k_experts sweep: {k_vals}")

    # ── Load dataset ───────────────────────────────────────────────────────────
    logger.info("Loading zoo…")
    zoo = ManifoldZoo.load(ds / "zoo.pt", device=device)
    logger.info(f"Zoo: {len(zoo.instances)} instances, {zoo.n_atoms} atoms")

    logger.info("Loading eval set…")
    eval_data = EvalData.load(ds / "eval.pt", device="cpu")
    logger.info(f"Eval set: {eval_data.x.shape[0]:,} samples")

    # ── Sweep ──────────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    best_r2 = -math.inf
    best_k = k_vals[0]
    best_model: SMIXAETraining | None = None
    # best_b_experts: list[int] = []

    for k_experts in k_vals:
        logger.info(f"\n{'=' * 60}\nTraining  k_experts={k_experts}\n{'=' * 60}")

        model = _build_smixae(
            zoo=zoo,
            n_experts=n_experts,
            d_expert=d_expert,
            d_bottleneck=d_bottleneck,
            k_experts=k_experts,
            device=device,
            seed=seed,
        )

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
        r2, cofiring, b_experts = compute_restricted_r2(model, zoo, eval_data, device=device)
        metrics = compute_metrics(model, zoo, eval_data, device=device)

        mean_r2       = float(r2.mean().item())
        mean_cofiring = float(cofiring.mean().item())
        logger.info(
            f"k={k_experts}  mean R²={mean_r2:.4f}  co-fire={mean_cofiring:.3f}  "
            f"MSE={metrics['mse']:.5f}  dead={metrics['dead_experts']}"
        )

        result: dict = {
            "k_experts":       k_experts,
            "r2":              r2,
            "cofiring":        cofiring,
            "best_experts":    b_experts,
            "instances":       zoo.instances,
            "l0_ground_truth": l0,
            **metrics,
        }
        all_results.append(result)

        if keep_all_models:
            model_dir = out / f"k{k_experts}" / "model"
            model.save_inference_model(str(model_dir))
            logger.info(f"Model saved → {model_dir}")

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_k = k_experts
            best_model = model
            # best_b_experts = b_experts

    assert best_model is not None

    if not keep_all_models:
        model_dir = out / f"k{best_k}" / "model"
        best_model.save_inference_model(str(model_dir))
        logger.info(f"Best model (k={best_k}) saved → {model_dir}")

    # ── CSV ────────────────────────────────────────────────────────────────────
    csv_path = out / "results.csv"
    fieldnames = [
        "k_experts", "n_experts",
        "manifold_type", "variant_idx", "k_i", "d_i",
        "r2_linear", "cofiring_rate", "best_expert",
        "mse", "effective_l0", "dead_experts",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            k_exp = result["k_experts"]
            r2_t  = result["r2"]
            cof_t = result["cofiring"]
            for inst_idx, inst in enumerate(zoo.instances):
                writer.writerow({
                    "k_experts":     k_exp,
                    "n_experts":     n_experts,
                    "manifold_type": inst.type_name,
                    "variant_idx":   inst.variant_idx,
                    "k_i":           inst.k_i,
                    "d_i":           inst.d_i,
                    "r2_linear":     round(float(r2_t[inst_idx].item()), 6),
                    "cofiring_rate": round(float(cof_t[inst_idx].item()), 6),
                    "best_expert":   result["best_experts"][inst_idx],
                    "mse":           round(result["mse"], 6),
                    "effective_l0":  round(result["effective_l0"], 4),
                    "dead_experts":  result["dead_experts"],
                })
    logger.info(f"CSV → {csv_path}")

    # ── Summary JSON ───────────────────────────────────────────────────────────
    summary = {
        "n_experts":           n_experts,
        "d_in":                zoo.d_in,
        "l0":                  l0,
        "training_samples":    training_samples,
        "seed":                seed,
        "best_k_experts":      best_k,
        "best_mean_r2_linear": round(best_r2, 6),
        "configs": [
            {
                "k_experts":       r["k_experts"],
                "mean_r2_linear":  round(float(r["r2"].mean().item()), 6),
                "mean_cofiring":   round(float(r["cofiring"].mean().item()), 6),
                "mse":             round(r["mse"], 6),
                "effective_l0":    round(r["effective_l0"], 4),
                "dead_experts":    r["dead_experts"],
            }
            for r in all_results
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Summary JSON → {out / 'summary.json'}")
    logger.info(f"\nDone.  Best k_experts={best_k}  mean R²(n=1)={best_r2:.4f}")
    logger.info(f"Artifacts in {out.resolve()}")


# ── smixae toy eval ───────────────────────────────────────────────────────────


@app.command(name="eval")
def eval_cmd(
    seed: Annotated[int, typer.Option(help="Dataset seed (used to locate the dataset).")] = 0,
    d_in: Annotated[int, typer.Option(help="Ambient dimension.")] = 128,
    l0: Annotated[int, typer.Option(help="Active sparse manifolds per sample.")] = 4,
    sigma_bias: Annotated[float, typer.Option(help="Global bias norm (used for dataset path).")] = 3.0,
    frac_dense: Annotated[float, typer.Option(help="Dense fraction (used for dataset path).")] = 0.0,
    norm_floor: Annotated[float, typer.Option(help="Norm floor (used for dataset path).")] = 0.1,
    toy_data_dir: Annotated[str, typer.Option(help="Root directory for datasets.")] = _DEFAULT_TOY_DATA,
    dataset_dir: Annotated[str, typer.Option(help="Explicit dataset path (overrides seed/d_in/l0).")] = "",
    n_experts: Annotated[int, typer.Option(help="Expert count (default 48 = ground-truth instance count).")] = 48,
    device: Annotated[str, typer.Option(help="Device (cuda / cpu).")] = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Re-evaluate saved model checkpoints; overwrite results.csv and summary.json without retraining."""
    import json as _json

    ds = Path(dataset_dir) if dataset_dir else _dataset_dir(toy_data_dir, seed, d_in, l0, sigma_bias, frac_dense, norm_floor)
    results_dir = ds / "results"
    if not (ds / "zoo.pt").exists():
        logger.error(f"Zoo not found at {ds / 'zoo.pt'}  (run `smixae toy generate` first)")
        raise typer.Exit(1)

    # ── Load dataset ───────────────────────────────────────────────────────────
    logger.info("Loading zoo…")
    zoo = ManifoldZoo.load(ds / "zoo.pt", device=device)
    logger.info(f"Zoo: {len(zoo.instances)} instances, {zoo.n_atoms} atoms")

    logger.info("Loading eval set…")
    eval_data = EvalData.load(ds / "eval.pt", device="cpu")
    logger.info(f"Eval set: {eval_data.x.shape[0]:,} samples")

    # Read n_experts from summary.json if available
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        n_experts = _json.loads(summary_path.read_text()).get("n_experts", n_experts)

    # ── Scan for saved checkpoints ─────────────────────────────────────────────
    model_dirs = sorted(
        results_dir.glob("k*/model"),
        key=lambda p: int(p.parent.name[1:]),
    )
    if not model_dirs:
        logger.error(f"No saved checkpoints found under {results_dir}  (run `smixae toy train` first)")
        raise typer.Exit(1)

    logger.info(f"Found {len(model_dirs)} checkpoint(s): {[str(p.parent.name) for p in model_dirs]}")

    # ── Re-evaluate each checkpoint ────────────────────────────────────────────
    all_results: list[dict] = []
    best_r2 = -math.inf
    best_k = int(model_dirs[0].parent.name[1:])

    for model_dir in model_dirs:
        k_val = int(model_dir.parent.name[1:])
        logger.info(f"\n{'=' * 60}\nEvaluating k_experts={k_val}\n{'=' * 60}")

        model = SMIXAETraining.load_from_pretrained(str(model_dir)).to(device)
        model.eval()

        n_experts_actual = model.cfg.n_experts

        logger.info("Computing metrics…")
        r2, cofiring, b_experts = compute_restricted_r2(model, zoo, eval_data, device=device)
        metrics = compute_metrics(model, zoo, eval_data, device=device)

        mean_r2       = float(r2.mean().item())
        mean_cofiring = float(cofiring.mean().item())
        logger.info(
            f"k={k_val}  mean R²={mean_r2:.4f}  co-fire={mean_cofiring:.3f}  "
            f"MSE={metrics['mse']:.5f}  dead={metrics['dead_experts']}"
        )

        result: dict = {
            "k_experts":       k_val,
            "n_experts":       n_experts_actual,
            "r2":              r2,
            "cofiring":        cofiring,
            "best_experts":    b_experts,
            "instances":       zoo.instances,
            "l0_ground_truth": l0,
            **metrics,
        }
        all_results.append(result)

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_k = k_val

    # ── Write fresh results.csv ────────────────────────────────────────────────
    csv_path = results_dir / "results.csv"
    fieldnames = [
        "k_experts", "n_experts",
        "manifold_type", "variant_idx", "k_i", "d_i",
        "r2_linear", "cofiring_rate", "best_expert",
        "mse", "effective_l0", "dead_experts",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            k_exp = result["k_experts"]
            r2_t  = result["r2"]
            cof_t = result["cofiring"]
            for inst_idx, inst in enumerate(zoo.instances):
                writer.writerow({
                    "k_experts":     k_exp,
                    "n_experts":     result["n_experts"],
                    "manifold_type": inst.type_name,
                    "variant_idx":   inst.variant_idx,
                    "k_i":           inst.k_i,
                    "d_i":           inst.d_i,
                    "r2_linear":     round(float(r2_t[inst_idx].item()), 6),
                    "cofiring_rate": round(float(cof_t[inst_idx].item()), 6),
                    "best_expert":   result["best_experts"][inst_idx],
                    "mse":           round(result["mse"], 6),
                    "effective_l0":  round(result["effective_l0"], 4),
                    "dead_experts":  result["dead_experts"],
                })
    logger.info(f"CSV → {csv_path}")

    # ── Write fresh summary.json ───────────────────────────────────────────────
    summary = {
        "n_experts":           all_results[0]["n_experts"] if all_results else n_experts,
        "d_in":                zoo.d_in,
        "l0":                  l0,
        "seed":                seed,
        "best_k_experts":      best_k,
        "best_mean_r2_linear": round(best_r2, 6),
        "configs": [
            {
                "k_experts":       r["k_experts"],
                "mean_r2_linear":  round(float(r["r2"].mean().item()), 6),
                "mean_cofiring":   round(float(r["cofiring"].mean().item()), 6),
                "mse":             round(r["mse"], 6),
                "effective_l0":    round(r["effective_l0"], 4),
                "dead_experts":    r["dead_experts"],
            }
            for r in all_results
        ],
    }
    summary_path.write_text(_json.dumps(summary, indent=2))
    logger.info(f"Summary JSON → {summary_path}")
    logger.info(f"\nDone.  Best k_experts={best_k}  mean R²={best_r2:.4f}")
    logger.info(f"Artifacts in {results_dir.resolve()}")


# ── smixae toy plot ────────────────────────────────────────────────────────────

@app.command(name="plot")
def plot(
    seed: Annotated[int, typer.Option(help="Dataset seed.")] = 0,
    d_in: Annotated[int, typer.Option(help="Ambient dimension.")] = 128,
    l0: Annotated[int, typer.Option(help="Active sparse manifolds per sample.")] = 4,
    sigma_bias: Annotated[float, typer.Option(help="Global bias norm (used for dataset path).")] = 3.0,
    frac_dense: Annotated[float, typer.Option(help="Dense fraction (used for dataset path).")] = 0.0,
    norm_floor: Annotated[float, typer.Option(help="Norm floor (used for dataset path).")] = 0.1,
    toy_data_dir: Annotated[str, typer.Option(help="Root directory for datasets.")] = _DEFAULT_TOY_DATA,
    dataset_dir: Annotated[str, typer.Option(help="Explicit dataset path (overrides seed/d_in/l0).")] = "",
    device: Annotated[str, typer.Option(help="Device (cuda / cpu).")] = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Regenerate HTML plots from a saved dataset and its training results.

    Reads zoo.pt, eval.pt, and results/summary.json from the dataset directory.
    Writes plots/ inside the dataset directory without retraining.
    """
    ds = Path(dataset_dir) if dataset_dir else _dataset_dir(toy_data_dir, seed, d_in, l0, sigma_bias, frac_dense, norm_floor)
    results_dir = ds / "results"
    if not (ds / "zoo.pt").exists():
        logger.error(f"Zoo not found at {ds / 'zoo.pt'}  (run `smixae toy generate` first)")
        raise typer.Exit(1)
    if not (results_dir / "summary.json").exists():
        logger.error(f"Training results not found at {results_dir}  (run `smixae toy train` first)")
        raise typer.Exit(1)

    plots_dir = ds / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading zoo…")
    zoo = ManifoldZoo.load(ds / "zoo.pt", device=device)

    logger.info("Loading eval set…")
    eval_data = EvalData.load(ds / "eval.pt", device="cpu")

    summary = json.loads((results_dir / "summary.json").read_text())
    # best_k = summary["best_k_experts"]

    # ── Collect results dicts from CSV ─────────────────────────────────────────
    rows: list[dict] = []
    with (results_dir / "results.csv").open() as fh:
        rows = list(csv.DictReader(fh))

    # Reconstruct per-k result dicts from CSV rows (r2 tensor rebuilt from CSV)
    n_inst = len(zoo.instances)
    k_vals_seen: list[int] = []
    seen: set[int] = set()
    for row in rows:
        k = int(row["k_experts"])
        if k not in seen:
            k_vals_seen.append(k)
            seen.add(k)

    # Build a lightweight results list sufficient for plotting
    all_results: list[dict] = []
    for k_val in k_vals_seen:
        k_rows = [r for r in rows if int(r["k_experts"]) == k_val]
        r2_tensor       = torch.zeros(n_inst)
        cofiring_tensor = torch.zeros(n_inst)
        best_experts: list[int] = []
        for i, row in enumerate(k_rows):
            r2_tensor[i]       = float(row["r2_linear"])
            cofiring_tensor[i] = float(row["cofiring_rate"])
            best_experts.append(int(row["best_expert"]))
        cfg_summary = next(
            c for c in summary["configs"] if c["k_experts"] == k_val
        )
        all_results.append({
            "k_experts":       k_val,
            "r2":              r2_tensor,
            "cofiring":        cofiring_tensor,
            "best_experts":    best_experts,
            "instances":       zoo.instances,
            "l0_ground_truth": summary["l0"],
            "mse":             cfg_summary["mse"],
            "effective_l0":    cfg_summary["effective_l0"],
            "dead_experts":    cfg_summary["dead_experts"],
        })

    # ── Metrics plot ───────────────────────────────────────────────────────────
    fig_metrics = plot_metrics_vs_k_experts(all_results)
    metrics_path = plots_dir / "metrics.html"
    fig_metrics.write_html(str(metrics_path))
    logger.info(f"Metrics plot → {metrics_path}")

    # ── Per-k bottleneck and all-experts plots ─────────────────────────────────
    from smixae.smixae import SMIXAETraining as _SMIXAETraining

    for result in all_results:
        k_val    = result["k_experts"]
        r2_t     = result["r2"]
        b_experts = result["best_experts"]

        model_dir = results_dir / f"k{k_val}" / "model"
        if not model_dir.exists():
            logger.warning(f"No model checkpoint for k={k_val} at {model_dir} — skipping.")
            continue

        model = _SMIXAETraining.load_from_pretrained(str(model_dir)).to(device)
        model.eval()

        cofiring_t = result["cofiring"]
        fig_bott = plot_bottlenecks(
            model, zoo, eval_data, b_experts, r2=r2_t, cofiring=cofiring_t, device=device
        )
        bott_path = plots_dir / f"bottlenecks_k{k_val}.html"
        bott_path.write_text(fig_bott.to_html())
        logger.info(f"Bottleneck plot (k={k_val}) → {bott_path}")

        fig_all = plot_all_experts_with_originals(
            model, zoo, eval_data, b_experts, k_experts=k_val,
            r2=r2_t, cofiring=cofiring_t, device=device
        )
        all_path = plots_dir / f"all_experts_k{k_val}.html"
        all_path.write_text(fig_all.to_html())
        logger.info(f"All-experts plot (k={k_val}) → {all_path}")

    logger.info(f"Plots in {plots_dir.resolve()}")


# ── smixae toy pipeline ────────────────────────────────────────────────────────

@app.command(name="pipeline")
def pipeline(
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    d_in: Annotated[int, typer.Option(help="Ambient dimension.")] = 128,
    l0: Annotated[int, typer.Option(help="Active sparse manifolds per sample.")] = 4,
    sigma_bias: Annotated[float, typer.Option(help="Global bias norm.")] = 3.0,
    frac_dense: Annotated[float, typer.Option(help="Fraction of instances that are dense.")] = 0.0,
    norm_floor: Annotated[float, typer.Option(help="Median min active norm for sparse instances.")] = 0.1,
    norm_floor_spread: Annotated[float, typer.Option(help="Lognormal spread of the per-instance norm floor.")] = 0.5,
    eval_samples: Annotated[int, typer.Option(help="Eval set size.")] = 200_000,
    grassmannian_steps: Annotated[int, typer.Option(help="Grassmannian optimisation steps.")] = 500,
    skip_grassmannian: Annotated[bool, typer.Option(help="Skip Grassmannian opt.")] = False,
    n_experts: Annotated[int, typer.Option(help="Expert count.")] = 48,
    d_expert: Annotated[int, typer.Option(help="Expert latent dimension.")] = 16,
    d_bottleneck: Annotated[int, typer.Option(help="Bottleneck dimension.")] = 3,
    k_experts_list: Annotated[str, typer.Option(help="Comma-separated k_experts values.")] = "2,4,6,8,12,16",
    training_samples: Annotated[int, typer.Option(help="Total training samples per config.")] = 50_000_000,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 2048,
    lr: Annotated[float, typer.Option(help="Adam learning rate.")] = 3e-3,
    lr_warm_up_steps: Annotated[int, typer.Option(help="LR warmup steps.")] = 2_000,
    keep_all_models: Annotated[bool, typer.Option(help="Save all models.")] = True,
    device: Annotated[str, typer.Option(help="Device (cuda / cpu).")] = "cuda" if torch.cuda.is_available() else "cpu",
    toy_data_dir: Annotated[str, typer.Option(help="Root directory for datasets.")] = _DEFAULT_TOY_DATA,
    force_generate: Annotated[bool, typer.Option(help="Re-generate even if dataset exists.")] = False,
) -> None:
    """Run generate (if needed) → train → plot in sequence.

    Generation is skipped if the dataset directory already exists for the given
    seed/d_in/l0, unless --force-generate is set.
    """
    ds = _dataset_dir(toy_data_dir, seed, d_in, l0, sigma_bias, frac_dense, norm_floor)

    # ── Generate (skip if already exists) ─────────────────────────────────────
    if ds.exists() and not force_generate:
        logger.info(f"Dataset already exists at {ds} — skipping generation.")
    else:
        logger.info("Running: smixae toy generate")
        generate(
            seed=seed,
            d_in=d_in,
            l0=l0,
            sigma_bias=sigma_bias,
            frac_dense=frac_dense,
            norm_floor=norm_floor,
            norm_floor_spread=norm_floor_spread,
            eval_samples=eval_samples,
            grassmannian_steps=grassmannian_steps,
            skip_grassmannian=skip_grassmannian,
            device=device,
            toy_data_dir=toy_data_dir,
            force=force_generate,
        )

    # ── Train ──────────────────────────────────────────────────────────────────
    logger.info("Running: smixae toy train")
    train(
        seed=seed,
        d_in=d_in,
        l0=l0,
        sigma_bias=sigma_bias,
        frac_dense=frac_dense,
        norm_floor=norm_floor,
        toy_data_dir=toy_data_dir,
        dataset_dir="",
        n_experts=n_experts,
        d_expert=d_expert,
        d_bottleneck=d_bottleneck,
        k_experts_list=k_experts_list,
        training_samples=training_samples,
        batch_size=batch_size,
        lr=lr,
        lr_warm_up_steps=lr_warm_up_steps,
        keep_all_models=keep_all_models,
        device=device,
    )

    # ── Plot ───────────────────────────────────────────────────────────────────
    logger.info("Running: smixae toy plot")
    plot(
        seed=seed,
        d_in=d_in,
        l0=l0,
        sigma_bias=sigma_bias,
        frac_dense=frac_dense,
        norm_floor=norm_floor,
        toy_data_dir=toy_data_dir,
        dataset_dir="",
        device=device,
    )
