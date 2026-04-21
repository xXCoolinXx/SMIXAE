"""Patch old SMIXAE final-model checkpoints for the correct fold.

Two bugs existed in fold_activation_norm_scaling_factor for models saved before
the fixes were applied:

  1. The threshold was not scaled at all (pre-c1127f7).  The first pass of this
     script corrected that by setting threshold /= scaling_factor.

  2. With rescale_acts_by_decoder_norm=True the base fold (W_dec /= sf) enlarges
     effective_decoder_norm by 1/sf.  smixae_encode then multiplies the bottleneck
     by this inflated norm, so the decode sees a 1/sf-enlarged bottleneck AND a
     1/sf-enlarged W_dec — output scale x/sf instead of x.  The correct fold is
     W_dec /= sqrt(sf) (net 1/sf combined), so W_dec needs to be multiplied back
     by sqrt(sf).  Threshold must match the corrected decoder norm, so it also
     needs to be multiplied by sqrt(sf) (going from threshold_train/sf to
     threshold_train/sqrt(sf)).

This script applies both corrections to results/{exp}/model/sae_weights.safetensors
in-place.  Training checkpoints are never touched.
"""

import json
import pathlib

import torch
from safetensors import safe_open
from safetensors.torch import save_file

RESULTS = pathlib.Path("results")
EXPERIMENTS = [
    "gemma_2_9b_l11",
    "gemma_2_9b_l20",
    "gemma_2_2b_l12",
]


def _earliest_step_dir(checkpoints_root: pathlib.Path) -> pathlib.Path:
    """Return the directory of the numerically earliest step checkpoint."""
    run_dir = next(checkpoints_root.iterdir())  # single wandb run sub-dir
    step_dirs = [d for d in run_dir.iterdir() if not d.name.startswith("final")]
    return sorted(step_dirs, key=lambda d: int(d.name))[0]


def patch_experiment(exp: str) -> None:
    """Apply W_dec and threshold corrections to the final model."""
    exp_dir = RESULTS / exp

    # ── Load scaling_factor from earliest intermediate checkpoint ──────────
    earliest = _earliest_step_dir(exp_dir / "checkpoints")
    scaler_file = earliest / "activation_scaler.json"
    with scaler_file.open() as f:
        scaling_factor = json.load(f)["scaling_factor"]
    if scaling_factor is None:
        raise RuntimeError(f"{exp}: scaling_factor is null even in early checkpoint {earliest}")

    sf_sqrt = scaling_factor**0.5

    # ── Read all tensors from the final model ─────────────────────────────
    model_path = exp_dir / "model" / "sae_weights.safetensors"
    tensors: dict = {}
    with safe_open(str(model_path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    old_threshold = tensors["threshold"].item()
    # threshold is currently threshold_train / sf (from first patch pass).
    # Correct value is threshold_train / sqrt(sf) = current * sqrt(sf).
    new_threshold = old_threshold * sf_sqrt
    tensors["threshold"] = tensors["threshold"].clone().fill_(new_threshold)

    # W_dec is currently W_dec_train / sf (from the base fold).
    # Correct value is W_dec_train / sqrt(sf) = current * sqrt(sf).
    tensors["W_dec"] = tensors["W_dec"].clone() * torch.tensor(sf_sqrt, dtype=tensors["W_dec"].dtype)

    print(
        f"{exp}: sf={scaling_factor:.6f}  sqrt(sf)={sf_sqrt:.6f}\n"
        f"  threshold {old_threshold:.6f} → {new_threshold:.6f}\n"
        f"  W_dec scaled ×{sf_sqrt:.6f}  (shape {tensors['W_dec'].shape})"
    )

    # ── Atomically overwrite ──────────────────────────────────────────────
    tmp = model_path.with_suffix(".tmp")
    try:
        save_file(tensors, str(tmp))
        tmp.replace(model_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"  → wrote {model_path}")


def main() -> None:
    """Patch all experiment models."""
    for exp in EXPERIMENTS:
        patch_experiment(exp)
    print("Done.")


if __name__ == "__main__":
    main()
