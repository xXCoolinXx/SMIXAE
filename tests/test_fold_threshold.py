"""Unit tests for fold_activation_norm_scaling_factor threshold correction.

The training-class fold is the primary code path: SAELens calls
fold_activation_norm_scaling_factor on the SMIXAETraining instance at the
end of training (sae_trainer.py), then saves inference weights via
save_inference_model.  The inference-class override handles the load-time
fallback (sae.py) when a checkpoint was saved without pre-folding.

Invariants under fold with scaling_factor sf (= 1/mean_activation_norm, < 1):

  threshold_after == threshold_before / sf
  threshold / effective_decoder_norm ratio is unchanged
  For x_raw = x_train / sf, SMIXAE.encode() returns the same zero/nonzero
    pattern with values scaled by 1/sf

The gating invariant holds exactly (not approximately) because:
  x_raw @ W_enc_fold = (x_train / sf) @ (W_enc * sf) = x_train @ W_enc
  b_enc is unchanged, so hidden_pre_latent is identical before and after fold
  LeakyReLU commutes with positive scaling, so h_latent is identical
  effective_decoder_norm_fold = N / sf  →  bottleneck norms scale by 1/sf
  threshold_fold = t / sf  →  comparison  (norm/sf > t/sf)  ↔  (norm > t)
"""

import torch
import pytest
from smixae.smixae import SMIXAE, SMIXAEConfig, SMIXAETraining, SMIXAETrainingConfig, smixae_encode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_training_model(seed: int = 0) -> SMIXAETraining:
    torch.manual_seed(seed)
    cfg = SMIXAETrainingConfig(
        n_experts=8,
        d_expert=4,
        d_bottleneck=2,
        k_experts=4,
        d_in=16,
        d_sae=32,
        rescale_acts_by_decoder_norm=True,
    )
    return SMIXAETraining(cfg)


def _make_inference_model(seed: int = 0) -> SMIXAE:
    torch.manual_seed(seed)
    cfg = SMIXAEConfig(
        n_experts=8,
        d_expert=4,
        d_bottleneck=2,
        d_in=16,
        d_sae=32,
        rescale_acts_by_decoder_norm=True,
    )
    return SMIXAE(cfg)


# ---------------------------------------------------------------------------
# Threshold value
# ---------------------------------------------------------------------------

class TestThresholdValueAfterFold:
    """threshold must equal threshold_before / scaling_factor exactly."""

    def test_training_class(self):
        model = _make_training_model()
        sf = 0.02
        model.threshold = torch.tensor(0.5, dtype=torch.double)
        model.fold_activation_norm_scaling_factor(sf)
        assert abs(model.threshold.item() - 0.5 / sf) < 1e-9

    def test_inference_class(self):
        model = _make_inference_model()
        sf = 0.02
        model.threshold = torch.tensor(0.5, dtype=torch.double)
        model.fold_activation_norm_scaling_factor(sf)
        assert abs(model.threshold.item() - 0.5 / sf) < 1e-9

    @pytest.mark.parametrize("sf", [0.01, 0.05, 0.1, 0.5])
    def test_various_scaling_factors(self, sf: float):
        model = _make_training_model(seed=42)
        t0 = 0.3
        model.threshold = torch.tensor(t0, dtype=torch.double)
        model.fold_activation_norm_scaling_factor(sf)
        assert abs(model.threshold.item() - t0 / sf) < 1e-9, f"failed for sf={sf}"

    def test_zero_threshold_stays_zero(self):
        """Threshold of 0 must remain 0 after fold."""
        model = _make_inference_model()
        model.threshold = torch.tensor(0.0, dtype=torch.double)
        model.fold_activation_norm_scaling_factor(0.02)
        assert model.threshold.item() == 0.0


# ---------------------------------------------------------------------------
# threshold / effective_decoder_norm ratio
# ---------------------------------------------------------------------------

class TestThresholdNormRatioPreserved:
    """threshold / effective_decoder_norm must be invariant to fold.

    The fold divides W_dec by sf, which scales effective_decoder_norm by 1/sf.
    The threshold fix divides threshold by sf as well.  Their ratio is preserved.
    """

    def test_mean_ratio_preserved(self):
        model = _make_inference_model(seed=7)
        sf = 0.05
        model.threshold = torch.tensor(1.0, dtype=torch.double)

        N_before = model.effective_decoder_norm.mean().item()
        t_before = model.threshold.item()

        model.fold_activation_norm_scaling_factor(sf)

        N_after = model.effective_decoder_norm.mean().item()
        t_after = model.threshold.item()

        ratio_before = t_before / N_before
        ratio_after = t_after / N_after
        assert abs(ratio_after / ratio_before - 1.0) < 1e-5, (
            f"ratio changed: {ratio_before:.6f} → {ratio_after:.6f}"
        )


# ---------------------------------------------------------------------------
# Gating invariance
# ---------------------------------------------------------------------------

class TestGatingInvariantToFold:
    """For equivalent inputs, threshold gating must be identical before and after fold.

    The equivalence relation is: x_raw = x_train / sf.
    Before fold the model sees x_train (training-normalised input).
    After fold(sf) the model sees x_raw (raw unnormalised input).
    Because the encoder fold exactly compensates for the input scale change
    (no approximation — the encoder is linear), h_latent is bit-identical.
    """

    def test_bottleneck_norm_gating_mask_identical(self):
        torch.manual_seed(0)
        model = _make_inference_model()
        sf = 0.02  # 1/mean_norm for Gemma-class mean norm of 50

        x_train = torch.randn(6, 16)

        _, _, h_bn_pre = smixae_encode(model, x_train)
        norms_pre = h_bn_pre.norm(dim=-1)  # (batch, n_experts)
        # Set threshold to the midpoint between two adjacent sorted norms so it
        # never falls exactly on a norm value (avoids fp boundary flips post-fold).
        sorted_norms = norms_pre[norms_pre > 0].sort().values
        mid = len(sorted_norms) // 2
        threshold_val = (sorted_norms[mid - 1].item() + sorted_norms[mid].item()) / 2.0
        model.threshold = torch.tensor(threshold_val, dtype=torch.double)
        mask_before = norms_pre > model.threshold.float()

        model.fold_activation_norm_scaling_factor(sf)
        x_raw = x_train / sf  # undo the normalization

        _, _, h_bn_post = smixae_encode(model, x_raw)
        norms_post = h_bn_post.norm(dim=-1)
        mask_after = norms_post > model.threshold.float()

        assert (mask_before == mask_after).all(), (
            f"{(mask_before != mask_after).sum().item()} expert(s) changed gating after fold"
        )

    def test_gating_breaks_without_threshold_scaling(self):
        """Confirm that omitting the threshold correction causes gating to break.

        Simulates the buggy behaviour by calling fold then undoing only the
        threshold correction.  Asserts that gating IS broken — i.e. this test
        PASSES by demonstrating the problem.  The complementary test
        test_bottleneck_norm_gating_mask_identical proves the fix restores
        correct gating.
        """
        torch.manual_seed(0)
        model = _make_inference_model()
        sf = 0.02  # mean_norm = 50 → threshold ends up ~50x too small without fix

        x_train = torch.randn(6, 16)

        _, _, h_bn_pre = smixae_encode(model, x_train)
        norms_pre = h_bn_pre.norm(dim=-1)
        sorted_norms = norms_pre[norms_pre > 0].sort().values
        mid = len(sorted_norms) // 2
        threshold_val = (sorted_norms[mid - 1].item() + sorted_norms[mid].item()) / 2.0
        model.threshold = torch.tensor(threshold_val, dtype=torch.double)
        mask_before = norms_pre > model.threshold.float()

        # Fold then deliberately undo the threshold correction — broken state.
        model.fold_activation_norm_scaling_factor(sf)
        model.threshold = model.threshold * sf  # undo fix: threshold is now 1/sf too small

        x_raw = x_train / sf
        _, _, h_bn_post = smixae_encode(model, x_raw)
        norms_post = h_bn_post.norm(dim=-1)
        mask_after = norms_post > model.threshold.float()

        n_changed = (mask_before != mask_after).sum().item()
        n_extra = (mask_after & ~mask_before).sum().item()
        n_total = mask_before.numel()

        assert n_changed > 0, (
            "Expected gating to break without threshold fix but all masks matched — "
            "check that threshold was not accidentally re-scaled."
        )
        assert n_extra > 0, (
            f"Expected extra experts to fire with unscaled threshold, got {n_changed} "
            "changed but none were false→true transitions."
        )
        # Print so the magnitude of the bug is always visible in pytest -s output.
        print(
            f"\n  [broken] {n_changed}/{n_total} experts changed gating "
            f"({n_extra} spuriously firing); "
            f"threshold should be {threshold_val / sf:.4f}, was left at {model.threshold.item():.4f}"
        )

    def test_encode_zero_pattern_preserved(self):
        """SMIXAE.encode() zero/nonzero pattern must be preserved."""
        torch.manual_seed(1)
        model = _make_inference_model()
        sf = 0.05

        x_train = torch.randn(4, 16)

        # Threshold at half the median active norm → roughly half the experts fire
        _, _, h_bn_pre = smixae_encode(model, x_train)
        norms_pre = h_bn_pre.norm(dim=-1)
        model.threshold = torch.tensor(
            norms_pre[norms_pre > 0].median().item() * 0.5, dtype=torch.double
        )

        out_before = model.encode(x_train)  # (batch, n_experts, d_bottleneck)

        model.fold_activation_norm_scaling_factor(sf)
        x_raw = x_train / sf

        out_after = model.encode(x_raw)

        zeros_before = out_before.norm(dim=-1) == 0
        zeros_after = out_after.norm(dim=-1) == 0
        assert (zeros_before == zeros_after).all(), (
            f"{(zeros_before != zeros_after).sum().item()} expert(s) changed zero/nonzero status"
        )

    def test_encode_values_scale_by_inverse_sf(self):
        """Non-zero encode outputs must scale by 1/sf after fold.

        With threshold=0 (all experts fire), SMIXAE.encode(x_raw) == encode(x_train) / sf.
        This follows from: bottleneck_fold = bottleneck_before / sf (same h_latent, N_fold = N/sf).
        """
        torch.manual_seed(2)
        model = _make_inference_model()
        sf = 0.04

        x_train = torch.randn(4, 16)
        model.threshold = torch.tensor(0.0, dtype=torch.double)  # all experts fire

        out_before = model.encode(x_train)  # (batch, n_experts, d_bottleneck)

        model.fold_activation_norm_scaling_factor(sf)
        x_raw = x_train / sf

        out_after = model.encode(x_raw)

        torch.testing.assert_close(out_after, out_before / sf, rtol=1e-4, atol=1e-6)
