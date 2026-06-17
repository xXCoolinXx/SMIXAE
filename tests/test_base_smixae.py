"""Tests for the single BaseSMIXAE and the SMIXAERebased implementation.

Coverage:
1. Instantiation of SMIXAERebased (one class for training and inference).
2. Forward-pass output shapes.
3. Training step (training_forward_pass) correctness.
4. Parameter name parity with the original SMIXAE.
5. SAELens architecture registration.
6. Save-to-disk / load-from-disk round-trip.
7. Numerical equivalence with the original SMIXAE given identical weights.
8. Base-class behaviour: process_sae_in without b_dec.
"""

import tempfile
from dataclasses import dataclass

import pytest
import torch
from sae_lens import register_sae_class
from sae_lens.saes.sae import SAE, TrainingSAEConfig, TrainStepInput

import smixae  # noqa: F401 — triggers SAELens registration
from smixae.base_smixae import (
    BaseSMIXAE,
    register_standard_linear_weights,
)
from smixae.smixae import SMIXAE, SMIXAEConfig, SMIXAETraining, SMIXAETrainingConfig, smixae_encode
from smixae.smixae_rebased import (
    SMIXAERebased,
    SMIXAERebasedConfig,
    _smixae_rebased_encode,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_EXPERTS = 8
D_EXPERT = 4
D_BOTTLENECK = 2
D_IN = 16
D_SAE = N_EXPERTS * D_EXPERT  # 32
BATCH = 6


def _make_inference_cfg() -> SMIXAERebasedConfig:
    return SMIXAERebasedConfig(
        n_experts=N_EXPERTS,
        d_expert=D_EXPERT,
        d_bottleneck=D_BOTTLENECK,
        d_in=D_IN,
        d_sae=D_SAE,
        rescale_acts_by_decoder_norm=True,
    )


def _make_training_cfg() -> SMIXAERebasedConfig:
    return SMIXAERebasedConfig(
        n_experts=N_EXPERTS,
        d_expert=D_EXPERT,
        d_bottleneck=D_BOTTLENECK,
        k_experts=4,
        d_in=D_IN,
        d_sae=D_SAE,
        rescale_acts_by_decoder_norm=True,
    )


def _make_rebased(seed: int = 0) -> SMIXAERebased:
    torch.manual_seed(seed)
    return SMIXAERebased(_make_inference_cfg())


def _make_rebased_training(seed: int = 0) -> SMIXAERebased:
    torch.manual_seed(seed)
    return SMIXAERebased(_make_training_cfg())


def _make_original(seed: int = 0) -> SMIXAE:
    torch.manual_seed(seed)
    return SMIXAE(
        SMIXAEConfig(
            n_experts=N_EXPERTS,
            d_expert=D_EXPERT,
            d_bottleneck=D_BOTTLENECK,
            d_in=D_IN,
            d_sae=D_SAE,
            rescale_acts_by_decoder_norm=True,
        )
    )


def _make_original_training(seed: int = 0) -> SMIXAETraining:
    torch.manual_seed(seed)
    return SMIXAETraining(
        SMIXAETrainingConfig(
            n_experts=N_EXPERTS,
            d_expert=D_EXPERT,
            d_bottleneck=D_BOTTLENECK,
            k_experts=4,
            d_in=D_IN,
            d_sae=D_SAE,
            rescale_acts_by_decoder_norm=True,
        )
    )


def _random_input(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(BATCH, D_IN)


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    """Both rebased classes construct without errors."""

    def test_inference_instantiation(self):
        model = _make_rebased()
        assert isinstance(model, SMIXAERebased)

    def test_training_instantiation(self):
        model = _make_rebased_training()
        assert isinstance(model, SMIXAERebased)

    def test_inference_has_expected_buffers(self):
        model = _make_rebased()
        assert hasattr(model, "threshold")
        assert hasattr(model, "n_passes_since_fired")

    def test_training_has_batchtopk(self):
        model = _make_rebased_training()
        assert hasattr(model, "batchtopk")

    def test_d_sae_auto_computed_in_training(self):
        cfg = _make_training_cfg()
        model = SMIXAERebased(cfg)
        assert model.cfg.d_sae == N_EXPERTS * D_EXPERT


# ---------------------------------------------------------------------------
# 2. Forward-pass shapes
# ---------------------------------------------------------------------------


class TestForwardPassShapes:
    """Output tensors have the expected shapes."""

    def test_forward_pass_shape(self):
        model = _make_rebased()
        x = _random_input()
        out = model(x)
        assert out.shape == (BATCH, D_IN)

    def test_encode_shape(self):
        model = _make_rebased()
        x = _random_input()
        enc = model.encode(x)
        assert enc.shape == (BATCH, N_EXPERTS, D_BOTTLENECK)

    def test_decode_shape(self):
        model = _make_rebased()
        x = _random_input()
        feature_acts = model.encode(x)
        dec = model.decode(feature_acts)
        assert dec.shape == (BATCH, D_IN)

    def test_encode_with_latents_shapes(self):
        model = _make_rebased()
        x = _random_input()
        bottleneck, h_latent = model.encode_with_latents(x)
        assert bottleneck.shape == (BATCH, N_EXPERTS, D_BOTTLENECK)
        assert h_latent.shape == (BATCH, D_SAE)


# ---------------------------------------------------------------------------
# 3. Training step
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """training_forward_pass returns a well-formed TrainStepOutput."""

    def _step_input(self) -> TrainStepInput:
        return TrainStepInput(
            sae_in=_random_input(),
            coefficients={},
            dead_neuron_mask=None,
            n_training_steps=100,
            is_logging_step=False,
        )

    def test_returns_train_step_output(self):
        model = _make_rebased_training()
        out = model.training_forward_pass(self._step_input())
        assert hasattr(out, "loss")
        assert hasattr(out, "sae_out")
        assert hasattr(out, "feature_acts")
        assert hasattr(out, "losses")

    def test_loss_is_finite(self):
        model = _make_rebased_training()
        out = model.training_forward_pass(self._step_input())
        assert torch.isfinite(out.loss)

    def test_sae_out_shape(self):
        model = _make_rebased_training()
        out = model.training_forward_pass(self._step_input())
        assert out.sae_out.shape == (BATCH, D_IN)

    def test_mse_loss_in_losses_dict(self):
        model = _make_rebased_training()
        out = model.training_forward_pass(self._step_input())
        assert "mse_loss" in out.losses

    def test_dead_expert_aux_loss_in_losses_dict(self):
        model = _make_rebased_training()
        out = model.training_forward_pass(self._step_input())
        assert "dead_expert_aux_loss" in out.losses

    def test_metrics_populated(self):
        model = _make_rebased_training()
        out = model.training_forward_pass(self._step_input())
        assert "experts_above_1e-3_L2" in out.metrics
        assert "act_threshold" in out.metrics

    def test_threshold_updates_after_step(self):
        model = _make_rebased_training()
        threshold_before = model.threshold.item()
        step_input = TrainStepInput(
            sae_in=torch.randn(BATCH, D_IN) * 10,  # large norms → threshold should move
            coefficients={},
            dead_neuron_mask=None,
            n_training_steps=100,
            is_logging_step=False,
        )
        model.training_forward_pass(step_input)
        # threshold should change from 0.0 after processing a non-trivial batch
        assert model.threshold.item() != threshold_before

    def test_feature_acts_shape_is_saelens_compatible(self):
        # SAELens trainer does: firing_feats.sum(-2).bool() to get a (d_sae,) mask.
        # This requires feature_acts to be 2D (batch, d_sae), not the 3D bottleneck.
        model = _make_rebased_training()
        out = model.training_forward_pass(self._step_input())
        assert out.feature_acts.ndim == 2
        assert out.feature_acts.shape == (BATCH, D_SAE)
        # Simulate what the SAELens trainer does for dead-neuron tracking.
        firing_feats = out.feature_acts.bool().float()
        did_fire = firing_feats.sum(-2).bool()
        assert did_fire.shape == (D_SAE,)


# ---------------------------------------------------------------------------
# 4. Parameter name parity with original SMIXAE
# ---------------------------------------------------------------------------


class TestParameterParity:
    """SMIXAERebased has the same state_dict keys as SMIXAE."""

    def test_inference_parameter_names_match(self):
        rebased = _make_rebased()
        original = _make_original()
        assert set(rebased.state_dict().keys()) == set(original.state_dict().keys()), (
            f"Key mismatch:\n"
            f"  rebased only: {set(rebased.state_dict().keys()) - set(original.state_dict().keys())}\n"
            f"  original only: {set(original.state_dict().keys()) - set(rebased.state_dict().keys())}"
        )

    def test_training_parameter_names_match(self):
        rebased = _make_rebased_training()
        original = _make_original_training()
        assert set(rebased.state_dict().keys()) == set(original.state_dict().keys()), (
            f"Key mismatch:\n"
            f"  rebased only: {set(rebased.state_dict().keys()) - set(original.state_dict().keys())}\n"
            f"  original only: {set(original.state_dict().keys()) - set(rebased.state_dict().keys())}"
        )

    def test_parameter_shapes_match(self):
        rebased = _make_rebased()
        original = _make_original()
        for key in original.state_dict():
            assert rebased.state_dict()[key].shape == original.state_dict()[key].shape, (
                f"Shape mismatch for '{key}': "
                f"rebased={rebased.state_dict()[key].shape}, "
                f"original={original.state_dict()[key].shape}"
            )


# ---------------------------------------------------------------------------
# 5. SAELens architecture registration
# ---------------------------------------------------------------------------


class TestSAELensRegistration:
    """SMIXAERebased integrates with the SAELens registry."""

    def test_architecture_name(self):
        assert SMIXAERebasedConfig.architecture() == "smixae_rebased"

    def test_get_sae_class_for_architecture(self):
        cls = SAE.get_sae_class_for_architecture("smixae_rebased")
        assert cls is SMIXAERebased

    def test_loaded_model_cfg_is_rebased_config(self):
        # SAE.get_sae_config_class_for_architecture always returns SAEConfig in
        # SAELens (hardcoded), so we verify via the loaded model instead.
        model = _make_rebased()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_model(tmp)
            loaded = SAE.load_from_disk(tmp, device="cpu")
        assert isinstance(loaded.cfg, SMIXAERebasedConfig)

    def test_register_sae_class_raises_on_duplicate(self):
        # SAELens does not allow overwriting an existing registration.
        with pytest.raises(ValueError, match="already registered"):
            register_sae_class("smixae_rebased", SMIXAERebased, SMIXAERebasedConfig)


# ---------------------------------------------------------------------------
# 6. Save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    """save_model → load_from_disk preserves forward-pass output."""

    def test_save_load_output_matches(self):
        torch.manual_seed(42)
        model = _make_rebased()
        model.eval()  # inference: threshold gating
        model.threshold = torch.tensor(0.0, dtype=torch.double)  # fire all experts
        x = _random_input(seed=7)
        out_before = model(x)

        with tempfile.TemporaryDirectory() as tmp:
            model.save_model(tmp)
            loaded = SAE.load_from_disk(tmp, device="cpu")

        loaded.eval()
        out_after = loaded(x)
        torch.testing.assert_close(out_before, out_after, rtol=1e-5, atol=1e-6)

    def test_loaded_model_is_correct_class(self):
        model = _make_rebased()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_model(tmp)
            loaded = SAE.load_from_disk(tmp, device="cpu")
        assert isinstance(loaded, SMIXAERebased)


# ---------------------------------------------------------------------------
# 7. Numerical equivalence with original SMIXAE
# ---------------------------------------------------------------------------


class TestNumericalEquivalence:
    """Given identical weights, SMIXAERebased and SMIXAE produce the same outputs."""

    def _clone_weights(self, src: SMIXAE, dst: SMIXAERebased) -> None:
        """Copy all parameters and buffers from src to dst."""
        dst.load_state_dict(src.state_dict())

    def test_encode_output_matches(self):
        original = _make_original(seed=3)
        rebased = _make_rebased(seed=0)  # different seed — weights will be overwritten
        self._clone_weights(original, rebased)

        # Inference mode → threshold gating, matching the original SMIXAE.
        original.eval()
        rebased.eval()
        # Set same threshold.
        original.threshold = torch.tensor(0.0, dtype=torch.double)
        rebased.threshold = torch.tensor(0.0, dtype=torch.double)

        x = _random_input(seed=5)
        enc_orig = original.encode(x)
        enc_rebased = rebased.encode(x)
        torch.testing.assert_close(enc_orig, enc_rebased, rtol=1e-5, atol=1e-6)

    def test_forward_output_matches(self):
        original = _make_original(seed=3)
        rebased = _make_rebased(seed=0)
        self._clone_weights(original, rebased)

        original.eval()
        rebased.eval()
        original.threshold = torch.tensor(0.0, dtype=torch.double)
        rebased.threshold = torch.tensor(0.0, dtype=torch.double)

        x = _random_input(seed=5)
        torch.testing.assert_close(original(x), rebased(x), rtol=1e-5, atol=1e-6)

    def test_encode_helper_matches(self):
        """_smixae_rebased_encode and smixae_encode produce identical intermediate tensors."""
        original = _make_original(seed=7)
        rebased = _make_rebased(seed=0)
        self._clone_weights(original, rebased)

        x = _random_input(seed=9)
        h_lat_o, h_pre_o, h_bn_o = smixae_encode(original, x)
        h_lat_r, h_pre_r, h_bn_r = _smixae_rebased_encode(rebased, x)

        torch.testing.assert_close(h_lat_o, h_lat_r, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(h_pre_o, h_pre_r, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(h_bn_o, h_bn_r, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 8. Base-class behaviour
# ---------------------------------------------------------------------------


class TestBaseClassBehaviour:
    """Unit tests for BaseSMIXAE properties independent of the rebased impl."""

    def test_process_sae_in_without_b_dec(self):
        """A minimal concrete subclass with no b_dec survives process_sae_in."""

        @dataclass
        class _MinimalConfig(TrainingSAEConfig):
            @classmethod
            def architecture(cls) -> str:
                return "_minimal_test_arch"

        class _MinimalModel(BaseSMIXAE[_MinimalConfig]):
            def initialize_weights(self) -> None:
                # Intentionally register NO b_dec — the whole point of this test.
                register_standard_linear_weights(self)

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.W_enc

            def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
                return feature_acts @ self.W_dec

            # Minimal training contract so the class is concrete (unused here).
            def encode_with_hidden_pre(self, x: torch.Tensor):
                h = self.encode(x)
                return h, h

            def get_coefficients(self):
                return {}

            def calculate_aux_loss(self, step_input, feature_acts, hidden_pre, sae_out):
                return {}

        register_sae_class("_minimal_test_arch", _MinimalModel, _MinimalConfig)

        cfg = _MinimalConfig(d_in=8, d_sae=16)
        model = _MinimalModel(cfg)

        # apply_b_dec_to_input=False is enforced by BaseSMIXAE — this must not crash.
        x = torch.randn(4, 8)
        processed = model.process_sae_in(x)
        assert processed.shape == (4, 8)

    def test_apply_b_dec_to_input_forced_false(self):
        """BaseSMIXAE always disables b_dec subtraction in pre-processing."""
        model = _make_rebased()
        assert model.cfg.apply_b_dec_to_input is False

    def test_fold_noop_default(self):
        """BaseSMIXAE.fold_activation_norm_scaling_factor resets normalize_activations."""
        model = _make_rebased()
        model.cfg.normalize_activations = "constant_norm_rescale"
        model.fold_activation_norm_scaling_factor(0.5)
        assert model.cfg.normalize_activations == "none"

    def test_rebased_inherits_from_base(self):
        """SMIXAERebased IS a BaseSMIXAE."""
        model = _make_rebased()
        assert isinstance(model, BaseSMIXAE)

    def test_rebased_training_inherits_from_base(self):
        """The single class trains and infers — a training-configured model is a BaseSMIXAE."""
        model = _make_rebased_training()
        assert isinstance(model, BaseSMIXAE)
