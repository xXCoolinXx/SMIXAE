"""SMIXAE v1 re-implemented on top of the single :class:`BaseSMIXAE`.

This module is functionally identical to :mod:`smixae.smixae` — same weight matrices,
same forward-pass logic, same threshold tracking and dead-expert recovery — but it
inherits from the single :class:`~smixae.base_smixae.BaseSMIXAE` (one class for both
training and inference) instead of the SAELens base classes directly.

Purpose:

1. **Proof of concept** — demonstrates that the base-class contract is sufficient to
   fully express the existing SMIXAE architecture in a single class.
2. **Template** — serves as a starting point when prototyping new architectures:
   copy this file, swap ``register_smixae_v1_bottleneck_weights`` for your own
   weight group, and implement new ``encode`` / ``decode`` / training logic.

The architecture is registered with SAELens under the name ``"smixae_rebased"``
(see :mod:`smixae.__init__`).
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from sae_lens.saes.batchtopk_sae import BatchTopK
from sae_lens.saes.sae import (
    TrainCoefficientConfig,
    TrainingSAEConfig,
    TrainStepInput,
    TrainStepOutput,
)
from torch import nn
from transformer_lens.hook_points import HookPoint
from typing_extensions import override

from smixae.base_smixae import (
    BaseSMIXAE,
    register_smixae_v1_bottleneck_weights,
    register_standard_linear_weights,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SMIXAERebasedConfig(TrainingSAEConfig):
    """Configuration for the single SMIXAERebased model (training and inference).

    Carries both the structural fields and the training fields. Inference paths simply
    ignore the training-only fields (``k_experts``, ``aux_loss_coefficient``,
    ``threshold_lr``, ``dead_after_n_passes``).
    """

    n_experts: int = 1024
    d_expert: int = 16
    d_bottleneck: int = 3
    rescale_acts_by_decoder_norm: bool = True

    k_experts: int = 8  # L0 = d_expert * k_experts
    aux_loss_coefficient: float = 1 / 32
    threshold_lr: float = 0.1
    # experts inactive for this many passes receive emergency auxiliary loss
    dead_after_n_passes: int = 1000

    @override
    @classmethod
    def architecture(cls) -> str:
        """Return the SAELens architecture identifier."""
        return "smixae_rebased"


# ---------------------------------------------------------------------------
# Shared encoding logic
# ---------------------------------------------------------------------------


def _smixae_rebased_encode(
    sae: "SMIXAERebased",
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Shared encoding logic.

    Applies:
    ``x → W_enc + b_enc → LeakyReLU → reshape (n_experts, d_expert) → W_bottleneck → bottleneck``.

    Optionally rescales bottleneck activations by the Frobenius norm of each
    expert's effective decoder projection (``W_latent_dec @ W_dec``), so that
    experts with larger decoders have proportionally higher selection pressure.

    Args:
        sae: An :class:`SMIXAERebased` instance.
        x: Input activations of shape ``(batch, d_model)``.

    Returns:
        h_latent: Post-activation expert-space activations,
            shape ``(batch, n_experts * d_expert)``.
        hidden_pre_latent: Pre-activation linear projections (before LeakyReLU),
            same shape as ``h_latent``.
        hidden_pre_bottleneck: Bottleneck activations before masking,
            shape ``(batch, n_experts, d_bottleneck)``.
    """
    sae_in = sae.process_sae_in(x)

    hidden_pre_latent = sae_in @ sae.W_enc + sae.b_enc
    h_latent = sae.activation_fn(hidden_pre_latent)
    h_latent_unflattened = h_latent.unflatten(-1, (sae.cfg.n_experts, sae.cfg.d_expert))

    hidden_pre_bottleneck = torch.einsum("bne,ned->bnd", h_latent_unflattened, sae.W_bottleneck)

    if sae.cfg.rescale_acts_by_decoder_norm:
        hidden_pre_bottleneck = hidden_pre_bottleneck * sae.effective_decoder_norm.unsqueeze(-1)

    return h_latent, hidden_pre_latent, hidden_pre_bottleneck


# ---------------------------------------------------------------------------
# Model (single class — train and infer with the same object)
# ---------------------------------------------------------------------------


class SMIXAERebased(BaseSMIXAE[SMIXAERebasedConfig]):
    """SMIXAE v1 on the single :class:`BaseSMIXAE` — edit this class to experiment.

    One class for both modes. Functionally identical to
    :class:`~smixae.smixae.SMIXAE` / :class:`~smixae.smixae.SMIXAETraining` combined:
    BatchTopK routing, EMA threshold tracking, dead-expert auxiliary loss, a custom
    ``training_forward_pass`` with per-step metrics, and threshold-gated inference
    ``encode``.

    Hot spots when prototyping a new architecture:
      training_forward_pass       — change loss terms or add new ones
      encode_with_hidden_pre      — change routing / bottleneck masking
      encode / encode_with_latents — change inference-time gating
      decode                      — change reconstruction path
      _smixae_rebased_encode      — change the core encode function above
    """

    W_enc: nn.Parameter
    b_enc: nn.Parameter
    W_dec: nn.Parameter
    b_dec: nn.Parameter
    W_bottleneck: nn.Parameter
    W_latent_dec: nn.Parameter

    def __init__(self, cfg: SMIXAERebasedConfig) -> None:
        cfg.d_sae = cfg.d_expert * cfg.n_experts
        super().__init__(cfg)

        self.hook_l0 = HookPoint()
        self.hook_sae_acts_bottleneck = HookPoint()
        self.batchtopk = BatchTopK(self.cfg.k_experts)

    @override
    def initialize_weights(self) -> None:
        """Register all parameters using the pre-built weight helpers."""
        register_standard_linear_weights(self)
        register_smixae_v1_bottleneck_weights(self)
        self.register_buffer(
            "threshold",
            torch.tensor(0.0, dtype=torch.double, device=self.device, requires_grad=False),
        )
        self.register_buffer(
            "n_passes_since_fired",
            torch.zeros(self.cfg.n_experts, dtype=torch.long),
        )

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return LeakyReLU(1e-4) to avoid dead neurons.

        Returns:
            LeakyReLU activation function.
        """
        return nn.LeakyReLU(negative_slope=1e-4)

    @property
    def effective_decoder_norm(self) -> torch.Tensor:
        """Frobenius norm of the effective bottleneck-to-residual projection.

        Returns:
            Tensor of shape ``(n_experts,)``.
        """
        W_dec_reshaped = self.W_dec.view(self.cfg.n_experts, self.cfg.d_expert, -1)
        W_eff = self.W_latent_dec @ W_dec_reshaped
        return torch.linalg.matrix_norm(W_eff, ord="fro", dim=(-2, -1))

    def gate_bottleneck(self, hidden_pre_bottleneck: torch.Tensor) -> torch.Tensor:
        """Mask bottleneck activations using the routing rule for the current mode.

        Training (``self.training``) uses the BatchTopK operation so a fixed number
        of experts fire per batch; inference (``eval``) uses the learned norm
        ``threshold`` instead.  All ``encode*`` paths and ``training_forward_pass``
        route through this single gate so train/eval behaviour is consistent.

        Args:
            hidden_pre_bottleneck: Pre-mask bottleneck activations,
                shape ``(batch, n_experts, d_bottleneck)``.

        Returns:
            Masked bottleneck activations, same shape.
        """
        norms = hidden_pre_bottleneck.norm(dim=-1)
        if self.training:
            mask = self.batchtopk(norms) > 0
        else:
            mask = norms > self.threshold  # type: ignore[operator]
        return hidden_pre_bottleneck * mask.unsqueeze(-1)

    # ── Inference ─────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input, gating per the current mode (BatchTopK if training, else threshold).

        Args:
            x: Input activations, shape ``(batch, d_in)``.

        Returns:
            Masked bottleneck activations, shape ``(batch, n_experts, d_bottleneck)``.
        """
        _, _, hidden_pre_bottleneck = _smixae_rebased_encode(self, x)
        return self.gate_bottleneck(hidden_pre_bottleneck)

    def encode_with_latents(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input, returning masked bottleneck and pre-bottleneck latents.

        Gating follows the current mode (BatchTopK if training, else threshold).

        Args:
            x: Input activations, shape ``(batch, d_in)``.

        Returns:
            bottleneck: Masked bottleneck activations,
                shape ``(batch, n_experts, d_bottleneck)``.
            h_latent: Post-activation expert-space activations,
                shape ``(batch, n_experts * d_expert)``.
        """
        h_latent, _, hidden_pre_bottleneck = _smixae_rebased_encode(self, x)
        return self.gate_bottleneck(hidden_pre_bottleneck), h_latent

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck activations back to input space.

        Args:
            feature_acts: Bottleneck activations,
                shape ``(batch, n_experts, d_bottleneck)``.

        Returns:
            Reconstructed input, shape ``(batch, d_in)``.
        """
        sae_out_pre = torch.einsum("bnd,nde->bne", feature_acts, self.W_latent_dec)
        sae_out_pre = sae_out_pre.flatten(-2, -1)
        sae_out_pre = sae_out_pre @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    # ── Training ──────────────────────────────────────────────────────────────

    @override
    def get_coefficients(self) -> dict[str, TrainCoefficientConfig | float]:
        """Return empty dict; loss weighting is managed internally.

        Returns:
            Empty dictionary.
        """
        return {}

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input and return bottleneck activations.

        Calls :func:`_smixae_rebased_encode` then applies BatchTopK masking on
        expert norms.

        Args:
            x: Input activations, shape ``(batch, d_in)``.

        Returns:
            h_bottleneck: BatchTopK-masked bottleneck activations,
                shape ``(batch, n_experts, d_bottleneck)``.  This is what
                ``decode()`` expects, so it maps to SAELens' ``feature_acts``.
            hidden_pre_bottleneck: Pre-mask bottleneck activations, same shape.
                Maps to SAELens' ``hidden_pre``.
        """
        h_latent, hidden_pre_latent, hidden_pre_bottleneck = _smixae_rebased_encode(self, x)
        h_bottleneck = self.gate_bottleneck(hidden_pre_bottleneck)

        self.hook_sae_acts_pre(hidden_pre_latent)
        self.hook_sae_acts_post(h_latent)
        self.hook_sae_acts_bottleneck(h_bottleneck)

        return h_bottleneck, hidden_pre_bottleneck

    @override
    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        """Run a full training forward pass and compute losses and metrics.

        Steps:

        1. Encode (BatchTopK routing on bottleneck norms).
        2. Decode the masked bottleneck to get the reconstruction.
        3. Update the running threshold from current batch bottleneck norms.
        4. Track per-expert firing; update ``n_passes_since_fired``.
        5. Compute MSE reconstruction loss.
        6. Compute dead-expert auxiliary loss.
        7. Return :class:`~sae_lens.saes.sae.TrainStepOutput` with all losses and metrics.

        Args:
            step_input: Contains ``sae_in`` (the residual stream slice to reconstruct).

        Returns:
            A :class:`~sae_lens.saes.sae.TrainStepOutput` with ``sae_in``,
            ``sae_out``, ``feature_acts``, ``hidden_pre``, ``loss``, ``losses``,
            and ``metrics``.
        """
        h_latent, hidden_pre_latent, hidden_pre_bottleneck = _smixae_rebased_encode(self, step_input.sae_in)
        h_bottleneck = self.gate_bottleneck(hidden_pre_bottleneck)

        self.hook_sae_acts_pre(hidden_pre_latent)
        self.hook_sae_acts_post(h_latent)
        self.hook_sae_acts_bottleneck(h_bottleneck)

        sae_out = self.decode(h_bottleneck)

        self.update_threshold(h_bottleneck.norm(dim=-1))

        with torch.no_grad():
            fired_in_batch = (h_bottleneck.norm(dim=-1) > 0).any(dim=0)
            self.n_passes_since_fired = torch.where(
                fired_in_batch,
                torch.zeros_like(self.n_passes_since_fired),
                self.n_passes_since_fired + 1,
            )

        per_item_mse_loss = self.mse_loss_fn(sae_out, step_input.sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        dead_aux_loss = self.calculate_pre_act_aux_loss(
            self.n_passes_since_fired > self.cfg.dead_after_n_passes,
            hidden_pre_bottleneck,
        )
        total_loss = mse_loss + dead_aux_loss
        losses = {"mse_loss": mse_loss, "dead_expert_aux_loss": dead_aux_loss}

        metrics: dict = {}
        metrics["experts_above_1e-3_L2"] = (
            (h_bottleneck.norm(dim=-1) > 1e-3).float().sum(dim=-1).mean()
        )
        metrics["experts_above_1e-1_L2"] = (
            (h_bottleneck.norm(dim=-1) > 1e-1).float().sum(dim=-1).mean()
        )
        post_act_norms = h_bottleneck.norm(dim=-1)
        metrics["expert_norm_mean"] = post_act_norms[post_act_norms > 0].mean()
        metrics["dead_experts"] = (
            self.n_passes_since_fired > self.cfg.dead_after_n_passes
        ).sum().item()
        metrics["act_threshold"] = self.threshold
        metrics["experts_above_threshold"] = (
            (hidden_pre_bottleneck.norm(dim=-1) > self.threshold).float().sum(dim=-1).mean()
        )
        metrics["nonzero_latent_l0"] = (h_latent > 0).float().sum(dim=-1).mean()

        # feature_acts and hidden_pre use the flat latent tensors so that SAELens'
        # trainer can compute L0 and dead-neuron statistics over the (batch, d_sae)
        # shape it expects; h_bottleneck is 3D and would break those paths.
        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=h_latent,
            hidden_pre=hidden_pre_latent,
            loss=total_loss,
            losses=losses,
            metrics=metrics,
        )

    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute dead-expert recovery auxiliary losses.

        Intended as a SAELens-compat fallback: when the default SAELens
        ``training_forward_pass`` calls this method, ``hidden_pre`` is
        ``hidden_pre_bottleneck`` (returned by :meth:`encode_with_hidden_pre`).

        The custom :meth:`training_forward_pass` override calls
        :meth:`calculate_pre_act_aux_loss` directly with local tensors and does
        not go through this method.

        Args:
            step_input: Training step input (unused).
            feature_acts: Post-activation expert activations (unused).
            hidden_pre: Pre-mask bottleneck activations,
                shape ``(batch, n_experts, d_bottleneck)``.
            sae_out: SAE reconstruction (unused).

        Returns:
            Dict mapping ``"dead_expert_aux_loss"`` to a scalar tensor.
        """
        return {
            "dead_expert_aux_loss": self.calculate_pre_act_aux_loss(
                self.n_passes_since_fired > self.cfg.dead_after_n_passes,
                hidden_pre,
            )
        }

    def calculate_pre_act_aux_loss(
        self, dead_expert_mask: torch.Tensor, hidden_pre_bottleneck: torch.Tensor
    ) -> torch.Tensor:
        """Push dead experts' bottleneck norms toward the selection threshold.

        Args:
            dead_expert_mask: Boolean tensor of shape ``(n_experts,)``;
                ``True`` for experts that have not fired recently.
            hidden_pre_bottleneck: Pre-mask bottleneck activations,
                shape ``(batch, n_experts, d_bottleneck)``.

        Returns:
            Scalar auxiliary loss tensor.
        """
        if dead_expert_mask is None or not dead_expert_mask.any():
            return self.threshold.new_tensor(0.0)

        expert_norms = hidden_pre_bottleneck.norm(dim=-1)
        dead_norms = expert_norms[:, dead_expert_mask]
        shortfall = torch.relu(self.threshold.detach().float() - dead_norms)
        dead_decoder_norms = self.effective_decoder_norm[dead_expert_mask].detach()
        return self.cfg.aux_loss_coefficient * (shortfall * dead_decoder_norms).sum(dim=-1).mean()

    @torch.no_grad()
    def update_threshold(self, norms_topk: torch.Tensor) -> None:
        """Update the running selection threshold via EMA of minimum positive norm.

        Args:
            norms_topk: Per-expert bottleneck norms from the current batch,
                shape ``(batch, n_experts)``.  Zero entries are inactive experts.
        """
        positive_mask = norms_topk > 0
        lr = self.cfg.threshold_lr
        with torch.autocast(self.threshold.device.type, enabled=False):
            if positive_mask.any():
                min_positive = norms_topk[positive_mask].min().to(self.threshold.dtype)
                self.threshold = (1 - lr) * self.threshold + lr * min_positive  # type: ignore[assignment]

    @override
    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float) -> None:
        """Fold activation scaling into weights and rescale threshold to match.

        Args:
            scaling_factor: The activation norm scaling factor.
        """
        # Manually apply the standard linear weight scaling (no SAELens super call).
        self.W_enc.data *= scaling_factor
        self.W_dec.data /= scaling_factor
        self.b_dec.data /= scaling_factor
        self.cfg.normalize_activations = "none"
        # SMIXAE-specific correction for decoder-norm rescaling.
        if self.cfg.rescale_acts_by_decoder_norm:
            sf_sqrt = scaling_factor**0.5
            self.W_dec.data *= sf_sqrt
            self.threshold = self.threshold / sf_sqrt  # type: ignore[assignment]
        else:
            self.threshold = self.threshold / scaling_factor  # type: ignore[assignment]
