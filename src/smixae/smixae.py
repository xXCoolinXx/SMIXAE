"""SMIXAE model architecture: inference and training classes for the Sparse Mixture of Autoencoders."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from sae_lens.saes.batchtopk_sae import BatchTopK
from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
    TrainStepOutput,
)
from torch import nn
from transformer_lens.hook_points import HookPoint
from typing_extensions import override


@dataclass
class SMIXAEConfig(SAEConfig):
    """Configuration class for a SMIXAE."""

    n_experts: int = 1024
    d_expert: int = 16
    d_bottleneck: int = 3
    rescale_acts_by_decoder_norm: bool = True

    @override
    @classmethod
    def architecture(cls) -> str:
        return "smixae"


class SMIXAE(SAE[SMIXAEConfig]):
    """Inference-only SMIXAE: Sparse Autoencoder with nonlinear expert bottleneck.

    Uses a linear encoder/decoder but routes activations through a per-expert
    low-dimensional bottleneck, allowing each expert to capture arbitrary multdimensional
    geometry rather than a linear direction.

    Implements the required abstract methods from BaseSAE:

      - initialize_weights: sets up simple parameter initializations for W_enc, b_enc, W_dec, and b_dec.
      - encode: computes the feature activations from an input.
      - decode: reconstructs the input from the feature activations.

    The BaseSAE.forward() method automatically calls encode and decode,
    including any error-term processing if configured.
    """

    # W_gate: nn.Parameter
    W_bottleneck: nn.Parameter
    W_latent_dec: nn.Parameter
    log_threshold: nn.Parameter
    b_enc: nn.Parameter
    # b_bottleneck: nn.Parameter

    def __init__(self, cfg: SMIXAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

        self.register_buffer(
            "threshold",
            # use double precision as otherwise we can run into numerical issues
            torch.tensor(0.0, dtype=torch.double, device=self.W_dec.device, requires_grad=False),
        )

        # Dead expert tracker - remove this later, not used for inference
        self.register_buffer(
            "n_passes_since_fired",
            torch.zeros(
                self.cfg.n_experts,
                dtype=torch.long,
            ),
        )

        # Let this be set by the loaded class
        self.cfg.apply_b_dec_to_input = False  # Remove bias term - destroys structure

    @override
    def initialize_weights(self) -> None:
        """Initialise base SAE weights then register SMIXAE-specific parameters."""
        # Initialize encoder weights and bias.
        super().initialize_weights()
        _init_weights_smixae(self)

    # @property
    # def threshold(self) -> torch.Tensor:
    #     return torch.exp(self.log_threshold)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor into the feature space."""
        _, _, hidden_pre_bottleneck = smixae_encode(self, x, decoder_norm_override=True)  # (batch, n_experts, d_bottleneck)

        bottleneck_mask = hidden_pre_bottleneck.norm(dim=-1) > self.threshold  # type: ignore # (batch, n_experts) mask

        return hidden_pre_bottleneck * bottleneck_mask.unsqueeze(-1)  # Apply mask per bottleneck

    def encode_with_latents(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input, returning masked bottleneck activations and pre-bottleneck latents.

        Returns:
            bottleneck:  ``(batch, n_experts, d_bottleneck)`` — same as ``encode()``.
            h_latent:    ``(batch, n_experts * d_expert)`` — latent activations before bottleneck projection.
        """
        h_latent, _, hidden_pre_bottleneck = smixae_encode(self, x, decoder_norm_override=True)
        bottleneck_mask = hidden_pre_bottleneck.norm(dim=-1) > self.threshold  # type: ignore
        bottleneck = hidden_pre_bottleneck * bottleneck_mask.unsqueeze(-1)
        return bottleneck, h_latent

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck activations back to the input space.

        Applies the latent decoder, flattens expert dimensions, and projects through
        ``W_dec``. The bias is added only at the output to avoid collapsing manifold structure.
        """
        # Undo rescaling from encoding: feature_acts shape is (batch, n_experts, d_bottleneck)
        # if self.cfg.rescale_acts_by_decoder_norm:
        #     # effective_decoder_norm shape is (n_experts,), reshape to (1, n_experts, 1) for broadcasting
        #     norm_factor = self.effective_decoder_norm.view(1, -1, 1)
        #     feature_acts = feature_acts / norm_factor

        sae_out_pre = torch.einsum("bnd,nde->bne", feature_acts, self.W_latent_dec)
        sae_out_pre = sae_out_pre.flatten(-2, -1)
        sae_out_pre = (
            sae_out_pre @ self.W_dec + self.b_dec
        )  # Bias term destroys manifold structure, so only add it to the output

        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return LeakyReLU(1e-4) to avoid dead neurons while preserving expert norms."""
        # use leaky relu to avoid dead neurons; small negative slope avoids impacting expert norm
        return nn.LeakyReLU(negative_slope=1e-4)


@dataclass
class SMIXAETrainingConfig(TrainingSAEConfig):
    """Configuration class for training a SMIXAETraining."""

    n_experts: int = 1024
    d_expert: int = 16
    d_bottleneck: int = 3
    k_experts: int = 8  # L0 = d_expert * k_experts
    aux_loss_coefficient: float = 1 / 32
    rescale_acts_by_decoder_norm: bool = True

    threshold_lr: float = 0.1
    # experts inactive for this many passes receive emergency auxiliary loss to recover them
    dead_after_n_passes: int = 1000

    @override
    @classmethod
    def architecture(cls) -> str:
        return "smixae"


class SMIXAETraining(TrainingSAE[SMIXAETrainingConfig]):
    """Training-mode SMIXAE with BatchTopK routing and dead-expert auxiliary loss.

    Implements the full training forward pass including MSE reconstruction loss,
    dead-expert recovery via auxiliary loss, and running threshold updates.
    """

    b_enc: nn.Parameter
    # b_bottleneck: nn.Parameter
    W_bottleneck: nn.Parameter
    W_latent_dec: nn.Parameter

    def __init__(self, cfg: SMIXAETrainingConfig):
        # Intercept d_sae
        cfg.d_sae = cfg.d_expert * cfg.n_experts

        super().__init__(cfg)

        self.hook_l0 = HookPoint()
        self.hook_sae_acts_bottleneck = HookPoint()

        self.batchtopk = BatchTopK(self.cfg.k_experts)

        self.register_buffer(
            "threshold",
            # use double precision as otherwise we can run into numerical issues
            torch.tensor(0.0, dtype=torch.double, device=self.W_dec.device),
        )

        # Dead expert tracker
        self.register_buffer(
            "n_passes_since_fired",
            torch.zeros(
                self.cfg.n_experts,
                dtype=torch.long,
            ),
        )

        self.cfg.apply_b_dec_to_input = False  # True  # True  # Remove bias term - destroys structure
        # self.b_dec.requires_grad_(False)

    def initialize_weights(self) -> None:
        """Initialise base SAE weights then register SMIXAE-specific parameters."""
        super().initialize_weights()
        _init_weights_smixae(self)

    @override
    def get_coefficients(self) -> dict[str, TrainCoefficientConfig | float]:
        """Return an empty coefficient dict; SMIXAE manages loss weighting internally."""
        return {}

    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input and stash intermediate activations needed for the training pass.

        Calls :func:`smixae_encode` then applies BatchTopK masking on expert norms to
        select the ``k_experts`` most active experts per example.  The masked bottleneck
        activations are stashed in ``self.h_bottleneck`` and the pre-mask activations in
        ``self.hidden_pre_bottleneck`` so that :meth:`training_forward_pass` and
        :meth:`calculate_aux_loss` can access them without re-running the encoder.

        Args:
            x: Input activations of shape ``(batch, d_model)``.

        Returns:
            h_latent: Post-activation expert space activations
                ``(batch, n_experts * d_expert)`` — returned for SAELens compatibility
                but not used by the training loop.
            hidden_pre_latent: Pre-activation projections, same shape — likewise
                returned for compatibility.
        """
        h_latent, hidden_pre_latent, hidden_pre_bottleneck = smixae_encode(self, x)

        batch_norm_mask = self.batchtopk(hidden_pre_bottleneck.norm(dim=-1)) > 0  # (batch_size, n_experts)

        # Stash
        self.hidden_pre_bottleneck = hidden_pre_bottleneck
        self.h_bottleneck = hidden_pre_bottleneck * batch_norm_mask.unsqueeze(-1)

        self.hook_sae_acts_pre(hidden_pre_latent)
        self.hook_sae_acts_post(h_latent)
        self.hook_sae_acts_bottleneck(self.h_bottleneck)

        return h_latent, hidden_pre_latent  # These get ignored

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck activations back to the input space.

        Applies the latent decoder, flattens expert dimensions, and projects through
        ``W_dec``. The bias is added only at the output to avoid collapsing manifold structure.
        """
        # Undo rescaling from encoding: feature_acts shape is (batch, n_experts, d_bottleneck)
        if self.cfg.rescale_acts_by_decoder_norm:
            # effective_decoder_norm shape is (n_experts,), reshape to (1, n_experts, 1) for broadcasting
            norm_factor = self.effective_decoder_norm.view(1, -1, 1).clamp(min=1e-8)
            feature_acts = feature_acts / norm_factor

        sae_out_pre = torch.einsum("bnd,nde->bne", feature_acts, self.W_latent_dec)
        sae_out_pre = sae_out_pre.flatten(-2, -1)
        sae_out_pre = (
            sae_out_pre @ self.W_dec + self.b_dec
        )  # Bias term destroys manifold structure, so only add it to the output

        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    def training_forward_pass(
        self,
        step_input: TrainStepInput,
    ) -> TrainStepOutput:
        """Run a full training forward pass and compute all losses and metrics.

        Steps:
        1. Encode the input via :meth:`encode_with_hidden_pre`, which stashes
           ``self.h_bottleneck`` (TopK-masked bottleneck activations).
        2. Decode ``self.h_bottleneck`` to get the reconstruction.
        3. Update the running threshold from the current batch's bottleneck norms.
        4. Track per-expert firing: reset counters for experts that fired this batch;
           increment counters for those that did not.
        5. Compute MSE reconstruction loss.
        6. Compute auxiliary losses (currently dead-expert recovery via
           :meth:`calculate_aux_loss`).
        7. Return a :class:`TrainStepOutput` with the combined loss and per-step metrics.

        Logged metrics include expert activity counts at two thresholds (1e-3, 1e-1
        L2 norm), mean active expert norm, dead expert count, threshold value, and
        latent L0 sparsity.

        Args:
            step_input: Contains ``sae_in`` (the residual stream slice to reconstruct).

        Returns:
            A :class:`TrainStepOutput` with ``sae_in``, ``sae_out``, ``feature_acts``,
            ``hidden_pre``, ``loss`` (scalar), ``losses`` (per-term dict), and
            ``metrics`` (dict of float tensors).
        """
        self._normalize_W_latent_dec()
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(self.h_bottleneck)

        self.update_threshold(self.h_bottleneck.norm(dim=-1))

        with torch.no_grad():
            # Check if an expert fired at least once in this batch (norm > 0)
            # h_bottleneck is (batch, n_experts, d_bottleneck)
            fired_in_batch = (self.h_bottleneck.norm(dim=-1) > 0).any(dim=0)

            # Reset counter to 0 if fired, otherwise increment by 1
            self.n_passes_since_fired = torch.where(
                fired_in_batch,
                torch.zeros_like(self.n_passes_since_fired),
                self.n_passes_since_fired + 1,
            )

        # Calculate MSE loss
        per_item_mse_loss = self.mse_loss_fn(sae_out, step_input.sae_in)

        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # Compute KL divergence between router distribution and

        # Calculate architecture-specific auxiliary losses
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )

        # Total loss is MSE plus all auxiliary losses
        total_loss = mse_loss

        # Create losses dictionary with mse_loss
        losses = {"mse_loss": mse_loss}

        # Add architecture-specific losses to the dictionary
        # Make sure aux_losses is a dictionary with string keys and tensor values
        if isinstance(aux_losses, dict):
            losses.update(aux_losses)

        # Sum all losses for total_loss
        if isinstance(aux_losses, dict):
            for loss_value in aux_losses.values():
                total_loss = total_loss + loss_value
        else:
            # Handle case where aux_losses is a tensor
            total_loss = total_loss + aux_losses

        metrics = {}

        metrics["experts_above_1e-3_L2"] = (self.h_bottleneck.norm(dim=-1) > 1e-3).float().sum(dim=-1).mean()

        metrics["experts_above_1e-1_L2"] = (self.h_bottleneck.norm(dim=-1) > 1e-1).float().sum(dim=-1).mean()

        post_act_norms = self.h_bottleneck.norm(dim=-1)

        metrics["expert_norm_mean"] = post_act_norms[post_act_norms > 0].mean()

        metrics["dead_experts"] = (self.n_passes_since_fired > self.cfg.dead_after_n_passes).sum().item()

        # Track this during training to make sure the threshold is working properly - woops
        metrics["act_threshold"] = self.threshold
        metrics["experts_above_threshold"] = (
            (self.hidden_pre_bottleneck.norm(dim=-1) > self.threshold).float().sum(dim=-1).mean()
        )

        # This is needed if we use something other than ReLU to avoid dead neurons - eg LeakyReLU or Swish
        metrics["nonzero_latent_l0"] = (feature_acts > 0).float().sum(dim=-1).mean()

        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
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
        """Compute architecture-specific auxiliary losses.

        Currently applies :meth:`calculate_pre_act_aux_loss` to recover dead experts
        by pushing their bottleneck norms toward the selection threshold.

        Args:
            step_input: Training step input (unused here; present for API compatibility).
            feature_acts: Post-activation expert activations (unused here).
            hidden_pre: Pre-activation projections (unused here).
            sae_out: SAE reconstruction (unused here).

        Returns:
            A dict mapping loss name to scalar tensor, e.g.
            ``{"dead_expert_aux_loss": tensor(...)}``.
        """
        losses = {}

        # losses["dead_expert_aux_loss"] = self.calculate_topk_aux_loss(
        #     step_input.sae_in,
        #     sae_out,
        #     self.hidden_pre_bottleneck,
        #     self.n_passes_since_fired > self.cfg.dead_after_n_passes,
        # )

        # hopefully this is better
        losses['dead_expert_aux_loss'] = self.calculate_pre_act_aux_loss(self.n_passes_since_fired > self.cfg.dead_after_n_passes)

        return losses

    def calculate_pre_act_aux_loss(self, dead_expert_mask):
        """Push dead experts' bottleneck norms toward the selection threshold.

        No residual peeking — purely about making dead experts competitive via norm pressure.
        """
        if dead_expert_mask is None or not dead_expert_mask.any():
            return self.threshold.new_tensor(0.0)

        # How far below the selection threshold are dead experts?
        expert_norms = self.hidden_pre_bottleneck.norm(dim=-1)  # (batch, n_experts)
        dead_norms = expert_norms[:, dead_expert_mask]  # (batch, n_dead)

        # Push norms toward threshold from below
        # Only penalize experts that are below threshold (relu clips those already above)
        shortfall = torch.relu(self.threshold.detach().float() - dead_norms)

        # Weight by decoder norm so experts with larger decoders get more pressure
        dead_decoder_norms = self.effective_decoder_norm[dead_expert_mask].detach()
        # Detachment avoids the model just making decoder norm 0 - this isn't a problem for JumpReLU since the trivial fix for the model is to just lower the threshold
        # But here decoder norm 0 is optimal

        return self.cfg.aux_loss_coefficient * (shortfall * dead_decoder_norms).sum(dim=-1).mean()

    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        z_pre_mask: torch.Tensor,  # (batch, n_experts, d_bottleneck) before TopK
        dead_expert_mask: torch.Tensor,  # (n_experts,) bool
    ) -> torch.Tensor:
        """Compute auxiliary reconstruction loss over dead experts (residual-peeking variant).

        Selects ``k_aux`` dead experts by bottleneck norm, decodes them through the full
        expert pipeline, and penalises the reconstruction error against the current residual.
        This variant is currently commented out in favour of :meth:`calculate_pre_act_aux_loss`.

        Args:
            sae_in: Original encoder input, shape ``(batch, d_in)``.
            sae_out: Current reconstruction, same shape.
            z_pre_mask: Pre-mask bottleneck activations, shape ``(batch, n_experts, d_bottleneck)``.
            dead_expert_mask: Boolean mask of shape ``(n_experts,)``; ``True`` for dead experts.

        Returns:
            Scaled auxiliary loss scalar tensor.
        """
        if dead_expert_mask is None or (num_dead := int(dead_expert_mask.sum())) == 0:
            return sae_out.new_tensor(0.0)

        residual = (sae_in - sae_out).detach()

        # Heuristic: use half of active experts as k_aux
        k_aux = (
            self.cfg.k_experts // 2
        )  # This is actually fine, just need to increase the aux loss coefficient

        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        # Select top-k_aux dead experts by bottleneck norm
        expert_norms = z_pre_mask.norm(dim=-1)  # (batch, n_experts)
        dead_norms = torch.where(dead_expert_mask[None], expert_norms, -torch.inf)
        topk = dead_norms.topk(k_aux, dim=-1, sorted=False)

        # Build sparse z with only selected dead experts
        aux_mask = torch.zeros_like(expert_norms)
        aux_mask.scatter_(1, topk.indices, 1.0)
        z_aux = z_pre_mask * aux_mask.unsqueeze(-1)  # (batch, n_experts, d_bottleneck)

        # Decode through full expert pipeline
        recons = self.decode(z_aux)

        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
        return self.cfg.aux_loss_coefficient * scale * auxk_loss

    @torch.no_grad()
    def update_threshold(self, norms_topk: torch.Tensor) -> None:
        """Update the running selection threshold via exponential moving average.

        Tracks the minimum positive bottleneck norm seen this batch and performs an
        EMA update: ``threshold ← (1 - lr) * threshold + lr * min_positive``.
        If no positive norms exist in the batch the threshold is left unchanged.

        The threshold is stored in double precision to avoid numerical instability
        when the threshold is very small.

        Args:
            norms_topk: Per-expert bottleneck norms from the current batch,
                shape ``(batch, n_experts)``.  Zero entries represent inactive experts.
        """
        positive_mask = norms_topk > 0
        lr = self.cfg.threshold_lr
        # autocast can cause numerical issues with the threshold update
        with torch.autocast(self.threshold.device.type, enabled=False):
            if positive_mask.any():
                min_positive = norms_topk[positive_mask].min().to(self.threshold.dtype)
                self.threshold = (1 - lr) * self.threshold + lr * min_positive

    @property
    def effective_decoder_norm(self) -> torch.Tensor:
        """Compute the Frobenius norm of the effective bottleneck-to-residual projection.

        Returns a tensor of shape ``(n_experts,)``.
        """
        W_dec_reshaped = self.W_dec.view(self.cfg.n_experts, self.cfg.d_expert, -1)

        # W_latent_dec: (n_experts, d_bottleneck, d_expert)
        # W_dec_reshaped: (n_experts, d_expert, d_model)
        W_eff = self.W_latent_dec @ W_dec_reshaped

        return torch.linalg.matrix_norm(W_eff, ord="fro", dim=(-2, -1))

    @torch.no_grad()
    def _normalize_W_latent_dec(self) -> None:
        """Normalize each expert's W_latent_dec slice to unit Frobenius norm.

        With symmetric decode (Fix B), effective_decoder_norm cancels from the
        reconstruction gradient path, removing the loss-side force that previously
        bounded it.  W_latent_dec gains a gauge degree of freedom (scale by alpha,
        compensate W_bottleneck by 1/alpha — loss invariant).  Normalising here
        mirrors what SAELens does to W_dec after each step, anchoring Eff to the
        W_dec row norms rather than letting it drift.
        """
        norms = self.W_latent_dec.data.view(self.cfg.n_experts, -1).norm(dim=-1)
        self.W_latent_dec.data /= norms.clamp(min=1e-8).view(self.cfg.n_experts, 1, 1)

    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float) -> None:
        """Fold the activation-norm scaling factor, capturing Eff_train before fold modifies W_dec.

        The base fold divides W_dec by scaling_factor, which would change
        effective_decoder_norm from Eff_train to Eff_train/sf. We capture
        Eff_train first and store it so process_state_dict_for_saving_inference
        folds the correct norm into W_bottleneck (giving hpb_inf = hpb_train
        exactly at inference, not hpb_train/sf).
        """
        eff_train = self.effective_decoder_norm.detach().clone().clamp(min=1e-8)
        super().fold_activation_norm_scaling_factor(scaling_factor)
        self.register_buffer("_eff_for_fold", eff_train)

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return LeakyReLU(1e-4) to avoid dead neurons while preserving expert norms."""
        # use leaky relu to avoid dead neurons; small negative slope avoids impacting expert norm
        return nn.LeakyReLU(negative_slope=1e-4)

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, Any]
    ) -> None:
        super().process_state_dict_for_saving_inference(state_dict)

        if self.cfg.rescale_acts_by_decoder_norm:
            # Use Eff captured before fold_activation_norm_scaling_factor changed W_dec.
            # Falling back to self.effective_decoder_norm uses Eff_train/sf instead of
            # Eff_train, which inflates hpb_inf by 1/sf and breaks threshold matching.
            eff = state_dict.pop("_eff_for_fold", None)
            if eff is None:
                eff = self.effective_decoder_norm.clamp(min=1e-8)
            _fold_effective_norm(
                W_bottleneck=state_dict["W_bottleneck"],
                W_latent_dec=state_dict["W_latent_dec"],
                effective_decoder_norm=eff,
            )


def _init_weights_smixae(
    sae: SAE[SMIXAEConfig] | TrainingSAE[SMIXAETrainingConfig],
) -> None:
    """Register SMIXAE-specific parameters on an SAE or TrainingSAE instance.

    Called by :meth:`initialize_weights` on both :class:`SMIXAE` and
    :class:`SMIXAETraining` after the base-class initialisation.  Registers three
    learnable parameters:

    - ``b_enc``: encoder bias, shape ``(n_experts * d_expert,)``, zero-initialised.
    - ``W_bottleneck``: expert-space → bottleneck projection,
      shape ``(n_experts, d_expert, d_bottleneck)``, Kaiming-uniform init.
    - ``W_latent_dec``: bottleneck → expert-space projection (decoder side),
      shape ``(n_experts, d_bottleneck, d_expert)``, Kaiming-uniform init.

    Args:
        sae: The SAE or TrainingSAE instance on which to register the parameters.
    """
    # Add gate bias term to allow more expressivity - pre relu
    sae.b_enc = nn.Parameter(
        torch.zeros(
            sae.cfg.n_experts * sae.cfg.d_expert,
            dtype=sae.dtype,
            device=sae.device,
        )
    )

    # sae.b_bottleneck = nn.Parameter(
    #     torch.zeros(
    #         sae.cfg.n_experts,
    #         sae.cfg.d_bottleneck,
    #         dtype=sae.dtype,
    #         device=sae.device,
    #     )
    # )

    sae.W_bottleneck = nn.Parameter(
        torch.empty(
            sae.cfg.n_experts,
            sae.cfg.d_expert,
            sae.cfg.d_bottleneck,
            dtype=sae.dtype,
            device=sae.device,
        )
    )

    sae.W_latent_dec = nn.Parameter(
        torch.empty(
            sae.cfg.n_experts,
            sae.cfg.d_bottleneck,
            sae.cfg.d_expert,
            dtype=sae.dtype,
            device=sae.device,
        )
    )

    nn.init.kaiming_uniform_(sae.W_bottleneck)
    nn.init.kaiming_uniform_(sae.W_latent_dec)


def smixae_encode(
    sae: SMIXAE | SMIXAETraining, x: torch.Tensor, decoder_norm_override = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared encoding logic for both inference and training SMIXAE models.

    Applies the full encoding pipeline:
    ``x → W_enc + b_enc → LeakyReLU → reshape (n_experts, d_expert) → W_bottleneck einsum → bottleneck``.

    Optionally rescales bottleneck activations by the Frobenius norm of each expert's
    effective decoder projection (``W_latent_dec @ W_dec``), so that experts with larger
    decoders have proportionally higher selection pressure under TopK.

    Args:
        sae: Either an :class:`SMIXAE` (inference) or :class:`SMIXAETraining` instance.
        x: Input activations of shape ``(batch, d_model)``.
        decoder_norm_override: If ``True``, skip the ``effective_decoder_norm`` rescale.
            Used at inference when the norm is already folded into ``W_bottleneck``.

    Returns:
        h_latent: Post-activation expert space activations, shape
            ``(batch, n_experts * d_expert)``.
        hidden_pre_latent: Pre-activation linear projections (before LeakyReLU), same
            shape as ``h_latent``. Used by the training forward pass for auxiliary loss.
        hidden_pre_bottleneck: Bottleneck activations before TopK or threshold masking,
            shape ``(batch, n_experts, d_bottleneck)``.
    """
    sae_in = sae.process_sae_in(x)

    # Standard forward
    hidden_pre_latent = sae_in @ sae.W_enc + sae.b_enc
    h_latent = sae.activation_fn(hidden_pre_latent)
    h_latent_unflattened = h_latent.unflatten(-1, (sae.cfg.n_experts, sae.cfg.d_expert))  # Unflatten

    # Bottleneck
    hidden_pre_bottleneck = (
        torch.einsum("bne,ned->bnd", h_latent_unflattened, sae.W_bottleneck)
        # + sae.b_bottleneck - this causes flattening
    )  # (batch_size, n_experts, d_bottelneck)

    if sae.cfg.rescale_acts_by_decoder_norm and not decoder_norm_override:
        hidden_pre_bottleneck = hidden_pre_bottleneck * sae.effective_decoder_norm.unsqueeze(-1)

    return h_latent, hidden_pre_latent, hidden_pre_bottleneck



@torch.no_grad()
def _fold_effective_norm(
    W_bottleneck : torch.Tensor,
    W_latent_dec : torch.Tensor,
    effective_decoder_norm : torch.Tensor
):
    norm = effective_decoder_norm.view(-1, 1, 1) # (n_experts, 1, 1)

    W_bottleneck.data = W_bottleneck.data * norm
    W_latent_dec.data = W_latent_dec.data / norm

