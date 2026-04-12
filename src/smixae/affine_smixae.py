"""AffineSMIXAE: a SMIXAE variant with a learned W_directions affine routing matrix.

.. warning::
    This module is **not actively used or maintained**. It is kept for historical reference only.
    Use :mod:`smixae.smixae` for all current work.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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
class AffineSMIXAEConfig(SAEConfig):
    """Configuration class for a AffineSMIXAE."""

    n_experts: int = 1024
    d_expert: int = 16
    d_bottleneck: int = 3
    rescale_acts_by_decoder_norm: bool = True

    @override
    @classmethod
    def architecture(cls) -> str:
        return "affine_smixae"


class AffineSMIXAE(SAE[AffineSMIXAEConfig]):
    """Inference-only AffineSMIXAE: SMIXAE variant with learned affine routing directions.

    **Not actively used or maintained** — kept for historical reference only.

    Extends the standard SMIXAE by adding a ``W_directions`` parameter
    ``(n_experts, d_in)`` that defines per-expert routing directions in the input space.
    Routing uses cosine similarity between the input and each direction rather than the
    bottleneck norm used in :class:`SMIXAE`.
    """

    # W_gate: nn.Parameter
    W_bottleneck: nn.Parameter
    W_latent_dec: nn.Parameter
    log_threshold: nn.Parameter
    b_enc: nn.Parameter
    W_directions : nn.Parameter
    # b_bottleneck: nn.Parameter

    def __init__(self, cfg: AffineSMIXAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

        self.register_buffer(
            "threshold",
            # use double precision as otherwise we can run into numerical issues
            torch.tensor(0.0, dtype=torch.double, device=self.W_dec.device),
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
        """Initialize base SAE weights then register AffineSMIXAE-specific parameters."""
        # Initialize encoder weights and bias.
        super().initialize_weights()
        _init_weights_affine_smixae(self)

    # @property
    # def threshold(self) -> torch.Tensor:
    #     return torch.exp(self.log_threshold)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor into the feature space."""
        _, _, _, bottleneck, _, _ = affine_smixae_encode(self, x)  # (batch, n_experts, d_bottleneck)

        return bottleneck

    # def encode_with_latents(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Encode the input tensor, returning both the masked bottleneck activations and
    #     the pre-bottleneck latent activations (post-LeakyReLU, shape ``(batch, n_experts * d_expert)``).

    #     Returns:
    #         bottleneck:  ``(batch, n_experts, d_bottleneck)`` — same as ``encode()``.
    #         h_latent:    ``(batch, n_experts * d_expert)`` — latent activations before bottleneck projection.
    #     """
    #     h_latent, _, hidden_pre_bottleneck = affine_smixae_encode(self, x)
    #     bottleneck_mask = hidden_pre_bottleneck.norm(dim=-1) > self.threshold  # type: ignore
    #     bottleneck = hidden_pre_bottleneck * bottleneck_mask.unsqueeze(-1)
    #     return bottleneck, h_latent

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck activations back to the input space.

        Applies the latent decoder, flattens expert dimensions, and projects through
        ``W_dec``. Reverses hook_z reshaping if it was applied during input processing.
        """
        sae_out_pre = torch.einsum("bnd,nde->bne", feature_acts, self.W_latent_dec)
        sae_out_pre = sae_out_pre.flatten(-2, -1)
        sae_out_pre = sae_out_pre @ self.W_dec + self.b_dec

        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return LeakyReLU(1e-4) to avoid dead neurons while preserving expert norms."""
        # use leaky relu to avoid dead neurons; small negative slope avoids impacting expert norm
        return nn.LeakyReLU(negative_slope=1e-4)

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


@dataclass
class AffineSMIXAETrainingConfig(TrainingSAEConfig):
    """Configuration class for training a AffineSMIXAETraining."""

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
        return "affine_smixae"


class AffineSMIXAETraining(TrainingSAE[AffineSMIXAETrainingConfig]):
    """Training-mode AffineSMIXAE with BatchTopK routing and dead-expert auxiliary loss.

    **Not actively used or maintained** — kept for historical reference only.

    Mirrors :class:`SMIXAETraining` but uses cosine-similarity routing via
    ``W_directions`` instead of bottleneck-norm routing.
    """

    b_enc: nn.Parameter
    # b_bottleneck: nn.Parameter
    W_bottleneck: nn.Parameter
    W_latent_dec: nn.Parameter
    W_directions : nn.Parameter # Directions for cos sim router

    def __init__(self, cfg: AffineSMIXAETrainingConfig):
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

        self.cfg.apply_b_dec_to_input = False # Remove bias term - destroys structure
        # self.b_dec.requires_grad_(False)

    def initialize_weights(self) -> None:
        """Initialize base SAE weights then register AffineSMIXAE-specific parameters."""
        super().initialize_weights()
        _init_weights_affine_smixae(self)

    @override
    def get_coefficients(self) -> dict[str, TrainCoefficientConfig | float]:
        """Return an empty coefficient dict (no sparsity penalties are used in AffineSMIXAE)."""
        return {}

    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ``x`` through the full affine pipeline, stashing bottleneck state for training.

        Args:
            x: Input activations of shape ``(batch, d_model)``.

        Returns:
            A 2-tuple ``(h_latent, hidden_pre_latent)`` returned for SAELens compatibility;
            the training loop reads ``self.h_bottleneck`` directly.
        """
        h_latent, hidden_pre_latent, hidden_pre_bottleneck, bottleneck, mask, cos_sims = affine_smixae_encode(self, x)

        # Stash
        self.hidden_pre_bottleneck = hidden_pre_bottleneck
        self.h_bottleneck = bottleneck
        self.mask = mask
        self.cos_sims = cos_sims

        self.hook_sae_acts_pre(hidden_pre_latent)
        self.hook_sae_acts_post(h_latent)
        self.hook_sae_acts_bottleneck(self.h_bottleneck)

        return h_latent, hidden_pre_latent  # These get ignored

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck activations back to the input space.

        Applies the latent decoder, flattens expert dimensions, and projects through
        ``W_dec``. The bias is added only at the output to avoid collapsing manifold structure.
        """
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
        """Forward pass during training."""
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(self.h_bottleneck)

        self.update_threshold(self.mask)

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

        metrics["experts_above_1e-3"] = (self.h_bottleneck.norm(dim=-1) > 1e-3).float().sum(dim=-1).mean()

        post_act_norms = self.h_bottleneck.norm(dim=-1)

        metrics["expert_norm_mean"] = post_act_norms[post_act_norms > 0].mean()

        metrics["dead_experts"] = (self.n_passes_since_fired > self.cfg.dead_after_n_passes).sum().item()

        # Track this during training to make sure the threshold is working properly - woops
        metrics["act_threshold"] = self.threshold
        metrics["experts_above_threshold"] = (
            (self.cos_sims > self.threshold).float().sum(dim=-1).mean()
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

        Returns:
            A dict mapping ``"dead_expert_aux_loss"`` to a scalar tensor.
        """
        losses = {}

        losses["dead_expert_aux_loss"] = self.calculate_topk_aux_loss(
            step_input.sae_in,
            sae_out,
            self.hidden_pre_bottleneck,
            self.n_passes_since_fired > self.cfg.dead_after_n_passes,
        )

        return losses

    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        z_pre_mask: torch.Tensor,  # (batch, n_experts, d_bottleneck) before TopK
        dead_expert_mask: torch.Tensor,  # (n_experts,) bool
    ) -> torch.Tensor:
        """Compute auxiliary reconstruction loss over dead experts.

        Selects ``k_aux`` dead experts by masked cosine similarity, decodes them through the
        full expert pipeline, and penalises the reconstruction error against the current residual.
        Also adds a direction-steering loss that pushes dead expert directions toward the residual.

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
        resid_normed = F.normalize(residual, p=2, dim=-1)
        W_dir_normed = F.normalize(self.W_directions, p=2, dim=-1)
        sims = (resid_normed @ W_dir_normed.T).float()

        # Heuristic: use half of active experts as k_aux
        k_aux = (
            self.cfg.k_experts // 2
        )  # This is actually fine, just need to increase the aux loss coefficient

        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        # Select top-k_aux dead experts by masked cos_sim
        dead_cs = torch.where(dead_expert_mask[None], self.cos_sims, -torch.inf)
        topk = dead_cs.topk(k_aux, dim=-1, sorted=False)

        # Build sparse z with only selected dead experts
        aux_mask = torch.zeros_like(self.cos_sims)
        aux_mask.scatter_(1, topk.indices, 1.0)
        z_aux = z_pre_mask * (aux_mask * self.cos_sims).unsqueeze(-1)  # (batch, n_experts, d_bottleneck)

        # Decode through full expert pipeline
        recons = self.decode(z_aux)

        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()

        steering_loss = (aux_mask * (1.0 - sims)).sum() / aux_mask.sum().clamp(1.0)

        return self.cfg.aux_loss_coefficient * scale * (auxk_loss + steering_loss)

    @torch.no_grad()
    def update_threshold(self, norms_topk: torch.Tensor) -> None:
        """Update the routing threshold via EMA over the minimum positive cosine similarity.

        Args:
            norms_topk: Per-expert cosine similarities from the current batch,
                shape ``(batch, n_experts)``.  Zero entries are inactive experts.
        """
        positive_mask = norms_topk > 0 # cos sim now, I don't care
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

    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return LeakyReLU(1e-4) to avoid dead neurons while preserving expert norms."""
        # use leaky relu to avoid dead neurons; small negative slope avoids impacting expert norm
        return nn.LeakyReLU(negative_slope=1e-4)


def _init_weights_affine_smixae(
    sae: SAE[AffineSMIXAEConfig] | TrainingSAE[AffineSMIXAETrainingConfig],
) -> None:
    """Register AffineSMIXAE-specific parameters on a SAE or TrainingSAE instance.

    Registers four parameters: ``b_enc`` (encoder bias), ``W_directions`` (affine routing,
    random normal init), ``W_bottleneck`` (expert space → bottleneck, Kaiming uniform), and
    ``W_latent_dec`` (bottleneck → expert space, Kaiming uniform).

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

    sae.W_directions = nn.Parameter(
        torch.randn(
            sae.cfg.n_experts,
            sae.cfg.d_in,
            dtype=sae.dtype,
            device=sae.device,
        )
    )

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


def affine_smixae_encode(sae: AffineSMIXAE | AffineSMIXAETraining, x: torch.Tensor, threshold : torch.Tensor = None) -> tuple[torch.Tensor, ...]:  # noqa: E501
    """Shared encoding logic for :class:`AffineSMIXAE` and :class:`AffineSMIXAETraining`.

    Routing is based on cosine similarity between the (normalised) input and each expert's
    ``W_directions`` direction.  During training, BatchTopK is applied over cosine similarities;
    during inference, a scalar threshold gates the experts.

    Args:
        sae: An :class:`AffineSMIXAE` or :class:`AffineSMIXAETraining` instance.
        x: Input activations, shape ``(batch, d_model)``.
        threshold: Scalar threshold tensor for inference-time gating.  Pass ``None`` during
            training to use BatchTopK routing instead.

    Returns:
        A 6-tuple ``(h_latent, hidden_pre_latent, hidden_pre_bottleneck, bottleneck, mask, cosine_similarities)``
        where shapes are ``(batch, n_experts * d_expert)``, same, ``(batch, n_experts, d_bottleneck)``,
        same, ``(batch, n_experts)``, ``(batch, n_experts)``.
    """
    sae_in = sae.process_sae_in(x) # (batch_size, d_in)

    # Compute cosine similarities
    x_norm = F.normalize(x, p=2, dim=-1)
    d_norm = F.normalize(sae.W_directions, p=2, dim=-1) # (n_experts, d_in)

    cosine_similarities = x_norm @ d_norm.T # (batch_size, n_experts)

    # Compute mask
    if threshold is None:
        # Training - Use BatchTopK over cosine similarity
        mask = sae.batchtopk(cosine_similarities)
    else:
        # Inference - Use threshold
        mask = cosine_similarities * (cosine_similarities > threshold)

    # Standard forward
    hidden_pre_latent = sae_in @ sae.W_enc + sae.b_enc
    h_latent = sae.activation_fn(hidden_pre_latent)

    h_latent_unflattened = h_latent.unflatten(-1, (sae.cfg.n_experts, sae.cfg.d_expert)) # (batch_size, n_experts, d_expert)  # noqa: E501

    # Bottleneck
    hidden_pre_bottleneck = (
        torch.einsum("bne,ned->bnd", h_latent_unflattened, sae.W_bottleneck)
        # + sae.b_bottleneck - this causes flattening
    )  # (batch_size, n_experts, d_bottelneck)

    if sae.cfg.rescale_acts_by_decoder_norm:
        hidden_pre_bottleneck = hidden_pre_bottleneck * sae.effective_decoder_norm.unsqueeze(-1)

    bottleneck = hidden_pre_bottleneck * mask.unsqueeze(-1)

    return h_latent, hidden_pre_latent, hidden_pre_bottleneck, bottleneck, mask, cosine_similarities
