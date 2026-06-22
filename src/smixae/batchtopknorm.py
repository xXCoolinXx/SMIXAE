from dataclasses import dataclass
from typing import ClassVar, override

import torch
from sae_lens.saes.batchtopk_sae import BatchTopK

from smixae.sparsity_layer import SparsityLayer, SparsityLayerConfig


@dataclass
class BatchTopKNormLayerConfig(SparsityLayerConfig):
    n_neurons : int = 2048
    dead_after_n_passes : int = 1000

    k : int = 64
    threshold_learning_rate : float = 0.01
    threshold_dtype : torch.dtype = torch.double

class BatchTopKNormLayer(SparsityLayer):
    r"""Computes the following function during training:

    $$ x * \mathbb{I}(BatchTopK(|x|_p) > 0)$$

    and the following function during inference

    $$ x * \mathbb{I}(|x|_p) > 0)$$
    
    Norm is computed on the last dimension of x

    Also updates inference threshold based on threshold_learning_rate. 
    """
    config_type : ClassVar[type[SparsityLayerConfig]] = BatchTopKNormLayerConfig

    def __init__(self, config : BatchTopKNormLayerConfig):
        super().__init__(config)

        self.register_buffer(
            'threshold',
            torch.tensor(0.0, dtype=self.cfg.threshold_dtype)
        )
        self.batchtopk = BatchTopK(self.cfg.k)

    @override
    def training_forward(self, x):
        norms = x.norm(dim=-1)

        norm_acts = self.batchtopk(norms)
        mask = norm_acts > 0
        self.update_threshold(norm_acts)

        return x * mask.unsqueeze(-1)

    @override
    def eval_forward(self, x):
        norms = x.norm(dim=-1)
        mask = norms > self.threshold

        return x * mask.unsqueeze(-1)

    def _apply(self, fn, *args, **kwargs):
        # Special apply function to avoid .to changing the data type of the threshold
        keep = self.threshold.dtype
        super()._apply(fn, *args, **kwargs)   # moves device + dtype as usual
        self.threshold = self.threshold.to(keep)  # restore dtype, keep new device
        return self

    @torch.no_grad()
    def update_threshold(self, mask: torch.Tensor) -> None:
        positive_mask = mask > 0
        lr = self.cfg.threshold_learning_rate
        with torch.autocast(self.threshold.device.type, enabled=False):
            if positive_mask.any():
                # Get the minimum activation in the batch (estimator for BatchTopK operation)
                min_positive = mask[positive_mask].min().to(self.threshold.dtype)

                # Exponential moving overage over threshold
                self.threshold = (1 - lr) * self.threshold + lr * min_positive  # type: ignore[assignment]

    @override
    def dead_neuron_loss(
        self, pre_act_x: torch.Tensor
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
        dead_mask = self.dead_mask

        if not dead_mask.any():
            return self.threshold.new_tensor(0.0)

        expert_norms = pre_act_x.norm(dim=-1)
        dead_norms = expert_norms[:, dead_mask]
        shortfall = torch.relu(self.threshold.detach().float() - dead_norms)

        return shortfall.sum(dim=-1).mean()
