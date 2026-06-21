from typing import Any

import einops as eo
import torch
from sae_lens.saes.batchtopk_sae import BatchTopK
from torch import nn


def rectangle_bandwidth(x : torch.Tensor, bandwidth : float) -> torch.Tensor:
    rectangle = (-bandwidth/2 < x) & (x < bandwidth / 2)

    return rectangle / bandwidth

def grump_relu_forward(x : torch.Tensor, threshold : torch.Tensor):
    """GrumpReLU forward implementation.
    
    Defined as a separate function since it will be used both in the forward pass of the Autograd function as well as in the GrumpReLU layer.
    """
    norms = x.norm(dim=-1)
    mask = norms > threshold

    return x * mask.unsqueeze(-1)

class GrumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(
        x : torch.Tensor,
        threshold : torch.Tensor,
        bandwidth : float
        ) -> torch.Tensor:

        # x, output : (..., n_experts, d_expert)
        return grump_relu_forward(x, threshold)

    @staticmethod
    def setup_context(
        ctx : Any,
        inputs: tuple[torch.Tensor, torch.Tensor, float],
        output: torch.Tensor
        ) -> None:
        x, threshold, bandwidth = inputs
        del output

        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(
        ctx : Any,
        grad_outputs : torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, None]:

        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # x, grad_output : ..., n_experts, d_expert
        # threshold: n_experts
        # Want: grad sizes to be ..., n_experts, d_expert and ..., n_experts

        norms = x.norm(dim=-1) # ..., n_experts
        mask = (norms > threshold).to(x)

        # All dimensions of the output have derivative 1 wrt the x
        x_grad = mask.unsqueeze(-1) * grad_outputs

        # x_grad = eo.reduce(partials * grad_outputs, '... n_experts d_expert -> n_experts d_expert', reduction='sum')

        # Threshold is the same for every sample in the batch (BROADCAST OVER THE BATCH) so requires a sum
        threshold_grad = eo.reduce(
             - x *
             (threshold * rectangle_bandwidth(norms - threshold, ctx.bandwidth) / norms).unsqueeze(-1)
             * grad_outputs,
             '... n_experts d_expert -> n_experts',
             reduction='sum'
        )

        return (x_grad, threshold_grad, None)

# BatchTopKNorm Layer

class BatchTopKNorm(nn.Module):
    r"""Computes the following function during training:

    $$ x * \mathbb{I}(BatchTopK(|x|_p) > 0)$$

    and the following function during inference

    $$ x * \mathbb{I}(|x|_p) > 0)$$
    
    Norm is computed on the last dimension of x

    Also updates inference threshold based on threshold_learning_rate. 
    """
    def __init__(self, k : int, threshold_learning_rate : float, device : torch.device, p : float = 2, dtype = torch.double):
        super().__init__()
        self.k = k
        self.threshold_learning_rate = threshold_learning_rate
        self.p = p
        self.batchtopk = BatchTopK(k)
        self.register_buffer('threshold', torch.tensor(0.0, dtype=dtype, device=device))

    def forward(self, x : torch.Tensor):
        norms = x.norm(dim=-1, p=self.p)

        if self.training:
            mask = self.batchtopk(norms) > 0
            self.update_threshold(mask)
        else:
            mask = norms > self.threshold

        return x * mask.unsqueeze(-1)

    @torch.no_grad()
    def update_threshold(self, norms: torch.Tensor) -> None:
        positive_mask = norms > 0
        lr = self.threshold_learning_rate
        with torch.autocast(self.threshold.device.type, enabled=False):
            if positive_mask.any():
                min_positive = norms[positive_mask].min().to(self.threshold.dtype)
                self.threshold = (1 - lr) * self.threshold + lr * min_positive  # type: ignore[assignment]

    def _apply(self, fn, *args, **kwargs):
        # Special apply function to avoid .to changing the data type of the threshold
        keep = self.threshold.dtype
        super()._apply(fn, *args, **kwargs)   # moves device + dtype as usual
        self.threshold = self.threshold.to(keep)  # restore dtype, keep new device
        return self

def calculate_dead_expert_aux_loss(
        self, pre_act_bottleneck: torch.Tensor, dead_expert_mask: torch.Tensor
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
        dead_expert_mask = self.n_passes_since_fired > self.cfg.dead_after_n_passes

        if dead_expert_mask is None or not dead_expert_mask.any():
            return self.batchtopk.threshold.new_tensor(0.0)

        expert_norms = pre_act_bottleneck.norm(dim=-1)
        dead_norms = expert_norms[:, dead_expert_mask]
        shortfall = torch.relu(self.batchtopk.threshold.detach().float() - dead_norms)

        return self.cfg.aux_loss_coefficient * shortfall.sum(dim=-1).mean()
