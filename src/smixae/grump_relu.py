from dataclasses import dataclass
from typing import Any, ClassVar, override

import einops as eo
import torch
from torch import nn

import smixae.einops_norms as eon
from smixae.sparsity_layer import SparsityLayer, SparsityLayerConfig


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

        norms = eon.l2(x, '... n_experts d_bottleneck -> n_experts')
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

@dataclass
class GrumpReLULayerConfig(SparsityLayerConfig):
    n_neurons : int = 2048
    dead_after_n_passes : int = 200

    bandwidth : float = 2.0
    init_threshold : float = 0.01
    hardness_coefficient : float = 4.0 # Hardness coefficient used for the soft tanh/sigmoid penalty

class GrumpReLULayer(SparsityLayer):
    config_type : ClassVar[type[SparsityLayerConfig]] = GrumpReLULayerConfig

    def __init__(self, config : GrumpReLULayerConfig):
        super().__init__(config)

        self.threshold = nn.Parameter(
            self.cfg.init_threshold * torch.ones(self.cfg.n_neurons),
        )

    @override
    def sparsity_loss(self, pre_act_x : torch.Tensor, post_act_x : torch.Tensor):
        """Anthropic JumpReLU loss version
        They use tanh to approximate the L0 function
        I am pretty skeptical of this and I think there are better L0 optimization methods that can be pulled from the L0 optimization literature
        For example, the L0 output depends on the expected feature norm, which is not ideal
        It should depend on the bottleneck
        """
        return eo.reduce(
            eo.reduce(
                torch.tanh(self.cfg.hardness_coefficient * eon.l2(post_act_x, '... n_experts d_bottleneck -> ... n_experts')),
                '... n_experts -> ...',
                reduction='sum'
            ),
            '... -> ',
            reduction = 'mean'
        )

    @override
    def dead_neuron_loss(self, pre_act_x):
        # Essentially just a hinge loss that provides upward pressure on dead expert norms to get them to cross their threshold

        dead_mask = self.dead_mask

        if not dead_mask.any():
            return self.threshold.new_tensor(0.0)

        shortfall = torch.relu(self.threshold - eon.l2(pre_act_x, '... n_experts d_bottleneck -> ... n_experts'))

        return eo.reduce(
            eo.reduce(
                shortfall * dead_mask,
                '... n_experts -> ...',
                reduction = 'sum'
            ),
            '... -> ',
            reduction='mean'
        )

    @override
    def training_forward(self, x):
        # Clamp value of threshold in place
        with torch.no_grad():
            self.threshold.clamp_(min=0.0)

        return GrumpReLU.apply(x, self.threshold, self.cfg.bandwidth)

    @override
    def eval_forward(self, x):
        return grump_relu_forward(x, self.threshold)
