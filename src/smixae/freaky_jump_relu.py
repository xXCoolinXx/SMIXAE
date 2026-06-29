from dataclasses import dataclass
from typing import Any, ClassVar, override

import einops as eo
import torch
from torch import nn

import smixae.einops_norms as eon
from smixae.sparsity_layer import SparsityLayer, SparsityLayerConfig
from smixae.grump_relu import grump_relu_forward


def rectangle_bandwidth(x : torch.Tensor, bandwidth : float) -> torch.Tensor:
    rectangle = (-bandwidth/2 < x) & (x < bandwidth / 2)

    return rectangle / bandwidth

@dataclass
class FreakyJumpReLULayerConfig(SparsityLayerConfig):
    n_neurons : int = 2048
    dead_after_n_passes : int = 200

    threshold : float = 0.1 # Fixed threshold by which we grade features. I don't recommend changing this
    hardness_coefficient : float = 1.0

class FreakyJumpReLULayer(SparsityLayer):
    config_type : ClassVar[type[SparsityLayerConfig]] = FreakyJumpReLULayerConfig

    def __init__(self, config : FreakyJumpReLULayerConfig):
        super().__init__(config)

        self.threshold = torch.tensor(self.cfg.threshold)

    @override
    def sparsity_loss(self, pre_act_x : torch.Tensor, post_act_x : torch.Tensor):
        """Anthropic JumpReLU loss version.

        They use tanh to approximate the L0 function
        I am pretty skeptical of this and I think there are better L0 optimization methods that can be pulled from the L0 optimization literature
        For example, the L0 output depends on the expected feature norm, which is not ideal
        It should depend on the bottleneck.
        """
        # Essentialy provides a tanh signal pushing features down in norm if they are only slightly above the threshold, though stops GAFing when it strongly activates

        # assert (torch.tanh(
        #             torch.relu(
        #                 self.cfg.hardness_coefficient * (eon.l2(post_act_x, '... n_experts d_bottleneck -> ... n_experts') - self.threshold)
        #                 )
        #             ) >= 0).all()

        # return eo.reduce(
        #     eo.reduce(
        #         torch.sigmoid(
        #                 self.cfg.hardness_coefficient * (eon.l2(pre_act_x, '... n_experts d_bottleneck -> ... n_experts') - self.threshold)
        #             ),
        #         '... n_experts -> ...',
        #         reduction='sum'
        #     ),
        #     '... -> ',
        #     reduction = 'mean'
        # )
    
        return eo.reduce(
            eo.reduce(
                torch.tanh(
                    torch.relu(
                        self.cfg.hardness_coefficient * (eon.l2(post_act_x, '... n_experts d_bottleneck -> ... n_experts') - self.threshold)
                        )
                    ),
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
        return grump_relu_forward(x, self.threshold)

    @override
    def eval_forward(self, x):
        return grump_relu_forward(x, self.threshold)
