import torch
import numpy as np
from typing import Any
import einops as eo
from torch import nn
from sae_lens.saes.batchtopk_sae import BatchTopK
from abc import ABC, abstractmethod
from dataclasses import dataclass
from smixae.sparsity_layer import SparsityLayerConfig, SparsityLayer

def rectangle_bandwidth(x : torch.Tensor, bandwidth : float) -> torch.Tensor:
    rectangle = (-bandwidth/2 < x) & (x < bandwidth / 2)
    
    return rectangle / bandwidth

def grump_relu_forward(x : torch.Tensor, threshold : torch.Tensor):
    """
    GrumpReLU forward implementation. 
    
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
    

@dataclass 
class GrumpReLULayerConfig(SparsityLayerConfig):
    pass

class GrumpReLULayer(SparsityLayer):
    def __init__(self, bandwidth : float, threshold_size : torch.Tensor, init_threshold : float):
        super().__init__()
        
        self.threshold = nn.Parameter(
            init_threshold * torch.ones(threshold_size),
        )
        self.bandwidth = bandwidth

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Clamp value of threshold in place
        with torch.no_grad():
            self.threshold.clamp_(min=0.0)

        return GrumpReLU.apply(x, self.threshold, self.bandwidth)