import torch
import numpy as np
from typing import Any
import einops as eo
from torch import nn
from sae_lens.saes.batchtopk_sae import BatchTopK
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class SparsityLayerConfig(ABC):
    n_neurons : int
    dead_after_n_passes : int
    dead_neuron_loss_coefficient : float
    sparsity_loss_coefficient : float

class SparsityLayer(nn.Module, ABC):
    def __init__(self, config : SparsityLayerConfig): 
        super().__init__()

        self.cfg = config

        self.register_buffer(
            "n_passes_since_fired",
            torch.zeros(self.cfg.n_neurons, dtype=torch.long)
        )
    
    # Loss Functions 
    @abstractmethod
    def dead_neuron_loss(self, pre_act_x : torch.Tensor) -> torch.Tensor:
        """Use self.dead_mask to get the dead neurons"""
        pass

    def sparsity_loss(self, post_act_x : torch.Tensor) -> torch.Tensor:
        return post_act_x.new_tensor(0.0) # Default to 0 because e.g. TopK-like methods do not have additional sparsity loss
    
    # Forward Functions
    @abstractmethod
    def training_forward(self, x : torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def eval_forward(self, x : torch.Tensor) -> torch.Tensor:
        pass
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.training:
            out = self.training_forward(x)

            # Update the dead neurons
            self._calculate_dead_experts(out)

            # Also calculate the loss functions
            self.loss_dict = {
                'dead_neuron_loss' : self.cfg.dead_neuron_loss_coefficient * self.dead_neuron_loss(x),
                'sparsity_loss' : self.cfg.sparsity_loss_coefficient * self.sparsity_loss(out)
            }
        else:
            out = self.eval_forward(x)

        return out

    # Dead Neuron Tracking
    def not_fired_critera(self, post_act_x : torch.Tensor):
        """
        Criteria for not firing - depends on whether you want to use norm and shape of the feature itself
        
        For standard SAEs you should override this to just be whether the feature value is > 0

        SMIXAE has to aggregate over the last dimension first
        """
        return post_act_x.norm(dim=-1) > 0 

    @torch.no_grad()
    def _calculate_dead_experts(self, post_act_x : torch.Tensor):
        """
        Private function used to update the dead neuron tracker

        We take a non-negative `scaler_criteria` variable to be compatible with future methods that may not use norm.

        However, we expect that typical inputs for scalar_criteria will simply be the feature norm
        """

        fired_in_batch = self.not_fired_critera(post_act_x).any(dim=0)

        self.n_passes_since_fired = torch.where(
                fired_in_batch, # Condition
                torch.zeros_like(self.n_passes_since_fired), # True condition (Resets)
                self.n_passes_since_fired + 1, # False condition (increases # passes since fired)
            )
        
    @property
    def dead_mask(self):
        return self.n_passes_since_fired > self.cfg.dead_after_n_passes