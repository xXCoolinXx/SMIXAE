"""SMIXAE public API and SAELens architecture registration.

Importing this module (directly or transitively) registers all SMIXAE architectures
with SAELens so they can be loaded via :meth:`SAE.load_from_disk` without any
additional setup.

Note: ``AffineSMIXAE`` is not actively used or maintained. It is exported here for
historical completeness only.
"""
from sae_lens import register_sae_class, register_sae_training_class

from smixae.affine_smixae import AffineSMIXAE, AffineSMIXAEConfig, AffineSMIXAETraining, AffineSMIXAETrainingConfig
from smixae.base_smixae import (
    BaseSMIXAE,
    BaseSMIXAETraining,
    register_smixae_v1_bottleneck_weights,
    register_standard_linear_weights,
)
from smixae.smixae import SMIXAE, SMIXAEConfig, SMIXAETraining, SMIXAETrainingConfig
from smixae.smixae_rebased import SMIXAERebased, SMIXAERebasedConfig, SMIXAERebasedTraining, SMIXAERebasedTrainingConfig

register_sae_class("smixae", SMIXAE, SMIXAEConfig)
register_sae_training_class("smixae", SMIXAETraining, SMIXAETrainingConfig)

register_sae_class("affine_smixae", AffineSMIXAE, AffineSMIXAEConfig)
register_sae_training_class("affine_smixae", AffineSMIXAETraining, AffineSMIXAETrainingConfig)

register_sae_class("smixae_rebased", SMIXAERebased, SMIXAERebasedConfig)
register_sae_training_class("smixae_rebased", SMIXAERebasedTraining, SMIXAERebasedTrainingConfig)

__all__ = [
    "SMIXAE",
    "SMIXAEConfig",
    "SMIXAETraining",
    "SMIXAETrainingConfig",
    "BaseSMIXAE",
    "BaseSMIXAETraining",
    "register_standard_linear_weights",
    "register_smixae_v1_bottleneck_weights",
    "SMIXAERebased",
    "SMIXAERebasedConfig",
    "SMIXAERebasedTraining",
    "SMIXAERebasedTrainingConfig",
]
