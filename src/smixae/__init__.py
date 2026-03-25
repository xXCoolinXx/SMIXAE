from sae_lens import register_sae_class, register_sae_training_class

from smixae.affine_smixae import AffineSMIXAE, AffineSMIXAEConfig, AffineSMIXAETraining, AffineSMIXAETrainingConfig
from smixae.smixae import SMIXAE, SMIXAEConfig, SMIXAETraining, SMIXAETrainingConfig

register_sae_class("smixae", SMIXAE, SMIXAEConfig)
register_sae_training_class("smixae", SMIXAETraining, SMIXAETrainingConfig)

register_sae_class("affine_smixae", AffineSMIXAE, AffineSMIXAEConfig)
register_sae_training_class("affine_smixae", AffineSMIXAETraining, AffineSMIXAETrainingConfig)

__all__ = ["SMIXAE", "SMIXAEConfig", "SMIXAETraining", "SMIXAETrainingConfig"]
