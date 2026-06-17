"""Base class for SMIXAE-style architectures.

Provides a single :class:`BaseSMIXAE` that serves **both training and inference**.
It removes the weight-matrix assumptions from the SAELens base classes while preserving
full SAELens compatibility (hooks, normalisation, hook_z reshaping, save/load,
architecture registration).

Because SAELens' :class:`~sae_lens.saes.sae.TrainingSAE` is itself a
:class:`~sae_lens.saes.sae.SAE`, a single subclass of ``TrainingSAE`` exposes the full
inference API (``encode``/``decode``/``forward``/``process_sae_in``/``save_model``/
``load_from_disk``) **and** the training API (``training_forward_pass``/
``encode_with_hidden_pre``/``get_coefficients``/``calculate_aux_loss``). One class, no
train/inference split.

Pre-built weight configuration helpers are exposed at module level so that
downstream ``initialize_weights`` implementations can compose them freely:

- :func:`register_standard_linear_weights` — W_enc, W_dec, b_dec (mirrors
  ``SAE.initialize_weights`` including ``decoder_init_norm`` normalisation).
- :func:`register_smixae_v1_bottleneck_weights` — b_enc, W_bottleneck, W_latent_dec
  (the per-expert bottleneck projection from SMIXAE v1).
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import torch
from sae_lens.saes.sae import TrainingSAE, TrainingSAEConfig
from torch import nn

T_BASE_CFG = TypeVar("T_BASE_CFG", bound=TrainingSAEConfig)


# ---------------------------------------------------------------------------
# Pre-built weight configuration helpers
# ---------------------------------------------------------------------------


def register_standard_linear_weights(sae: Any) -> None:
    """Register W_enc, W_dec, and b_dec with standard Kaiming initialisation.

    Matches the shapes and initialisation of ``SAE.initialize_weights()``.
    Also applies the optional ``decoder_init_norm`` column-normalisation step
    from ``TrainingSAE.initialize_weights()`` when that attribute is set.

    Call from ``initialize_weights()`` whenever the architecture uses a
    standard linear encoder/decoder projection.

    Args:
        sae: A :class:`BaseSMIXAE` instance.
            Must expose ``cfg.d_in``, ``cfg.d_sae``, ``dtype``, and ``device``.
    """
    sae.b_dec = nn.Parameter(
        torch.zeros(sae.cfg.d_in, dtype=sae.dtype, device=sae.device)
    )
    w_dec_data = torch.empty(sae.cfg.d_sae, sae.cfg.d_in, dtype=sae.dtype, device=sae.device)
    nn.init.kaiming_uniform_(w_dec_data)
    sae.W_dec = nn.Parameter(w_dec_data)
    sae.W_enc = nn.Parameter(sae.W_dec.data.T.clone().detach().contiguous())

    # TrainingSAEConfig optionally normalises decoder columns on init.
    decoder_init_norm = getattr(sae.cfg, "decoder_init_norm", None)
    if decoder_init_norm is not None:
        with torch.no_grad():
            sae.W_dec.data /= sae.W_dec.norm(dim=-1, keepdim=True)
            sae.W_dec.data *= decoder_init_norm
        sae.W_enc.data = sae.W_dec.data.T.clone().detach().contiguous()


def register_smixae_v1_bottleneck_weights(sae: Any) -> None:
    """Register the SMIXAE v1 per-expert bottleneck parameters.

    Registers:

    - ``b_enc`` — encoder bias, shape ``(n_experts * d_expert,)``, zero init.
    - ``W_bottleneck`` — expert→bottleneck projection,
      shape ``(n_experts, d_expert, d_bottleneck)``, Kaiming init.
    - ``W_latent_dec`` — bottleneck→expert decoder projection,
      shape ``(n_experts, d_bottleneck, d_expert)``, Kaiming init.

    Args:
        sae: A :class:`BaseSMIXAE` instance.
            Must expose ``cfg.n_experts``, ``cfg.d_expert``, ``cfg.d_bottleneck``,
            ``dtype``, and ``device``.
    """
    sae.b_enc = nn.Parameter(
        torch.zeros(
            sae.cfg.n_experts * sae.cfg.d_expert,
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


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseSMIXAE(TrainingSAE[T_BASE_CFG], Generic[T_BASE_CFG], ABC):
    """Abstract base class for SMIXAE-style architectures (training **and** inference).

    A single class that subclasses SAELens' :class:`~sae_lens.saes.sae.TrainingSAE`, so one
    instance can be trained (``train_toy_sae`` / ``SAETrainer`` duck-type it) and used for
    inference/eval — no separate inference class. It removes the weight-matrix assumptions
    baked into SAELens' :class:`~sae_lens.saes.sae.SAE`:

    - ``initialize_weights()`` is **abstract** — no default W_enc/W_dec/b_dec are
      created.  Subclasses register exactly the parameters they need.
    - ``process_sae_in()`` guards ``b_dec`` access so architectures without a
      ``b_dec`` parameter still work.
    - ``fold_activation_norm_scaling_factor()`` is a **no-op** by default.
      Override to rescale architecture-specific weights.

    All SAELens infrastructure (hooks, normalisation, hook_z reshaping, save/load,
    architecture registration, ``mse_loss_fn``) is inherited unchanged from
    :class:`~sae_lens.saes.sae.TrainingSAE`.

    Subclasses must use a :class:`~sae_lens.saes.sae.TrainingSAEConfig` (the single config
    carries both structural and training fields; inference simply ignores the training ones).

    Use :func:`register_standard_linear_weights` and
    :func:`register_smixae_v1_bottleneck_weights` inside ``initialize_weights()``
    to compose standard weight groups without reimplementing boilerplate.

    Example::

        @dataclass
        class MyConfig(TrainingSAEConfig):
            n_experts: int = 64
            d_expert: int = 8
            d_bottleneck: int = 3

            @classmethod
            def architecture(cls) -> str:
                return "my_arch"

        class MyModel(BaseSMIXAE[MyConfig]):
            def initialize_weights(self) -> None:
                register_standard_linear_weights(self)
                register_smixae_v1_bottleneck_weights(self)
                self.register_buffer(
                    "threshold",
                    torch.tensor(0.0, dtype=torch.double, device=self.device),
                )

            # Training contract (TrainingSAE):
            def encode_with_hidden_pre(self, x): ...
            def get_coefficients(self): ...
            def calculate_aux_loss(self, *a, **k): ...

            # Inference contract (SAE):
            def encode(self, x): ...
            def decode(self, feature_acts): ...
    """

    def __init__(self, cfg: T_BASE_CFG, use_error_term: bool = False) -> None:
        cfg.apply_b_dec_to_input = False
        super().__init__(cfg, use_error_term)

    @abstractmethod
    def initialize_weights(self) -> None:
        """Register all learnable parameters and buffers for this architecture.

        Do **not** call ``super().initialize_weights()`` here — the default SAELens
        implementation unconditionally creates W_enc, W_dec, and b_dec.  Use the
        pre-built helpers (:func:`register_standard_linear_weights`,
        :func:`register_smixae_v1_bottleneck_weights`) to compose the weight groups
        you need, then register any architecture-specific parameters directly.
        """

    def process_sae_in(self, sae_in: torch.Tensor) -> torch.Tensor:
        """Pre-process input, skipping the b_dec subtraction when b_dec is absent.

        Mirrors ``SAE.process_sae_in`` but wraps the ``b_dec`` attribute access in
        ``hasattr``, so architectures that do not register a ``b_dec`` parameter
        pass through cleanly.

        Args:
            sae_in: Raw input tensor.

        Returns:
            Pre-processed tensor ready for ``encode()`` / ``encode_with_hidden_pre()``.
        """
        sae_in = sae_in.to(self.dtype)
        sae_in = self.reshape_fn_in(sae_in)
        sae_in = self.hook_sae_input(sae_in)
        sae_in = self.run_time_activation_norm_fn_in(sae_in)
        if self.cfg.apply_b_dec_to_input and hasattr(self, "b_dec"):
            sae_in = sae_in - self.b_dec
        return sae_in

    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float) -> None:
        """No-op fold — only resets the normalisation mode to ``"none"``.

        Override to rescale architecture-specific weight matrices after training.

        Args:
            scaling_factor: The activation norm scaling factor from the trainer.
        """
        self.cfg.normalize_activations = "none"

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input activations to feature space.

        Args:
            x: Input activations, shape ``(batch, d_in)``.

        Returns:
            Feature activations.
        """

    @abstractmethod
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode feature activations back to input space.

        Args:
            feature_acts: Feature activations returned by ``encode()``.

        Returns:
            Reconstructed input, shape ``(batch, d_in)``.
        """
