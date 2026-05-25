"""Marimo notebook: iterative SMIXAE development on the toy model.

Run with:
    uv run marimo run notebooks/toy_smixae.py
    uv run marimo edit notebooks/toy_smixae.py   # editable
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import plotly.graph_objects as go
    import torch
    from plotly.subplots import make_subplots

    # Ensure src/ is on the path when run from the repo root
    _src = Path(__file__).parent.parent / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    from collections.abc import Callable
    from dataclasses import dataclass

    from sae_lens.saes.batchtopk_sae import BatchTopK
    from sae_lens.saes.sae import (
        SAEConfig,
        SAEMetadata,
        TrainCoefficientConfig,
        TrainingSAEConfig,
        TrainStepInput,
        TrainStepOutput,
    )
    from sae_lens.synthetic import train_toy_sae
    from transformer_lens.hook_points import HookPoint
    from typing_extensions import override

    import smixae  # noqa: F401 — registers architecture with SAELens
    from smixae.base_smixae import (
        BaseSMIXAE,
        BaseSMIXAETraining,
        register_smixae_v1_bottleneck_weights,
        register_standard_linear_weights,
    )
    from toy.metrics import compute_cofiring_matrix, compute_metrics, compute_restricted_r2
    from toy.plot import plot_all_experts_with_originals
    from toy.zoo import (
        EvalData,
        ManifoldActivationGenerator,
        ManifoldZoo,
        build_manifold_zoo,
        generate_eval_set,
    )

    import einops as eo

    from torch import nn

    mo.md("## Imports loaded")
    return (
        BaseSMIXAE,
        BaseSMIXAETraining,
        BatchTopK,
        Callable,
        EvalData,
        HookPoint,
        ManifoldActivationGenerator,
        ManifoldZoo,
        Path,
        SAEConfig,
        SAEMetadata,
        TrainCoefficientConfig,
        TrainStepInput,
        TrainStepOutput,
        TrainingSAEConfig,
        build_manifold_zoo,
        compute_cofiring_matrix,
        compute_metrics,
        compute_restricted_r2,
        dataclass,
        eo,
        generate_eval_set,
        go,
        make_subplots,
        mo,
        nn,
        override,
        plot_all_experts_with_originals,
        torch,
        train_toy_sae,
    )


@app.cell
def _(Path, mo, torch):
    # ── Dataset params ──────────────────────────────────────
    seed        = 0
    d_in        = 128
    l0          = 4           # active manifolds per sample
    sigma_bias  = 5.0

    # ── Model architecture ──────────────────────────────────
    n_experts    = 48         # should match n_instances in zoo (48)
    d_expert     = 16
    d_bottleneck = 3          # 3-D for direct visualisation
    d_sae = n_experts * d_expert

    # ── Training ────────────────────────────────────────────
    k_experts         = 4     # active experts per sample (BatchTopK budget)
    training_samples  = 50_000_000
    batch_size        = 2048
    lr                = 3e-3
    lr_warm_up_steps  = 2_000
    n_snapshots       = 20    # how many times compute_restricted_r2 is called during training

    # ── Infrastructure ──────────────────────────────────────
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    toy_data_dir = Path("toy_data")

    mo.md(f"""
    **Config**

    | Param | Value |
    |---|---|
    | seed | {seed} |
    | d_in | {d_in} |
    | l0 | {l0} |
    | sigma_bias | {sigma_bias} |
    | n_experts | {n_experts} |
    | k_experts | {k_experts} |
    | training_samples | {training_samples:,} |
    | device | {device} |
    """)
    return (
        batch_size,
        d_bottleneck,
        d_expert,
        d_in,
        device,
        k_experts,
        l0,
        lr,
        lr_warm_up_steps,
        n_experts,
        n_snapshots,
        seed,
        sigma_bias,
        toy_data_dir,
        training_samples,
    )


@app.cell
def _(
    EvalData,
    ManifoldZoo,
    build_manifold_zoo,
    d_in,
    device,
    generate_eval_set,
    l0,
    mo,
    seed,
    sigma_bias,
    toy_data_dir,
):
    def _dataset_dir(base, s, d, l0_val, b):
        return base / f"seed{s}_d{d}_l{l0_val}_b{b:g}"

    _ds = _dataset_dir(toy_data_dir, seed, d_in, l0_val=l0, b=sigma_bias)
    _zoo_path  = _ds / "zoo.pt"
    _eval_path = _ds / "eval.pt"

    if _zoo_path.exists() and _eval_path.exists():
        zoo       = ManifoldZoo.load(_zoo_path, device=device)
        eval_data = EvalData.load(_eval_path, device="cpu")
        _source   = f"Loaded from `{_ds}`"
    else:
        _ds.mkdir(parents=True, exist_ok=True)
        zoo = build_manifold_zoo(d_in=d_in, seed=seed, device=device, sigma_bias=sigma_bias)
        zoo.save(_zoo_path)
        eval_data = generate_eval_set(zoo, n_samples=200_000, l0=l0, seed=seed + 1, device=device)
        eval_data.save(_eval_path)
        _source = f"Generated and saved to `{_ds}`"

    _type_counts = {}
    for _inst in zoo.instances:
        _type_counts[_inst.type_name] = _type_counts.get(_inst.type_name, 0) + 1

    mo.md(f"""
    **Toy Data** — {_source}

    - Zoo: {len(zoo.instances)} instances, {zoo.n_atoms} atoms, d_in={zoo.d_in}
    - Eval set: {eval_data.x.shape[0]:,} samples of shape {tuple(eval_data.x.shape)}
    - Manifold types: {dict(sorted(_type_counts.items()))}
    """)
    return eval_data, zoo


@app.cell
def _(
    BaseSMIXAE,
    BaseSMIXAETraining,
    BatchTopK,
    Callable,
    HookPoint,
    SAEConfig,
    SAEMetadata,
    TrainCoefficientConfig,
    TrainStepInput,
    TrainStepOutput,
    TrainingSAEConfig,
    d_bottleneck,
    d_expert,
    dataclass,
    device,
    eo,
    k_experts,
    mo,
    n_experts,
    nn,
    override,
    seed,
    torch,
    zoo,
):
    # ══════════════════════════════════════════════════════════════════════════════
    # SMIXAERebased — self-contained copy.  Edit freely; does NOT affect the library.
    # Base classes and weight helpers are stable library imports — edit the classes below.
    # ══════════════════════════════════════════════════════════════════════════════

    # ── Shared encode ─────────────────────────────────────────────────────────────

    def register_smixae_v2_weights(sae : BaseSMIXAE | BaseSMIXAETraining):
        """
        Update weight configuration for SMIXAEv2. 

        Key principles:
        - Remove unnneccessary additional linear layer
        - Encoders and decoder are 3-tensors - no flattening!
        - Bias is now a 2-tensor
        """
        sae.W_enc = nn.Parameter(
            torch.zeros(
                sae.cfg.d_in,
                sae.cfg.n_experts,
                sae.cfg.d_expert,
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

        sae.W_dec = nn.Parameter(
            torch.zeros(
                sae.cfg.n_experts,
                sae.cfg.d_bottleneck,
                sae.cfg.d_in,
                dtype=sae.dtype,
                device=sae.device,
            )
        )

        sae.b_enc = nn.Parameter(
            torch.zeros(
                sae.cfg.n_experts,
                sae.cfg.d_expert,
                dtype=sae.dtype,
                device=sae.device,
            )
        )

        sae.b_dec = nn.Parameter(
            torch.zeros(
                sae.cfg.d_in,
                dtype=sae.dtype,
                device=sae.device,
            )
        )

        nn.init.kaiming_uniform_(sae.W_enc)
        nn.init.kaiming_uniform_(sae.W_bottleneck)
        nn.init.kaiming_uniform_(sae.W_dec)


    def _smixae_rebased_encode(sae, x: torch.Tensor):  # noqa: ANN001
        """X → W_enc + b_enc → LeakyReLU → (n_experts, d_expert) → W_bottleneck → bottleneck.

        Do not fear the einsums, they are just matmuls but for the 3-tensors.

        Returns (h_charts, pre_act_charts, pre_act_bottleneck).
        """
        sae_in = sae.process_sae_in(x)


        pre_act_charts = eo.einsum(sae_in, sae.W_enc, "batch_size d_in, d_in n_experts d_expert -> batch_size n_experts d_expert") + sae.b_enc
        h_charts = sae.activation_fn(pre_act_charts)

        pre_act_bottleneck = eo.einsum(h_charts, sae.W_bottleneck, "batch_size n_experts d_expert, n_experts d_expert d_bottleneck -> batch_size n_experts d_bottleneck")

        return pre_act_charts, h_charts, pre_act_bottleneck

    def _smixae_decode(sae : SMIXAERebased | SMIXAERebasedTraining, z : torch.Tensor) -> torch.Tensor:
        """
        Decode feature acts
        """
        # Divide by frob norm per expert, provides stability
        W_dec_normed = sae.W_dec / sae.effective_decoder_norm.view(-1, 1, 1)

        # Apply Decoder
        return eo.einsum(W_dec_normed, z, "n_experts d_bottleneck d_in, batch_size n_experts d_bottleneck -> batch_size d_in") + sae.b_dec

    # ── Inference config + class ──────────────────────────────────────────────────

    @dataclass
    class SMIXAERebasedConfig(SAEConfig):
        """Inference config."""

        n_experts: int = 1024
        d_expert: int = 16
        d_bottleneck: int = 3
        rescale_acts_by_decoder_norm: bool = True
        d_sae : int = 1024 * 3

        @override
        @classmethod
        def architecture(cls) -> str:
            return "smixae_nb"

    class SMIXAERebased(BaseSMIXAE[SMIXAERebasedConfig]):
        """Inference-only SMIXAERebased."""

        @override
        def initialize_weights(self) -> None:
            register_smixae_v2_weights(self)

            self.register_buffer("threshold", torch.tensor(0.0, dtype=torch.double, device=self.device, requires_grad=False))
            self.register_buffer("n_passes_since_fired", torch.zeros(self.cfg.n_experts, dtype=torch.long))

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            _, _, pre_act_bottleneck = _smixae_rebased_encode(self, x)
            mask = pre_act_bottleneck.norm(dim=-1) > self.threshold  # type: ignore[operator]
            return pre_act_bottleneck * mask.unsqueeze(-1)

        def encode_with_charts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h_charts, _, pre_act_bottleneck = _smixae_rebased_encode(self, x)
            mask = pre_act_bottleneck.norm(dim=-1) > self.threshold  # type: ignore[operator]
            return pre_act_bottleneck * mask.unsqueeze(-1), h_charts

        def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
            out = _smixae_decode(self, feature_acts)

            out = self.hook_sae_recons(out)
            out = self.run_time_activation_norm_fn_out(out)
            return self.reshape_fn_out(out, self.d_head)

        def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
            return torch.nn.LeakyReLU(negative_slope=1e-4)

        @override
        @torch.no_grad()
        def fold_activation_norm_scaling_factor(self, scaling_factor: float) -> None:
            self.W_enc.data *= scaling_factor
            self.W_dec.data /= scaling_factor
            self.b_dec.data /= scaling_factor
            self.cfg.normalize_activations = "none"
            if self.cfg.rescale_acts_by_decoder_norm:
                sf_sqrt = scaling_factor**0.5
                self.W_dec.data *= sf_sqrt
                self.threshold = self.threshold / sf_sqrt  # type: ignore[assignment]
            else:
                self.threshold = self.threshold / scaling_factor  # type: ignore[assignment]

        @property
        def effective_decoder_norm(self) -> torch.Tensor:
            W_dec_r = self.W_dec.view(self.cfg.n_experts, self.cfg.d_expert, -1)
            return torch.linalg.matrix_norm(self.W_latent_dec @ W_dec_r, ord="fro", dim=(-2, -1))

    # ── Training config + class ───────────────────────────────────────────────────

    @dataclass
    class SMIXAERebasedTrainingConfig(TrainingSAEConfig):
        """Training config."""

        n_experts: int = 1024
        d_expert: int = 16
        d_bottleneck: int = 3
        k_experts: int = 8
        aux_loss_coefficient: float = 1 / 32
        rescale_acts_by_decoder_norm: bool = True
        threshold_lr: float = 0.1
        dead_after_n_passes: int = 1000
        d_sae : int = 1024 * 3

        @override
        @classmethod
        def architecture(cls) -> str:
            return "smixae_nb"

    class SMIXAERebasedTraining(BaseSMIXAETraining[SMIXAERebasedTrainingConfig]):
        """Training SMIXAERebased — edit this class to experiment.

        Key override points:
          training_forward_pass       — change loss terms or add new ones
          calculate_pre_act_aux_loss  — swap dead-expert recovery strategy
          encode_with_hidden_pre      — change routing / bottleneck projection
          decode                      — change reconstruction path
          _smixae_rebased_encode      — change the core encode function above
        """

        def __init__(self, cfg: SMIXAERebasedTrainingConfig) -> None:
            cfg.d_sae = cfg.d_expert * cfg.n_experts
            super().__init__(cfg)
            self.hook_l0 = HookPoint()
            self.hook_sae_acts_bottleneck = HookPoint()
            self.batchtopk = BatchTopK(self.cfg.k_experts)

        @override
        def initialize_weights(self) -> None:
            # register_standard_linear_weights(self)
            # register_smixae_v1_bottleneck_weights(self)
            register_smixae_v2_weights(self)

            self.register_buffer("threshold", torch.tensor(0.0, dtype=torch.double, device=self.device))
            self.register_buffer("n_passes_since_fired", torch.zeros(self.cfg.n_experts, dtype=torch.long))

        @override
        def get_coefficients(self) -> dict[str, TrainCoefficientConfig | float]:
            return {}

        def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
            return torch.nn.LeakyReLU(negative_slope=1e-4)

        @property
        def effective_decoder_norm(self) -> torch.Tensor:
            # W_dec_r = self.W_dec.view(self.cfg.n_experts, self.cfg.d_expert, -1)
            # return torch.linalg.matrix_norm(self.W_latent_dec @ W_dec_r, ord="fro", dim=(-2, -1))

            # Frobenius norm per expert, treating them as if they are separate matrices
            return eo.reduce(model.W_dec ** 2, "n_experts d_bottleneck d_in -> n_experts", 'sum') ** 0.5

        def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            pre_act_charts, h_charts, pre_act_bottleneck = _smixae_rebased_encode(self, x)

            # Apply BatchTopK
            batch_norm_mask = self.batchtopk(pre_act_bottleneck.norm(dim=-1)) > 0
            h_bottleneck = pre_act_bottleneck * batch_norm_mask.unsqueeze(-1)

            # Hooks and stashes
            self.h_bottleneck = h_bottleneck  # stored for compute_restricted_r2 / plot helpers
            self.hook_sae_acts_pre(pre_act_charts)
            self.hook_sae_acts_post(h_charts)
            self.hook_sae_acts_bottleneck(h_bottleneck)

            return h_bottleneck, pre_act_bottleneck, h_charts, pre_act_charts

        def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
            out = _smixae_decode(self, feature_acts)

            # Hooks
            out = self.hook_sae_recons(out)
            out = self.run_time_activation_norm_fn_out(out)

            return self.reshape_fn_out(out, self.d_head)

        @override
        def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
            # Encode
            # pre_act_charts, h_charts, pre_act_bottleneck = _smixae_rebased_encode(self, step_input.sae_in)
            h_bottleneck, pre_act_bottleneck, h_charts, pre_act_charts = self.encode_with_hidden_pre(step_input.sae_in)
            h_bottleneck_norms = h_bottleneck.norm(dim=-1)

            # ???
            # batch_norm_mask = self.batchtopk(pre_act_bottleneck.norm(dim=-1)) > 0
            # h_bottleneck = pre_act_bottleneck * batch_norm_mask.unsqueeze(-1)
            # self.h_bottleneck = h_bottleneck
            # self.hook_sae_acts_pre(pre_act_charts)
            # self.hook_sae_acts_post(h_charts)
            # self.hook_sae_acts_bottleneck(h_bottleneck)

            # Decode and update threshold
            sae_out = self.decode(h_bottleneck)
            self.update_threshold(h_bottleneck_norms)

            # Update the dead expert tracker
            with torch.no_grad():
                fired_in_batch = (h_bottleneck_norms > 0).any(dim=0)
                self.n_passes_since_fired = torch.where(
                    fired_in_batch,
                    torch.zeros_like(self.n_passes_since_fired),
                    self.n_passes_since_fired + 1,
                )

            # Calculate MSE and dead expert aux loss
            mse_loss = self.mse_loss_fn(sae_out, step_input.sae_in).sum(dim=-1).mean()
            dead_aux_loss = self.calculate_pre_act_aux_loss(
                self.n_passes_since_fired > self.cfg.dead_after_n_passes,
                pre_act_bottleneck,
            )

            total_loss = mse_loss + dead_aux_loss
            losses = {"mse_loss": mse_loss, "dead_expert_aux_loss": dead_aux_loss}

            # WandB metrics
            metrics: dict = {
                "experts_above_1e-3_L2":   (h_bottleneck_norms > 1e-3).float().sum(dim=-1).mean(),
                "experts_above_1e-1_L2":   (h_bottleneck_norms > 1e-1).float().sum(dim=-1).mean(),
                "expert_norm_mean":        h_bottleneck_norms[h_bottleneck_norms > 0].mean(),
                "dead_experts":            (self.n_passes_since_fired > self.cfg.dead_after_n_passes).sum().item(),
                "act_threshold":           self.threshold,
                "experts_above_threshold": (pre_act_bottleneck.norm(dim=-1) > self.threshold).float().sum(dim=-1).mean(),
                "nonzero_latent_l0":       (h_charts > 0).float().sum(dim=-1).mean(),
            }

            # SAELens trainer expects (batch, d_sae) shape for its book-keeping.
            return TrainStepOutput(
                sae_in=step_input.sae_in, sae_out=sae_out,
                feature_acts=h_charts.flatten(start_dim=1), hidden_pre=pre_act_charts.flatten(start_dim=1),
                loss=total_loss, losses=losses, metrics=metrics,
            )

        def calculate_pre_act_aux_loss(
            self, dead_expert_mask: torch.Tensor, pre_act_bottleneck: torch.Tensor
        ) -> torch.Tensor:
            if dead_expert_mask is None or not dead_expert_mask.any():
                return self.threshold.new_tensor(0.0)
            expert_norms = pre_act_bottleneck.norm(dim=-1)
            dead_norms = expert_norms[:, dead_expert_mask]
            shortfall = torch.relu(self.threshold.detach().float() - dead_norms)

            return self.cfg.aux_loss_coefficient * (shortfall).sum(dim=-1).mean()

        @torch.no_grad()
        def update_threshold(self, norms_topk: torch.Tensor) -> None:
            positive_mask = norms_topk > 0
            lr = self.cfg.threshold_lr
            with torch.autocast(self.threshold.device.type, enabled=False):
                if positive_mask.any():
                    min_positive = norms_topk[positive_mask].min().to(self.threshold.dtype)
                    self.threshold = (1 - lr) * self.threshold + lr * min_positive  # type: ignore[assignment]

        @override
        @torch.no_grad()
        def fold_activation_norm_scaling_factor(self, scaling_factor: float) -> None:
            self.W_enc.data *= scaling_factor
            self.W_dec.data /= scaling_factor
            self.b_dec.data /= scaling_factor
            self.cfg.normalize_activations = "none"

            # Fold norm rescaling into the decoder
            self.W_dec.data /= self.effective_decoder_norm.view(-1, 1, 1)

            # if self.cfg.rescale_acts_by_decoder_norm:
            #     sf_sqrt = scaling_factor**0.5
            #     self.W_dec.data *= sf_sqrt
            #     self.threshold = self.threshold / sf_sqrt  # type: ignore[assignment]
            # else:
            #     self.threshold = self.threshold / scaling_factor  # type: ignore[assignment]

        @override
        def calculate_aux_loss(
            self,
            step_input: TrainStepInput,
            feature_acts: torch.Tensor,
            hidden_pre: torch.Tensor,
            sae_out: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            return {
                "dead_expert_aux_loss": self.calculate_pre_act_aux_loss(
                    self.n_passes_since_fired > self.cfg.dead_after_n_passes,
                    hidden_pre,
                )
            }

    # ── Instantiate (mirrors _build_smixae in cli/toy.py) ─────────────────────────

    torch.manual_seed(seed)
    model = SMIXAERebasedTraining(
        SMIXAERebasedTrainingConfig(
            d_in=zoo.d_in,
            n_experts=n_experts,
            d_expert=d_expert,
            d_sae=n_experts*d_expert,
            d_bottleneck=d_bottleneck,
            k_experts=k_experts,
            normalize_activations="none",
            apply_b_dec_to_input=False,
            device=device,
            dead_after_n_passes=200,
            metadata=SAEMetadata(model_name="synthetic_toy", hook_name="ambient"),
        )
    ).to(device)

    _n_params = sum(p.numel() for p in model.parameters())
    mo.md(f"**Model**: SMIXAERebasedTraining (local copy) — {_n_params:,} parameters")
    return (model,)


@app.cell
def _(eo, model):
    print(model.W_dec.shape)

    model.W_dec.norm(dim=1).shape

    a = eo.reduce(model.W_dec ** 2, "n_experts d_bottleneck d_in -> n_experts", 'sum') ** 0.5
    a.shape

    # print(eo.repeat(model.effective_decoder_norm, "n_experts -> n_experts i j"), i=1, j=1)
    model.W_dec / model.effective_decoder_norm.view(-1, 1, 1)
    return


@app.cell
def _(
    ManifoldActivationGenerator,
    batch_size,
    compute_cofiring_matrix,
    compute_metrics,
    compute_restricted_r2,
    device,
    eval_data,
    l0,
    lr,
    lr_warm_up_steps,
    mo,
    model,
    n_snapshots,
    toy_data_dir,
    train_toy_sae,
    training_samples,
    zoo,
):
    # History is collected via snapshot callbacks — these are co-firing matched
    # metrics, NOT the reconstruction R² that SAELens reports internally.
    history: dict[str, list] = {
        "sample":   [],
        "r2":       [],   # mean co-firing matched R² across all instances
        "cofiring": [],   # mean co-firing rate of matched expert
        "mse":      [],
        "dead":     [],
    }

    _total_samples = training_samples

    def _snapshot_fn(trainer) -> None:
        """Called n_snapshots times, evenly spaced during training."""
        _sae = trainer.sae
        _sae.eval()

        _r2, _cf, _ = compute_restricted_r2(_sae, zoo, eval_data, device=device)
        _m          = compute_metrics(_sae, zoo, eval_data, device=device)

        history["sample"].append(trainer.n_training_samples)
        history["r2"].append(float(_r2.mean()))
        history["cofiring"].append(float(_cf.mean()))
        history["mse"].append(_m["mse"])
        history["dead"].append(_m["dead_experts"])

        _sae.train()

    _manifold_ag = ManifoldActivationGenerator(zoo=zoo, l0=l0, device=device)

    _save_dir = toy_data_dir / "notebook_model"
    _save_dir.mkdir(parents=True, exist_ok=True)

    train_toy_sae(
        sae=model,
        feature_dict=zoo.feature_dict,
        activations_generator=_manifold_ag,
        training_samples=_total_samples,
        batch_size=batch_size,
        lr=lr,
        lr_warm_up_steps=lr_warm_up_steps,
        device=device,
        n_snapshots=n_snapshots,
        snapshot_fn=_snapshot_fn,
    )
    model.eval()

    # Final evaluation
    r2_final, cofiring_final, best_experts = compute_restricted_r2(model, zoo, eval_data, device=device)
    metrics_final = compute_metrics(model, zoo, eval_data, device=device)
    cofiring_matrix = compute_cofiring_matrix(model, zoo, eval_data, device=device)

    mo.md(f"""
    **Training complete**

    | Metric | Value |
    |---|---|
    | Mean R² (co-firing matched) | {r2_final.mean():.4f} |
    | Mean co-firing rate | {cofiring_final.mean():.3f} |
    | MSE | {metrics_final['mse']:.5f} |
    | Effective L0 | {metrics_final['effective_l0']:.2f} |
    | Dead experts | {metrics_final['dead_experts']} |
    """)
    return best_experts, cofiring_final, cofiring_matrix, history, r2_final


@app.cell
def _(go, history: dict[str, list], make_subplots, mo):
    if not history["sample"]:
        mo.stop(True, mo.md("No snapshots recorded — increase n_snapshots."))

    _fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Mean R² (co-firing matched)", "Mean co-firing rate",
            "MSE", "Dead experts",
        ],
    )

    _xs = history["sample"]

    _fig.add_trace(go.Scatter(x=_xs, y=history["r2"], mode="lines+markers", name="R²"), row=1, col=1)
    _fig.add_trace(go.Scatter(x=_xs, y=history["cofiring"], mode="lines+markers", name="co-fire"), row=1, col=2)
    _fig.add_trace(go.Scatter(x=_xs, y=history["mse"], mode="lines+markers", name="MSE"), row=2, col=1)
    _fig.add_trace(go.Scatter(x=_xs, y=history["dead"], mode="lines+markers", name="dead"), row=2, col=2)

    _fig.update_xaxes(title_text="Training samples")
    _fig.update_layout(title="Training curves (co-firing matched metrics)", height=600, showlegend=False)
    _fig
    return


@app.cell
def _(go, r2_final, zoo):
    import collections as _col

    _type_r2: dict[str, list] = _col.defaultdict(list)
    for _i, _inst in enumerate(zoo.instances):
        _type_r2[_inst.type_name].append(float(r2_final[_i]))

    _types  = sorted(_type_r2)
    _means  = [sum(_type_r2[t]) / len(_type_r2[t]) for t in _types]
    _mins   = [min(_type_r2[t]) for t in _types]
    _maxs   = [max(_type_r2[t]) for t in _types]

    _fig_bar = go.Figure([
        go.Bar(
            name="mean R²", x=_types, y=_means,
            error_y=dict(
                type="data",
                symmetric=False,
                array=[_maxs[i] - _means[i] for i in range(len(_types))],
                arrayminus=[_means[i] - _mins[i] for i in range(len(_types))],
            ),
        )
    ])
    _fig_bar.update_layout(
        title="R² by manifold type (mean ± min/max across variants)",
        yaxis_title="R²",
        xaxis_title="Manifold type",
        yaxis_range=[0, 1],
    )
    _fig_bar
    return


@app.cell
def _(cofiring_matrix, go, zoo):
    _labels = [f"{inst.type_name}[{inst.variant_idx}]" for inst in zoo.instances]
    _fig_heat = go.Figure(go.Heatmap(
        z=cofiring_matrix.numpy(),
        x=[str(e) for e in range(cofiring_matrix.shape[1])],
        y=_labels,
        colorscale="Viridis",
        colorbar=dict(title="P(fire | active)"),
    ))
    _fig_heat.update_layout(
        title="Co-firing matrix: P(expert e fires | manifold i active)",
        xaxis_title="Expert index",
        yaxis_title="Manifold instance",
        height=800,
    )
    _fig_heat
    return


@app.cell
def _(mo):
    mo.md("""
    ### All 48 experts: learned (red) vs original (blue)
    """)
    return


@app.cell
def _(
    best_experts,
    cofiring_final,
    device,
    eval_data,
    k_experts,
    model,
    plot_all_experts_with_originals,
    r2_final,
    zoo,
):
    plot_all_experts_with_originals(
        model, zoo, eval_data, best_experts,
        k_experts=k_experts, r2=r2_final, cofiring=cofiring_final, device=device,
    )
    return


if __name__ == "__main__":
    app.run()
