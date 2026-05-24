"""ManifoldZoo, EvalData, ManifoldActivationGenerator, and zoo construction helpers.

Contains:
- ManifoldZoo dataclass (48-instance zoo with Grassmannian-optimised embeddings)
- EvalData dataclass (held-out evaluation set)
- optimize_subspaces (Grassmannian projected-gradient optimisation)
- build_manifold_zoo (zoo construction entry point)
- ManifoldActivationGenerator (ActivationGenerator subclass for manifold coords)
- generate_eval_set (fixed eval set generator)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from sae_lens.synthetic import ActivationGenerator, FeatureDictionary

from toy.manifolds import (
    _D_INTRINSIC,
    _K_PER_TYPE,
    _MANIFOLD_ORDER,
    _SAMPLERS,
    _VARIANT_PARAMS,
    N_VARIANTS,
    ManifoldInstance,
)

# ── ManifoldZoo ───────────────────────────────────────────────────────────────


@dataclass
class ManifoldZoo:
    """The 48-instance manifold zoo with Grassmannian-optimised ambient embeddings.

    Attributes:
        instances: 48 ManifoldInstance objects (8 types × 6 variants).
        feature_dict: FeatureDictionary mapping (n_atoms,) coordinates → (d_in,).
            The FeatureDictionary bias stores the global affine offset b_global
            (a fixed random unit direction scaled to sigma_bias), added to every
            generated activation before RMSNorm.  Access via feature_dict.bias.
        n_atoms: Total manifold atom count = Σ k_i = 120.
        d_in: Ambient dimension.
        sigma_bias: Norm of the global bias vector (0 = no bias).
    """

    instances: list[ManifoldInstance]
    feature_dict: FeatureDictionary
    n_atoms: int
    d_in: int
    sigma_bias: float = 0.0

    @property
    def b_global(self) -> torch.Tensor:
        """Global bias vector (d_in,) stored in the FeatureDictionary bias."""
        return self.feature_dict.bias.data

    def sample_raw(
        self,
        instance_idx: int,
        n: int,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample n normalised points from manifold instance instance_idx.

        Args:
            instance_idx: Index into self.instances.
            n: Number of samples.
            device: Target device.

        Returns:
            Tuple of:
            - color: (n,) scalar intrinsic parameter for visualisation.
            - z_norm: (n, k_i) normalised manifold coordinates.
        """
        inst = self.instances[instance_idx]
        params = _VARIANT_PARAMS[inst.type_name][inst.variant_idx]
        sampler = _SAMPLERS[inst.type_name]
        with torch.no_grad():
            if isinstance(params, tuple):
                color, z_raw = sampler(*params, n, device)
            else:
                color, z_raw = sampler(params, n, device)
        z_norm = (z_raw - inst.mu.to(device)) / inst.sigma
        return color, z_norm

    def save(self, path: Path) -> None:
        """Serialise the zoo to a .pt file.

        Args:
            path: Destination file path (will be created or overwritten).
        """
        state = {
            "feature_vectors": self.feature_dict.feature_vectors.data.cpu(),
            "bias": self.feature_dict.bias.data.cpu(),
            "n_atoms": self.n_atoms,
            "d_in": self.d_in,
            "sigma_bias": self.sigma_bias,
            "instances": [
                {
                    "type_name":   inst.type_name,
                    "variant_idx": inst.variant_idx,
                    "k_i":         inst.k_i,
                    "d_i":         inst.d_i,
                    "params":      inst.params,
                    "mu":          inst.mu,
                    "sigma":       inst.sigma,
                    "atom_offset": inst.atom_offset,
                }
                for inst in self.instances
            ],
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "ManifoldZoo":
        """Deserialise a zoo saved by :meth:`save`.

        Args:
            path: Source .pt file.
            device: Target device for tensors.

        Returns:
            Reconstructed ManifoldZoo.
        """
        state = torch.load(path, map_location=device, weights_only=False)
        instances = [
            ManifoldInstance(
                type_name=d["type_name"],
                variant_idx=d["variant_idx"],
                k_i=d["k_i"],
                d_i=d["d_i"],
                params=d["params"],
                mu=d["mu"].cpu(),
                sigma=d["sigma"],
                atom_offset=d["atom_offset"],
            )
            for d in state["instances"]
        ]
        fd = FeatureDictionary(
            num_features=state["n_atoms"],
            hidden_dim=state["d_in"],
            bias=False,
            initializer=None,
            device=device,
        )
        fd.feature_vectors.data.copy_(state["feature_vectors"].to(device))
        fd.bias.data.copy_(state["bias"].to(device))
        return cls(
            instances=instances,
            feature_dict=fd,
            n_atoms=state["n_atoms"],
            d_in=state["d_in"],
            sigma_bias=state["sigma_bias"],
        )


# ── EvalData ──────────────────────────────────────────────────────────────────


@dataclass
class EvalData:
    """Held-out evaluation set with stored per-atom feature activations.

    Per-manifold ground-truth contribution can be recovered as:
        M_i = feature_acts[:, off:off+k] @ V_i
    where V_i = zoo.feature_dict.feature_vectors[off:off+k].

    Attributes:
        x: (N, d_in) ambient activations fed to SMIXAE (includes global bias).
        feature_acts: (N, n_atoms) signed manifold coordinates (zero where inactive).
        color_param: (N, n_instances) intrinsic colour parameter; 0 where inactive.
    """

    x: torch.Tensor            # (N, d_in)
    feature_acts: torch.Tensor  # (N, n_atoms)
    color_param: torch.Tensor   # (N, n_instances)

    def save(self, path: Path) -> None:
        """Serialise to a .pt file.

        Args:
            path: Destination file path.
        """
        torch.save(
            {
                "x": self.x,
                "feature_acts": self.feature_acts,
                "color_param": self.color_param,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "EvalData":
        """Deserialise a file saved by :meth:`save`.

        Args:
            path: Source .pt file.
            device: Target device.

        Returns:
            Loaded EvalData on the specified device.
        """
        state = torch.load(path, map_location=device, weights_only=True)
        return cls(
            x=state["x"],
            feature_acts=state["feature_acts"],
            color_param=state["color_param"],
        )


# ── Zoo construction ───────────────────────────────────────────────────────────


def optimize_subspaces(
    k_list: list[int],
    d_in: int,
    n_steps: int = 500,
    temperature: float = 50.0,
    lr: float = 5e-3,
    seed: int = 0,
    verbose: bool = True,
) -> list[torch.Tensor]:
    r"""Find low-mutual-coherence subspaces via projected gradient descent.

    Minimises a smooth surrogate of the max pairwise coherence across all
    ``len(k_list)`` subspaces:

        cost = (1/T) * log(Σ_{i<j} exp(T * ||Q_i^T Q_j||_F^2))

    using Adam + QR retraction (projected gradient on the Stiefel manifold).
    Each iteration: gradient step on the Euclidean cost, then re-orthogonalise
    each frame via QR to retract back onto the Stiefel manifold.

    Args:
        k_list: Subspace dimensions k_i for each of the n instances.
        d_in: Ambient dimension.
        n_steps: Number of optimisation steps.
        temperature: Log-sum-exp smoothing temperature (higher → closer to max).
        lr: Adam learning rate.
        seed: Reproducibility seed.
        verbose: Whether to log progress every 100 steps.

    Returns:
        List of n tensors, each shape ``(k_i, d_in)`` with orthonormal rows.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    n = len(k_list)

    # Initialise: random orthonormal frames represented as (d_in, k_i) matrices.
    frames: list[torch.nn.Parameter] = []
    for k in k_list:
        G = torch.randn(d_in, k, generator=rng)
        Q, _ = torch.linalg.qr(G)
        frames.append(torch.nn.Parameter(Q.clone()))

    opt = torch.optim.Adam(frames, lr=lr)  # type: ignore[arg-type]

    def _max_spectral_coherence() -> float:
        """Max pairwise spectral coherence = max singular value of Q_i^T Q_j, in [0,1]."""
        with torch.no_grad():
            vals = [
                float(torch.linalg.matrix_norm(frames[i].T @ frames[j], ord=2).item())
                for i in range(n) for j in range(i + 1, n)
            ]
        return max(vals) if vals else 0.0

    if verbose:
        from loguru import logger
        logger.info(f"Grassmannian optimisation: {n} subspaces in R^{d_in}")
        logger.info(f"  max spectral coherence (before): {_max_spectral_coherence():.6f}")

    for step in range(n_steps):
        opt.zero_grad()
        # Cost uses Frobenius-squared as smooth proxy for spectral norm.
        coh_list = [
            (frames[i].T @ frames[j]).pow(2).sum()
            for i in range(n) for j in range(i + 1, n)
        ]
        cost = (1.0 / temperature) * torch.logsumexp(
            temperature * torch.stack(coh_list), dim=0
        )
        cost.backward()
        opt.step()

        # QR retraction: project back onto the Stiefel manifold.
        with torch.no_grad():
            for p in frames:
                Q, _ = torch.linalg.qr(p.data)
                p.data.copy_(Q)

        if verbose and (step + 1) % 100 == 0:
            from loguru import logger
            logger.info(
                f"  step {step + 1:4d}/{n_steps}  "
                f"max_spectral_coh={_max_spectral_coherence():.6f}  "
                f"cost={float(cost.item()):.6f}"
            )

    if verbose:
        from loguru import logger
        logger.info(f"  max spectral coherence (after):  {_max_spectral_coherence():.6f}")

    # Return V_i = Q_i^T in (k_i, d_in) convention matching ManifoldInstance.
    return [p.data.T.detach().clone() for p in frames]


def build_manifold_zoo(
    d_in: int = 128,
    seed: int = 0,
    device: str = "cpu",
    n_calibration: int = 50_000,
    sigma_bias: float = 3.0,
    skip_grassmannian: bool = False,
    grassmannian_steps: int = 500,
) -> ManifoldZoo:
    """Build the 48-instance manifold zoo with Grassmannian-optimised embeddings.

    Each instance's ambient subspace V_i ∈ R^{k_i × d_in} (orthonormal rows) is
    determined by Grassmannian optimisation (projected gradient on Stiefel) that
    minimises the max pairwise mutual coherence across all 48 subspaces.  Set
    ``skip_grassmannian=True`` for a fast random-QR fallback (debugging only).

    Global bias: a single random unit-direction vector scaled to ``sigma_bias``
    is stored as the FeatureDictionary bias and added to every generated
    activation, simulating the DC offset of real LLM residual streams.  Set
    ``sigma_bias=0`` to disable.

    Args:
        d_in: Ambient dimension.
        seed: Random seed for calibration, Grassmannian optimisation, and bias.
        device: Device for calibration and final tensor placement.
        n_calibration: Points drawn per instance for mean/RMS calibration.
        sigma_bias: Norm of the global bias vector (0 = no bias).
        skip_grassmannian: If True, use independent random QR embeddings instead
            of Grassmannian optimisation (fast, low coherence control).
        grassmannian_steps: Optimisation steps passed to optimize_subspaces.

    Returns:
        ManifoldZoo with 48 instances and a FeatureDictionary of shape
        (n_atoms=120, d_in) with bias = b_global (zero if sigma_bias=0).
    """
    rng = torch.Generator(device=device).manual_seed(seed)

    instances: list[ManifoldInstance] = []
    atom_offset = 0
    k_list: list[int] = []
    calibration_data: list[tuple[torch.Tensor, float]] = []  # (mu, sigma) per instance

    # ── Pass 1: calibrate all instances ──────────────────────────────────────
    for type_name in _MANIFOLD_ORDER:
        k_i = _K_PER_TYPE[type_name]
        d_i = _D_INTRINSIC[type_name]
        for variant_idx in range(N_VARIANTS):
            params = _VARIANT_PARAMS[type_name][variant_idx]
            sampler = _SAMPLERS[type_name]

            with torch.no_grad():
                if isinstance(params, tuple):
                    _, z_cal = sampler(*params, n_calibration, device)
                else:
                    _, z_cal = sampler(params, n_calibration, device)

            mu = z_cal.mean(0)
            sigma = float((z_cal - mu).norm(dim=1).pow(2).mean().sqrt().item())
            if sigma < 1e-8:
                sigma = 1.0

            calibration_data.append((mu.cpu(), sigma))
            k_list.append(k_i)
            instances.append(ManifoldInstance(
                type_name=type_name,
                variant_idx=variant_idx,
                k_i=k_i,
                d_i=d_i,
                params=params,
                mu=mu.cpu(),
                sigma=sigma,
                atom_offset=atom_offset,
            ))
            atom_offset += k_i

    n_manifold_atoms = atom_offset  # 120

    # ── Pass 2: optimise subspace directions ─────────────────────────────────
    if skip_grassmannian:
        v_rows: list[torch.Tensor] = []
        for k_i in k_list:
            G = torch.randn(d_in, k_i, device=device, generator=rng)
            Q, _ = torch.linalg.qr(G)
            v_rows.append(Q.T)          # (k_i, d_in)
    else:
        frames = optimize_subspaces(
            k_list=k_list,
            d_in=d_in,
            n_steps=grassmannian_steps,
            seed=seed,
        )
        # frames[i] is (k_i, d_in) already in our V_i convention
        v_rows = [f.to(device) for f in frames]

    # ── Build FeatureDictionary ───────────────────────────────────────────────
    fd = FeatureDictionary(
        num_features=n_manifold_atoms,
        hidden_dim=d_in,
        bias=False,
        initializer=None,
        device=device,
    )
    fd.feature_vectors.data.copy_(torch.cat(v_rows, dim=0))

    # Global bias: single random unit direction scaled to sigma_bias.
    # Stored in FeatureDictionary.bias so it is added automatically by fd.forward().
    if sigma_bias > 0.0:
        b_rng = torch.Generator(device=device).manual_seed(seed + 1)
        b = torch.randn(d_in, device=device, generator=b_rng)
        fd.bias.data.copy_(b / b.norm().clamp(min=1e-8) * sigma_bias)

    return ManifoldZoo(
        instances=instances,
        feature_dict=fd,
        n_atoms=n_manifold_atoms,
        d_in=d_in,
        sigma_bias=sigma_bias,
    )


# ── ManifoldActivationGenerator ───────────────────────────────────────────────


class ManifoldActivationGenerator(ActivationGenerator):
    """ActivationGenerator subclass that produces signed manifold coordinates.

    The standard ActivationGenerator applies relu() to all outputs, which would
    zero negative manifold coordinates (e.g. cos θ < 0 on a circle).  This
    subclass overrides sample() to perform exact-L0 manifold instance selection
    and returns the full signed normalised coordinates for each active atom.

    The output of sample() is compatible with FeatureDictionary.forward():
        x = feature_dict(generator.sample(B))
    which computes x = feature_acts @ V (a linear projection, no relu).

    Args:
        zoo: ManifoldZoo produced by build_manifold_zoo.
        l0: Exact number of active manifold instances per sample.
        device: Device for tensor operations.
    """

    def __init__(
        self,
        zoo: ManifoldZoo,
        l0: int = 4,
        device: str = "cpu",
    ) -> None:
        """Initialise with a ManifoldZoo and sparsity parameter.

        Args:
            zoo: ManifoldZoo produced by build_manifold_zoo.
            l0: Exact number of active manifold instances per sample.
            device: Device for tensor operations.
        """
        # Initialise parent with dummy firing probs; sample() is fully overridden.
        super().__init__(
            num_features=zoo.n_atoms,
            firing_probabilities=float(l0) / len(zoo.instances),
            device=device,
        )
        self.zoo = zoo
        self.l0 = l0
        self._device = device
        self._n_instances = len(zoo.instances)

    @torch.no_grad()
    def sample(self, batch_size: int) -> torch.Tensor:
        """Return signed manifold coordinates, shape (batch_size, n_atoms).

        For each sample, exactly l0 manifold instances are chosen uniformly at
        random (without replacement).  The normalised coordinates z̃_i of each
        active manifold are written into the corresponding atom slots; inactive
        slots remain zero.

        Args:
            batch_size: Number of samples to generate.

        Returns:
            feature_acts: (batch_size, n_atoms) — signed, not relu'd.
        """
        B = batch_size
        n_inst = self._n_instances
        device = self._device
        feature_acts = torch.zeros(B, self.zoo.n_atoms, device=device)

        # Exact-L0 manifold selection: top-l0 of random priorities per row
        priorities = torch.rand(B, n_inst, device=device)
        _, topk_idx = priorities.topk(self.l0, dim=1)          # (B, l0)
        instance_mask = torch.zeros(B, n_inst, device=device)
        instance_mask.scatter_(1, topk_idx, 1.0)               # (B, n_inst) binary

        for inst_idx, inst in enumerate(self.zoo.instances):
            active_rows = instance_mask[:, inst_idx].nonzero(as_tuple=True)[0]
            if active_rows.numel() == 0:
                continue
            _, z_norm = self.zoo.sample_raw(inst_idx, active_rows.numel(), device)
            off = inst.atom_offset
            feature_acts[active_rows, off: off + inst.k_i] = z_norm

        return feature_acts

    def sample_with_colors(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like sample() but also returns the per-instance colour parameter.

        Args:
            batch_size: Number of samples to generate.

        Returns:
            Tuple of:
            - feature_acts: (batch_size, n_atoms).
            - color_param: (batch_size, n_instances) intrinsic param; 0 where inactive.
        """
        B = batch_size
        n_inst = self._n_instances
        device = self._device
        feature_acts = torch.zeros(B, self.zoo.n_atoms, device=device)
        color_param  = torch.zeros(B, n_inst, device=device)

        priorities = torch.rand(B, n_inst, device=device)
        _, topk_idx = priorities.topk(self.l0, dim=1)
        instance_mask = torch.zeros(B, n_inst, device=device)
        instance_mask.scatter_(1, topk_idx, 1.0)

        for inst_idx, inst in enumerate(self.zoo.instances):
            active_rows = instance_mask[:, inst_idx].nonzero(as_tuple=True)[0]
            if active_rows.numel() == 0:
                continue
            color, z_norm = self.zoo.sample_raw(inst_idx, active_rows.numel(), device)
            off = inst.atom_offset
            feature_acts[active_rows, off: off + inst.k_i] = z_norm
            color_param[active_rows, inst_idx] = color

        return feature_acts, color_param


# ── Eval set ──────────────────────────────────────────────────────────────────


def generate_eval_set(
    zoo: ManifoldZoo,
    n_samples: int = 200_000,
    l0: int = 4,
    seed: int = 42,
    device: str = "cpu",
    chunk_size: int = 10_000,
) -> EvalData:
    """Generate a fixed evaluation set with stored feature activations.

    Per-manifold ground-truth contributions are recoverable from feature_acts
    without storing an N × 42 × d_in tensor.

    Args:
        zoo: ManifoldZoo produced by build_manifold_zoo.
        n_samples: Total eval samples.
        l0: Exact active manifolds per sample.
        seed: Reproducibility seed.
        device: Generation device (data is moved to CPU for storage).
        chunk_size: Generation chunk size to bound peak memory.

    Returns:
        EvalData on CPU.
    """
    torch.manual_seed(seed)
    ag = ManifoldActivationGenerator(zoo=zoo, l0=l0, device=device)

    x_chunks: list[torch.Tensor] = []
    fa_chunks: list[torch.Tensor] = []
    cp_chunks: list[torch.Tensor] = []

    generated = 0
    while generated < n_samples:
        n_batch = min(chunk_size, n_samples - generated)
        fa_chunk, cp_chunk = ag.sample_with_colors(n_batch)
        with torch.no_grad():
            x_chunk = zoo.feature_dict(fa_chunk)
        x_chunks.append(x_chunk.cpu())
        fa_chunks.append(fa_chunk.cpu())
        cp_chunks.append(cp_chunk.cpu())
        generated += n_batch

    return EvalData(
        x=torch.cat(x_chunks, dim=0),
        feature_acts=torch.cat(fa_chunks, dim=0),
        color_param=torch.cat(cp_chunks, dim=0),
    )
