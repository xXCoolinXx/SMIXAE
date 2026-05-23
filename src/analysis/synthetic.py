"""Synthetic manifold toy-model benchmark for SMIXAE.

Builds a zoo of 7 low-dimensional manifolds (the Torus is omitted — its Clifford
embedding requires k_i=4 > 3, incompatible with SMIXAE's 3-D bottleneck) embedded
sparsely in a 128-dimensional ambient space via sae_lens.synthetic.FeatureDictionary.

Sparse instance selection uses a ManifoldActivationGenerator subclass of
sae_lens.synthetic.ActivationGenerator, overriding sample() to produce signed
manifold coordinates (standard ActivationGenerator applies relu which would zero
negative coordinate values like cos θ on a circle).

This module is meant to be consumed by src/cli/synthetic.py, which calls
train_toy_sae(smixae, feature_dict, manifold_ag) for training.

Public API
----------
build_manifold_zoo          Build the 42-instance zoo with normalised embeddings.
ManifoldActivationGenerator ActivationGenerator subclass producing manifold coords.
generate_eval_set           Fixed held-out eval set with stored feature activations.
compute_restricted_r2       Per-manifold R²(i,n) for n=1,2,3 greedy expert steps.
compute_metrics             Overall MSE, effective L0, dead experts.
plot_metrics_vs_k_experts   Plotly figure of metric curves vs k_experts sweep.
plot_bottlenecks            Plotly figure of 3-D bottleneck codes per manifold type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sae_lens.synthetic import ActivationGenerator, FeatureDictionary

if TYPE_CHECKING:
    from smixae.smixae import SMIXAETraining

# ── Manifold parameter sets ────────────────────────────────────────────────────

_CIRCLE_RADII     = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
_SPHERE_RADII     = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
_MOBIUS_WIDTHS    = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
_SWISS_ROLL_PARAMS = [
    (2.0 * math.pi, 1.5),
    (2.5 * math.pi, 2.0),
    (3.0 * math.pi, 3.0),
    (3.5 * math.pi, 4.0),
    (4.0 * math.pi, 5.0),
    (4.5 * math.pi, 6.0),
]
_HELIX_ALPHAS     = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
_DISK_RADII       = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
_SEGMENT_LENGTHS  = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

N_VARIANTS = 6

_MANIFOLD_ORDER = [
    "circle",
    "sphere",
    "mobius",
    "swiss_roll",
    "helix",
    "flat_disk",
    "segment",
]

_K_PER_TYPE: dict[str, int] = {
    "circle":     2,
    "sphere":     3,
    "mobius":     3,
    "swiss_roll": 3,
    "helix":      3,
    "flat_disk":  2,
    "segment":    1,
}

_D_INTRINSIC: dict[str, int] = {
    "circle":     1,
    "sphere":     2,
    "mobius":     2,
    "swiss_roll": 2,
    "helix":      1,
    "flat_disk":  2,
    "segment":    1,
}

_VARIANT_PARAMS: dict[str, list[Any]] = {
    "circle":     _CIRCLE_RADII,
    "sphere":     _SPHERE_RADII,
    "mobius":     _MOBIUS_WIDTHS,
    "swiss_roll": _SWISS_ROLL_PARAMS,
    "helix":      _HELIX_ALPHAS,
    "flat_disk":  _DISK_RADII,
    "segment":    _SEGMENT_LENGTHS,
}

# ── Per-manifold raw samplers ──────────────────────────────────────────────────
# Each returns (color_param: (n,), z_raw: (n, k_i)) with k_i = ambient dim.
# Coordinates are raw (unnormalised); normalisation is applied by sample_raw().


def _sample_circle(r: float, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    theta = torch.rand(n, device=device) * (2 * math.pi)
    return theta, torch.stack([r * theta.cos(), r * theta.sin()], dim=1)


def _sample_sphere(r: float, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = torch.randn(n, 3, device=device)
    xyz = xyz / xyz.norm(dim=1, keepdim=True).clamp(min=1e-8)
    color = torch.atan2(xyz[:, 1], xyz[:, 0])
    return color, r * xyz


def _sample_mobius(w: float, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    phi = torch.rand(n, device=device) * (2 * math.pi)
    t = (torch.rand(n, device=device) - 0.5) * w
    rim = 1.0 + t * (phi * 0.5).cos()
    x = rim * phi.cos()
    y = rim * phi.sin()
    z = t * (phi * 0.5).sin()
    return phi, torch.stack([x, y, z], dim=1)


def _sample_swiss_roll(
    theta_max: float, h_max: float, n: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = torch.rand(n, device=device) * theta_max
    h     = torch.rand(n, device=device) * h_max
    return theta, torch.stack([theta * theta.cos(), h, theta * theta.sin()], dim=1)


def _sample_helix(alpha: float, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    theta = torch.rand(n, device=device) * (4 * math.pi)
    return theta, torch.stack([theta.cos(), theta.sin(), alpha * theta], dim=1)


def _sample_flat_disk(R: float, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    r   = R * torch.rand(n, device=device).sqrt()
    phi = torch.rand(n, device=device) * (2 * math.pi)
    return phi, torch.stack([r * phi.cos(), r * phi.sin()], dim=1)


def _sample_segment(length: float, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.rand(n, device=device)
    return t, (t * length).unsqueeze(1)


_SAMPLERS: dict[str, Callable[..., tuple[torch.Tensor, torch.Tensor]]] = {
    "circle":     _sample_circle,
    "sphere":     _sample_sphere,
    "mobius":     _sample_mobius,
    "swiss_roll": _sample_swiss_roll,
    "helix":      _sample_helix,
    "flat_disk":  _sample_flat_disk,
    "segment":    _sample_segment,
}

# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class ManifoldInstance:
    """Metadata for one manifold instance (type × variant)."""

    type_name: str
    variant_idx: int
    k_i: int       # ambient embedding dimension (1, 2, or 3)
    d_i: int       # intrinsic dimension
    params: Any    # type-specific variant params (float or tuple)
    mu: torch.Tensor   # (k_i,) calibration mean, on CPU
    sigma: float       # RMS norm after centering
    atom_offset: int   # start index in the joint FeatureDictionary
    bias_atom_offset: int = 0   # index of the per-instance bias atom (set by build_manifold_zoo)


@dataclass
class ManifoldZoo:
    """The 42-instance manifold zoo with shared ambient-embedding FeatureDictionary.

    Attributes:
        instances: 42 ManifoldInstance objects (7 types × 6 variants).
        feature_dict: FeatureDictionary mapping (n_atoms,) coordinates → (d_in,).
        n_atoms: Total atom count = Σ k_i + 42 = 102 + 42 = 144 (includes bias atoms).
        d_in: Ambient dimension.
        sigma_bias: Amplitude of per-instance bias atoms (0 = no bias).
    """

    instances: list[ManifoldInstance]
    feature_dict: FeatureDictionary
    n_atoms: int
    d_in: int
    sigma_bias: float = 0.0

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


@dataclass
class EvalData:
    """Held-out evaluation set with stored per-atom feature activations.

    Per-manifold ground-truth contribution can be recovered as:
        M_i = feature_acts[:, off:off+k] @ V_i
    where V_i = zoo.feature_dict.feature_vectors[off:off+k].

    Attributes:
        x: (N, d_in) ambient activations fed to SMIXAE.
        feature_acts: (N, n_atoms) signed manifold coordinates (zero where inactive).
        color_param: (N, n_instances) intrinsic colour parameter; 0 where inactive.
    """

    x: torch.Tensor            # (N, d_in)
    feature_acts: torch.Tensor  # (N, n_atoms)
    color_param: torch.Tensor   # (N, n_instances)


# ── Zoo construction ───────────────────────────────────────────────────────────


def build_manifold_zoo(
    d_in: int = 128,
    seed: int = 0,
    device: str = "cpu",
    n_calibration: int = 50_000,
    sigma_bias: float = 3.0,
) -> ManifoldZoo:
    """Build the 42-instance manifold zoo with normalised ambient embeddings.

    Each instance gets an independent QR-orthonormal ambient embedding
    V_i ∈ R^{k_i × d_in} (orthonormal rows within each instance; no
    cross-instance orthogonality, matching the paper's construction).

    Bias atoms: one extra random unit-direction atom per instance is appended
    after the 102 manifold atoms (indices 102..143).  When instance i fires,
    its bias atom receives amplitude sigma_bias, so the ambient contribution
    includes a constant shift sigma_bias * b_dir_i — simulating the affine
    offsets present in real LLM residual streams.  Set sigma_bias=0 to disable.

    Args:
        d_in: Ambient dimension.
        seed: Random seed for V_i matrices and bias directions.
        device: Device for QR computation.
        n_calibration: Points drawn per instance for mean/RMS calibration.
        sigma_bias: Amplitude of per-instance bias atoms.

    Returns:
        ManifoldZoo with 42 instances and a FeatureDictionary of shape
        (n_atoms=144, d_in) (102 manifold atoms + 42 bias atoms).
    """
    rng = torch.Generator(device=device).manual_seed(seed)

    instances: list[ManifoldInstance] = []
    atom_offset = 0
    v_rows: list[torch.Tensor] = []

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

            # Per-instance QR-orthonormal V_i
            G = torch.randn(d_in, k_i, device=device, generator=rng)
            Q, _ = torch.linalg.qr(G)   # (d_in, k_i)
            V_i = Q.T                    # (k_i, d_in) — orthonormal rows

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
            v_rows.append(V_i)
            atom_offset += k_i

    n_manifold_atoms = atom_offset  # 102

    # Append one bias atom per instance (indices n_manifold_atoms..n_manifold_atoms+41)
    for inst_idx, inst in enumerate(instances):
        inst.bias_atom_offset = n_manifold_atoms + inst_idx
        b = torch.randn(d_in, device=device, generator=rng)
        b = b / b.norm().clamp(min=1e-8)
        v_rows.append(b.unsqueeze(0))   # (1, d_in)

    n_atoms = n_manifold_atoms + len(instances)  # 144

    fd = FeatureDictionary(
        num_features=n_atoms,
        hidden_dim=d_in,
        bias=False,
        initializer=None,
        device=device,
    )
    fd.feature_vectors.data.copy_(torch.cat(v_rows, dim=0))

    return ManifoldZoo(
        instances=instances,
        feature_dict=fd,
        n_atoms=n_atoms,
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
            if self.zoo.sigma_bias > 0.0:
                feature_acts[active_rows, inst.bias_atom_offset] = self.zoo.sigma_bias

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
            if self.zoo.sigma_bias > 0.0:
                feature_acts[active_rows, inst.bias_atom_offset] = self.zoo.sigma_bias
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


# ── Metrics ────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _encode_all(
    model: "SMIXAETraining",
    x: torch.Tensor,
    device: str,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Run the trained model's encoder on x and return h_bottleneck on CPU.

    Args:
        model: Trained SMIXAETraining on device.
        x: (N, d_in) on CPU.
        device: Model device.
        chunk_size: Encoding chunk size.

    Returns:
        h_bottleneck: (N, n_experts, d_bottleneck) on CPU.
    """
    N = x.shape[0]
    n_exp = model.cfg.n_experts
    d_b   = model.cfg.d_bottleneck
    out   = torch.zeros(N, n_exp, d_b)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        model.encode_with_hidden_pre(x[start:end].to(device))
        out[start:end] = model.h_bottleneck.detach().cpu()
    return out


@torch.no_grad()
def compute_restricted_r2(
    model: "SMIXAETraining",
    zoo: ManifoldZoo,
    eval_data: EvalData,
    device: str = "cpu",
    max_n: int = 3,
) -> tuple[torch.Tensor, list[int]]:
    """Per-manifold restricted R²(i, n) via greedy expert selection.

    For each manifold instance i, greedily picks up to max_n experts whose
    decoded contribution most reduces the residual variance of the ground-truth
    manifold contribution M_i.  The ideal outcome is R²(i, 1) ≈ 1 for all i.

    Args:
        model: Trained SMIXAETraining.
        zoo: ManifoldZoo used to generate eval_data.
        eval_data: EvalData on CPU.
        device: Model device.
        max_n: Greedy steps (default 3 — R²(n=1,2,3)).

    Returns:
        Tuple of:
        - r2: (n_instances, max_n) — R²(i, n) for i=0..41, n=0..2.
        - best_experts: list[int] of length n_instances — top-1 expert per manifold.
    """
    model.eval()
    n_inst    = len(zoo.instances)
    n_exp     = model.cfg.n_experts
    d_exp     = model.cfg.d_expert
    d_in      = zoo.d_in

    h_bott = _encode_all(model, eval_data.x, device)  # (N, n_exp, d_bott)

    W_latent_dec = model.W_latent_dec.detach().cpu()          # (n_exp, d_bott, d_exp)
    W_dec_grouped = (
        model.W_dec.detach().cpu().reshape(n_exp, d_exp, d_in)
    )                                                          # (n_exp, d_exp, d_in)

    r2            = torch.zeros(n_inst, max_n)
    best_experts: list[int] = []

    for inst_idx, inst in enumerate(zoo.instances):
        off = inst.atom_offset
        ki  = inst.k_i

        active_mask = eval_data.feature_acts[:, off: off + ki].abs().sum(1) > 0
        active_rows = active_mask.nonzero(as_tuple=True)[0]
        if active_rows.numel() < 10:
            best_experts.append(0)
            continue

        # Ground-truth per-manifold contribution M_i = fa_i @ V_i
        V_i  = zoo.feature_dict.feature_vectors[off: off + ki].detach().cpu()
        fa_i = eval_data.feature_acts[active_rows, off: off + ki]
        M_i  = fa_i @ V_i                                  # (n_i, d_in)

        # Per-expert contributions for active rows
        h_i      = h_bott[active_rows]                     # (n_i, n_exp, d_bott)
        latents  = torch.einsum("bnd,nde->bne", h_i, W_latent_dec)   # (n_i, n_exp, d_exp)
        contrib  = torch.einsum("bne,ned->bnd", latents, W_dec_grouped)  # (n_i, n_exp, d_in)

        # Remove constant offsets from each expert's contribution (affine bias directions,
        # decoder b_dec components, etc.).  Without centering, sigma_bias² adds a fixed
        # penalty to ||contrib_e||² that is invisible to the residual dot-product, causing
        # R² to understate manifold geometry capture.
        contrib_c = contrib - contrib.mean(0, keepdim=True)   # (n_i, n_exp, d_in)

        # Eligibility: expert must fire (post-BatchTopK norm > 0) on ≥75% of
        # manifold i's active rows.  Experts below this threshold are excluded
        # from all greedy steps — they don't genuinely track this manifold.
        firing_frac = (h_i.norm(dim=-1) > 0).float().mean(0)  # (n_exp,)
        ineligible  = firing_frac < 0.75                       # (n_exp,) bool

        if ineligible.all():
            best_experts.append(0)
            continue

        # Greedy R²
        residual  = M_i - M_i.mean(0, keepdim=True)
        total_var = residual.pow(2).sum().clamp(min=1e-8)
        selected: list[int] = []

        for n in range(max_n):
            # gain_e = 2(residual · contrib_e_centered) - ||contrib_e_centered||²
            gains = (
                2.0 * (residual.unsqueeze(1) * contrib_c).sum(dim=(0, 2))
                - contrib_c.pow(2).sum(dim=(0, 2))
            )   # (n_exp,)
            gains[ineligible] = float("-inf")
            for e_sel in selected:
                gains[e_sel] = float("-inf")

            e_star = int(gains.argmax().item())
            selected.append(e_star)
            residual = residual - contrib_c[:, e_star, :]
            r2[inst_idx, n] = 1.0 - residual.pow(2).sum() / total_var

        best_experts.append(selected[0])

    return r2, best_experts


@torch.no_grad()
def compute_metrics(
    model: "SMIXAETraining",
    zoo: ManifoldZoo,
    eval_data: EvalData,
    device: str = "cpu",
    chunk_size: int = 4096,
) -> dict[str, float]:
    """Compute overall MSE, effective L0, and dead-expert count on the eval set.

    Args:
        model: Trained SMIXAETraining.
        zoo: ManifoldZoo (unused, kept for signature consistency).
        eval_data: EvalData on CPU.
        device: Model device.
        chunk_size: Evaluation chunk size.

    Returns:
        Dict with keys ``mse``, ``effective_l0``, ``dead_experts``.
    """
    model.eval()
    N = eval_data.x.shape[0]
    mse_sum = 0.0
    l0_sum  = 0.0
    ever_fired = torch.zeros(model.cfg.n_experts, dtype=torch.bool)

    for start in range(0, N, chunk_size):
        end     = min(start + chunk_size, N)
        x_chunk = eval_data.x[start:end].to(device)
        model.encode_with_hidden_pre(x_chunk)
        sae_out = model.decode(model.h_bottleneck)

        mse_sum += (sae_out - x_chunk).pow(2).sum(dim=-1).sum().item()
        norms    = model.h_bottleneck.norm(dim=-1)          # (B, n_experts)
        l0_sum  += (norms > 0).float().sum(dim=-1).sum().item()
        ever_fired |= (norms > 0).any(0).cpu()

    return {
        "mse":          mse_sum / N,
        "effective_l0": l0_sum / N,
        "dead_experts": int((~ever_fired).sum().item()),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

_TYPE_COLORS = {
    "circle":     "#1f77b4",
    "sphere":     "#ff7f0e",
    "mobius":     "#2ca02c",
    "swiss_roll": "#d62728",
    "helix":      "#9467bd",
    "flat_disk":  "#8c564b",
    "segment":    "#7f7f7f",
}

_TYPE_LABELS = {
    "circle":     "Circle",
    "sphere":     "Sphere",
    "mobius":     "Möbius",
    "swiss_roll": "Swiss Roll",
    "helix":      "Helix",
    "flat_disk":  "Flat Disk",
    "segment":    "Segment",
}


def plot_metrics_vs_k_experts(results: list[dict]) -> go.Figure:
    """Plot R²(n=1,2,3), MSE, effective L0, and dead experts vs k_experts.

    Args:
        results: List of result dicts, one per swept k_experts value.  Each
            must contain ``k_experts`` (int), ``r2`` (tensor n_instances×3),
            ``mse``, ``effective_l0``, ``dead_experts``, ``instances``
            (list[ManifoldInstance]), and ``l0_ground_truth`` (int).

    Returns:
        Plotly figure with four subplots.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Mean R²(n=1,2,3) vs k_experts",
            "MSE vs k_experts",
            "Effective L0 vs k_experts",
            "Dead experts vs k_experts",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    k_vals    = [r["k_experts"] for r in results]
    instances = results[0]["instances"]

    # R² aggregate traces (n=1,2,3)
    for n_idx, dash in enumerate(["solid", "dash", "dot"]):
        n_label  = n_idx + 1
        mean_r2  = [float(r["r2"][:, n_idx].mean().item()) for r in results]
        fig.add_trace(
            go.Scatter(
                x=k_vals, y=mean_r2,
                mode="lines+markers",
                name=f"R²(n={n_label})",
                line={"width": 2, "dash": dash},
            ),
            row=1, col=1,
        )

    # Per-type R²(n=1) as faded dashed traces (hidden by default)
    for type_name in _MANIFOLD_ORDER:
        type_idxs = [i for i, inst in enumerate(instances) if inst.type_name == type_name]
        if not type_idxs:
            continue
        mean_r2_type = [float(r["r2"][type_idxs, 0].mean().item()) for r in results]
        fig.add_trace(
            go.Scatter(
                x=k_vals, y=mean_r2_type,
                mode="lines+markers",
                name=_TYPE_LABELS[type_name],
                line={"color": _TYPE_COLORS[type_name], "dash": "dash", "width": 1},
                visible="legendonly",
            ),
            row=1, col=1,
        )

    # MSE
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=[r["mse"] for r in results],
            mode="lines+markers", name="MSE",
            line={"color": "#e377c2", "width": 2},
        ),
        row=1, col=2,
    )

    # Effective L0
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=[r["effective_l0"] for r in results],
            mode="lines+markers", name="Eff. L0",
            line={"color": "#17becf", "width": 2},
        ),
        row=2, col=1,
    )
    fig.add_hline(
        y=results[0].get("l0_ground_truth", 4),
        line_dash="dot", line_color="gray",
        annotation_text="Ground-truth L0",
        row=2, col=1,
    )

    # Dead experts
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=[r["dead_experts"] for r in results],
            mode="lines+markers", name="Dead experts",
            line={"color": "#bcbd22", "width": 2},
        ),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="k_experts", dtick=2)
    fig.update_yaxes(title_text="R²",                    row=1, col=1)
    fig.update_yaxes(title_text="MSE",                   row=1, col=2)
    fig.update_yaxes(title_text="Active experts / sample", row=2, col=1)
    fig.update_yaxes(title_text="Count",                 row=2, col=2)
    fig.update_layout(
        title="SMIXAE synthetic benchmark — k_experts sweep",
        height=800, width=1200,
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


def plot_bottlenecks(
    model: "SMIXAETraining",
    zoo: ManifoldZoo,
    eval_data: EvalData,
    best_experts: list[int],
    device: str = "cpu",
    max_samples_per_panel: int = 3000,
    representative_variant: int = 2,
) -> go.Figure:
    """3-D scatter of the best expert's bottleneck codes per manifold type.

    One panel per manifold type, showing the eval points where that manifold
    instance was active, coloured by the manifold's intrinsic parameter θ.
    The top-1 greedy expert (best_experts[inst_idx]) is used for each panel.

    Args:
        model: Trained SMIXAETraining.
        zoo: ManifoldZoo.
        eval_data: EvalData on CPU.
        best_experts: list[int] length n_instances — greedy top-1 expert index
            (from compute_restricted_r2).
        device: Model device.
        max_samples_per_panel: Cap on plotted points per panel.
        representative_variant: Which variant (0-5) to show per type.

    Returns:
        Plotly figure with 7 3-D subplots.
    """
    model.eval()
    h_bott = _encode_all(model, eval_data.x, device, chunk_size=2048)

    n_types = len(_MANIFOLD_ORDER)
    cols    = 4
    rows    = math.ceil(n_types / cols)

    specs = [[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)]
    subplot_titles = [_TYPE_LABELS[t] for t in _MANIFOLD_ORDER]
    subplot_titles += [""] * (rows * cols - n_types)

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )

    instance_by_type: dict[str, list[int]] = {t: [] for t in _MANIFOLD_ORDER}
    for idx, inst in enumerate(zoo.instances):
        instance_by_type[inst.type_name].append(idx)

    for subplot_idx, type_name in enumerate(_MANIFOLD_ORDER):
        row = subplot_idx // cols + 1
        col = subplot_idx % cols + 1

        inst_idxs = instance_by_type[type_name]
        inst_idx  = inst_idxs[representative_variant % len(inst_idxs)]
        inst      = zoo.instances[inst_idx]

        off  = inst.atom_offset
        ki   = inst.k_i
        active_mask = eval_data.feature_acts[:, off: off + ki].abs().sum(1) > 0
        active_rows = active_mask.nonzero(as_tuple=True)[0]
        if active_rows.numel() < 5:
            continue

        # Sub-sample for performance
        n_pts = min(max_samples_per_panel, active_rows.numel())
        perm  = torch.randperm(active_rows.numel())[:n_pts]
        rows_plot = active_rows[perm]

        expert_idx = best_experts[inst_idx]
        codes  = h_bott[rows_plot, expert_idx, :].numpy()             # (n_pts, 3)
        color  = eval_data.color_param[rows_plot, inst_idx].numpy()

        fig.add_trace(
            go.Scatter3d(
                x=codes[:, 0], y=codes[:, 1], z=codes[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=color,
                    colorscale="Viridis",
                    opacity=0.7,
                    showscale=False,
                ),
                name=_TYPE_LABELS[type_name],
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title="SMIXAE best-model bottleneck reconstruction (top-1 expert per manifold type)",
        height=350 * rows,
        width=1200,
        paper_bgcolor="white",
    )
    return fig


def plot_all_experts_with_originals(
    model: "SMIXAETraining",
    zoo: ManifoldZoo,
    eval_data: EvalData,
    best_experts: list[int],
    k_experts: int,
    device: str = "cpu",
    max_samples_per_panel: int = 1000,
) -> go.Figure:
    """3-D scatter of all 42 manifold instances: learned bottleneck + original geometry.

    For each of the 42 manifold instances, shows two overlaid traces:
    - Original: normalised manifold coordinates z̃_i zero-padded to 3-D.
    - Learned: best expert's bottleneck codes (h_bott[:, expert, :]).
    Both are coloured by the manifold's intrinsic parameter θ.  Panel titles
    embed the k_experts value and the assigned expert index.

    Args:
        model: Trained SMIXAETraining.
        zoo: ManifoldZoo.
        eval_data: EvalData on CPU.
        best_experts: list[int] length n_instances — greedy top-1 expert per instance
            (from compute_restricted_r2).
        k_experts: Active-expert budget used during training (shown in titles).
        device: Model device.
        max_samples_per_panel: Cap on plotted points per panel.

    Returns:
        Plotly figure with 42 3-D subplots arranged in 7 rows × 6 cols.
    """
    model.eval()
    h_bott = _encode_all(model, eval_data.x, device, chunk_size=2048)  # (N, n_exp, 3)

    n_inst = len(zoo.instances)
    cols   = 6
    rows   = math.ceil(n_inst / cols)  # 7

    specs = [[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)]
    subplot_titles = [
        f"{_TYPE_LABELS[inst.type_name]} v{inst.variant_idx + 1} | E{best_experts[i]} | k={k_experts}"
        for i, inst in enumerate(zoo.instances)
    ]
    # Pad to fill the grid
    subplot_titles += [""] * (rows * cols - n_inst)

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    for inst_idx, inst in enumerate(zoo.instances):
        row = inst_idx // cols + 1
        col = inst_idx % cols + 1

        off = inst.atom_offset
        ki  = inst.k_i
        active_mask = eval_data.feature_acts[:, off: off + ki].abs().sum(1) > 0
        active_rows = active_mask.nonzero(as_tuple=True)[0]
        if active_rows.numel() < 5:
            continue

        n_pts     = min(max_samples_per_panel, active_rows.numel())
        perm      = torch.randperm(active_rows.numel())[:n_pts]
        rows_plot = active_rows[perm]

        color = eval_data.color_param[rows_plot, inst_idx].numpy()

        # --- Original manifold (z̃_i zero-padded to 3D) ---
        orig = eval_data.feature_acts[rows_plot, off: off + ki]
        if ki < 3:
            orig = torch.cat([orig, torch.zeros(n_pts, 3 - ki)], dim=1)
        orig_np = orig.numpy()

        fig.add_trace(
            go.Scatter3d(
                x=orig_np[:, 0], y=orig_np[:, 1], z=orig_np[:, 2],
                mode="markers",
                marker=dict(size=2, color=color, colorscale="Blues", opacity=0.45),
                name="Original",
                showlegend=(inst_idx == 0),
                legendgroup="original",
            ),
            row=row, col=col,
        )

        # --- Learned bottleneck codes ---
        exp_idx = best_experts[inst_idx]
        codes   = h_bott[rows_plot, exp_idx, :].numpy()

        fig.add_trace(
            go.Scatter3d(
                x=codes[:, 0], y=codes[:, 1], z=codes[:, 2],
                mode="markers",
                marker=dict(size=2, color=color, colorscale="Reds", opacity=0.70),
                name="Learned",
                showlegend=(inst_idx == 0),
                legendgroup="learned",
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title=f"All manifold instances — original (blue) vs learned (red) | k_experts={k_experts}",
        height=320 * rows,
        width=1400,
        paper_bgcolor="white",
    )
    return fig
