"""Manifold parameter sets, samplers, and ManifoldInstance dataclass.

Contains all per-type parameter lists, the _MANIFOLD_ORDER / _K_PER_TYPE /
_D_INTRINSIC / _VARIANT_PARAMS registries, the eight _sample_* functions,
the _SAMPLERS dispatch dict, and the ManifoldInstance dataclass.

This module has no internal dependencies — only standard library and torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

# ── Manifold parameter sets ────────────────────────────────────────────────────

_CIRCLE_RADII     = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
_SPHERE_RADII     = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
# Torus (R, r): major radius R, tube radius r — standard 3-D embedding k_i=3.
# Parameters adapted from the paper's Clifford-embedding (R,r) pairs.
_TORUS_PARAMS: list[tuple[float, float]] = [
    (2.0, 0.5),
    (2.0, 1.0),
    (3.0, 1.0),
    (3.0, 1.5),
    (4.0, 1.0),
    (4.0, 2.0),
]
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
    "torus",
    "mobius",
    "swiss_roll",
    "helix",
    "flat_disk",
    "segment",
]

_K_PER_TYPE: dict[str, int] = {
    "circle":     2,
    "sphere":     3,
    "torus":      3,
    "mobius":     3,
    "swiss_roll": 3,
    "helix":      3,
    "flat_disk":  2,
    "segment":    1,
}

_D_INTRINSIC: dict[str, int] = {
    "circle":     1,
    "sphere":     2,
    "torus":      2,
    "mobius":     2,
    "swiss_roll": 2,
    "helix":      1,
    "flat_disk":  2,
    "segment":    1,
}

_VARIANT_PARAMS: dict[str, list[Any]] = {
    "circle":     _CIRCLE_RADII,
    "sphere":     _SPHERE_RADII,
    "torus":      _TORUS_PARAMS,
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
    """Sample n points uniformly from a circle of radius r.

    Args:
        r: Circle radius.
        n: Number of sample points.
        device: Target device.

    Returns:
        Tuple of (color: (n,) angle θ, z_raw: (n, 2)).
    """
    theta = torch.rand(n, device=device) * (2 * math.pi)
    return theta, torch.stack([r * theta.cos(), r * theta.sin()], dim=1)


def _sample_sphere(r: float, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n points uniformly from a sphere of radius r.

    Args:
        r: Sphere radius.
        n: Number of sample points.
        device: Target device.

    Returns:
        Tuple of (color: (n,) azimuthal angle, z_raw: (n, 3)).
    """
    xyz = torch.randn(n, 3, device=device)
    xyz = xyz / xyz.norm(dim=1, keepdim=True).clamp(min=1e-8)
    color = torch.atan2(xyz[:, 1], xyz[:, 0])
    return color, r * xyz


def _sample_mobius(w: float, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n points from a Möbius strip of half-width w.

    Args:
        w: Half-width of the strip.
        n: Number of sample points.
        device: Target device.

    Returns:
        Tuple of (color: (n,) angle φ, z_raw: (n, 3)).
    """
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
    """Sample n points from a Swiss roll.

    Args:
        theta_max: Maximum toroidal angle.
        h_max: Maximum height.
        n: Number of sample points.
        device: Target device.

    Returns:
        Tuple of (color: (n,) angle θ, z_raw: (n, 3)).
    """
    theta = torch.rand(n, device=device) * theta_max
    h     = torch.rand(n, device=device) * h_max
    return theta, torch.stack([theta * theta.cos(), h, theta * theta.sin()], dim=1)


def _sample_helix(alpha: float, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n points from a helix with pitch alpha.

    Args:
        alpha: Pitch (z-axis rise per radian).
        n: Number of sample points.
        device: Target device.

    Returns:
        Tuple of (color: (n,) angle θ, z_raw: (n, 3)).
    """
    theta = torch.rand(n, device=device) * (4 * math.pi)
    return theta, torch.stack([theta.cos(), theta.sin(), alpha * theta], dim=1)


def _sample_flat_disk(R: float, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n points uniformly from a flat disk of radius R.

    Args:
        R: Disk radius.
        n: Number of sample points.
        device: Target device.

    Returns:
        Tuple of (color: (n,) angle φ, z_raw: (n, 2)).
    """
    r   = R * torch.rand(n, device=device).sqrt()
    phi = torch.rand(n, device=device) * (2 * math.pi)
    return phi, torch.stack([r * phi.cos(), r * phi.sin()], dim=1)


def _sample_segment(length: float, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n points uniformly from a line segment of given length.

    Args:
        length: Segment length.
        n: Number of sample points.
        device: Target device.

    Returns:
        Tuple of (color: (n,) position t, z_raw: (n, 1)).
    """
    t = torch.rand(n, device=device)
    return t, (t * length).unsqueeze(1)


def _sample_torus(
    R: float, r: float, n: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n points from a torus with major radius R and tube radius r.

    Uses the standard 3-D embedding (k_i=3):
        x = (R + r cos φ) cos θ
        y = (R + r cos φ) sin θ
        z = r sin φ

    Args:
        R: Major radius (centre of tube to centre of torus).
        r: Tube (minor) radius.
        n: Number of sample points.
        device: Target device.

    Returns:
        Tuple of (color: (n,) toroidal angle θ, z_raw: (n, 3)).
    """
    theta = torch.rand(n, device=device) * (2 * math.pi)   # toroidal angle
    phi   = torch.rand(n, device=device) * (2 * math.pi)   # poloidal angle
    x = (R + r * phi.cos()) * theta.cos()
    y = (R + r * phi.cos()) * theta.sin()
    z = r * phi.sin()
    return theta, torch.stack([x, y, z], dim=1)


_SAMPLERS: dict[str, Callable[..., tuple[torch.Tensor, torch.Tensor]]] = {
    "circle":     _sample_circle,
    "sphere":     _sample_sphere,
    "torus":      _sample_torus,
    "mobius":     _sample_mobius,
    "swiss_roll": _sample_swiss_roll,
    "helix":      _sample_helix,
    "flat_disk":  _sample_flat_disk,
    "segment":    _sample_segment,
}

# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class ManifoldInstance:
    """Metadata for one manifold instance (type × variant).

    Attributes:
        type_name: Manifold type key (e.g. "circle", "sphere").
        variant_idx: Variant index within the type (0–5).
        k_i: Ambient embedding dimension (1, 2, or 3).
        d_i: Intrinsic dimension.
        params: Type-specific variant params (float or tuple).
        mu: (k_i,) calibration mean, on CPU.
        sigma: RMS norm after centering.
        atom_offset: Start index in the joint FeatureDictionary.
        shift: (k_i,) translation in subspace (intrinsic) coords applied after
            centre/scale normalisation. Pushes a *sparse* manifold off the origin
            so its minimum active norm equals ``norm_floor``; zero for dense
            instances.
        is_dense: If True the instance is *dense* — active on every sample and
            origin-passing (no shift). If False it is *sparse* — sampled within
            the L0 budget and shifted off-origin.
        norm_floor: Target minimum active norm tᵢ for sparse instances (0.0 for
            dense). Recorded for provenance.
    """

    type_name: str
    variant_idx: int
    k_i: int       # ambient embedding dimension (1, 2, or 3)
    d_i: int       # intrinsic dimension
    params: Any    # type-specific variant params (float or tuple)
    mu: torch.Tensor   # (k_i,) calibration mean, on CPU
    sigma: float       # RMS norm after centering
    atom_offset: int   # start index in the joint FeatureDictionary
    shift: torch.Tensor = None  # type: ignore[assignment]  # (k_i,) subspace translation, on CPU
    is_dense: bool = False      # always-active + origin-passing when True
    norm_floor: float = 0.0     # target min active norm tᵢ (0 for dense)
