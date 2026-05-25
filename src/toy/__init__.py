"""Synthetic manifold toy-model benchmark for SMIXAE — public API.

Re-exports the full public API from the sub-modules so that
``from toy import ManifoldZoo`` etc. continue to work.
"""

from toy.manifolds import (
    _D_INTRINSIC,
    _K_PER_TYPE,
    _MANIFOLD_ORDER,
    _SAMPLERS,
    _VARIANT_PARAMS,
    N_VARIANTS,
    ManifoldInstance,
)
from toy.metrics import compute_cofiring_matrix, compute_metrics, compute_restricted_r2
from toy.plot import (
    plot_all_experts_with_originals,
    plot_bottlenecks,
    plot_metrics_vs_k_experts,
)
from toy.zoo import (
    EvalData,
    ManifoldActivationGenerator,
    ManifoldZoo,
    build_manifold_zoo,
    generate_eval_set,
    optimize_subspaces,
)

__all__ = [
    # manifolds
    "ManifoldInstance",
    "N_VARIANTS",
    "_D_INTRINSIC",
    "_K_PER_TYPE",
    "_MANIFOLD_ORDER",
    "_SAMPLERS",
    "_VARIANT_PARAMS",
    # zoo
    "ManifoldZoo",
    "EvalData",
    "ManifoldActivationGenerator",
    "build_manifold_zoo",
    "generate_eval_set",
    "optimize_subspaces",
    # metrics
    "compute_restricted_r2",
    "compute_metrics",
    "compute_cofiring_matrix",
    # plot
    "plot_metrics_vs_k_experts",
    "plot_bottlenecks",
    "plot_all_experts_with_originals",
]
