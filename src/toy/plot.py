"""Plotting utilities for the synthetic toy-model benchmark.

Contains:
- _TYPE_COLORS: per-manifold-type colour map
- _TYPE_LABELS: per-manifold-type display labels
- plot_metrics_vs_k_experts: Plotly figure of metric curves vs k_experts sweep
- plot_bottlenecks: Plotly figure of 3-D bottleneck codes per manifold type
- plot_all_experts_with_originals: 3-D scatter of all 48 instances
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from toy.manifolds import _MANIFOLD_ORDER
from toy.zoo import EvalData, ManifoldZoo

if TYPE_CHECKING:
    from smixae.smixae import SMIXAETraining

# ── Plotting ───────────────────────────────────────────────────────────────────

_TYPE_COLORS = {
    "circle":     "#1f77b4",
    "sphere":     "#ff7f0e",
    "torus":      "#aec7e8",
    "mobius":     "#2ca02c",
    "swiss_roll": "#d62728",
    "helix":      "#9467bd",
    "flat_disk":  "#8c564b",
    "segment":    "#7f7f7f",
}

_TYPE_LABELS = {
    "circle":     "Circle",
    "sphere":     "Sphere",
    "torus":      "Torus",
    "mobius":     "Möbius",
    "swiss_roll": "Swiss Roll",
    "helix":      "Helix",
    "flat_disk":  "Flat Disk",
    "segment":    "Segment",
}


def _encode_all_plot(
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


def plot_metrics_vs_k_experts(results: list[dict]) -> go.Figure:
    """Plot linear OLS R², co-firing rate, MSE, effective L0, and dead experts vs k_experts.

    Args:
        results: List of result dicts, one per swept k_experts value.  Each
            must contain ``k_experts`` (int), ``r2`` (tensor n_instances),
            ``cofiring`` (tensor n_instances), ``mse``, ``effective_l0``,
            ``dead_experts``, ``instances`` (list[ManifoldInstance]), and
            ``l0_ground_truth`` (int).

    Returns:
        Plotly figure with four subplots.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "R² (linear OLS) & co-firing rate vs k_experts",
            "MSE vs k_experts",
            "Effective L0 vs k_experts",
            "Dead experts vs k_experts",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    k_vals    = [r["k_experts"] for r in results]
    instances = results[0]["instances"]

    # Mean linear R²
    mean_r2 = [float(r["r2"].mean().item()) for r in results]
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=mean_r2,
            mode="lines+markers",
            name="R² (linear OLS)",
            line={"width": 2, "color": "#1f77b4"},
        ),
        row=1, col=1,
    )

    # Mean co-firing rate
    mean_cofiring = [float(r["cofiring"].mean().item()) for r in results]
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=mean_cofiring,
            mode="lines+markers",
            name="Co-firing rate",
            line={"width": 2, "color": "#ff7f0e", "dash": "dash"},
        ),
        row=1, col=1,
    )

    # Per-type R² as faded dashed traces (hidden by default)
    for type_name in _MANIFOLD_ORDER:
        type_idxs = [i for i, inst in enumerate(instances) if inst.type_name == type_name]
        if not type_idxs:
            continue
        mean_r2_type = [float(r["r2"][type_idxs].mean().item()) for r in results]
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
    fig.update_yaxes(title_text="R² / co-firing",          row=1, col=1)
    fig.update_yaxes(title_text="MSE",                     row=1, col=2)
    fig.update_yaxes(title_text="Active experts / sample", row=2, col=1)
    fig.update_yaxes(title_text="Count",                   row=2, col=2)
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
    r2: torch.Tensor | None = None,
    cofiring: torch.Tensor | None = None,
    device: str = "cpu",
    max_samples_per_panel: int = 3000,
    representative_variant: int = 2,
) -> go.Figure:
    """3-D scatter of the best expert's bottleneck codes per manifold type.

    One panel per manifold type, showing the eval points where that manifold
    instance was active, coloured by the manifold's intrinsic parameter θ.
    The matched expert (best_experts[inst_idx]) is used for each panel.

    Args:
        model: Trained SMIXAETraining.
        zoo: ManifoldZoo.
        eval_data: EvalData on CPU.
        best_experts: list[int] length n_instances — matched expert per instance
            (from compute_restricted_r2).
        r2: Optional tensor of shape (n_instances,) — linear OLS R² per instance.
            When provided, mean R² across the type's variants is shown in titles.
        cofiring: Optional tensor of shape (n_instances,) — co-firing rate per instance.
            When provided, mean co-firing rate is shown alongside R² in titles.
        device: Model device.
        max_samples_per_panel: Cap on plotted points per panel.
        representative_variant: Which variant (0-5) to show per type.

    Returns:
        Plotly figure with 8 3-D subplots.
    """
    model.eval()
    h_bott = _encode_all_plot(model, eval_data.x, device, chunk_size=2048)

    n_types = len(_MANIFOLD_ORDER)
    cols    = 4
    rows    = math.ceil(n_types / cols)

    instance_by_type: dict[str, list[int]] = {t: [] for t in _MANIFOLD_ORDER}
    for idx, inst in enumerate(zoo.instances):
        instance_by_type[inst.type_name].append(idx)

    subplot_titles = []
    for t in _MANIFOLD_ORDER:
        inst_idxs = instance_by_type[t]
        label = _TYPE_LABELS[t]
        if r2 is not None:
            mean_r2 = float(r2[inst_idxs].mean().item())
            label += f" | R²={mean_r2:.2f}"
        if cofiring is not None:
            mean_fire = float(cofiring[inst_idxs].mean().item())
            label += f" | fire={mean_fire:.0%}"
        subplot_titles.append(label)
    subplot_titles += [""] * (rows * cols - n_types)

    specs = [[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)]

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )

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
    r2: torch.Tensor | None = None,
    cofiring: torch.Tensor | None = None,
    device: str = "cpu",
    max_samples_per_panel: int = 1000,
) -> go.Figure:
    """3-D scatter of all 48 manifold instances: learned bottleneck + original geometry.

    For each of the 48 manifold instances, shows two overlaid traces:
    - Original: normalised manifold coordinates z̃_i zero-padded to 3-D.
    - Learned: matched expert's bottleneck codes (h_bott[:, expert, :]).
    Both are coloured by the manifold's intrinsic parameter θ.  Panel titles
    embed the k_experts value, the matched expert index, and per-instance R² /
    co-firing rate when provided.

    Args:
        model: Trained SMIXAETraining.
        zoo: ManifoldZoo.
        eval_data: EvalData on CPU.
        best_experts: list[int] length n_instances — matched expert per instance
            (from compute_restricted_r2).
        k_experts: Active-expert budget used during training (shown in titles).
        r2: Optional tensor of shape (n_instances,) — linear OLS R² per instance.
        cofiring: Optional tensor of shape (n_instances,) — co-firing rate per instance.
        device: Model device.
        max_samples_per_panel: Cap on plotted points per panel.

    Returns:
        Plotly figure with 48 3-D subplots arranged in 8 rows × 6 cols.
    """
    model.eval()
    h_bott = _encode_all_plot(model, eval_data.x, device, chunk_size=2048)  # (N, n_exp, 3)

    n_inst = len(zoo.instances)
    cols   = 6
    rows   = math.ceil(n_inst / cols)

    specs = [[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)]
    subplot_titles = [
        f"{_TYPE_LABELS[inst.type_name]} v{inst.variant_idx + 1}"
        f" | E{best_experts[i]} | k={k_experts}"
        + (f" | R²={float(r2[i].item()):.2f}" if r2 is not None else "")
        + (f" | fire={float(cofiring[i].item()):.0%}" if cofiring is not None else "")
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
