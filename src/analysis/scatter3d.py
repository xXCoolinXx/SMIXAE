"""
scatter3d.py — Flexible 3-D scatter with per-class means, labels, and colorbar
===============================================================================

Public API (all importable directly)
-------------------------------------
Color helpers
    build_color_map(classes, colorscale)  → dict[label, "rgb(...)"]
    rgb_with_alpha(rgb_str, alpha)         → "rgba(...)"

Label layout
    compute_label_offsets(mean_xyz, display_labels, ...)  → np.ndarray (N,2)

Figure building (composable — each returns the modified fig)
    add_continuous_scatter_trace(fig, xyz, values, colorscale, ...)
    add_scatter_trace(fig, xyz, labels, cmap, ...)
    add_mean_trace(fig, mean_xyz, classes, cmap, names, ...)
    add_label_annotations(fig, mean_xyz, classes, cmap, names, ...)
    add_colorbar_trace(fig, classes, colorscale_name, label_range, ...)

Top-level
    plot_3d_scatter(xyz, labels, ...)  → go.Figure

Data helpers
    make_hour_ring(...)
    make_continuous_ring(n_classes, ...)

Demo
    demo(which)   — "hour" | "plasma" | "dict" | "list" | "continuous"


Colorscale formats
------------------
None / "auto"        HSV rainbow, one hue per unique class
"Plasma", "Viridis"  Any Plotly colorscale name (used for both colors and colorbar)
list[color_str]      One Plotly-native color string per class in sorted order
dict[label, color]   Explicit per-label color mapping

All user-supplied color strings must be Plotly-native (hex, "rgb(...)", CSS names).

Legend
------
``plot_3d_scatter`` auto-selects the legend type based on colorscale:
- Named colorscale (e.g. "Viridis", "HSV") → vertical colorbar with class-name
  tick labels at each class position.
- HSV / dict / list colorscale → discrete colored marker entries in a Plotly
  legend panel.
Pass ``show_legend=False`` to suppress entirely.
"""

from __future__ import annotations

from colorsys import hsv_to_rgb
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from matplotlib.colors import to_rgb as _mpl_to_rgb


# ══════════════════════════════════════════════════════════════════════════════
#  COLOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_to_rgb(color: str) -> str:
    """Normalize any color string (named CSS, hex, rgb(...), rgba(...)) to 'rgb(R,G,B)'."""
    if color.startswith("rgb("):
        return color
    if color.startswith("rgba("):
        parts = color[5:-1].split(",")
        return f"rgb({parts[0].strip()},{parts[1].strip()},{parts[2].strip()})"
    # handles named CSS colors, hex strings, etc.
    r, g, b = _mpl_to_rgb(color)
    return f"rgb({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))})"


def _rgb_from_floats(r: float, g: float, b: float) -> str:
    """(r,g,b) in [0,1] → 'rgb(R,G,B)'."""
    return f"rgb({int(round(r*255))},{int(round(g*255))},{int(round(b*255))})"


def rgb_with_alpha(rgb_str: str, alpha: float) -> str:
    """
    Append alpha to a canonical 'rgb(R,G,B)' string.

    Parameters
    ----------
    rgb_str : str   Must be in 'rgb(R,G,B)' format (use build_color_map to
                    ensure this).
    alpha   : float Opacity in [0, 1].

    Returns
    -------
    str  'rgba(R,G,B,alpha)'
    """
    if not rgb_str.startswith("rgb("):
        rgb_str = _normalize_to_rgb(rgb_str)
    return f"rgba({rgb_str[4:-1]},{alpha:.3f})"


def _darken_rgb(rgb_str: str, factor: float = 0.55) -> str:
    """Return a darkened version of an 'rgb(R,G,B)' string (for font legibility)."""
    if not rgb_str.startswith("rgb("):
        rgb_str = _normalize_to_rgb(rgb_str)
    vals = [int(v) for v in rgb_str[4:-1].split(",")]
    return f"rgb({int(vals[0]*factor)},{int(vals[1]*factor)},{int(vals[2]*factor)})"


def _border_rgba(rgb_str: str, alpha: float = 0.5) -> str:
    """'rgb(R,G,B)' → 'rgba(R,G,B,alpha)' for annotation borders."""
    vals = rgb_str[4:-1]
    return f"rgba({vals},{alpha})"


def build_color_map(
    classes: list,
    colorscale: Optional[Union[str, list, dict]],
) -> Dict[Any, str]:
    """
    Build a mapping from class label → canonical 'rgb(R,G,B)' color string.

    Parameters
    ----------
    classes    : list of unique class labels in the desired order.
    colorscale : One of:
        None / "auto"  — evenly-spaced HSV hues
        str            — named Plotly colorscale (e.g. "Plasma", "Viridis")
        list[str]      — one Plotly-native color string per class
        dict           — {label: color_str} explicit mapping

    Returns
    -------
    dict mapping each class label to a normalized 'rgb(R,G,B)' string.
    All values are guaranteed to be in 'rgb(...)' format so rgb_with_alpha
    can be applied without further parsing.
    """
    n = len(classes)

    if isinstance(colorscale, dict):
        missing = [c for c in classes if c not in colorscale]
        if missing:
            raise ValueError(f"colorscale dict missing labels: {missing}")
        return {c: _normalize_to_rgb(colorscale[c]) for c in classes}

    if isinstance(colorscale, list):
        if len(colorscale) < n:
            raise ValueError(
                f"colorscale list has {len(colorscale)} entries "
                f"but there are {n} unique classes."
            )
        return {c: _normalize_to_rgb(colorscale[i]) for i, c in enumerate(classes)}

    name = colorscale if isinstance(colorscale, str) else "auto"

    if name in ("auto", "hsv", None):
        return {
            c: _rgb_from_floats(*hsv_to_rgb(i / max(n, 1), 0.88, 0.90))
            for i, c in enumerate(classes)
        }

    try:
        scale = pc.get_colorscale(name)
        sampled = pc.sample_colorscale(scale, [i / max(n - 1, 1) for i in range(n)])
        return {c: sampled[i] for i, c in enumerate(classes)}
    except Exception as exc:
        raise ValueError(
            f"Unknown colorscale {name!r}. Pass a Plotly colorscale name, "
            "a list of color strings, a dict, or None."
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
#  LABEL LAYOUT  (adjustText)
# ══════════════════════════════════════════════════════════════════════════════

def _camera_project(pts: np.ndarray, eye: np.ndarray) -> np.ndarray:
    """Orthographic projection of (N, 3) points onto the screen plane for a
    Plotly camera at ``eye`` looking at the origin with world-up = (0, 0, 1).

    Returns (N, 2) screen coordinates in the same unit system as ``pts``.
    """
    fwd = -eye / np.linalg.norm(eye)           # camera → origin
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    return pts @ np.column_stack([right, up])   # (N, 2)


def compute_label_offsets(
    mean_xyz: np.ndarray,
    display_labels: list,
    base_r: float = 120,
    fig_w: int = 1100,
    fig_h: int = 850,
    camera_eye: tuple = (1.5, -1.3, 0.8),
) -> np.ndarray:
    """
    Compute non-overlapping (ax, ay) pixel offsets for Plotly 3-D annotations
    using adjustText.

    Projects mean_xyz using the actual camera orthographic projection (so the
    2-D layout matches what appears on screen), normalizes to [0,1] space,
    places initial positions radially from the centroid, then runs adjustText
    with strong forces to resolve overlaps.

    Parameters
    ----------
    mean_xyz   : (K, 3) array of class mean positions in data space.
    display_labels : list of K label strings.
    base_r     : initial radial offset in pixels before adjustText runs.
    fig_w, fig_h : figure dimensions for pixel↔normalized conversion.
    camera_eye : Plotly camera eye tuple (x, y, z); must match the figure.

    Returns
    -------
    offsets : (K, 2) array of (ax, ay) pixel offsets for Plotly annotations.
    """
    from adjustText import adjust_text
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(mean_xyz)
    if n == 0:
        return np.empty((0, 2))

    # Project using the actual camera orientation so the 2-D layout
    # matches what Plotly renders on screen.
    eye = np.asarray(camera_eye, dtype=float)
    pts = _camera_project(mean_xyz.astype(float), eye)

    # Normalize to [0.05, 0.95]
    lo, hi = pts.min(0), pts.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    pts_n = (pts - lo) / span * 0.90 + 0.05

    # Radial initial positions to give adjustText a head-start
    base_r_n = base_r / np.array([fig_w, fig_h])
    centroid = pts_n.mean(0)
    dirs = pts_n - centroid
    norms = np.linalg.norm(dirs, axis=1, keepdims=True).clip(min=1e-6)
    init_n = pts_n + dirs / norms * base_r_n

    dpi = 100
    fig, ax = plt.subplots(figsize=(fig_w / dpi, fig_h / dpi), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.scatter(pts_n[:, 0], pts_n[:, 1], s=0)

    texts = [
        ax.text(init_n[i, 0], init_n[i, 1], display_labels[i], fontsize=8)
        for i in range(n)
    ]
    adjust_text(
        texts,
        x=pts_n[:, 0], y=pts_n[:, 1],
        ax=ax,
        force_text=(0.8, 0.8),
        force_points=(0.5, 0.5),
        expand=(2.5, 2.5),
    )
    fig.canvas.draw()

    # Normalized offset → pixels; flip y (mpl y↑, Plotly ay screen y↓)
    offsets = np.array([
        [(t.get_position()[0] - pts_n[i, 0]) * fig_w,
         -(t.get_position()[1] - pts_n[i, 1]) * fig_h]
        for i, t in enumerate(texts)
    ])
    plt.close(fig)
    return offsets


# ══════════════════════════════════════════════════════════════════════════════
#  AXIS HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _make_axis(label: str = "", dtick: Optional[float] = None) -> dict:
    """Grid and background visible; axis titles, tick labels, and spikes hidden."""
    d = dict(
        title=dict(text=""),
        showgrid=True, gridcolor="rgb(200,200,200)", gridwidth=2,
        showbackground=True, backgroundcolor="rgb(235,235,240)",
        zeroline=True, zerolinecolor="rgb(150,150,150)", zerolinewidth=2,
        showticklabels=False,
        showspikes=False,
    )
    if dtick is not None:
        d["dtick"] = dtick
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  COMPOSABLE TRACE / ANNOTATION BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def add_scatter_trace(
    fig: go.Figure,
    xyz: np.ndarray,
    labels: np.ndarray,
    cmap: Dict[Any, str],
    scatter_alpha: float = 0.5,
    scatter_size: float = 3,
    hovertext: Optional[List[str]] = None,
) -> go.Figure:
    """
    Add the raw point-cloud trace to fig.

    Each point is colored by its class using cmap, with scatter_alpha applied.
    Returns fig for chaining.

    Parameters
    ----------
    hovertext : list of per-point hover strings, or None to use (x,y,z) coordinates.
    """
    point_colors = [rgb_with_alpha(cmap[lbl], scatter_alpha) for lbl in labels]
    if hovertext is None:
        hovertext = [f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in zip(xyz[:,0], xyz[:,1], xyz[:,2])]
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker=dict(size=scatter_size, color=point_colors, line=dict(width=0)),
        hovertext=hovertext,
        hoverinfo="text",
        showlegend=False,
        name="scatter",
    ))
    return fig


def add_continuous_scatter_trace(
    fig: go.Figure,
    xyz: np.ndarray,
    values: np.ndarray,
    colorscale: str = "Viridis",
    scatter_alpha: float = 0.8,
    scatter_size: float = 3,
    hovertext: Optional[List[str]] = None,
    colorbar_title: str = "",
    colorbar_thickness: int = 20,
    colorbar_len: float = 0.75,
    colorbar_x: float = 1.02,
) -> go.Figure:
    """
    Add a scatter trace colored by continuous float values.

    Bypasses the class-label color machinery — color is applied directly from
    the ``values`` array via a Plotly colorscale, with a built-in colorbar.

    Parameters
    ----------
    values   : (N,) float array used as continuous color signal.
    colorscale : named Plotly colorscale string (default "Viridis").
    hovertext  : per-point hover strings; defaults to (x,y,z) coordinates.

    Returns
    -------
    fig for chaining.
    """
    if hovertext is None:
        hovertext = [
            f"({x:.2f}, {y:.2f}, {z:.2f})"
            for x, y, z in zip(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        ]
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker=dict(
            size=scatter_size,
            color=values,
            colorscale=colorscale,
            opacity=scatter_alpha,
            showscale=True,
            colorbar=dict(
                title=dict(text=colorbar_title, side="right"),
                thickness=colorbar_thickness,
                len=colorbar_len,
                x=colorbar_x,
                outlinewidth=1,
                outlinecolor="black",
                ticks="outside",
                ticklen=6,
                tickwidth=1,
                tickcolor="black",
            ),
            line=dict(width=0),
        ),
        hovertext=hovertext,
        hoverinfo="text",
        showlegend=False,
        name="scatter",
    ))
    return fig


def add_mean_trace(
    fig: go.Figure,
    mean_xyz: np.ndarray,
    classes: list,
    cmap: Dict[Any, str],
    names: Dict[Any, str],
    mean_alpha: float = 1.0,
    mean_size: float = 6,
    mean_marker_line_width: float = 1,
) -> go.Figure:
    """
    Add one marker per class mean to fig, colored to match the scatter trace.
    Returns fig for chaining.
    """
    fig.add_trace(go.Scatter3d(
        x=mean_xyz[:, 0], y=mean_xyz[:, 1], z=mean_xyz[:, 2],
        mode="markers",
        marker=dict(
            size=mean_size,
            color=[cmap[c] for c in classes],
            opacity=mean_alpha,
            line=dict(width=mean_marker_line_width, color="black"),
        ),
        hovertext=[
            f"{names[c]}<br>({mean_xyz[i,0]:.2f}, {mean_xyz[i,1]:.2f}, {mean_xyz[i,2]:.2f})"
            for i, c in enumerate(classes)
        ],
        hoverinfo="text",
        showlegend=False,
        name="means",
    ))
    return fig


def add_mean_line_trace(
    fig: go.Figure,
    mean_xyz: np.ndarray,
    classes: list,
    cmap: Dict[Any, str],
    line_width: float = 3,
) -> go.Figure:
    """
    Connect adjacent class means (in sorted class order) with a gradient line.
    No wraparound — the last point does NOT connect back to the first.

    The line color interpolates smoothly from each point's class color to the
    next, matching the mean sphere colors exactly. Achieved via a single
    Scatter3d trace with a custom colorscale built from cmap.

    Parameters
    ----------
    mean_xyz   : (K, 3) array of class mean positions, already in sorted order.
    classes    : sorted list of K class labels.
    cmap       : {label: 'rgb(R,G,B)'} — used to derive the gradient colorscale.
    line_width : stroke width in pixels (default 3).

    Returns
    -------
    fig for chaining.
    """
    n = len(classes)
    # Build a Plotly colorscale from the per-class rgb colors.
    # Format: [[frac, color], ...] with fracs in [0, 1].
    colorscale = [
        [i / max(n - 1, 1), cmap[c]]
        for i, c in enumerate(classes)
    ]
    fig.add_trace(go.Scatter3d(
        x=mean_xyz[:, 0], y=mean_xyz[:, 1], z=mean_xyz[:, 2],
        mode="lines",
        line=dict(
            color=list(range(n)),   # numeric index per point drives the gradient
            colorscale=colorscale,
            width=line_width,
            cauto=False,
            cmin=0,
            cmax=n - 1,
        ),
        hoverinfo="skip",
        showlegend=False,
        name="mean_line",
    ))
    return fig


def add_label_annotations(
    fig: go.Figure,
    mean_xyz: np.ndarray,
    classes: list,
    cmap: Dict[Any, str],
    names: Dict[Any, str],
    label_font_size: int = 10,
    label_bg: str = "rgba(255,255,255,0.88)",
    label_offset_base_r: float = 120,
    fig_w: int = 1100,
    fig_h: int = 850,
    camera_eye: tuple = (1.5, -1.3, 0.8),
    label_every_n: int = 1,
) -> go.Figure:
    """
    Compute adjustText-based offsets and attach 3-D annotations to fig.scene.
    Returns fig for chaining.
    """
    # Subsample classes if label_every_n > 1 (reduces clutter for dense sets).
    labeled_classes = classes[::label_every_n]
    labeled_xyz = mean_xyz[::label_every_n]

    display_labels = [names[c] for c in labeled_classes]
    offsets = compute_label_offsets(
        labeled_xyz, display_labels,
        base_r=label_offset_base_r, fig_w=fig_w, fig_h=fig_h,
        camera_eye=camera_eye,
    )

    annotations = []
    for i, c in enumerate(labeled_classes):
        rgb = cmap[c]
        annotations.append(dict(
            x=labeled_xyz[i, 0], y=labeled_xyz[i, 1], z=labeled_xyz[i, 2],
            text=f"<b>{names[c]}</b>",
            font=dict(size=label_font_size, color="black"),
            bgcolor=label_bg,
            bordercolor=_border_rgba(rgb),
            borderwidth=1, borderpad=3,
            showarrow=True,
            arrowhead=2, arrowsize=1, arrowwidth=1.2,
            arrowcolor="rgba(100,100,100,0.5)",
            ax=int(offsets[i, 0]),
            ay=int(offsets[i, 1]),
        ))

    # Merge with any existing annotations
    existing = list(fig.layout.scene.annotations or [])
    fig.update_layout(scene=dict(annotations=existing + annotations))
    return fig


def add_origin_marker(
    fig: go.Figure,
    color: str = "rgba(30,30,30,0.7)",
    size: float = 5,
    symbol: str = "cross",
) -> go.Figure:
    """
    Place a subtle crosshair marker at (0, 0, 0).

    Uses a 'cross' symbol (or any Plotly 3-D marker symbol) so it reads
    clearly as an origin reference without adding clutter.  No hover text
    — it is purely a visual anchor.

    Parameters
    ----------
    color  : Plotly color string for the marker.
    size   : marker size in pixels.
    symbol : Plotly 3-D marker symbol (default "cross").

    Returns
    -------
    fig for chaining.
    """
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(
            symbol=symbol,
            size=size,
            color=color,
            line=dict(width=1, color=color),
        ),
        hoverinfo="skip",
        showlegend=False,
        name="origin",
    ))
    return fig


def add_discrete_legend(
    fig: go.Figure,
    classes: list,
    cmap: Dict[Any, str],
    names: Dict[Any, str],
    marker_size: float = 8,
) -> go.Figure:
    """
    Add one invisible marker trace per class to populate Plotly's 2-D legend.
    Returns fig for chaining.
    """
    for c in classes:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=marker_size, color=cmap[c], line=dict(width=0)),
            name=names[c],
            showlegend=True,
        ))
    return fig


def add_colorbar_trace(
    fig: go.Figure,
    classes: list,
    colorscale_name: str,
    names: Optional[Dict] = None,
    label_range: Optional[tuple] = None,
    colorbar_title: str = "",
    colorbar_thickness: int = 20,
    colorbar_len: float = 0.75,
    colorbar_x: float = 1.02,
    tick_increment: Optional[float] = 20,
) -> go.Figure:
    """
    Add an invisible scatter trace whose sole purpose is to render a colorbar.

    Parameters
    ----------
    classes          : sorted list of class labels.
    colorscale_name  : named Plotly colorscale string, e.g. "Plasma".
    names            : when provided, tick every class with its display name
                       (overrides label_range and tick_increment).
    label_range      : (start_label, end_label) for endpoint-only ticks.
    colorbar_title   : optional title text above the colorbar.
    tick_increment   : numeric tick spacing; ignored when ``names`` is given.
    colorbar_thickness, colorbar_len, colorbar_x : colorbar geometry.
    """
    n = len(classes)
    if names is not None:
        # One tick per class, labeled with display names.
        lo, hi = 0.0, float(n - 1)
        tickvals = list(range(n))
        ticktext = [str(names.get(c, c)) for c in classes]
    else:
        lo = float(classes[0]) if not isinstance(classes[0], str) else 0.0
        hi = float(classes[-1]) if not isinstance(classes[-1], str) else float(n - 1)
        if tick_increment is not None and tick_increment > 0:
            first_tick = np.ceil(lo / tick_increment) * tick_increment
            tickvals = list(np.arange(first_tick, hi + 1e-9, tick_increment))
            if lo not in tickvals:
                tickvals = [lo] + tickvals
            tickvals = sorted(set(tickvals))
            ticktext = [str(int(v)) if v == int(v) else f"{v:.2f}" for v in tickvals]
        else:
            start_label, end_label = (
                label_range if label_range is not None
                else (str(classes[0]), str(classes[-1]))
            )
            tickvals = [lo, hi]
            ticktext = [start_label, end_label]

    fig.add_trace(go.Scatter3d(
        x=[None, None], y=[None, None], z=[None, None],
        mode="markers",
        marker=dict(
            color=[lo, hi],
            colorscale=colorscale_name,
            showscale=True,
            opacity=0,
            colorbar=dict(
                title=dict(text=colorbar_title, side="right"),
                thickness=colorbar_thickness,
                len=colorbar_len,
                x=colorbar_x,
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(size=11),
                outlinewidth=1,
                outlinecolor="black",
                ticks="outside",
                ticklen=6,
                tickwidth=1,
                tickcolor="black",
            ),
        ),
        hoverinfo="skip",
        showlegend=False,
        name="_colorbar",
    ))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  TOP-LEVEL FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_scatter(
    xyz: np.ndarray,
    labels: Sequence,
    *,
    # ── labelling ────────────────────────────────────────────────────────────
    label_names: Optional[Union[Dict, List, Sequence]] = None,
    # ── color ────────────────────────────────────────────────────────────────
    colorscale: Optional[Union[str, list, dict]] = None,
    scatter_alpha: float = 1.0,
    mean_alpha: float = 1.0,
    # ── markers ──────────────────────────────────────────────────────────────
    scatter_size: float = 1,
    mean_size: float = 6,
    mean_marker_line_width: float = 1,
    # ── mean line ────────────────────────────────────────────────────────────
    connect_means: bool = False,
    mean_line_width: float = 3,
    # ── legend ───────────────────────────────────────────────────────────────
    show_legend: bool = True,
    colorbar_title: str = "",
    colorbar_tick_increment: Optional[float] = 20,
    # ── origin marker ────────────────────────────────────────────────────────
    show_origin: bool = True,
    # ── layout ───────────────────────────────────────────────────────────────
    width: int = 1100,
    height: int = 850,
    camera_eye: Optional[dict] = None,
    axis_dtick: Optional[dict] = None,
    title: str = "",
    show: bool = False,
) -> go.Figure:
    """
    3-D scatter plot with per-class mean spheres, optional annotations, and
    optional continuous colorbar.

    Parameters
    ----------
    xyz            : (N, 3) array of point coordinates.
    labels         : (N,) array of class labels (any hashable type), OR a
                     float array for continuous-value coloring.  When a float
                     array is passed the class-label machinery (means,
                     annotations, colorbar dummy trace) is bypassed;
                     ``add_continuous_scatter_trace`` is used instead.
    label_names    : dict or list mapping labels to display strings.
    colorscale     : see module docstring for accepted formats.
    scatter_alpha  : opacity of individual scatter points (default 0.7).
    mean_alpha     : opacity of class-mean spheres (default 1.0).
    scatter_size   : marker size for scatter points.
    mean_size      : marker size for class-mean spheres.
    mean_marker_line_width : width of the black outline on mean spheres (default 1, pass 0 to disable).
    connect_means  : if True, draw a line connecting adjacent means in sorted
                     class order (no wraparound).
    mean_line_width: stroke width in pixels (default 3).
    show_legend    : if True (default), auto-select legend type: named colorscale
                     → colorbar with class-name ticks; otherwise → discrete
                     colored marker entries.
    colorbar_title : optional title shown above the colorbar.
    colorbar_tick_increment : spacing between colorbar tick labels (default 20).
    width, height  : figure pixel dimensions.
    camera_eye     : Plotly camera dict, e.g. dict(x=1.5, y=-1.3, z=0.8).
    axis_dtick     : dict with optional keys "x", "y", "z" for axis tick spacing.
    title          : figure title string.
    show           : if True, call fig.show() before returning.

    Returns
    -------
    go.Figure
    """
    # ── validate inputs ──────────────────────────────────────────────────────
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")

    labels = np.asarray(labels)
    if labels.shape[0] != xyz.shape[0]:
        raise ValueError("len(labels) must equal len(xyz)")

    if camera_eye is None:
        camera_eye = dict(x=1.5, y=-1.3, z=0.8)
    dtick = axis_dtick or {}

    _layout_kwargs = dict(
        title=dict(text=title, x=0.5) if title else None,
        scene=dict(
            xaxis=_make_axis("x", dtick.get("x")),
            yaxis=_make_axis("y", dtick.get("y")),
            zaxis=_make_axis("z", dtick.get("z")),
            aspectmode="data",
            camera=dict(eye=camera_eye, up=dict(x=0, y=0, z=1)),
        ),
        width=width, height=height,
        paper_bgcolor="white",
    )

    # ── continuous float labels: bypass class machinery ───────────────────────
    if np.issubdtype(labels.dtype, np.floating):
        _cs = colorscale if isinstance(colorscale, str) and colorscale not in ("auto", "hsv") else "Viridis"
        fig = go.Figure()
        add_continuous_scatter_trace(
            fig, xyz, labels,
            colorscale=_cs,
            scatter_alpha=scatter_alpha,
            scatter_size=scatter_size,
            colorbar_title=colorbar_title,
        )
        if show_origin:
            add_origin_marker(fig)
        fig.update_layout(**_layout_kwargs, margin=dict(l=0, r=80, t=40 if title else 10, b=0))
        if show:
            fig.show()
        return fig

    # ── class-based path (integer / string / hashable labels) ────────────────
    classes = sorted(set(labels.tolist()), key=lambda v: (str(type(v)), v))

    # ── display names ────────────────────────────────────────────────────────
    if label_names is None:
        names = {c: str(c) for c in classes}
    elif isinstance(label_names, dict):
        names = {c: str(label_names.get(c, c)) for c in classes}
    else:
        ln = list(label_names)
        names = {c: str(ln[i]) for i, c in enumerate(classes)}

    # ── color map ────────────────────────────────────────────────────────────
    cmap = build_color_map(classes, colorscale)

    # ── class means ──────────────────────────────────────────────────────────
    mean_xyz = np.array([xyz[labels == c].mean(axis=0) for c in classes])

    # ── build figure ─────────────────────────────────────────────────────────
    _colorscale_is_named = isinstance(colorscale, str) and colorscale not in ("auto", "hsv")

    fig = go.Figure()
    add_scatter_trace(fig, xyz, labels, cmap, scatter_alpha, scatter_size)
    add_mean_trace(fig, mean_xyz, classes, cmap, names, mean_alpha, mean_size, mean_marker_line_width)
    if show_origin:
        add_origin_marker(fig)

    if connect_means and len(classes) > 1:
        add_mean_line_trace(fig, mean_xyz, classes, cmap, line_width=mean_line_width)

    # ── legend: colorbar for named scales, discrete markers otherwise ─────────
    right_margin = 0
    legend_kwargs: dict = {}
    if show_legend:
        if _colorscale_is_named:
            add_colorbar_trace(
                fig, classes, colorscale,
                names=names,
                colorbar_title=colorbar_title,
                tick_increment=colorbar_tick_increment,
            )
            _max_label_chars = max((len(str(names.get(c, c))) for c in classes), default=6)
            right_margin = 40 + _max_label_chars * 9
            legend_kwargs["showlegend"] = False
        else:
            add_discrete_legend(fig, classes, cmap, names)
            _max_label_chars = max((len(str(names.get(c, c))) for c in classes), default=6)
            right_margin = 40 + _max_label_chars * 9
            legend_kwargs["showlegend"] = True
            legend_kwargs["legend"] = dict(
                x=1.02, y=0.5, xanchor="left", yanchor="middle",
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
            )

    fig.update_layout(
        **_layout_kwargs,
        **legend_kwargs,
        margin=dict(l=0, r=right_margin, t=40 if title else 10, b=0),
    )

    if show:
        fig.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_hour_ring(
    points_per_hour: int = 60,
    noise: float = 4.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Synthetic 24-class hour ring.

    Returns
    -------
    xyz          : (N, 3) point cloud
    labels       : (N,) integer labels 0–23
    label_names  : {int: str} hour label strings
    """
    rng = np.random.default_rng(seed)
    label_names = [
        "12AM","1AM","2AM","3AM","4AM","5AM",
        "6AM","7AM","8AM","9AM","10AM","11AM",
        "12PM","1PM","2PM","3PM","4PM","5PM",
        "6PM","7PM","8PM","9PM","10PM","11PM",
    ]
    all_xyz, all_labels = [], []
    for hour in range(24):
        theta = 2 * np.pi * hour / 24 + rng.normal(0, 0.06, points_per_hour)
        x = 80 * np.cos(theta) + rng.normal(0, noise, points_per_hour)
        y = 48 * np.sin(theta) + rng.normal(0, noise, points_per_hour)
        z = 100 * np.sin(theta) + 24 * np.cos(theta) + rng.normal(0, noise, points_per_hour)
        all_xyz.append(np.column_stack([x, y, z]))
        all_labels.append(np.full(points_per_hour, hour))
    return (
        np.vstack(all_xyz),
        np.concatenate(all_labels),
        dict(enumerate(label_names)),
    )


def make_continuous_ring(
    n_classes: int = 150,
    points_per_class: int = 20,
    noise: float = 4.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic ring with n_classes integer labels (0 … n_classes-1).
    Designed to demonstrate the colorbar mode.

    Returns
    -------
    xyz    : (N, 3) point cloud
    labels : (N,) integer labels 0 … n_classes-1
    """
    rng = np.random.default_rng(seed)
    all_xyz, all_labels = [], []
    for cls in range(n_classes):
        theta = 2 * np.pi * cls / n_classes + rng.normal(0, 0.02, points_per_class)
        x = 80 * np.cos(theta) + rng.normal(0, noise, points_per_class)
        y = 48 * np.sin(theta) + rng.normal(0, noise, points_per_class)
        z = 100 * np.sin(theta) + 24 * np.cos(theta) + rng.normal(0, noise, points_per_class)
        all_xyz.append(np.column_stack([x, y, z]))
        all_labels.append(np.full(points_per_class, cls))
    return np.vstack(all_xyz), np.concatenate(all_labels)


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════════════════════

def demo(which: str = "hour") -> go.Figure:
    """
    Notebook-friendly demos.

    Parameters
    ----------
    which : str
        "hour"       — 24-class hour ring, HSV rainbow with labels
        "plasma"     — same data, Plotly Plasma colorscale
        "dict"       — 4 Gaussian blobs, explicit rgb() color dict
        "list"       — 3 Gaussian blobs, hex color list
        "continuous" — 150-class ring, Viridis colorscale, no per-class labels,
                       colorbar with start/end annotation
    """
    rng = np.random.default_rng(7)

    if which == "hour":
        xyz, labels, label_names = make_hour_ring()
        return plot_3d_scatter(
            xyz, labels,
            label_names=label_names,
            axis_dtick={"x": 50, "y": 20, "z": 50},
            title="Hour Ring – HSV rainbow",
        )

    elif which == "plasma":
        xyz, labels, label_names = make_hour_ring()
        return plot_3d_scatter(
            xyz, labels,
            label_names=label_names,
            colorscale="Plasma",
            title="Hour Ring – Plasma colorscale",
        )

    elif which == "dict":
        centres = [(3, 0, 0), (-3, 0, 0), (0, 3, 0), (0, -3, 0)]
        xyz  = np.vstack([rng.standard_normal((200, 3)) + c for c in centres])
        labs = np.concatenate([np.full(200, i) for i in range(4)])
        return plot_3d_scatter(
            xyz, labs,
            label_names={0: "East", 1: "West", 2: "North", 3: "South"},
            colorscale={
                0: "rgb(230,57,70)",
                1: "rgb(42,157,143)",
                2: "rgb(233,196,106)",
                3: "rgb(155,93,229)",
            },
            title="4 Blobs – explicit rgb() dict",
        )

    elif which == "list":
        centres = [(2, 2, 0), (-2, -2, 0), (0, 0, 3)]
        xyz  = np.vstack([rng.standard_normal((150, 3)) + c for c in centres])
        labs = np.concatenate([np.full(150, f"cls_{i}") for i in range(3)])
        return plot_3d_scatter(
            xyz, labs,
            colorscale=["#ff006e", "#3a86ff", "#8338ec"],
            title="3 Blobs – hex color list",
        )

    elif which == "continuous":
        # 150-class ring: named colorscale → colorbar with class-name ticks auto-selected.
        xyz, labs = make_continuous_ring(n_classes=150)
        return plot_3d_scatter(
            xyz, labs,
            colorscale="Viridis",
            scatter_alpha=0.0,          # hide individual points; means tell the story
            connect_means=True,         # draw the sorted path through all means
            colorbar_title="class",
            title="150-class Ring – Viridis + colorbar + mean path",
        )

    raise ValueError(
        f"Unknown demo {which!r}. "
        "Choose: hour | plasma | dict | list | continuous"
    )