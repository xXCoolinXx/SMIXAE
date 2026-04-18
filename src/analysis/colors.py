"""Shared color, colorbar, and legend backend.

Single source of truth for every bit of color math used by
:mod:`analysis.scatter3d` (interactive Plotly figures) and
:mod:`latex.camera_ready` (camera-ready PIL legends). Both call sites import
from here; neither reimplements sampling, tick selection, or legend rendering.

Sections
--------
1. Color parsing / conversion — hex / rgb(...) / rgba(...) / CSS names → canonical forms.
2. Colorscale sampling — maps ``n`` discrete labels onto a named Plotly scale.
   Applies the endpoint-skip fix: when ``skip_endpoints=True`` (default) the
   first and last labels land at ``1/(n+1)`` and ``n/(n+1)`` of the source
   scale rather than ``0`` and ``1``. Required for circular scales (``phase``,
   ``mygbm``, ``hsv``) where both endpoints are the same colour.
3. Plotly rendering — ``add_discrete_legend`` and ``add_colorbar_trace``.
   ``add_colorbar_trace`` builds a clipped colorscale list so the rendered
   colorbar widget matches the dot colours produced by
   ``sample_named_scale_discrete``.
4. PIL rendering — ``render_discrete_legend_png`` and
   ``render_continuous_colorbar_png`` for standalone PNG exports used by
   LaTeX figures.
5. ``export_legend_png`` — unified standalone-PNG entry point. Defaults to the
   PIL backend; ``backend="plotly"`` preserves the kaleido-based path for
   callers that need pixel parity with an interactive figure.
"""

from __future__ import annotations

import math
import re
from colorsys import hsv_to_rgb
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from matplotlib.colors import to_rgb as _mpl_to_rgb
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFont

# ══════════════════════════════════════════════════════════════════════════════
#  1. COLOR PARSING / CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_to_rgb(color: str) -> str:
    """Normalize any color string (named CSS, hex, rgb(...), rgba(...)) to 'rgb(R,G,B)'."""
    if color.startswith("rgb("):
        return color
    if color.startswith("rgba("):
        parts = color[5:-1].split(",")
        return f"rgb({parts[0].strip()},{parts[1].strip()},{parts[2].strip()})"
    r, g, b = _mpl_to_rgb(color)
    return f"rgb({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))})"


def rgb_from_floats(r: float, g: float, b: float) -> str:
    """(r,g,b) in [0,1] → 'rgb(R,G,B)'."""
    return f"rgb({int(round(r*255))},{int(round(g*255))},{int(round(b*255))})"


def rgb_with_alpha(rgb_str: str, alpha: float) -> str:
    """Append alpha to a canonical 'rgb(R,G,B)' string → 'rgba(R,G,B,alpha)'."""
    if not rgb_str.startswith("rgb("):
        rgb_str = normalize_to_rgb(rgb_str)
    return f"rgba({rgb_str[4:-1]},{alpha:.3f})"


def darken_rgb(rgb_str: str, factor: float = 0.55) -> str:
    """Return a darkened version of an 'rgb(R,G,B)' string (for font legibility)."""
    if not rgb_str.startswith("rgb("):
        rgb_str = normalize_to_rgb(rgb_str)
    vals = [int(v) for v in rgb_str[4:-1].split(",")]
    return f"rgb({int(vals[0]*factor)},{int(vals[1]*factor)},{int(vals[2]*factor)})"


def border_rgba(rgb_str: str, alpha: float = 0.5) -> str:
    """'rgb(R,G,B)' → 'rgba(R,G,B,alpha)' for annotation borders."""
    vals = rgb_str[4:-1]
    return f"rgba({vals},{alpha})"


def parse_rgb_tuple(c) -> tuple[int, int, int]:
    """Parse hex, rgb(...), rgba(...), or named colors to an (R, G, B) int tuple (for PIL)."""
    if c is None:
        return (0, 0, 0)
    s = str(c).strip()
    if s.startswith("rgb(") or s.startswith("rgba("):
        nums = re.findall(r"[\d.]+", s)
        if len(nums) >= 3:
            return (int(float(nums[0])), int(float(nums[1])), int(float(nums[2])))
    try:
        return ImageColor.getrgb(s)
    except Exception:
        return (0, 0, 0)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load DejaVuSans / Arial at the requested size; fall back to PIL default."""
    for name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def format_tick(v: float) -> str:
    """Format a numeric tick label: int if integral, else trimmed decimal."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    if abs(v) >= 10:
        return str(int(round(v)))
    return f"{v:.2f}"


# ══════════════════════════════════════════════════════════════════════════════
#  2. COLORSCALE SAMPLING (endpoint fix lives here)
# ══════════════════════════════════════════════════════════════════════════════

def _discrete_positions(n: int, *, skip_endpoints: bool) -> list[float]:
    """The ``n`` source-scale positions used for discrete label sampling.

    ``skip_endpoints=True`` maps label ``i`` to ``(i + 1) / (n + 1)`` so the
    first and last labels sit strictly inside ``(0, 1)`` — required for
    circular scales where ``t=0`` and ``t=1`` are identical.
    """
    if n <= 0:
        return []
    if skip_endpoints:
        return [(i + 1) / (n + 1) for i in range(n)]
    if n == 1:
        return [0.5]
    return [i / (n - 1) for i in range(n)]


def sample_named_scale(name: str, t: float) -> tuple[int, int, int]:
    """Sample a named Plotly colorscale at scalar position ``t`` → (R, G, B) ints."""
    t = max(0.0, min(1.0, t))
    col = pc.sample_colorscale(name, [t])[0]
    return parse_rgb_tuple(col)


def sample_named_scale_discrete(
    name: str,
    n: int,
    *,
    skip_endpoints: bool = True,
) -> list[str]:
    """Sample ``n`` 'rgb(R,G,B)' strings from a named Plotly scale.

    ``skip_endpoints=True`` (default) samples at ``(i + 1) / (n + 1)`` so the
    first and last labels never land on the colorscale extremes.
    """
    positions = _discrete_positions(n, skip_endpoints=skip_endpoints)
    if not positions:
        return []
    return list(pc.sample_colorscale(name, positions))


def build_color_map(
    classes: list,
    colorscale: Optional[Union[str, list, dict]],
    *,
    skip_endpoints: bool = True,
) -> Dict[Any, str]:
    """Map class labels → canonical 'rgb(R,G,B)' color strings.

    Parameters
    ----------
    classes        : list of unique class labels in the desired order.
    colorscale     : one of:
        ``None`` / ``"auto"`` / ``"hsv"`` — evenly-spaced HSV hues
        ``str``  — named Plotly colorscale (e.g. ``"Plasma"``, ``"phase"``)
        ``list`` — one Plotly-native color string per class
        ``dict`` — ``{label: color_str}`` explicit mapping
    skip_endpoints : only affects the named-colorscale branch. When ``True``
        (default), labels are sampled at ``(i+1)/(n+1)`` positions instead of
        at endpoints, preventing circular scales from showing duplicate colours
        at the first and last class.
    """
    n = len(classes)

    if isinstance(colorscale, dict):
        missing = [c for c in classes if c not in colorscale]
        if missing:
            raise ValueError(f"colorscale dict missing labels: {missing}")
        return {c: normalize_to_rgb(colorscale[c]) for c in classes}

    if isinstance(colorscale, list):
        if len(colorscale) < n:
            raise ValueError(
                f"colorscale list has {len(colorscale)} entries "
                f"but there are {n} unique classes."
            )
        return {c: normalize_to_rgb(colorscale[i]) for i, c in enumerate(classes)}

    name = colorscale if isinstance(colorscale, str) else "auto"

    if name in ("auto", "hsv", None):
        return {
            c: rgb_from_floats(*hsv_to_rgb(i / max(n, 1), 0.88, 0.90))
            for i, c in enumerate(classes)
        }

    try:
        sampled = sample_named_scale_discrete(name, n, skip_endpoints=skip_endpoints)
        return {c: sampled[i] for i, c in enumerate(classes)}
    except Exception as exc:
        raise ValueError(
            f"Unknown colorscale {name!r}. Pass a Plotly colorscale name, "
            "a list of color strings, a dict, or None."
        ) from exc


def build_clipped_colorscale(
    name: str,
    n: int,
    *,
    skip_endpoints: bool = True,
    n_stops: int = 256,
) -> list[list]:
    """Build a Plotly colorscale list whose endpoints match discrete-dot sampling.

    When ``skip_endpoints=True`` the returned list spans ``[0, 1]`` of Plotly's
    gradient space but internally remaps to ``[1/(n+1), n/(n+1)]`` of ``name``
    — making the Plotly colorbar widget visually consistent with dot colours
    produced by :func:`sample_named_scale_discrete`.
    """
    if n <= 0:
        return [[0.0, "rgb(0,0,0)"], [1.0, "rgb(0,0,0)"]]

    if skip_endpoints and n > 1:
        lo = 1.0 / (n + 1)
        hi = n / (n + 1)
    else:
        lo, hi = 0.0, 1.0

    source_positions = [lo + (hi - lo) * (k / (n_stops - 1)) for k in range(n_stops)]
    colors = pc.sample_colorscale(name, source_positions)
    return [[k / (n_stops - 1), colors[k]] for k in range(n_stops)]


# ══════════════════════════════════════════════════════════════════════════════
#  3. PLOTLY RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def auto_tick_increment(
    lo: float, hi: float, n_classes: int, requested: Optional[float]
) -> Optional[float]:
    """Pick a sensible colorbar tick increment for datasets with >25 classes.

    Selects the largest increment from ``{20, 10, 5}`` that still produces at
    least 5 visible ticks across ``[lo, hi]``. Returns ``requested`` unchanged
    when ``n_classes <= 25``.
    """
    if n_classes <= 25:
        return requested
    span = hi - lo
    for inc in [20, 10, 5]:
        if span / inc >= 5:
            return float(inc)
    return 5.0


def add_discrete_legend(
    fig: go.Figure,
    classes: list,
    cmap: Dict[Any, str],
    names: Dict[Any, str],
    marker_size: float = 8,
) -> go.Figure:
    """Add one invisible marker trace per class to populate Plotly's 2-D legend.

    Returns ``fig`` for chaining.
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
    skip_endpoints: bool = True,
) -> go.Figure:
    """Add an invisible scatter trace whose sole purpose is to render a colorbar.

    When ``skip_endpoints=True`` the colorscale is clipped so its visible range
    matches the ``(i+1)/(n+1)`` discrete sampling used by
    :func:`sample_named_scale_discrete` — required for circular scales.

    Parameters
    ----------
    classes          : sorted list of class labels.
    colorscale_name  : named Plotly colorscale string, e.g. ``"Plasma"``.
    names            : when provided, tick every class with its display name
                       (overrides ``label_range`` and ``tick_increment``).
    label_range      : ``(start_label, end_label)`` for endpoint-only ticks.
    colorbar_title   : optional title text above the colorbar.
    tick_increment   : numeric tick spacing; ignored when ``names`` is given.
                       When ``n_classes > 25`` and ``names`` is ``None``, this
                       value is overridden by :func:`auto_tick_increment`.
    skip_endpoints   : clip the colorscale to ``[1/(n+1), n/(n+1)]``.
    """
    n = len(classes)
    if names is not None:
        lo, hi = 0.0, float(n - 1)
        inc = auto_tick_increment(lo, hi, n, tick_increment)
        if inc is not None and inc > 0 and n > 25:
            raw = np.arange(np.ceil(lo / inc) * inc, hi + 1e-9, inc)
            indices = sorted(int(v) for v in raw if 0 <= int(v) < n)
            tickvals = indices
            ticktext = [str(names.get(classes[i], classes[i])) for i in indices]
        else:
            tickvals = list(range(n))
            ticktext = [str(names.get(c, c)) for c in classes]
    else:
        lo = float(classes[0]) if not isinstance(classes[0], str) else 0.0
        hi = float(classes[-1]) if not isinstance(classes[-1], str) else float(n - 1)
        tick_increment = auto_tick_increment(lo, hi, n, tick_increment)
        if tick_increment is not None and tick_increment > 0:
            first_tick = np.ceil(lo / tick_increment) * tick_increment
            tickvals = sorted(set(np.arange(first_tick, hi + 1e-9, tick_increment)))
            ticktext = [str(int(v)) if v == int(v) else f"{v:.2f}" for v in tickvals]
        else:
            start_label, end_label = (
                label_range if label_range is not None
                else (str(classes[0]), str(classes[-1]))
            )
            tickvals = [lo, hi]
            ticktext = [start_label, end_label]

    marker_colorscale: Any
    if skip_endpoints and n > 1:
        marker_colorscale = build_clipped_colorscale(
            colorscale_name, n, skip_endpoints=True
        )
    else:
        marker_colorscale = colorscale_name

    fig.add_trace(go.Scatter3d(
        x=[None, None], y=[None, None], z=[None, None],
        mode="markers",
        marker=dict(
            color=[lo, hi],
            colorscale=marker_colorscale,
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
#  4. PIL RENDERING (camera-ready defaults)
# ══════════════════════════════════════════════════════════════════════════════

# Large defaults so PNGs stay legible when \includegraphics shrinks them.
_LEGEND_BG_DEFAULT = (255, 255, 255, 0)
_LEGEND_FONT_SIZE_DEFAULT = 56
_LEGEND_TICK_FONT_SIZE_DEFAULT = 50
_LEGEND_SWATCH_PAD_DEFAULT = 14
_LEGEND_LINE_PAD_DEFAULT = 14


def render_discrete_legend_png(
    *,
    labels: list,
    label_to_color: dict,
    output_path: Path,
    display_names: Optional[dict] = None,
    font_size: int = _LEGEND_FONT_SIZE_DEFAULT,
    swatch_pad: int = _LEGEND_SWATCH_PAD_DEFAULT,
    line_pad: int = _LEGEND_LINE_PAD_DEFAULT,
    bg: tuple = _LEGEND_BG_DEFAULT,
) -> None:
    """Render a PIL swatch+text legend PNG.

    ``labels`` are rendered in the given order. ``display_names`` (optional)
    maps each label to the display string; otherwise ``str(label)`` is used.
    """
    names = display_names or {lbl: str(lbl) for lbl in labels}
    font = load_font(font_size)

    dummy = Image.new("RGBA", (10, 10), bg)
    d = ImageDraw.Draw(dummy)
    text_ws: list[int] = []
    text_hs: list[int] = []
    for lbl in labels:
        bbox = d.textbbox((0, 0), names.get(lbl, str(lbl)), font=font)
        text_ws.append(bbox[2] - bbox[0])
        text_hs.append(bbox[3] - bbox[1])
    text_w = max(text_ws) if text_ws else 1
    text_h = max(text_hs) if text_hs else font_size

    sw = int(text_h * 0.85)
    line_h = text_h + line_pad

    W = swatch_pad * 3 + sw + text_w
    H = swatch_pad * 2 + line_h * len(labels)

    im = Image.new("RGBA", (W, H), bg)
    draw = ImageDraw.Draw(im)

    x0 = swatch_pad
    y = swatch_pad
    for lbl in labels:
        color = parse_rgb_tuple(label_to_color.get(lbl))
        draw.rectangle(
            [x0, y + 2, x0 + sw, y + 2 + sw],
            fill=(*color, 255),
            outline=(0, 0, 0, 40),
        )
        draw.text(
            (x0 + sw + swatch_pad, y),
            names.get(lbl, str(lbl)),
            fill=(0, 0, 0, 255),
            font=font,
        )
        y += line_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(output_path)


def render_continuous_colorbar_png(
    *,
    colorscale: str,
    labels: list,
    output_path: Path,
    bar_h: int = 550,
    bar_w: int = 60,
    pad: int = 16,
    tick_len: int = 14,
    gap: int = 12,
    tick_font_size: int = _LEGEND_TICK_FONT_SIZE_DEFAULT,
    bg: tuple = _LEGEND_BG_DEFAULT,
) -> None:
    """Render a PIL continuous colorbar PNG with auto-selected 'nice' ticks.

    ``labels`` must be numeric (or numeric-convertible). ``vmin`` and ``vmax``
    are derived from them; endpoints are preserved (no ``skip_endpoints`` here
    — true continuous axes have meaningful endpoints).
    """
    vals: list[float] = []
    for v in labels:
        try:
            vals.append(float(v))
        except Exception:
            pass
    if not vals:
        raise ValueError("Continuous legend requested but labels are not numeric.")
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    tick_font = load_font(tick_font_size)

    dummy = Image.new("RGBA", (10, 10), bg)
    d = ImageDraw.Draw(dummy)
    sample_bbox = d.textbbox((0, 0), "0", font=tick_font)
    tick_h = sample_bbox[3] - sample_bbox[1]
    min_tick_gap_px = tick_h + 8

    span = vmax - vmin
    nice_increments = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
    increment = nice_increments[-1]
    for inc in nice_increments:
        n_interior = span / inc
        if n_interior <= 0:
            continue
        px_per_unit = bar_h / span
        if inc * px_per_unit >= min_tick_gap_px:
            increment = inc
            break

    first_tick = math.ceil(vmin / increment) * increment
    last_tick = math.floor(vmax / increment) * increment
    if first_tick > last_tick:
        first_tick = math.ceil(vmin)
        last_tick = math.floor(vmax)
    tick_vals: list[float] = []
    v = first_tick
    while v <= last_tick + 1e-9:
        tick_vals.append(v)
        v += increment
    tick_text = [format_tick(t) for t in tick_vals]

    tw = 0
    th = 0
    for t in tick_text:
        bbox = d.textbbox((0, 0), t, font=tick_font)
        tw = max(tw, bbox[2] - bbox[0])
        th = max(th, bbox[3] - bbox[1])

    W = pad + bar_w + gap + tick_len + gap + tw + pad
    H = pad + bar_h + pad + th // 2

    im = Image.new("RGBA", (W, H), bg)
    draw = ImageDraw.Draw(im)

    x_bar = pad
    y_bar = pad
    for yi in range(bar_h):
        t = 1.0 - yi / (bar_h - 1)
        col = sample_named_scale(colorscale, t)
        draw.line([(x_bar, y_bar + yi), (x_bar + bar_w, y_bar + yi)], fill=(*col, 255))

    draw.rectangle(
        [x_bar, y_bar, x_bar + bar_w, y_bar + bar_h], outline=(0, 0, 0, 80), width=1
    )

    for v, t in zip(tick_vals, tick_text):
        frac = (v - vmin) / (vmax - vmin)
        y = y_bar + (1.0 - frac) * bar_h
        y = int(round(y))
        x1 = x_bar + bar_w + gap
        x2 = x1 + tick_len
        draw.line([(x1, y), (x2, y)], fill=(0, 0, 0, 160), width=2)

        bbox = draw.textbbox((0, 0), t, font=tick_font)
        text_top = bbox[1]
        text_bot = bbox[3]
        text_h = text_bot - text_top
        text_x = x2 + gap
        text_y = y - text_h // 2 - text_top
        draw.text((text_x, text_y), t, fill=(0, 0, 0, 255), font=tick_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(output_path)


def render_discrete_from_scale_png(
    *,
    colorscale: str,
    labels: list,
    output_path: Path,
    display_names: Optional[dict] = None,
    skip_endpoints: bool = True,
    **kwargs,
) -> None:
    """Render discrete swatches whose colors are sampled from a named scale.

    Thin wrapper over :func:`render_discrete_legend_png` that sources colors
    via :func:`sample_named_scale_discrete` (endpoint-skip aware).
    """
    n = len(labels)
    if n == 0:
        return
    sampled = sample_named_scale_discrete(colorscale, n, skip_endpoints=skip_endpoints)
    label_to_color = {lbl: sampled[i] for i, lbl in enumerate(labels)}
    render_discrete_legend_png(
        labels=labels,
        label_to_color=label_to_color,
        output_path=output_path,
        display_names=display_names,
        **kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  5. UNIFIED STANDALONE PNG EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_legend_png(
    colorscale: Optional[Union[str, list, dict]],
    labels: list,
    output_path: Path,
    label_names: Optional[Union[Dict, List, Sequence]] = None,
    continuous_color: bool = False,
    backend: str = "pil",
) -> None:
    """Render a standalone legend PNG.

    Parameters
    ----------
    colorscale       : same formats as :func:`build_color_map`.
    labels           : sorted list of class labels.
    output_path      : destination path (parent dirs are created).
    label_names      : dict / list mapping label → display string; ``None``
                       uses ``str(label)``.
    continuous_color : ``True`` renders a continuous colorbar. ``False`` (default)
                       renders discrete swatches — for named scales the colors
                       are sampled via :func:`sample_named_scale_discrete` with
                       ``skip_endpoints=True``.
    backend          : ``"pil"`` (default) writes directly with PIL.
                       ``"plotly"`` builds a Plotly figure and exports via
                       kaleido — only needed for pixel parity with an
                       interactive figure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(labels)
    if n == 0:
        return

    names = _resolve_display_names(labels, label_names)

    if backend == "pil":
        if continuous_color:
            if not isinstance(colorscale, str):
                raise ValueError(
                    "continuous_color=True requires a named colorscale string."
                )
            render_continuous_colorbar_png(
                colorscale=colorscale,
                labels=labels,
                output_path=output_path,
            )
            return

        if isinstance(colorscale, str) and colorscale not in ("auto", "hsv"):
            render_discrete_from_scale_png(
                colorscale=colorscale,
                labels=labels,
                output_path=output_path,
                display_names=names,
            )
            return

        cmap = build_color_map(labels, colorscale)
        render_discrete_legend_png(
            labels=labels,
            label_to_color=cmap,
            output_path=output_path,
            display_names=names,
        )
        return

    if backend == "plotly":
        _export_legend_png_plotly(
            colorscale, labels, output_path, names=names, continuous_color=continuous_color
        )
        return

    raise ValueError(f"Unknown backend {backend!r}; choose 'pil' or 'plotly'.")


def _resolve_display_names(
    labels: list, label_names: Optional[Union[Dict, List, Sequence]]
) -> Dict[Any, str]:
    if label_names is None:
        return {c: str(c) for c in labels}
    if isinstance(label_names, dict):
        return {c: str(label_names.get(c, c)) for c in labels}
    ln = list(label_names)
    return {c: str(ln[i]) for i, c in enumerate(labels)}


def _export_legend_png_plotly(
    colorscale: Optional[Union[str, list, dict]],
    labels: list,
    output_path: Path,
    names: Dict[Any, str],
    continuous_color: bool,
) -> None:
    """Plotly+kaleido standalone PNG export. Preserved for API parity."""
    _colorscale_is_named = (
        isinstance(colorscale, str) and colorscale not in ("auto", "hsv", None)
    )

    fig = go.Figure()

    if _colorscale_is_named or continuous_color:
        _all_numeric = all(isinstance(lb, (int, float)) for lb in labels)
        names_for_cb = None if _all_numeric else names
        add_colorbar_trace(
            fig, labels,
            colorscale_name=colorscale if isinstance(colorscale, str) else "Viridis",
            names=names_for_cb,
            colorbar_len=0.95,
            skip_endpoints=not continuous_color,
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False,
            margin=dict(l=5, r=190, t=10, b=10),
            paper_bgcolor="white",
        )
        img_bytes = fig.to_image(format="png", width=480, height=1040, scale=1)
    else:
        cmap = build_color_map(labels, colorscale)
        add_discrete_legend(fig, labels, cmap, names)
        png_height = max(320, len(labels) * 52 + 120)
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            legend=dict(
                x=0.5, xanchor="center",
                y=0.5, yanchor="middle",
                bgcolor="rgba(255,255,255,0)",
                font=dict(size=18),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="white",
        )
        img_bytes = fig.to_image(format="png", width=480, height=png_height, scale=1)

    output_path.write_bytes(img_bytes)

    try:
        import io as _io

        img = Image.open(_io.BytesIO(img_bytes))
        if img.mode == "RGBA":
            bg_img = Image.new("RGBA", img.size, (255, 255, 255, 255))
        else:
            bg_img = Image.new("RGB", img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg_img)
        bbox = diff.getbbox()
        if bbox:
            img.crop(bbox).save(output_path)
    except ImportError:
        pass
