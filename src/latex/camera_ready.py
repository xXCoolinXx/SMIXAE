"""Assemble camera-ready LaTeX figure files from saved PNG snapshots.

Workflow:
1. Browse experts.html files and click "Save scatter" / "Save means" to download PNGs.
   PNGs are saved with a structured filename:
       {experiment_id}__{task}__{E{expert_id}}__{hyp_name}__{score_type}__{score:.4f}__{scatter|means}.png
2. Place all saved PNGs into a single camera-ready directory
   (e.g. results/camera_ready/).
3. Run:
       smixae latex figures \\
           --camera-ready-dir results/camera_ready/ \\
           --output-dir       results/paper/ \\
           --dataset-config   datasets/probing/dataset_config.json

Output structure::

    results/paper/
      legends/
        legend_gemma_2_9b_l11_hours.png
        legend_gemma_2_9b_l11_months.png
        ...
      figure_gemma_2_9b_l11_hours.tex
      figure_gemma_2_9b_l11_months.tex
      ...

Required LaTeX packages: subcaption, graphicx
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import typer

app = typer.Typer()

# ── Filename parsing ───────────────────────────────────────────────────────────

_FILENAME_RE = re.compile(
    r"^(?P<experiment_id>[^_].*?)__"
    r"(?P<task>[^_].*?)__"
    r"E(?P<expert_id>\d+)__"
    r"(?P<hyp_name>[^_].*?)__"
    r"(?P<score_type>[^_]+)__"
    r"(?P<score>[0-9.]+)__"
    r"(?P<figure_type>scatter|means)"
    r"\.png$"
)

_SCORE_LABEL: dict[str, str] = {
    "r2":    r"$R^2$",
    "acc":   "Acc.",
    "score": "Score",
}


@dataclass
class PNGEntry:
    path: Path
    experiment_id: str
    task: str
    expert_id: int
    hyp_name: str
    score_type: str
    score: float
    figure_type: str  # "scatter" or "means"


def scan_camera_ready(camera_ready_dir: Path) -> list[PNGEntry]:
    """Parse all PNG files in *camera_ready_dir* that match the expected filename convention."""
    entries: list[PNGEntry] = []
    for p in sorted(camera_ready_dir.glob("*.png")):
        m = _FILENAME_RE.match(p.name)
        if not m:
            typer.echo(f"  [skip] {p.name} — does not match naming convention", err=True)
            continue
        entries.append(PNGEntry(
            path=p,
            experiment_id=m.group("experiment_id"),
            task=m.group("task"),
            expert_id=int(m.group("expert_id")),
            hyp_name=m.group("hyp_name"),
            score_type=m.group("score_type"),
            score=float(m.group("score")),
            figure_type=m.group("figure_type"),
        ))
    return entries


# ── Color-info loading ─────────────────────────────────────────────────────────

def _task_from_dataframe_path(path_str: str) -> str:
    return Path(path_str).stem


def load_color_info(dataset_config_path: Path) -> dict[str, dict]:
    """Return a mapping from task name to its color configuration.

    Each value is a dict with keys:
    - ``color_map``: explicit {label: color} dict if present, else None
    - ``color_scale``: named Plotly colorscale string if present, else None
    - ``hypothesis_color_overrides``: {hyp_name: {label: color}} if present, else {}
    - ``continuous_color``: bool
    """
    with open(dataset_config_path) as f:
        raw = json.load(f)
    info: dict[str, dict] = {}
    for entry in raw:
        task = _task_from_dataframe_path(entry["dataframe_path"])
        info[task] = {
            "color_map":                  entry.get("color_map"),
            "color_scale":                entry.get("color_scale"),
            "hypothesis_color_overrides": entry.get("hypothesis_color_overrides", {}),
            "continuous_color":           entry.get("continuous_color", False),
        }
    return info


# ── Grouping ───────────────────────────────────────────────────────────────────

@dataclass
class LegendGroup:
    """A set of PNGs that share a single legend image."""
    key: str          # e.g. "gemma_2_9b_l11__hours" or "gemma_2_9b_l11__living_things__plant_animal"
    experiment_id: str
    task: str
    hyp_filter: str | None  # None = all hypotheses; str = only this hypothesis
    entries: list[PNGEntry] = field(default_factory=list)
    color_map: dict | None = None
    color_scale: str | None = None
    continuous_color: bool = False


def infer_legend_groups(
    entries: list[PNGEntry],
    color_info: dict[str, dict],
) -> list[LegendGroup]:
    """Group PNG entries into legend groups.

    Rules:
    - All entries for the same (experiment_id, task) share one legend, **unless**
      the task has ``hypothesis_color_overrides`` — in that case each overridden
      hypothesis gets its own sub-group; remaining hypotheses share one group.
    - Groups with no PNGs are not produced.
    """
    from collections import defaultdict

    # Index entries by (experiment_id, task)
    by_exp_task: dict[tuple[str, str], list[PNGEntry]] = defaultdict(list)
    for e in entries:
        by_exp_task[(e.experiment_id, e.task)].append(e)

    groups: list[LegendGroup] = []

    for (exp_id, task), task_entries in sorted(by_exp_task.items()):
        ci = color_info.get(task, {})
        overrides: dict[str, dict] = ci.get("hypothesis_color_overrides", {})

        if overrides:
            # Split overridden hypotheses into their own sub-groups
            overridden_hyps = set(overrides.keys())
            override_entries: dict[str, list[PNGEntry]] = defaultdict(list)
            default_entries: list[PNGEntry] = []

            for e in task_entries:
                if e.hyp_name in overridden_hyps:
                    override_entries[e.hyp_name].append(e)
                else:
                    default_entries.append(e)

            for hyp_name, hyp_entries in sorted(override_entries.items()):
                g = LegendGroup(
                    key=f"{exp_id}__{task}__{hyp_name}",
                    experiment_id=exp_id,
                    task=task,
                    hyp_filter=hyp_name,
                    entries=hyp_entries,
                    color_map=overrides[hyp_name],
                    continuous_color=False,
                )
                groups.append(g)

            if default_entries:
                g = LegendGroup(
                    key=f"{exp_id}__{task}",
                    experiment_id=exp_id,
                    task=task,
                    hyp_filter=None,
                    entries=default_entries,
                    color_map=ci.get("color_map"),
                    color_scale=ci.get("color_scale"),
                    continuous_color=ci.get("continuous_color", False),
                )
                groups.append(g)
        else:
            g = LegendGroup(
                key=f"{exp_id}__{task}",
                experiment_id=exp_id,
                task=task,
                hyp_filter=None,
                entries=task_entries,
                color_map=ci.get("color_map"),
                color_scale=ci.get("color_scale"),
                continuous_color=ci.get("continuous_color", False),
            )
            groups.append(g)

    return groups


# ── Legend rendering ───────────────────────────────────────────────────────────

def _plotly_colorscale_to_rgba(scale_name: str, n: int) -> list[str]:
    """Sample *n* evenly-spaced colors from a named Plotly colorscale."""
    from plotly.colors import sample_colorscale
    positions = [i / max(n - 1, 1) for i in range(n)]
    return sample_colorscale(scale_name, positions)


def _parse_rgb(color_str: str) -> tuple[float, float, float]:
    """Parse an 'rgb(r,g,b)' or CSS hex color string to (r,g,b) floats in [0,1]."""
    color_str = color_str.strip()
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color_str)
    if m:
        return tuple(int(m.group(i)) / 255 for i in range(1, 4))  # type: ignore[return-value]
    # CSS hex
    color_str = color_str.lstrip("#")
    if len(color_str) == 6:
        r, g, b = (int(color_str[i:i+2], 16) / 255 for i in (0, 2, 4))
        return r, g, b
    # Fall back to matplotlib parsing
    import matplotlib.colors as mcolors
    return mcolors.to_rgb(color_str)


def render_legend_png(group: LegendGroup, output_path: Path) -> None:
    """Render a clean legend swatch image for *group* and save to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if group.continuous_color and group.color_scale:
        _render_colorbar_legend(group.color_scale, output_path)
        return

    if group.color_map:
        _render_swatch_legend(group.color_map, output_path)
        return

    if group.color_scale:
        # Named colorscale with discrete classes — try to infer labels from task name.
        # Without live data we can only render a generic colorbar strip.
        _render_colorbar_legend(group.color_scale, output_path)
        return

    # No color info: skip
    typer.echo(f"  [skip legend] no color info for group {group.key}", err=True)


def _render_swatch_legend(color_map: dict, output_path: Path) -> None:
    """Render a horizontal swatch legend for a discrete color_map."""
    labels = list(color_map.keys())
    colors = [_parse_rgb(str(v)) for v in color_map.values()]
    n = len(labels)

    swatch_w = 0.25   # inches per swatch
    label_w  = 0.80   # inches per label
    height   = 0.35
    fig_w = max(4.0, n * (swatch_w + label_w) + 0.2)

    fig, ax = plt.subplots(figsize=(fig_w, height))
    ax.set_axis_off()

    x = 0.0
    step = (swatch_w + label_w) / fig_w
    for label, rgb in zip(labels, colors):
        ax.add_patch(mpatches.Rectangle(
            (x, 0.1), swatch_w / fig_w, 0.8,
            color=rgb, transform=ax.transAxes, clip_on=False,
        ))
        ax.text(
            x + (swatch_w + 3) / fig_w / 72 * fig_w,
            0.5,
            label,
            transform=ax.transAxes,
            va="center", ha="left",
            fontsize=7,
        )
        x += step

    fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _render_colorbar_legend(scale_name: str, output_path: Path) -> None:
    """Render a horizontal colorbar strip for a continuous/named colorscale."""
    n_steps = 256
    colors = _plotly_colorscale_to_rgba(scale_name, n_steps)
    rgb_array = np.array([_parse_rgb(c) for c in colors])

    fig, ax = plt.subplots(figsize=(5.0, 0.4))
    ax.imshow(rgb_array[np.newaxis, :, :], aspect="auto", extent=[0, 1, 0, 1])
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel(scale_name, fontsize=8)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ── LaTeX generation ───────────────────────────────────────────────────────────

_TASK_DISPLAY: dict[str, str] = {
    "weekdays":     "Weekdays",
    "hours":        "Hours",
    "temperatures": "Temperature",
    "time_units":   "Time Units",
    "body_parts":   "Body Parts",
    "living_things":"Living Things",
    "months":       "Months",
    "colors":       "Colors",
    "emotions":     "Emotions",
}

_HYP_DISPLAY: dict[str, str] = {
    "cyc_7d":          "7-Day Ring",
    "weekday_weekend": "Weekday vs Weekend",
    "cyc_24h":         "24-Hour Ring",
    "cyc_12h":         "12-Hour Ring",
    "am_pm":           "AM vs PM",
    "cyc_12m":         "12-Month Ring",
    "season":          "Season",
    "linear_f":        "Linear °F",
    "log_f":           "Log °F",
    "log_duration":    "log Duration",
    "plant_animal":    "Plant vs Animal",
    "taxonomy":        "Taxonomy",
    "hue_wheel":       "Hue Ring",
    "rgb":             "RGB",
    "warm_nat_cool":   "Warm/Cool",
    "valence_arousal": "Valence-Arousal",
    "quadrant":        "Quadrant",
}


def _esc_text(s: str) -> str:
    """Escape LaTeX special chars in plain text (not math mode)."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _subcaption(entry: PNGEntry, png_rel: str) -> str:
    """Return a \\subcaptionbox{...}{...} string for one PNG."""
    task_disp = _TASK_DISPLAY.get(entry.task, entry.task.replace("_", " ").title())
    hyp_disp  = _HYP_DISPLAY.get(entry.hyp_name, entry.hyp_name.replace("_", " "))
    score_lbl = _SCORE_LABEL.get(entry.score_type, entry.score_type.upper())
    # score_lbl may contain LaTeX math ($R^2$) — don't escape it; escape the rest
    caption = (
        f"{_esc_text(task_disp)}, Expert {entry.expert_id}, "
        f"{_esc_text(hyp_disp)} ({score_lbl}\\,=\\,{entry.score:.3f})"
    )
    return (
        f"  \\subcaptionbox{{{caption}}}{{%\n"
        f"    \\includegraphics[width=\\columnwidth]{{{png_rel}}}}}"
    )


def generate_latex_figure(
    group: LegendGroup,
    camera_ready_dir: Path,
    legend_path: Path | None,
    output_tex: Path,
    cols: int = 2,
) -> None:
    """Write one \\begin{{figure}}…\\end{{figure}} .tex file for *group*."""
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    # Use only scatter entries; pair with means when both exist for same expert+hyp
    scatter_entries = [e for e in group.entries if e.figure_type == "scatter"]
    means_entries   = {(e.expert_id, e.hyp_name): e for e in group.entries if e.figure_type == "means"}

    if not scatter_entries:
        typer.echo(f"  [skip .tex] no scatter entries in group {group.key}", err=True)
        return

    task_disp = _TASK_DISPLAY.get(group.task, group.task.replace("_", " ").title())
    model_disp = group.experiment_id.replace("_", "-")

    lines: list[str] = [
        "% Requires: \\usepackage{subcaption} \\usepackage{graphicx}",
        "% Image paths are relative to this .tex file's location.",
        f"% Generated by: smixae latex figures  (group: {group.key})",
        r"\begin{figure*}[t]",
    ]

    subcaps: list[str] = []
    for entry in scatter_entries:
        png_rel = str(Path("camera_ready") / entry.path.name)
        subcaps.append(_subcaption(entry, png_rel))
        # If a means figure exists for this expert+hyp, add it alongside
        means_entry = means_entries.get((entry.expert_id, entry.hyp_name))
        if means_entry is not None:
            means_rel = str(Path("camera_ready") / means_entry.path.name)
            subcaps.append(_subcaption(means_entry, means_rel))

    # Lay out in rows of `cols` subfigures
    row_parts: list[list[str]] = []
    for i in range(0, len(subcaps), cols):
        row_parts.append(subcaps[i:i + cols])

    subfig_width = f"{0.98 / cols:.2f}" + r"\textwidth"
    for row in row_parts:
        # Override width based on cols
        row_adjusted = [s.replace(r"\columnwidth", subfig_width) for s in row]
        lines.append("  " + r" \hfill".join(row_adjusted) + r" \\[4pt]")

    # Legend
    if legend_path is not None and legend_path.exists():
        legend_rel = str(Path("legends") / legend_path.name)
        lines.append(f"  \\includegraphics[width=\\textwidth]{{{legend_rel}}}")

    lines.append(
        f"  \\caption{{{task_disp} manifold structure ({model_disp}).}}"
    )
    lines.append(r"\end{figure*}")

    output_tex.write_text("\n".join(lines) + "\n")
    typer.echo(f"  Written {output_tex}")


# ── CLI entry point ────────────────────────────────────────────────────────────

@app.command()
def figures(
    camera_ready_dir: Path = typer.Option(..., help="Directory containing saved camera-ready PNGs"),
    output_dir: Path = typer.Option(..., help="Directory to write .tex files and legend images"),
    dataset_config: Path = typer.Option(
        Path("datasets/probing/dataset_config.json"),
        help="Path to dataset_config.json for color scheme information",
    ),
    cols: int = typer.Option(2, help="Number of subfigures per row"),
) -> None:
    """Assemble camera-ready PNGs into LaTeX figure files with shared legend images.

    Scans *camera-ready-dir* for PNGs saved by the browser Save buttons in experts.html,
    groups them by (experiment, task), renders per-group legend images, and writes one
    .tex file per group under *output-dir*.
    """
    if not camera_ready_dir.exists():
        typer.echo(f"Error: camera-ready-dir does not exist: {camera_ready_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Scanning {camera_ready_dir} …")
    entries = scan_camera_ready(camera_ready_dir)
    if not entries:
        typer.echo("No matching PNGs found.")
        raise typer.Exit(0)

    typer.echo(f"Found {len(entries)} PNG(s) across "
               f"{len({(e.experiment_id, e.task) for e in entries})} (experiment, task) pairs.")

    color_info = load_color_info(dataset_config) if dataset_config.exists() else {}
    groups = infer_legend_groups(entries, color_info)
    typer.echo(f"Grouped into {len(groups)} legend group(s).")

    legends_dir = output_dir / "legends"
    legends_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        typer.echo(f"\nGroup: {group.key}  ({len(group.entries)} PNG(s))")

        # Render legend
        legend_filename = f"legend_{group.key}.png"
        legend_path = legends_dir / legend_filename
        render_legend_png(group, legend_path)
        if legend_path.exists():
            typer.echo(f"  Legend: {legend_path}")

        # Write .tex
        tex_path = output_dir / f"figure_{group.key}.tex"
        generate_latex_figure(
            group=group,
            camera_ready_dir=camera_ready_dir,
            legend_path=legend_path if legend_path.exists() else None,
            output_tex=tex_path,
            cols=cols,
        )

    typer.echo(f"\nDone. Output written to {output_dir}")
