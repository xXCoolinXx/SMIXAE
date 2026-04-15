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
        legend_hours.png
        legend_months.png
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

import typer

app = typer.Typer()

# ── Filename parsing ───────────────────────────────────────────────────────────

_FILENAME_RE = re.compile(
    r"^(?P<experiment_id>[^_].*?)__"
    r"(?P<task>[^_].*?)__"
    r"E(?P<expert_id>\d+)__"
    r"(?P<hyp_name>[^_].*?)__"
    r"(?P<score_type>[^_].*?)__"
    r"(?P<score>[0-9.]+)__"
    r"(?P<figure_type>scatter|means)"
    r"\.png$"
)

_SCORE_LABEL: dict[str, str] = {
    "r2":    r"$R^2$",
    "acc":   "Acc.",
    "score": "Score",
    "per_gain": r"$\Delta$per",
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


# ── Task-name canonicalisation ────────────────────────────────────────────────

def _canonical_task(raw_task: str) -> str:
    """Strip the ``_—_…`` suffix added by ``utils.py`` when building HTML titles.

    The save workflow lowercases the dataset title and replaces spaces with
    underscores, producing task names like ``"hours_—_expert_analysis"`` in the
    PNG filename.  The dataset config key is just ``"hours"`` (the CSV stem).
    This helper strips the suffix so lookups work correctly.

    Examples::

        'hours_—_expert_analysis' → 'hours'
        'pile-uncopyrighted'       → 'pile-uncopyrighted'  (no suffix, unchanged)
    """
    return raw_task.split("_—_")[0]


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
    key: str          # e.g. "hours" or "living_things__plant_animal"
    task: str         # canonical task name (e.g. "hours")
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
    - All entries for the same canonical task share one legend, **unless**
      the task has ``hypothesis_color_overrides`` — in that case each overridden
      hypothesis gets its own sub-group; remaining hypotheses share one group.
    - Task names are canonicalised via :func:`_canonical_task` before config lookup.
    """
    from collections import defaultdict

    # Index entries by canonical task
    by_task: dict[str, list[PNGEntry]] = defaultdict(list)
    for e in entries:
        by_task[_canonical_task(e.task)].append(e)

    groups: list[LegendGroup] = []

    for task, task_entries in sorted(by_task.items()):
        ci = color_info.get(task, {})
        overrides: dict[str, dict] = ci.get("hypothesis_color_overrides", {})

        if overrides:
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
                    key=f"{task}__{hyp_name}",
                    task=task,
                    hyp_filter=hyp_name,
                    entries=hyp_entries,
                    color_map=overrides[hyp_name],
                    continuous_color=False,
                )
                groups.append(g)

            if default_entries:
                g = LegendGroup(
                    key=task,
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
                key=task,
                task=task,
                hyp_filter=None,
                entries=task_entries,
                color_map=ci.get("color_map"),
                color_scale=ci.get("color_scale"),
                continuous_color=ci.get("continuous_color", False),
            )
            groups.append(g)

    return groups


def build_all_config_groups(color_info: dict[str, dict]) -> list[LegendGroup]:
    """Build legend groups for **every** task in the config, even those without PNGs.

    This ensures a complete library of legend images is generated.
    """
    groups: list[LegendGroup] = []

    for task, ci in sorted(color_info.items()):
        overrides: dict[str, dict] = ci.get("hypothesis_color_overrides", {})

        if overrides:
            for hyp_name in sorted(overrides.keys()):
                groups.append(LegendGroup(
                    key=f"{task}__{hyp_name}",
                    task=task,
                    hyp_filter=hyp_name,
                    color_map=overrides[hyp_name],
                    continuous_color=False,
                ))
            # Default group (no override)
            groups.append(LegendGroup(
                key=task,
                task=task,
                hyp_filter=None,
                color_map=ci.get("color_map"),
                color_scale=ci.get("color_scale"),
                continuous_color=ci.get("continuous_color", False),
            ))
        else:
            groups.append(LegendGroup(
                key=task,
                task=task,
                hyp_filter=None,
                color_map=ci.get("color_map"),
                color_scale=ci.get("color_scale"),
                continuous_color=ci.get("continuous_color", False),
            ))

    return groups


# ── Legend rendering ───────────────────────────────────────────────────────────

def _render_group_legend(group: LegendGroup, output_path: Path) -> None:
    """Render a legend PNG for *group* using scatter3d's exact Plotly rendering."""
    from analysis.scatter3d import render_legend_png

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Hard-coded newline exception: use Viridis with integer range [1, line_length]
    if group.task == "pile-uncopyrighted":
        # Determine wraparound point from any PNG entry's experiment path,
        # or default to 150.  The newline output dirs are newline_80, newline_150, etc.
        line_length = _infer_newline_wrap(group.entries)
        labels = list(range(1, line_length + 1))
        render_legend_png(
            colorscale="Viridis",
            labels=labels,
            output_path=output_path,
            continuous_color=True,
        )
        return

    if group.color_map:
        labels = sorted(group.color_map.keys())
        render_legend_png(
            colorscale=group.color_map,
            labels=labels,
            output_path=output_path,
        )
    elif group.color_scale:
        # Named colorscale without explicit label→color map.
        # Build numeric labels matching the class count that scatter3d would see.
        # The colorbar renders by range; exact tick labels come from the data.
        labels = list(range(24))  # placeholder — colorbar will auto-range
        render_legend_png(
            colorscale=group.color_scale,
            labels=labels,
            output_path=output_path,
            continuous_color=group.continuous_color,
        )
    else:
        typer.echo(f"  [skip legend] no color info for group {group.key}", err=True)


def _infer_newline_wrap(entries: list[PNGEntry]) -> int:
    """Infer the newline wraparound point from entries' experiment paths.

    Looks for a ``newline_<N>`` pattern in the source directory tree.
    Defaults to 150 if no hint is found.
    """
    for e in entries:
        for part in e.path.parts:
            m = re.match(r"newline_(\d+)", part)
            if m:
                return int(m.group(1))
    return 150


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
    "pile-uncopyrighted": "Newline Position",
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
    "periodic_gain":   "Periodic Gain",
}


def _esc_text(s: str) -> str:
    """Escape LaTeX special chars in plain text (not math mode)."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _subcaption(entry: PNGEntry, png_rel: str) -> str:
    """Return a \\subcaptionbox{...}{...} string for one PNG."""
    canon = _canonical_task(entry.task)
    task_disp = _TASK_DISPLAY.get(canon, canon.replace("_", " ").title())
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
    # Derive model name from the first entry's experiment_id
    model_disp = scatter_entries[0].experiment_id.replace("_", "-")

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
    groups them by task, renders legend images for ALL config tasks, and writes one
    .tex file per group under *output-dir*.
    """
    if not camera_ready_dir.exists():
        typer.echo(f"Error: camera-ready-dir does not exist: {camera_ready_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Scanning {camera_ready_dir} …")
    entries = scan_camera_ready(camera_ready_dir)

    color_info = load_color_info(dataset_config) if dataset_config.exists() else {}

    # ── Generate legends for ALL config tasks ──────────────────────────────
    legends_dir = output_dir / "legends"
    legends_dir.mkdir(parents=True, exist_ok=True)

    all_groups = build_all_config_groups(color_info)

    typer.echo(f"\nGenerating {len(all_groups)} legend(s) from config …")
    legend_paths: dict[str, Path] = {}  # group.key → Path
    for group in all_groups:
        legend_filename = f"legend_{group.key}.png"
        legend_path = legends_dir / legend_filename
        _render_group_legend(group, legend_path)
        if legend_path.exists():
            typer.echo(f"  Legend: {legend_path}")
            legend_paths[group.key] = legend_path

    # ── Group PNG entries and generate .tex files ──────────────────────────
    if not entries:
        typer.echo("No matching PNGs found.")
        raise typer.Exit(0)

    typer.echo(f"\nFound {len(entries)} PNG(s) across "
               f"{len({_canonical_task(e.task) for e in entries})} task(s).")

    groups = infer_legend_groups(entries, color_info)
    typer.echo(f"Grouped into {len(groups)} figure group(s).")

    for group in groups:
        typer.echo(f"\nGroup: {group.key}  ({len(group.entries)} PNG(s))")
        tex_path = output_dir / f"figure_{group.key}.tex"
        generate_latex_figure(
            group=group,
            camera_ready_dir=camera_ready_dir,
            legend_path=legend_paths.get(group.key),
            output_tex=tex_path,
            cols=cols,
        )

    typer.echo(f"\nDone. Output written to {output_dir}")
