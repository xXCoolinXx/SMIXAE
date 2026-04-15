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
           --dataset-config   datasets/probing/dataset_config.json \\
           --results-json     results/results.json

Output structure::

    results/paper/
      legends/
        legend_hours.png
        legend_pile-uncopyrighted__80.png
        legend_pile-uncopyrighted__150.png
        ...
      probe_gemma_2_9b_l11.tex      ← all probing tasks for this experiment
      probe_gemma_2_2b_l12.tex
      newline_gemma_2_9b_l11.tex    ← newline tasks (one figure per wrap length)
      newline_gemma_2_2b_l12.tex
      ...

Each .tex file contains one \\begin{figure*} block per task/wrap, using a
minipage-based grid with the legend on the right side.

Required LaTeX packages: subcaption, graphicx
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    "r2":       r"$R^2$",
    "acc":      "Acc.",
    "score":    "Score",
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

    Examples::

        'hours_—_expert_analysis' → 'hours'
        'pile-uncopyrighted'       → 'pile-uncopyrighted'
    """
    return raw_task.split("_—_")[0]


# ── Color-info loading ─────────────────────────────────────────────────────────

def _task_from_dataframe_path(path_str: str) -> str:
    return Path(path_str).stem


def load_color_info(
    dataset_config_path: Path,
    csv_base_dir: Path | None = None,
) -> dict[str, dict]:
    """Return a mapping from task name to its color configuration.

    Each value is a dict with keys:
    - ``color_map``: explicit {label: color} dict if present, else None
    - ``color_scale``: named Plotly colorscale string if present, else None
    - ``hypothesis_color_overrides``: {hyp_name: {label: color}} if present, else {}
    - ``continuous_color``: bool
    - ``labels``: sorted unique Label-column values read from the CSV when
      *csv_base_dir* is provided and the CSV exists; ``None`` otherwise.
    """
    import csv as _csv

    with open(dataset_config_path) as f:
        raw = json.load(f)
    info: dict[str, dict] = {}
    for entry in raw:
        task = _task_from_dataframe_path(entry["dataframe_path"])

        labels: list | None = None
        if csv_base_dir is not None:
            csv_path = csv_base_dir / f"{task}.csv"
            if csv_path.exists():
                with open(csv_path, newline="") as cf:
                    reader = _csv.DictReader(cf)
                    raw_labels = {row["Label"] for row in reader if "Label" in row}
                # Attempt numeric sort + conversion; fall back to lexicographic.
                try:
                    float_vals = {v: float(v) for v in raw_labels}
                    sorted_str = sorted(raw_labels, key=lambda v: float_vals[v])
                    labels = [
                        int(float_vals[v]) if float_vals[v] == int(float_vals[v]) else float_vals[v]
                        for v in sorted_str
                    ]
                except ValueError:
                    labels = sorted(raw_labels)

        info[task] = {
            "color_map":                  entry.get("color_map"),
            "color_scale":                entry.get("color_scale"),
            "hypothesis_color_overrides": entry.get("hypothesis_color_overrides", {}),
            "continuous_color":           entry.get("continuous_color", False),
            "labels":                     labels,
        }
    return info


# ── Newline wrap lookup ────────────────────────────────────────────────────────

def load_newline_wrap_lookup(results_json_path: Path) -> dict[tuple, int]:
    """Build ``(experiment_id, expert_id, round(score, 4)) → line_length`` from results.json.

    Used to assign the correct wrap length to each newline PNG entry even when
    the flat camera_ready directory has no ``newline_<N>`` path component.
    """
    with open(results_json_path) as f:
        data = json.load(f)

    lookup: dict[tuple, int] = {}
    for exp_id, exp_data in data.items():
        for _wrap_key, wrap_data in exp_data.get("newline", {}).items():
            line_length = wrap_data.get("line_length")
            if line_length is None:
                continue
            for expert in wrap_data.get("top10_experts", []):
                key = (exp_id, expert["expert_id"], round(expert["periodic_gain"], 4))
                lookup[key] = line_length
    return lookup


def _get_wrap(entry: PNGEntry, lookup: dict | None) -> int:
    """Determine newline wrap length for *entry* from lookup table or path scan."""
    if lookup:
        key = (entry.experiment_id, entry.expert_id, round(entry.score, 4))
        if key in lookup:
            return lookup[key]
    return _infer_newline_wrap([entry])


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
    labels: list | None = None  # sorted unique label values; None = unknown


def infer_legend_groups(
    entries: list[PNGEntry],
    color_info: dict[str, dict],
    newline_wrap_lookup: dict | None = None,
) -> list[LegendGroup]:
    """Group PNG entries into legend groups.

    Rules:
    - All entries for the same canonical task share one legend, **unless**
      the task has ``hypothesis_color_overrides`` — in that case each overridden
      hypothesis gets its own sub-group; remaining hypotheses share one group.
    - ``pile-uncopyrighted`` entries are split by wrap length.  Wrap is resolved
      via *newline_wrap_lookup* (keyed by experiment_id, expert_id, score) when
      available, otherwise falls back to scanning path components.
    - Task names are canonicalised via :func:`_canonical_task` before config lookup.
    """
    by_task: dict[str, list[PNGEntry]] = defaultdict(list)
    for e in entries:
        by_task[_canonical_task(e.task)].append(e)

    groups: list[LegendGroup] = []

    for task, task_entries in sorted(by_task.items()):
        ci = color_info.get(task, {})

        # ── newline task: one group per (experiment_id, wrap length) ────────
        if task == "pile-uncopyrighted":
            by_exp_wrap: dict[tuple[str, int], list[PNGEntry]] = defaultdict(list)
            for e in task_entries:
                n = _get_wrap(e, newline_wrap_lookup)
                by_exp_wrap[(e.experiment_id, n)].append(e)
            for (exp_id_nl, n), wrap_entries in sorted(by_exp_wrap.items()):
                groups.append(LegendGroup(
                    key=f"pile-uncopyrighted__{n}",   # shared legend key across experiments
                    task="pile-uncopyrighted",
                    hyp_filter=None,
                    entries=wrap_entries,
                    color_scale="Viridis",
                    continuous_color=True,
                    labels=list(range(1, n + 1)),
                ))
            continue

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
                groups.append(LegendGroup(
                    key=f"{task}__{hyp_name}",
                    task=task,
                    hyp_filter=hyp_name,
                    entries=hyp_entries,
                    color_map=overrides[hyp_name],
                    continuous_color=False,
                    labels=ci.get("labels"),
                ))

            if default_entries:
                groups.append(LegendGroup(
                    key=task,
                    task=task,
                    hyp_filter=None,
                    entries=default_entries,
                    color_map=ci.get("color_map"),
                    color_scale=ci.get("color_scale"),
                    continuous_color=ci.get("continuous_color", False),
                    labels=ci.get("labels"),
                ))
        else:
            groups.append(LegendGroup(
                key=task,
                task=task,
                hyp_filter=None,
                entries=task_entries,
                color_map=ci.get("color_map"),
                color_scale=ci.get("color_scale"),
                continuous_color=ci.get("continuous_color", False),
                labels=ci.get("labels"),
            ))

    return groups


def build_all_config_groups(color_info: dict[str, dict]) -> list[LegendGroup]:
    """Build legend groups for every task in the config, even those without PNGs.

    ``pile-uncopyrighted`` is excluded — its legends depend on the actual wrap
    length present in the PNGs and are built by :func:`infer_legend_groups` instead.
    """
    groups: list[LegendGroup] = []

    for task, ci in sorted(color_info.items()):
        if task == "pile-uncopyrighted":
            continue

        overrides: dict[str, dict] = ci.get("hypothesis_color_overrides", {})

        if overrides:
            for hyp_name in sorted(overrides.keys()):
                groups.append(LegendGroup(
                    key=f"{task}__{hyp_name}",
                    task=task,
                    hyp_filter=hyp_name,
                    color_map=overrides[hyp_name],
                    continuous_color=False,
                    labels=ci.get("labels"),
                ))
            groups.append(LegendGroup(
                key=task,
                task=task,
                hyp_filter=None,
                color_map=ci.get("color_map"),
                color_scale=ci.get("color_scale"),
                continuous_color=ci.get("continuous_color", False),
                labels=ci.get("labels"),
            ))
        else:
            groups.append(LegendGroup(
                key=task,
                task=task,
                hyp_filter=None,
                color_map=ci.get("color_map"),
                color_scale=ci.get("color_scale"),
                continuous_color=ci.get("continuous_color", False),
                labels=ci.get("labels"),
            ))

    return groups


# ── Legend rendering ───────────────────────────────────────────────────────────

def _display_names(labels: list) -> dict:
    """Strip leading ``NN_`` sort prefixes for display (e.g. ``'06_insect'`` → ``'insect'``)."""
    return {lbl: re.sub(r"^\d+_", "", str(lbl)) for lbl in labels}


def _render_group_legend(group: LegendGroup, output_path: Path) -> None:
    """Render a legend PNG for *group* using scatter3d's exact Plotly rendering."""
    from analysis.scatter3d import render_legend_png

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if group.color_map:
        labels = sorted(group.color_map.keys())
        render_legend_png(
            colorscale=group.color_map,
            labels=labels,
            output_path=output_path,
            label_names=_display_names(labels),
        )
    elif group.color_scale:
        if not group.labels:
            typer.echo(f"  [skip legend] no labels available for group {group.key}", err=True)
            return
        render_legend_png(
            colorscale=group.color_scale,
            labels=group.labels,
            output_path=output_path,
            label_names=_display_names(group.labels),
            continuous_color=group.continuous_color,
        )
    elif group.labels:
        render_legend_png(
            colorscale=None,
            labels=group.labels,
            output_path=output_path,
            label_names=_display_names(group.labels),
        )
    else:
        typer.echo(f"  [skip legend] no color info for group {group.key}", err=True)


def _infer_newline_wrap(entries: list[PNGEntry]) -> int:
    """Infer newline wrap from ``newline_<N>`` path components; defaults to 150."""
    for e in entries:
        for part in e.path.parts:
            m = re.match(r"newline_(\d+)", part)
            if m:
                return int(m.group(1))
    return 150


# ── LaTeX generation ───────────────────────────────────────────────────────────

_TASK_DISPLAY: dict[str, str] = {
    "weekdays":          "Weekdays",
    "hours":             "Hours",
    "months":            "Months",
    "temperatures":      "Temperature",
    "time_units":        "Time Units",
    "body_parts":        "Body Parts",
    "living_things":     "Living Things",
    "colors":            "Colors",
    "emotions":          "Emotions",
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
    "linear_f":        r"Linear \textdegree F",
    "log_f":           r"Log \textdegree F",
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

# Canonical display order for probe tasks (pile-uncopyrighted always last in newline files)
_TASK_ORDER: list[str] = [
    "weekdays", "hours", "months", "temperatures",
    "time_units", "body_parts", "living_things",
    "colors", "emotions",
]


def _esc_text(s: str) -> str:
    """Escape LaTeX special chars in plain text (not math mode)."""
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _format_experiment_id(exp_id: str) -> str:
    """Convert a snake_case experiment ID to a human-readable model name.

    Examples::

        'gemma_2_9b_l11' → 'Gemma 2 9B, Layer 11'
        'gemma_2_2b_l12' → 'Gemma 2 2B, Layer 12'
    """
    m = re.match(r"gemma_(\d+)_(\d+)([bBkK])_l(\d+)$", exp_id)
    if m:
        gen, size, unit, layer = m.groups()
        return f"Gemma {gen} {size}{unit.upper()}, Layer {layer}"
    return exp_id.replace("_", " ").replace("-", " ").title()


def _subfigure_cell(
    entry: PNGEntry,
    png_rel: str,
    cell_frac: float,
    is_paired_means: bool = False,
) -> str:
    """Return lines for one ``\\begin{subfigure}…\\end{subfigure}`` block.

    For a means entry paired with a scatter, the caption is shortened to
    ``E{id} (means)`` since the hypothesis label already appears on the scatter.
    Standalone means entries (e.g. newline) get the full caption.
    """
    if is_paired_means:
        caption = f"E{entry.expert_id} (means)"
    else:
        hyp_disp  = _HYP_DISPLAY.get(entry.hyp_name, entry.hyp_name.replace("_", " "))
        score_lbl = _SCORE_LABEL.get(entry.score_type, entry.score_type.upper())
        caption = (
            f"E{entry.expert_id}, {_esc_text(hyp_disp)}"
            f" ({score_lbl}\\,=\\,{entry.score:.3f})"
        )
    width_spec = f"{cell_frac:.2f}\\linewidth"
    return (
        f"  \\begin{{subfigure}}[t]{{{width_spec}}}\n"
        f"    \\centering\n"
        f"    \\includegraphics[width=\\linewidth]{{{png_rel}}}\n"
        f"    \\caption{{{caption}}}\n"
        f"  \\end{{subfigure}}"
    )


def _figure_block(
    group: LegendGroup,
    camera_ready_dir: Path,
    legend_path: Path | None,
    cols: int,
) -> list[str]:
    """Return lines for one complete ``\\begin{figure*}…\\end{figure*}`` block.

    Image sequence:
    - If scatter entries exist: [scatter1, means1, scatter2, means2, …]
      (means silently omitted when no matching means exists for a scatter).
    - If only means entries (e.g. newline): means entries in expert_id order.
    """
    scatter_entries = [e for e in group.entries if e.figure_type == "scatter"]
    means_map = {
        (e.expert_id, e.hyp_name): e
        for e in group.entries if e.figure_type == "means"
    }

    if scatter_entries:
        images: list[tuple[PNGEntry, bool]] = []
        for se in scatter_entries:
            images.append((se, False))
            me = means_map.get((se.expert_id, se.hyp_name))
            if me is not None:
                images.append((me, True))
    else:
        images = [
            (e, False)
            for e in sorted(group.entries, key=lambda x: (-x.score, x.expert_id))
        ]

    if not images:
        return []

    model_disp = _format_experiment_id(images[0][0].experiment_id)

    # Caption
    if group.task == "pile-uncopyrighted":
        wrap = group.key.split("__")[-1] if "__" in group.key else "?"
        caption_text = f"Newline Position ({wrap} chars) --- {model_disp}."
    elif group.hyp_filter:
        hyp_disp = _HYP_DISPLAY.get(group.hyp_filter, group.hyp_filter.replace("_", " "))
        task_disp = _TASK_DISPLAY.get(group.task, group.task.replace("_", " ").title())
        caption_text = f"{_esc_text(task_disp)} ({_esc_text(hyp_disp)}) --- {model_disp}."
    else:
        task_disp = _TASK_DISPLAY.get(group.task, group.task.replace("_", " ").title())
        caption_text = f"{_esc_text(task_disp)} --- {model_disp}."

    cell_frac = 0.98 / cols

    lines: list[str] = [r"\begin{figure*}[t]", r"\centering"]

    # ── Content minipage ───────────────────────────────────────────────────────
    lines.append(r"\begin{minipage}[c]{0.88\linewidth}")
    lines.append(r"\centering")

    rows: list[list[tuple[PNGEntry, bool]]] = [
        images[i:i + cols] for i in range(0, len(images), cols)
    ]

    for row_idx, row in enumerate(rows):
        is_last_row = (row_idx == len(rows) - 1)
        for cell_idx, (entry, is_pm) in enumerate(row):
            png_rel = str(Path("camera_ready") / entry.path.name)
            cell = _subfigure_cell(entry, png_rel, cell_frac, is_pm)
            is_last_in_row = (cell_idx == len(row) - 1)
            if not is_last_in_row:
                cell += "\n  \\hfill"
            elif not is_last_row:
                cell += "\n  \\\\[4pt]"
            lines.append(cell)

    lines.append(r"\end{minipage}\hfill")

    # ── Legend minipage ────────────────────────────────────────────────────────
    if legend_path is not None and legend_path.exists():
        legend_rel = str(Path("legends") / legend_path.name)
        lines += [
            r"\begin{minipage}[c]{0.10\linewidth}",
            r"  \centering",
            f"  \\includegraphics[width=\\linewidth]{{{legend_rel}}}",
            r"\end{minipage}",
        ]

    lines += [
        f"\\caption{{{caption_text}}}",
        r"\end{figure*}",
    ]
    return lines


def generate_probe_tex(
    experiment_id: str,
    groups: list[LegendGroup],
    legend_paths: dict[str, Path],
    camera_ready_dir: Path,
    output_tex: Path,
    cols: int = 2,
) -> None:
    """Write one .tex file containing all probe-task figure blocks for *experiment_id*."""
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    def _sort_key(g: LegendGroup) -> tuple[int, str]:
        try:
            return (_TASK_ORDER.index(g.task), g.key)
        except ValueError:
            return (len(_TASK_ORDER), g.key)

    all_lines: list[str] = [
        "% Requires: \\usepackage{subcaption} \\usepackage{graphicx}",
        "% Image paths are relative to this .tex file's location.",
        f"% Generated by: smixae latex figures  (experiment: {experiment_id})",
        "",
    ]

    for group in sorted(groups, key=_sort_key):
        block = _figure_block(group, camera_ready_dir, legend_paths.get(group.key), cols)
        if block:
            all_lines.extend(block)
            all_lines.append("")

    output_tex.write_text("\n".join(all_lines) + "\n")
    typer.echo(f"  Written {output_tex}")


def generate_newline_tex(
    experiment_id: str,
    groups: list[LegendGroup],
    legend_paths: dict[str, Path],
    camera_ready_dir: Path,
    output_tex: Path,
    cols: int = 2,
) -> None:
    """Write one .tex file containing all newline figure blocks for *experiment_id*."""
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    def _wrap_key(g: LegendGroup) -> int:
        try:
            return int(g.key.split("__")[-1])
        except (ValueError, IndexError):
            return 0

    all_lines: list[str] = [
        "% Requires: \\usepackage{subcaption} \\usepackage{graphicx}",
        "% Image paths are relative to this .tex file's location.",
        f"% Generated by: smixae latex figures  (experiment: {experiment_id}, newline)",
        "",
    ]

    for group in sorted(groups, key=_wrap_key):
        block = _figure_block(group, camera_ready_dir, legend_paths.get(group.key), cols)
        if block:
            all_lines.extend(block)
            all_lines.append("")

    output_tex.write_text("\n".join(all_lines) + "\n")
    typer.echo(f"  Written {output_tex}")


def _split_groups_by_experiment(groups: list[LegendGroup]) -> list[LegendGroup]:
    """Ensure each LegendGroup contains entries from exactly one experiment_id.

    ``pile-uncopyrighted`` groups are already per-experiment (handled in
    :func:`infer_legend_groups`).  All other groups may have entries from
    multiple experiments and are duplicated here — one copy per experiment,
    sharing the same legend key so :func:`figures` can look up the right legend.
    """
    result: list[LegendGroup] = []
    for group in groups:
        if group.task == "pile-uncopyrighted":
            result.append(group)
            continue
        by_exp: dict[str, list[PNGEntry]] = defaultdict(list)
        for e in group.entries:
            by_exp[e.experiment_id].append(e)
        if len(by_exp) <= 1:
            result.append(group)
            continue
        for exp_id, exp_entries in sorted(by_exp.items()):
            result.append(LegendGroup(
                key=group.key,
                task=group.task,
                hyp_filter=group.hyp_filter,
                entries=exp_entries,
                color_map=group.color_map,
                color_scale=group.color_scale,
                continuous_color=group.continuous_color,
                labels=group.labels,
            ))
    return result


def generate_latex_figure(
    group: LegendGroup,
    camera_ready_dir: Path,
    legend_path: Path | None,
    output_tex: Path,
    cols: int = 2,
) -> None:
    """Write one figure .tex file for *group* (legacy single-task format).

    Kept for backwards compatibility; the ``figures`` command now uses
    :func:`generate_probe_tex` / :func:`generate_newline_tex` instead.
    """
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    block = _figure_block(group, camera_ready_dir, legend_path, cols)
    if not block:
        typer.echo(f"  [skip .tex] no renderable entries in group {group.key}", err=True)
        return
    header = [
        "% Requires: \\usepackage{subcaption} \\usepackage{graphicx}",
        f"% Generated by: smixae latex figures  (group: {group.key})",
        "",
    ]
    output_tex.write_text("\n".join(header + block) + "\n")
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
    results_json: Optional[Path] = typer.Option(
        None,
        help="Path to results.json; used to resolve newline wrap lengths per expert",
    ),
    cols: int = typer.Option(2, help="Number of subfigures per row"),
) -> None:
    """Assemble camera-ready PNGs into per-experiment LaTeX figure files.

    Produces one ``probe_{experiment_id}.tex`` and/or one ``newline_{experiment_id}.tex``
    per experiment found in *camera-ready-dir*.  Each file contains one
    ``\\begin{figure*}`` block per task (probe) or wrap length (newline), using a
    minipage grid with the legend on the right side.
    """
    if not camera_ready_dir.exists():
        typer.echo(f"Error: camera-ready-dir does not exist: {camera_ready_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Scanning {camera_ready_dir} …")
    entries = scan_camera_ready(camera_ready_dir)

    csv_base_dir = dataset_config.parent if dataset_config.exists() else None
    color_info = load_color_info(dataset_config, csv_base_dir=csv_base_dir) if dataset_config.exists() else {}

    # ── Generate legends for all config tasks (excluding pile-uncopyrighted) ──
    legends_dir = output_dir / "legends"
    legends_dir.mkdir(parents=True, exist_ok=True)

    all_config_groups = build_all_config_groups(color_info)
    typer.echo(f"\nGenerating {len(all_config_groups)} legend(s) from config …")
    legend_paths: dict[str, Path] = {}
    for group in all_config_groups:
        legend_path = legends_dir / f"legend_{group.key}.png"
        _render_group_legend(group, legend_path)
        if legend_path.exists():
            typer.echo(f"  Legend: {legend_path}")
            legend_paths[group.key] = legend_path

    # ── Group PNG entries ──────────────────────────────────────────────────────
    if not entries:
        typer.echo("No matching PNGs found.")
        raise typer.Exit(0)

    typer.echo(
        f"\nFound {len(entries)} PNG(s) across "
        f"{len({_canonical_task(e.task) for e in entries})} task(s)."
    )

    newline_wrap_lookup: dict | None = None
    if results_json is not None and results_json.exists():
        typer.echo(f"Loading newline wrap lookup from {results_json} …")
        newline_wrap_lookup = load_newline_wrap_lookup(results_json)
        typer.echo(f"  {len(newline_wrap_lookup)} expert→wrap entries loaded.")
    elif results_json is not None:
        typer.echo(f"  [warn] --results-json path not found: {results_json}", err=True)

    groups = infer_legend_groups(entries, color_info, newline_wrap_lookup)
    groups = _split_groups_by_experiment(groups)
    typer.echo(f"Grouped into {len(groups)} figure group(s).")

    # ── Render per-wrap newline legends ────────────────────────────────────────
    for group in groups:
        if group.task == "pile-uncopyrighted":
            legend_path = legends_dir / f"legend_{group.key}.png"
            _render_group_legend(group, legend_path)
            if legend_path.exists():
                typer.echo(f"  Legend: {legend_path}")
                legend_paths[group.key] = legend_path

    # ── Partition groups by experiment_id ─────────────────────────────────────
    by_exp: dict[str, dict[str, list[LegendGroup]]] = defaultdict(
        lambda: {"probe": [], "newline": []}
    )
    for group in groups:
        if not group.entries:
            continue
        exp_id = group.entries[0].experiment_id
        slot = "newline" if group.task == "pile-uncopyrighted" else "probe"
        by_exp[exp_id][slot].append(group)

    # ── Generate .tex files ────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    for exp_id in sorted(by_exp):
        probe_groups   = by_exp[exp_id]["probe"]
        newline_groups = by_exp[exp_id]["newline"]

        if probe_groups:
            tex_path = output_dir / f"probe_{exp_id}.tex"
            typer.echo(f"\nProbe ({exp_id}): {len(probe_groups)} group(s)")
            generate_probe_tex(exp_id, probe_groups, legend_paths, camera_ready_dir, tex_path, cols)

        if newline_groups:
            tex_path = output_dir / f"newline_{exp_id}.tex"
            typer.echo(f"\nNewline ({exp_id}): {len(newline_groups)} wrap(s)")
            generate_newline_tex(exp_id, newline_groups, legend_paths, camera_ready_dir, tex_path, cols)

    typer.echo(f"\nDone. Output written to {output_dir}")
