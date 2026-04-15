"""
Assemble camera-ready LaTeX figure files from saved PNG snapshots.

This version fixes the layout issues by:
- NEVER using \\hfill for horizontal layout (it causes huge, stretchy whitespace).
- Avoiding subfigure grids for the layout; instead it uses rigid minipage blocks
  packed into rows, centered with \\makebox[\\linewidth][c]{...}.
- Enforcing "rows of up to N plots" (legends do NOT count toward N).
- For each TASK, placing EXACTLY ONE legend immediately after the last plot of that task.
- Cropping legend PNGs tightly with PIL to remove all extra whitespace.

Legend rendering still uses your existing Plotly renderer (analysis.scatter3d.render_legend_png),
then we crop the output PNG with PIL.

LaTeX requirements:
- \\usepackage{graphicx}
(subcaption is not required by this generated layout)

Run:
  smixae latex figures --camera-ready-dir results/camera_ready --output-dir results/paper ...

It will write:
  results/paper/
    camera_ready/   (copies of the used PNGs)
    legends/        (cropped legend PNGs)
    probe_<exp>.tex
    newline_<exp>.tex
"""

from __future__ import annotations

import json
import re
import shutil
import string as _string
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer
from PIL import Image, ImageChops

app = typer.Typer()

# ------------------------- Tunable layout constants ----------------------------
# Max fraction of \linewidth to use for content. Leave slack for inter-minipage whitespace.
_USABLE_FRAC = 0.98

# Legend width relative to plot width (for layout slots).
# Keep this SMALL; legends are cropped and typically narrow.
_LEGEND_SCALE = 0.18

# Horizontal gap between TASK blocks (physical space, not stretch).
_BLOCK_GAP = r"\hspace{2.5mm}"

# Vertical gap between physical rows.
_ROW_VSPACE = r"\vspace{5pt}"

# Constrain both plots and legends to this height (max).
# If your plots look too small, increase this (e.g. 4.2cm).
_PANEL_HEIGHT = "4.0cm"

# --------------------------- Filename parsing ---------------------------------

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


# ------------------------ Task-name canonicalisation ---------------------------

def _canonical_task(raw_task: str) -> str:
    return raw_task.split("_—_")[0]


# ----------------------------- Color-info loading ------------------------------

def _task_from_dataframe_path(path_str: str) -> str:
    return Path(path_str).stem


def load_color_info(
    dataset_config_path: Path,
    csv_base_dir: Path | None = None,
) -> dict[str, dict]:
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


# ------------------------------ Newline wrap lookup ----------------------------

def load_newline_wrap_lookup(results_json_path: Path) -> dict[tuple, int]:
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


def _infer_newline_wrap(entries: list[PNGEntry]) -> int:
    for e in entries:
        for part in e.path.parts:
            m = re.match(r"newline_(\d+)", part)
            if m:
                return int(m.group(1))
    return 150


def _get_wrap(entry: PNGEntry, lookup: dict | None) -> int:
    if lookup:
        key = (entry.experiment_id, entry.expert_id, round(entry.score, 4))
        if key in lookup:
            return lookup[key]
    return _infer_newline_wrap([entry])


# -------------------------------- Grouping ------------------------------------

@dataclass
class LegendGroup:
    key: str
    task: str
    hyp_filter: str | None
    entries: list[PNGEntry] = field(default_factory=list)
    color_map: dict | None = None
    color_scale: str | None = None
    continuous_color: bool = False
    labels: list | None = None


def infer_legend_groups(
    entries: list[PNGEntry],
    color_info: dict[str, dict],
    newline_wrap_lookup: dict | None = None,
) -> list[LegendGroup]:
    by_task: dict[str, list[PNGEntry]] = defaultdict(list)
    for e in entries:
        by_task[_canonical_task(e.task)].append(e)

    groups: list[LegendGroup] = []

    for task, task_entries in sorted(by_task.items()):
        ci = color_info.get(task, {})

        # newline task: one group per (experiment_id, wrap length)
        if task == "pile-uncopyrighted":
            by_exp_wrap: dict[tuple[str, int], list[PNGEntry]] = defaultdict(list)
            for e in task_entries:
                n = _get_wrap(e, newline_wrap_lookup)
                by_exp_wrap[(e.experiment_id, n)].append(e)
            for (exp_id_nl, n), wrap_entries in sorted(by_exp_wrap.items()):
                groups.append(LegendGroup(
                    key=f"pile-uncopyrighted__{n}",
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


# ------------------------------ Legend rendering -------------------------------

def _display_names(labels: list) -> dict:
    return {lbl: re.sub(r"^\d+_", "", str(lbl)) for lbl in labels}


def _crop_png_to_content(path: Path, *, pad: int = 6, white_eps: int = 10) -> None:
    im = Image.open(path).convert("RGBA")

    alpha = im.getchannel("A")
    if alpha.getextrema()[0] < 255:
        mask = alpha.point(lambda a: 255 if a > 0 else 0)
        bbox = mask.getbbox()
    else:
        rgb = im.convert("RGB")
        bg = Image.new("RGB", rgb.size, (255, 255, 255))
        diff = ImageChops.difference(rgb, bg).convert("L")
        diff = diff.point(lambda p: 255 if p > white_eps else 0)
        bbox = diff.getbbox()

    if not bbox:
        return

    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(im.width, right + pad)
    bottom = min(im.height, bottom + pad)

    im.crop((left, top, right, bottom)).save(path)


def _render_group_legend(group: LegendGroup, output_path: Path) -> None:
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
        return

    if output_path.exists():
        _crop_png_to_content(output_path)


# ------------------------------ LaTeX generation -------------------------------

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

_TASK_ORDER: list[str] = [
    "weekdays", "hours", "months", "temperatures",
    "time_units", "body_parts", "living_things",
    "colors", "emotions",
]


def _esc_text(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def _format_experiment_id(exp_id: str) -> str:
    m = re.match(r"gemma_(\d+)_(\d+)([bBkK])_l(\d+)$", exp_id)
    if m:
        gen, size, unit, layer = m.groups()
        return f"Gemma {gen} {size}{unit.upper()}, Layer {layer}"
    return exp_id.replace("_", " ").replace("-", " ").title()


def _ordered_images(group: LegendGroup) -> list[PNGEntry]:
    scatter_entries = [e for e in group.entries if e.figure_type == "scatter"]
    means_map = {(e.expert_id, e.hyp_name): e for e in group.entries if e.figure_type == "means"}

    if scatter_entries:
        images: list[PNGEntry] = []
        for se in scatter_entries:
            images.append(se)
            me = means_map.get((se.expert_id, se.hyp_name))
            if me is not None:
                images.append(me)
        return images
    return sorted(group.entries, key=lambda x: (-x.score, x.expert_id))


def _next_letter(idx: int) -> str:
    if idx < 26:
        return _string.ascii_uppercase[idx]
    return _next_letter(idx // 26 - 1) + _string.ascii_uppercase[idx % 26]


def _entry_description(entry: PNGEntry) -> str:
    hyp_disp = _HYP_DISPLAY.get(entry.hyp_name, entry.hyp_name.replace("_", " "))
    score_lbl = _SCORE_LABEL.get(entry.score_type, entry.score_type.upper())
    return f"E{entry.expert_id}, {_esc_text(hyp_disp)} ({score_lbl}\\,=\\,{entry.score:.3f})."


def _caption_text(
    model_disp: str,
    labeled: list[tuple[str, PNGEntry]],
    newline_wrap: int | None = None,
) -> str:
    if newline_wrap is not None:
        prefix = f"Newline Position ({newline_wrap} chars) --- {model_disp}."
    else:
        prefix = f"{model_disp}."

    task_groups: list[tuple[str, list[tuple[str, PNGEntry]]]] = []
    current_task: str | None = None
    current_items: list[tuple[str, PNGEntry]] = []

    for letter, entry in labeled:
        task = _canonical_task(entry.task)
        if task != current_task:
            if current_items:
                task_groups.append((current_task, current_items))
            current_task = task
            current_items = []
        current_items.append((letter, entry))
    if current_items:
        task_groups.append((current_task, current_items))

    parts: list[str] = []
    for task, items in task_groups:
        task_disp = _TASK_DISPLAY.get(task, task.replace("_", " ").title())
        descriptions = " ".join(
            f"({letter.lower()}) {_entry_description(e)}" for letter, e in items
        )
        parts.append(f"\\textbf{{{_esc_text(task_disp)}}}: {descriptions}")

    return f"{prefix} {'  '.join(parts)}"


# ------------------------------ Block row planning -----------------------------

@dataclass
class _Unit:
    entry: PNGEntry
    letter: str


@dataclass
class _Block:
    group: LegendGroup
    units: list[_Unit]
    legend_path: Path | None


@dataclass
class _Row:
    blocks: list[_Block]
    # total plot count only (legends not counted)
    plot_count: int


def _split_groups_by_experiment(groups: list[LegendGroup]) -> list[LegendGroup]:
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
        for _exp_id, exp_entries in sorted(by_exp.items()):
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


def _plan_rows_for_experiment(
    groups: list[LegendGroup],
    cols: int,
    legend_paths: dict[str, Path],
) -> tuple[list[_Row], list[tuple[str, PNGEntry]]]:
    """
    Physical rows contain up to `cols` PLOTS TOTAL (across tasks).
    Each task is a block: plots then ONE legend after the final plot.
    """
    rows: list[_Row] = []
    labeled: list[tuple[str, PNGEntry]] = []
    letter_idx = 0

    cur_blocks: list[_Block] = []
    cur_plots = 0

    def flush() -> None:
        nonlocal cur_blocks, cur_plots
        if cur_blocks:
            rows.append(_Row(blocks=cur_blocks, plot_count=cur_plots))
            cur_blocks = []
            cur_plots = 0

    for group in groups:
        imgs = _ordered_images(group)
        if not imgs:
            continue

        units: list[_Unit] = []
        for e in imgs:
            letter = _next_letter(letter_idx)
            letter_idx += 1
            units.append(_Unit(entry=e, letter=letter))
            labeled.append((letter, e))

        leg = legend_paths.get(group.key)

        # If task has >cols plots, it gets its own dedicated rows (wrapped internally).
        if len(units) > cols:
            flush()
            rows.append(_Row(blocks=[_Block(group=group, units=units, legend_path=leg)], plot_count=len(units)))
            continue

        # pack compact tasks by plot count
        if cur_plots + len(units) > cols:
            flush()

        cur_blocks.append(_Block(group=group, units=units, legend_path=leg))
        cur_plots += len(units)

        if cur_plots == cols:
            flush()

    flush()
    return rows, labeled


# ------------------------------ LaTeX rendering --------------------------------

def _task_title(group: LegendGroup) -> str:
    return _TASK_DISPLAY.get(group.task, group.task.replace("_", " ").title())


def _include_plot(png_rel: str) -> str:
    # Use BOTH width and height caps to avoid giant/tiny scaling surprises.
    return (
        rf"\includegraphics[width=\linewidth,height={_PANEL_HEIGHT},keepaspectratio]"
        rf"{{{png_rel}}}"
    )


def _include_legend(legend_rel: str) -> str:
    return (
        rf"\includegraphics[width=\linewidth,height={_PANEL_HEIGHT},keepaspectratio]"
        rf"{{{legend_rel}}}"
    )


def _render_compact_row(row: _Row, cols: int) -> list[str]:
    """
    Render a physical row of multiple compact task blocks.

    Key properties:
    - plots never get bigger than a "3-plot row" size (cap)
    - no \\hfill (no stretchy whitespace)
    - blocks are centered using \\makebox
    """
    blocks = row.blocks
    if not blocks:
        return []

    # Count plots and legends that exist
    G = sum(len(b.units) for b in blocks)
    L = sum(1 for b in blocks if b.legend_path is not None and b.legend_path.exists())

    # Cap plot width so single-plot rows do not blow up.
    # This matches roughly "full row of cols plots + one legend".
    plot_w_cap = _USABLE_FRAC / (cols + _LEGEND_SCALE)

    # Choose plot_w to fit this row (legends consume width too), but never exceed cap.
    denom = (G + _LEGEND_SCALE * max(L, 0))
    plot_w = plot_w_cap if denom <= 0 else min(_USABLE_FRAC / denom, plot_w_cap)
    legend_w = _LEGEND_SCALE * plot_w

    # Build row content as a centered makebox of minipage blocks.
    lines: list[str] = []
    lines.append(r"\noindent\makebox[\linewidth][c]{%")

    for bi, block in enumerate(blocks):
        show_leg = block.legend_path is not None and block.legend_path.exists()
        k = len(block.units)

        block_w = k * plot_w + (legend_w if show_leg else 0.0)
        if block_w <= 0:
            continue

        # Start block minipage
        lines.append(rf"\begin{{minipage}}[t]{{{block_w:.4f}\linewidth}}%")
        lines.append(r"\vspace{0pt}\centering%")

        # Render k plot slots
        plot_frac = plot_w / block_w
        for i, u in enumerate(block.units):
            png_rel = str(Path("camera_ready") / u.entry.path.name)
            lines.append(rf"\begin{{minipage}}[t]{{{plot_frac:.4f}\linewidth}}%")
            lines.append(r"\vspace{0pt}\centering%")
            lines.append(_include_plot(png_rel) + "%")
            lines.append(r"\end{minipage}%")

        # Render legend immediately after last plot of the task
        if show_leg:
            leg_frac = legend_w / block_w
            legend_rel = str(Path("legends") / block.legend_path.name)
            lines.append(rf"\begin{{minipage}}[t]{{{leg_frac:.4f}\linewidth}}%")
            lines.append(r"\vspace{0pt}\centering%")
            lines.append(_include_legend(legend_rel) + "%")
            lines.append(r"\end{minipage}%")

        # Task label under the block
        title = _task_title(block.group)
        lines.append(r"\par\vspace{1pt}%")
        lines.append(rf"{{\footnotesize\textit{{{_esc_text(title)}}}}}%")
        lines.append(r"\end{minipage}%")

        if bi < len(blocks) - 1:
            lines.append(_BLOCK_GAP + "%")

    lines.append(r"}")  # end makebox
    return lines


def _render_multiline_task_block(block: _Block, cols: int) -> list[str]:
    """
    A single task with >cols plots. It becomes its own multi-row block:
    - internal rows of exactly cols plots (pad empties so size stays consistent)
    - ONE legend shown only after the final plot (i.e., in the final internal row)
    - legend slot width is reserved on all internal rows so plots don't change size
    """
    show_leg = block.legend_path is not None and block.legend_path.exists()

    # Fix plot width for internal rows as if legend exists (reserve legend slot).
    plot_w = _USABLE_FRAC / (cols + (_LEGEND_SCALE if show_leg else 0.0))
    legend_w = _LEGEND_SCALE * plot_w if show_leg else 0.0

    units = block.units
    chunks = [units[i:i + cols] for i in range(0, len(units), cols)]

    lines: list[str] = []
    lines.append(r"\noindent\makebox[\linewidth][c]{%")
    lines.append(r"\begin{minipage}[t]{\linewidth}%")
    lines.append(r"\vspace{0pt}\centering%")

    for ci, chunk in enumerate(chunks):
        is_last = (ci == len(chunks) - 1)

        # exactly cols plot slots (pad empties)
        for i in range(cols):
            lines.append(rf"\begin{{minipage}}[t]{{{plot_w:.4f}\linewidth}}%")
            lines.append(r"\vspace{0pt}\centering%")
            if i < len(chunk):
                png_rel = str(Path("camera_ready") / chunk[i].entry.path.name)
                lines.append(_include_plot(png_rel) + "%")
            else:
                lines.append(r"\vspace{0pt}%")
            lines.append(r"\end{minipage}%")

        # reserved legend slot (only filled on last internal row)
        if show_leg:
            lines.append(rf"\begin{{minipage}}[t]{{{legend_w:.4f}\linewidth}}%")
            lines.append(r"\vspace{0pt}\centering%")
            if is_last:
                legend_rel = str(Path("legends") / block.legend_path.name)
                lines.append(_include_legend(legend_rel) + "%")
            else:
                lines.append(r"\vspace{0pt}%")
            lines.append(r"\end{minipage}%")

        if not is_last:
            lines.append(r"\par\vspace{3pt}%")

    title = _task_title(block.group)
    lines.append(r"\par\vspace{1pt}%")
    lines.append(rf"{{\footnotesize\textit{{{_esc_text(title)}}}}}%")
    lines.append(r"\end{minipage}%")
    lines.append(r"}")
    return lines


def generate_figure_tex(
    experiment_id: str,
    groups: list[LegendGroup],
    legend_paths: dict[str, Path],
    output_tex: Path,
    cols: int = 3,
    *,
    is_newline: bool = False,
) -> None:
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    model_disp = _format_experiment_id(experiment_id)

    rows, labeled = _plan_rows_for_experiment(groups, cols, legend_paths)
    if not rows:
        typer.echo(f"  [skip] no renderable rows for {experiment_id}", err=True)
        return

    newline_wrap: int | None = None
    if is_newline and groups:
        wrap_str = groups[0].key.split("__")[-1] if "__" in groups[0].key else None
        if wrap_str:
            try:
                newline_wrap = int(wrap_str)
            except ValueError:
                pass

    caption = _caption_text(model_disp, labeled, newline_wrap)

    lines: list[str] = [
        "% Requires: \\usepackage{graphicx}",
        "% Image paths are relative to this .tex file's location.",
        f"% Generated by: smixae latex figures  (experiment: {experiment_id})",
        "",
        r"\begin{figure*}[t]",
        r"\centering",
    ]

    for ri, row in enumerate(rows):
        # Dedicated multiline task row: exactly one block and it has >cols units
        if len(row.blocks) == 1 and len(row.blocks[0].units) > cols:
            lines.extend(_render_multiline_task_block(row.blocks[0], cols))
        else:
            lines.extend(_render_compact_row(row, cols))

        if ri < len(rows) - 1:
            lines.append(_ROW_VSPACE)
            lines.append("")

    lines += [
        f"\\caption{{{caption}}}",
        r"\end{figure*}",
    ]

    output_tex.write_text("\n".join(lines) + "\n")
    typer.echo(f"  Written {output_tex}")


# ------------------------------ Output syncing --------------------------------

def _sync_camera_ready_images(entries: list[PNGEntry], output_dir: Path) -> None:
    """Copy used camera-ready PNGs into output_dir/camera_ready for LaTeX portability."""
    dst_dir = output_dir / "camera_ready"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for e in entries:
        dst = dst_dir / e.path.name
        if not dst.exists():
            shutil.copy2(e.path, dst)


# -------------------------------- CLI entry -----------------------------------

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
    cols: int = typer.Option(3, help="Max number of PLOTS per physical row (legends do not count)"),
) -> None:
    if not camera_ready_dir.exists():
        typer.echo(f"Error: camera-ready-dir does not exist: {camera_ready_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Scanning {camera_ready_dir} …")
    entries = scan_camera_ready(camera_ready_dir)

    if not entries:
        typer.echo("No matching PNGs found.")
        raise typer.Exit(0)

    # Make TeX portable: copy images under output_dir/camera_ready
    output_dir.mkdir(parents=True, exist_ok=True)
    _sync_camera_ready_images(entries, output_dir)

    csv_base_dir = dataset_config.parent if dataset_config.exists() else None
    color_info = load_color_info(dataset_config, csv_base_dir=csv_base_dir) if dataset_config.exists() else {}

    # Generate legends from config groups
    legends_dir = output_dir / "legends"
    legends_dir.mkdir(parents=True, exist_ok=True)

    all_config_groups = build_all_config_groups(color_info)
    typer.echo(f"\nGenerating {len(all_config_groups)} legend(s) from config …")
    legend_paths: dict[str, Path] = {}
    for group in all_config_groups:
        legend_path = legends_dir / f"legend_{group.key}.png"
        _render_group_legend(group, legend_path)
        if legend_path.exists():
            legend_paths[group.key] = legend_path

    newline_wrap_lookup: dict | None = None
    if results_json is not None and results_json.exists():
        typer.echo(f"Loading newline wrap lookup from {results_json} …")
        newline_wrap_lookup = load_newline_wrap_lookup(results_json)
        typer.echo(f"  {len(newline_wrap_lookup)} expert→wrap entries loaded.")
    elif results_json is not None:
        typer.echo(f"  [warn] --results-json path not found: {results_json}", err=True)

    groups = infer_legend_groups(entries, color_info, newline_wrap_lookup)
    groups = _split_groups_by_experiment(groups)

    # Render per-wrap newline legends (depend on wraps present)
    for group in groups:
        if group.task == "pile-uncopyrighted":
            legend_path = legends_dir / f"legend_{group.key}.png"
            _render_group_legend(group, legend_path)
            if legend_path.exists():
                legend_paths[group.key] = legend_path

    # Partition groups by experiment_id
    by_exp: dict[str, dict[str, list[LegendGroup]]] = defaultdict(lambda: {"probe": [], "newline": []})
    for group in groups:
        if not group.entries:
            continue
        exp_id = group.entries[0].experiment_id
        slot = "newline" if group.task == "pile-uncopyrighted" else "probe"
        by_exp[exp_id][slot].append(group)

    # Sort probe tasks into desired order
    task_rank = {t: i for i, t in enumerate(_TASK_ORDER)}

    def sort_probe(gs: list[LegendGroup]) -> list[LegendGroup]:
        return sorted(gs, key=lambda g: (task_rank.get(g.task, 999), g.key))

    def sort_newline(gs: list[LegendGroup]) -> list[LegendGroup]:
        def _wrap_key(g: LegendGroup) -> int:
            try:
                return int(g.key.split("__")[-1])
            except Exception:
                return 10**9
        return sorted(gs, key=_wrap_key)

    for exp_id in sorted(by_exp):
        probe_groups = sort_probe(by_exp[exp_id]["probe"])
        newline_groups = sort_newline(by_exp[exp_id]["newline"])

        if probe_groups:
            tex_path = output_dir / f"probe_{exp_id}.tex"
            typer.echo(f"\nProbe ({exp_id}): {len(probe_groups)} group(s)")
            generate_figure_tex(exp_id, probe_groups, legend_paths, tex_path, cols, is_newline=False)

        if newline_groups:
            tex_path = output_dir / f"newline_{exp_id}.tex"
            typer.echo(f"\nNewline ({exp_id}): {len(newline_groups)} wrap(s)")
            generate_figure_tex(exp_id, newline_groups, legend_paths, tex_path, cols, is_newline=True)

    typer.echo(f"\nDone. Output written to {output_dir}")


if __name__ == "__main__":
    app()
