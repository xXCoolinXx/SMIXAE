r"""Camera-ready LaTeX figure assembly + legend generation (PIL) with correct layout.

Fixes vs previous versions:
- Labels are sorted BEFORE stripping numeric prefixes (e.g. 01_Sunday, 02_Monday ...),
  then displayed without the prefix.
- Legends are rendered with PIL (no Plotly/Kaleido image export), giving:
  - large, legible tick labels on continuous colorbars
  - fewer ticks (configurable)
  - discrete legends as a readable swatch+text list
- Discrete legends are vertically centered in the plot-height box via fixed-height
  minipages in LaTeX (plots are also placed in fixed-height boxes, centered).

Layout:
- Physical rows contain up to --cols PLOTS total (legends do NOT count toward cols).
- Each TASK is a block: its plots, then ONE legend immediately after its last plot.
- Tasks with >cols plots get their own wrapped block (internal rows of exactly cols plots),
  with ONE legend only on the final internal row.

LaTeX requirements:
  \\usepackage{graphicx,tikz}

Dependencies:
  pip install pillow
  pip install plotly        (used only to sample named Plotly colorscales; no kaleido)

Run:
  smixae latex figures --camera-ready-dir results/camera_ready --output-dir results/ ...

Output (everything the LaTeX document needs lives under <output-dir>/paper/):
  <output-dir>/paper/
    camera_ready/       (copied PNGs)
    legends/            (PIL legends)
    probe_<exp>.tex
    newline_<exp>.tex
    random_<exp>.tex

Usage in a LaTeX project: copy the ``paper/`` folder next to your main .tex, then
``\\input{paper/probe_<exp>.tex}`` — the ``\\includegraphics{paper/...}`` paths
inside resolve relative to the main document.
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

from analysis.colors import (
    render_continuous_colorbar_png,
    render_discrete_from_scale_png,
    render_discrete_legend_png,
)

app = typer.Typer()

# ------------------------- Tunable layout constants ----------------------------

_USABLE_FRAC = 0.98                 # fraction of \linewidth used for content
_LEGEND_SCALE = 0.35                # legend slot width relative to one plot slot
_BLOCK_GAP = r"\hspace{2.5mm}"      # gap between TASK blocks in the same physical row
_ROW_VSPACE = r"\vspace{5pt}"       # gap between physical rows
_PANEL_HEIGHT = "4.0cm"             # fixed-height boxes for plots+legends

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
    "acc":      "Accuracy",
    "score":    "Score",
    "per_gain": r"$\Delta R^2_{\mathrm{per}}$",
    "cont":     "Cont.",
}


@dataclass
class PNGEntry:
    """Parsed metadata for a single camera-ready PNG file (filename follows the convention in docs/LATEX.md)."""

    path: Path
    experiment_id: str
    task: str
    expert_id: int
    hyp_name: str
    score_type: str
    score: float
    figure_type: str  # "scatter" or "means"


def scan_camera_ready(camera_ready_dir: Path) -> list[PNGEntry]:
    """Return all conformant PNG entries in ``camera_ready_dir``, skipping files that don't match the naming convention."""
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


# ----------------------------- Sorting helpers --------------------------------

_PREFIX_RE = re.compile(r"^(?P<num>\d+)_")

def _strip_numeric_prefix(s: str) -> str:
    return _PREFIX_RE.sub("", s)

def _labels_all_numeric(values: list) -> bool:
    for v in values:
        if isinstance(v, (int, float)):
            continue
        try:
            float(str(v))
        except Exception:
            return False
    return True

def _labels_all_prefixed(values: list) -> bool:
    ss = [str(v) for v in values]
    return bool(ss) and all(_PREFIX_RE.match(s) for s in ss)

def _sorted_labels(values: list) -> list:
    r"""Sort labels BEFORE stripping numeric prefixes.

    - If numeric -> numeric ascending
    - Else if all match ^\d+_ -> sort by that integer prefix, then by full string
    - Else -> lexicographic by string.
    """
    if not values:
        return []
    if _labels_all_numeric(values):
        # preserve ints if they look integral
        nums = []
        for v in values:
            f = float(v)
            nums.append(int(f) if f.is_integer() else f)
        return sorted(nums)
    if _labels_all_prefixed(values):
        def key(x):
            s = str(x)
            m = _PREFIX_RE.match(s)
            return (int(m.group("num")) if m else 10**9, s)
        return sorted(values, key=key)
    return sorted(values, key=lambda x: str(x))

def _display_names_from_sorted(labels: list) -> dict:
    """Assumes *labels are already sorted*; maps label -> display text with prefix removed."""
    return {lbl: _strip_numeric_prefix(str(lbl)) for lbl in labels}


# ----------------------------- Color-info loading ------------------------------

def _task_from_dataframe_path(path_str: str) -> str:
    return Path(path_str).stem


def load_color_info(
    dataset_config_path: Path,
    csv_base_dir: Path | None = None,
) -> dict[str, dict]:
    """Return per-task color info from ``dataset_config_path``.

    The shape is ``{task: {color_map, color_scale, hypothesis_color_overrides,
    continuous_color, labels}}``. Labels (if found) are sorted with
    ``_sorted_labels()``.
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
                labels = _sorted_labels(list(raw_labels))

        info[task] = {
            "color_map":                  entry.get("color_map"),
            "color_scale":                entry.get("color_scale"),
            "hypothesis_color_overrides": entry.get("hypothesis_color_overrides", {}),
            "continuous_color":           entry.get("continuous_color", False),
            "labels":                     labels,
        }
    return info


def load_newline_color_info(newline_config_path: Path) -> dict[str, dict]:
    """Load color info from newline_config.json.

    Unlike the probing config, entries use ``"task"`` directly instead of
    deriving it from ``"dataframe_path"``.
    """
    with open(newline_config_path) as f:
        raw = json.load(f)

    info: dict[str, dict] = {}
    for entry in raw:
        task = entry.get("task", "")
        if not task:
            continue
        info[task] = {
            "color_map":                  entry.get("color_map"),
            "color_scale":                entry.get("color_scale"),
            "hypothesis_color_overrides": entry.get("hypothesis_color_overrides", {}),
            "continuous_color":           entry.get("continuous_color", False),
            "labels":                     None,
        }
    return info


# ------------------------------ Newline wrap lookup ----------------------------

def load_newline_wrap_lookup(results_json_path: Path) -> dict[tuple, int]:
    """Build a ``(experiment_id, expert_id, periodic_gain) → line_length`` lookup from ``results.json``."""
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
    """Bucket of PNG entries that share a single rendered legend (color map or colorbar)."""

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
    """Partition PNG entries into ``LegendGroup`` buckets that each map to a single legend image."""
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
            for (_exp_id_nl, n), wrap_entries in sorted(by_exp_wrap.items()):
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

        # random-sample unlabeled pass: one group per experiment_id (no wrap, no labels)
        if task == "continuity":
            by_exp_r: dict[str, list[PNGEntry]] = defaultdict(list)
            for e in task_entries:
                by_exp_r[e.experiment_id].append(e)
            for _exp_id_r, r_entries in sorted(by_exp_r.items()):
                groups.append(LegendGroup(
                    key="continuity",
                    task="continuity",
                    hyp_filter=None,
                    entries=r_entries,
                    color_scale="Viridis",
                    continuous_color=True,
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
    """Create a legend group per entry in ``color_info``, used for the config-only legend gallery."""
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

def _render_group_legend(group: LegendGroup, output_path: Path) -> None:
    """Dispatch a ``LegendGroup`` to the appropriate PIL renderer in ``analysis.colors``.

    - ``color_map``      → discrete swatch legend using the provided mapping.
    - ``color_scale`` + ``continuous_color`` → continuous colorbar PNG.
    - ``color_scale`` alone → discrete swatch legend with colors sampled from
      the named scale (endpoint-skip applied in
      :func:`analysis.colors.sample_named_scale_discrete`).
    - ``labels`` alone   → fallback discrete legend using a qualitative palette.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if group.color_map:
        if group.labels:
            labels = _sorted_labels(group.labels)
            label_to_color = {}
            for lbl in labels:
                stripped = _strip_numeric_prefix(str(lbl))
                label_to_color[lbl] = group.color_map.get(stripped, group.color_map.get(lbl))
        else:
            labels = _sorted_labels(list(group.color_map.keys()))
            label_to_color = dict(group.color_map)
        render_discrete_legend_png(
            labels=labels,
            label_to_color=label_to_color,
            output_path=output_path,
            display_names=_display_names_from_sorted(labels),
        )
        return

    if group.color_scale:
        if not group.labels:
            typer.echo(f"  [skip legend] no labels for group {group.key}", err=True)
            return
        labels = _sorted_labels(group.labels)

        if group.continuous_color:
            render_continuous_colorbar_png(
                colorscale=group.color_scale,
                labels=labels,
                output_path=output_path,
            )
        else:
            render_discrete_from_scale_png(
                colorscale=group.color_scale,
                labels=labels,
                output_path=output_path,
                display_names=_display_names_from_sorted(labels),
            )
        return

    if group.labels:
        labels = _sorted_labels(group.labels)
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                   "#bcbd22", "#17becf"]
        label_to_color = {lbl: palette[i % len(palette)] for i, lbl in enumerate(labels)}
        render_discrete_legend_png(
            labels=labels,
            label_to_color=label_to_color,
            output_path=output_path,
            display_names=_display_names_from_sorted(labels),
        )
        return

    typer.echo(f"  [skip legend] no color info for group {group.key}", err=True)


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
    "continuity":        "Random Experts (Unlabeled)",
}

_HYP_DISPLAY: dict[str, str] = {
    "random":          "Random Sample",
    "continuity":      "Continuity",
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
    return f"Expert {entry.expert_id}, {_esc_text(hyp_disp)} ({score_lbl}\\,=\\,{entry.score:.3f})."


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

    if newline_wrap is not None:
        suffix = " Points represent individual token activations in the bottleneck space, colored by distance since the last newline."
    else:
        suffix = " Larger points denote class mean activations in the bottleneck space; smaller points are individual token activations."
    return f"{prefix} {'  '.join(parts)}{suffix}"


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
    plot_count: int  # plots only; legends not counted


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

        # If task has >cols plots, it gets its own dedicated row (wrapped internally).
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


def _include_graphics(relpath: str) -> str:
    return rf"\includegraphics[width=\linewidth,height={_PANEL_HEIGHT},keepaspectratio]{{{relpath}}}"


def _render_compact_row(row: _Row, cols: int) -> list[str]:
    blocks = row.blocks
    if not blocks:
        return []

    # plots and legends present
    G = sum(len(b.units) for b in blocks)
    L = sum(1 for b in blocks if b.legend_path is not None and b.legend_path.exists())
    if G <= 0:
        return []

    # Cap plot width so single-plot rows do not blow up.
    plot_w_cap = _USABLE_FRAC / (cols + _LEGEND_SCALE)

    denom = (G + _LEGEND_SCALE * L)
    plot_w = min(_USABLE_FRAC / denom, plot_w_cap) if denom > 0 else plot_w_cap
    legend_w = _LEGEND_SCALE * plot_w

    lines: list[str] = []
    lines.append(r"\noindent\makebox[\linewidth][c]{%")

    for bi, block in enumerate(blocks):
        show_leg = block.legend_path is not None and block.legend_path.exists()
        k = len(block.units)

        block_w = k * plot_w + (legend_w if show_leg else 0.0)
        if block_w <= 0:
            continue

        lines.append(rf"\begin{{minipage}}[t]{{{block_w:.4f}\linewidth}}%")
        lines.append(r"\vspace{0pt}\centering%")

        plot_frac = plot_w / block_w

        # plots in fixed-height centered boxes with overlaid panel label
        for u in block.units:
            png_rel = str(Path("paper/camera_ready") / u.entry.path.name)
            lines.append(rf"\begin{{minipage}}[c][{_PANEL_HEIGHT}][c]{{{plot_frac:.4f}\linewidth}}%")
            lines.append(r"\centering%")
            lines.append(r"\begin{tikzpicture}[baseline=(img.base)]")
            lines.append(r"  \node[inner sep=0pt] (img) {" + _include_graphics(png_rel) + "};")
            lines.append(rf"  \node[anchor=north west, inner sep=2pt, overlay] at (img.north west) {{\small\textbf{{({u.letter.lower()})}}}};")
            lines.append(r"\end{tikzpicture}%")
            lines.append(r"\end{minipage}%")

        # legend in fixed-height centered box (this centers discrete legends vertically)
        if show_leg:
            leg_frac = legend_w / block_w
            legend_rel = str(Path("paper/legends") / block.legend_path.name)
            lines.append(rf"\begin{{minipage}}[c][{_PANEL_HEIGHT}][c]{{{leg_frac:.4f}\linewidth}}%")
            lines.append(r"\centering%")
            lines.append(_include_graphics(legend_rel) + "%")
            lines.append(r"\end{minipage}%")

        # task label
        title = _task_title(block.group)
        lines.append(r"\par\vspace{1pt}%")
        lines.append(rf"{{\footnotesize\textit{{{_esc_text(title)}}}}}%")
        lines.append(r"\end{minipage}%")

        if bi < len(blocks) - 1:
            lines.append(_BLOCK_GAP + "%")

    lines.append(r"}")
    return lines


def _render_multiline_task_block(block: _Block, cols: int) -> list[str]:
    """Render a task block that wraps across multiple internal rows.

    - internal rows of exactly cols plots (pad empties)
    - reserve a legend slot on every internal row so plot sizes don't change
    - draw legend only after final plot (in final internal row).
    """
    show_leg = block.legend_path is not None and block.legend_path.exists()

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

        # plots: exactly cols slots
        for i in range(cols):
            lines.append(rf"\begin{{minipage}}[c][{_PANEL_HEIGHT}][c]{{{plot_w:.4f}\linewidth}}%")
            lines.append(r"\centering%")
            if i < len(chunk):
                png_rel = str(Path("paper/camera_ready") / chunk[i].entry.path.name)
                lines.append(r"\begin{tikzpicture}[baseline=(img.base)]")
                lines.append(r"  \node[inner sep=0pt] (img) {" + _include_graphics(png_rel) + "};")
                lines.append(rf"  \node[anchor=north west, inner sep=2pt, overlay] at (img.north west) {{\small\textbf{{({chunk[i].letter.lower()})}}}};")
                lines.append(r"\end{tikzpicture}%")
            else:
                lines.append(r"\vspace{0pt}%")
            lines.append(r"\end{minipage}%")

        # legend slot (centered vertically); only filled on last internal row
        if show_leg:
            lines.append(rf"\begin{{minipage}}[c][{_PANEL_HEIGHT}][c]{{{legend_w:.4f}\linewidth}}%")
            lines.append(r"\centering%")
            if is_last:
                legend_rel = str(Path("paper/legends") / block.legend_path.name)
                lines.append(_include_graphics(legend_rel) + "%")
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
    """Write a LaTeX ``figure*`` block for one experiment, laying out plots and legends into rows."""
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
        "% Requires: \\usepackage{graphicx,tikz}",
        "% Image paths are relative to this .tex file's location.",
        f"% Generated by: smixae latex figures  (experiment: {experiment_id})",
        "",
        r"\begin{figure*}[t]",
        r"\centering",
    ]

    for ri, row in enumerate(rows):
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

def _sync_camera_ready_images(entries: list[PNGEntry], dst_dir: Path) -> None:
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
    """CLI entry: assemble camera-ready PNGs into per-experiment LaTeX figure files."""
    if not camera_ready_dir.exists():
        typer.echo(f"Error: camera-ready-dir does not exist: {camera_ready_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Scanning {camera_ready_dir} …")
    entries = scan_camera_ready(camera_ready_dir)
    if not entries:
        typer.echo("No matching PNGs found.")
        raise typer.Exit(0)

    paper_dir = output_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    _sync_camera_ready_images(entries, paper_dir / "camera_ready")

    csv_base_dir = dataset_config.parent if dataset_config.exists() else None
    color_info = load_color_info(dataset_config, csv_base_dir=csv_base_dir) if dataset_config.exists() else {}

    newline_config = dataset_config.parent / "newline_config.json"
    if newline_config.exists():
        typer.echo(f"Loading newline color info from {newline_config} …")
        color_info.update(load_newline_color_info(newline_config))

    legends_dir = paper_dir / "legends"
    legends_dir.mkdir(parents=True, exist_ok=True)

    # Build legend set from config (except pile-uncopyrighted which depends on wrap)
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

    # Render per-wrap newline legends
    for group in groups:
        if group.task == "pile-uncopyrighted":
            legend_path = legends_dir / f"legend_{group.key}.png"
            _render_group_legend(group, legend_path)
            if legend_path.exists():
                legend_paths[group.key] = legend_path

    # Partition by experiment_id
    by_exp: dict[str, dict[str, list[LegendGroup]]] = defaultdict(lambda: {"probe": [], "newline": [], "random": []})
    for group in groups:
        if not group.entries:
            continue
        exp_id = group.entries[0].experiment_id
        if group.task == "pile-uncopyrighted":
            slot = "newline"
        elif group.task == "continuity":
            slot = "random"
        else:
            slot = "probe"
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
        random_groups = by_exp[exp_id]["random"]

        if probe_groups:
            tex_path = paper_dir / f"probe_{exp_id}.tex"
            typer.echo(f"\nProbe ({exp_id}): {len(probe_groups)} group(s)")
            generate_figure_tex(exp_id, probe_groups, legend_paths, tex_path, cols, is_newline=False)

        if newline_groups:
            tex_path = paper_dir / f"newline_{exp_id}.tex"
            typer.echo(f"\nNewline ({exp_id}): {len(newline_groups)} wrap(s)")
            generate_figure_tex(exp_id, newline_groups, legend_paths, tex_path, cols, is_newline=True)

        if random_groups:
            tex_path = paper_dir / f"random_{exp_id}.tex"
            typer.echo(f"\nRandom sample ({exp_id}): {len(random_groups)} group(s)")
            generate_figure_tex(exp_id, random_groups, legend_paths, tex_path, cols, is_newline=False)

    typer.echo(f"\nDone. Output written to {output_dir}")


if __name__ == "__main__":
    app()
