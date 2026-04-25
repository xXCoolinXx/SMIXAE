"""On-disk format for decoupled probing-task outputs.

Both ``categorize_all.py`` and ``anthropic_newline.py`` build a list of
:class:`ExpertRecord` plus a :class:`TaskIndex` and call
:func:`write_probing_task`. The save server reads the same files back via
:func:`read_task_index`, :func:`read_expert_meta`, and :func:`read_expert_tensors`
and renders Plotly scatters client-side.

Layout written under ``task_dir``::

    {task_dir}/
        index.json                     # task-level metadata
        experts/
            E{expert_id}.json          # per-expert metrics + hover text + tensor map
            E{expert_id}.pth           # per-expert tensors via torch.save

The format is intentionally minimal — tensors live in ``.pth`` and everything else
is JSON so the save server can mmap, cache, and return tensor bytes raw without
re-encoding.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Iterator, Literal

import torch

SCHEMA_VERSION = 1

TaskType = Literal["labeled_probe", "unlabeled_probe", "newline"]


# ── Schemas ───────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ColorSpec:
    """How the viewer should colour points for this task.

    Attributes:
        mode: ``"discrete"`` (per-class swatches) or ``"continuous"`` (colorbar).
        scale: Plotly colorscale name (e.g. ``"Plasma"``, ``"Viridis"``).
        color_map: Optional explicit ``{class_label: 'rgb(...)'}`` map. When set,
            overrides ``scale`` for discrete-mode views.
        continuous_label: Axis label for the colorbar in continuous mode.
        skip_endpoints: Whether to use the endpoint-skip sampling rule for
            circular colorscales (matches ``colors.sample_named_scale_discrete``).
    """

    mode: Literal["discrete", "continuous"]
    scale: str
    color_map: dict[str, str] | None = None
    continuous_label: str | None = None
    skip_endpoints: bool = True


@dataclasses.dataclass
class HypothesisSpec:
    """One regression hypothesis attached to a labeled probing task.

    Attributes:
        name: Hypothesis key used in ``experts_by_view`` and PNG filenames.
        description: Human-readable description shown in the viewer.
        regression_type: ``"linear"`` / ``"ridge"`` / ``"logistic"`` / ``"multinomial"``.
        score_type: Short label used in PNG filenames (``"r2"`` / ``"acc"``).
        color_override: Optional per-hypothesis ``{class_label: 'rgb(...)'}`` map
            used in place of the task-level :class:`ColorSpec` when this hypothesis
            view is active.
    """

    name: str
    description: str
    regression_type: str
    score_type: str
    color_override: dict[str, str] | None = None


@dataclasses.dataclass
class ExpertRanking:
    """One row in ``experts_by_view``: an expert's position under some ranking.

    Attributes:
        rank: 1-based position in this view.
        expert_id: Global expert index.
        score: Numeric score used for the ranking (or ``None`` if unranked).
        score_std: Optional std (e.g. CV std for regression scores).
    """

    rank: int
    expert_id: int
    score: float | None
    score_std: float | None = None


@dataclasses.dataclass
class ExpertRecord:
    """Full per-expert payload to write: metadata + tensors.

    Attributes:
        expert_id: Global expert index.
        metrics: Free-form dict of computed metrics (Fisher, continuity, regression
            scores, periodic gain, etc.). All values must be JSON-serialisable.
        hover_text: Per-point HTML hover strings, length ``n_points``. May be empty.
        points: ``(n_points, d_bottleneck)`` float32 tensor of bottleneck activations.
        labels: ``(n_points,)`` int64 tensor of class ids, or ``None``.
        continuity: ``(n_points,)`` float32 per-point continuity scores, or ``None``.
    """

    expert_id: int
    metrics: dict[str, Any]
    hover_text: list[str]
    points: torch.Tensor
    labels: torch.Tensor | None = None
    continuity: torch.Tensor | None = None


@dataclasses.dataclass
class TaskIndex:
    """Task-level metadata written as ``index.json``.

    Attributes:
        task_type: Discriminates the three task families.
        experiment_id: Run identifier (e.g. ``"gemma_2_9b_l11"``); used in PNG filenames.
        dataset_name: Dataset slug (e.g. ``"weekdays"``, ``"newline_80"``).
        title: Human-readable title shown in the viewer.
        model_name: HF model name.
        hook_name: Hook point string.
        d_bottleneck: SMIXAE bottleneck dim.
        n_experts_total: Total experts in the SAE (not just those plotted).
        color: Default :class:`ColorSpec` for the task.
        label_names: Mapping ``{int_id: display_name}`` or ``None``.
        hypotheses: Regression hypotheses, empty for unlabeled / newline tasks.
        experts_by_view: ``{view_name: [ExpertRanking, ...]}``. View names are
            things like ``"fisher"``, ``"continuity"``, ``"periodic_gain"``, or
            a hypothesis name. The viewer builds one tab strip per view.
        scatter_size: Marker size hint (1 for labeled, 5 for unlabeled).
    """

    task_type: TaskType
    experiment_id: str
    dataset_name: str
    title: str
    model_name: str
    hook_name: str
    d_bottleneck: int
    n_experts_total: int
    color: ColorSpec
    label_names: dict[int, str] | None
    hypotheses: list[HypothesisSpec]
    experts_by_view: dict[str, list[ExpertRanking]]
    scatter_size: float = 1.0


# ── Writer ────────────────────────────────────────────────────────────────────


def _clean_float(v: Any) -> Any:
    """Replace NaN with None so json.dumps doesn't choke."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            out[k] = {kk: _clean_float(vv) for kk, vv in v.items()}
        else:
            out[k] = _clean_float(v)
    return out


def _index_to_dict(idx: TaskIndex) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_type": idx.task_type,
        "experiment_id": idx.experiment_id,
        "dataset_name": idx.dataset_name,
        "title": idx.title,
        "model_name": idx.model_name,
        "hook_name": idx.hook_name,
        "d_bottleneck": idx.d_bottleneck,
        "n_experts_total": idx.n_experts_total,
        "scatter_size": idx.scatter_size,
        "color": dataclasses.asdict(idx.color),
        "label_names": (
            {str(k): v for k, v in idx.label_names.items()} if idx.label_names else None
        ),
        "hypotheses": [dataclasses.asdict(h) for h in idx.hypotheses],
        "experts_by_view": {
            view: [dataclasses.asdict(r) for r in rankings]
            for view, rankings in idx.experts_by_view.items()
        },
    }


def _expert_meta_dict(rec: ExpertRecord, tensor_keys: list[str]) -> dict[str, Any]:
    n_points = int(rec.points.shape[0])
    return {
        "schema_version": SCHEMA_VERSION,
        "expert_id": rec.expert_id,
        "n_points": n_points,
        "metrics": _clean_metrics(rec.metrics),
        "hover_text": rec.hover_text,
        "tensor_path": f"experts/E{rec.expert_id}.pth",
        "tensor_keys": tensor_keys,
    }


def write_probing_task(
    task_dir: str | Path,
    *,
    index: TaskIndex,
    experts: list[ExpertRecord],
) -> None:
    """Write a probing task to disk in the canonical format.

    Atomic at the per-file level (each JSON / pth is written then renamed).
    Removes any stale ``E*.json`` / ``E*.pth`` from prior runs in the same
    directory whose expert ids are not in ``experts``, plus any leftover
    ``experts.html`` / ``top_experts.html``.

    Args:
        task_dir: Output directory. Created if it doesn't exist.
        index: Task-level metadata.
        experts: List of per-expert records to write.
    """
    task_dir = Path(task_dir)
    experts_dir = task_dir / "experts"
    experts_dir.mkdir(parents=True, exist_ok=True)

    # ── per-expert files
    fresh_ids: set[int] = set()
    for rec in experts:
        fresh_ids.add(rec.expert_id)
        tensors: dict[str, torch.Tensor] = {"points": rec.points.detach().cpu().contiguous().float()}
        if rec.labels is not None:
            tensors["labels"] = rec.labels.detach().cpu().contiguous().long()
        if rec.continuity is not None:
            tensors["continuity"] = rec.continuity.detach().cpu().contiguous().float()

        pth_path = experts_dir / f"E{rec.expert_id}.pth"
        torch.save(tensors, pth_path)

        meta = _expert_meta_dict(rec, list(tensors.keys()))
        meta_path = experts_dir / f"E{rec.expert_id}.json"
        meta_path.write_text(json.dumps(meta, indent=2))

    # ── stale expert cleanup
    for old in experts_dir.glob("E*.pth"):
        try:
            old_id = int(old.stem[1:])
        except ValueError:
            continue
        if old_id not in fresh_ids:
            old.unlink()
            json_sibling = old.with_suffix(".json")
            if json_sibling.exists():
                json_sibling.unlink()

    # ── index
    index_path = task_dir / "index.json"
    index_path.write_text(json.dumps(_index_to_dict(index), indent=2))

    # ── stale html cleanup (decoupling: viewer is now save-server-rendered)
    for stale_html in ("experts.html", "top_experts.html"):
        p = task_dir / stale_html
        if p.exists():
            p.unlink()


# ── Reader ────────────────────────────────────────────────────────────────────


def read_task_index(task_dir: str | Path) -> dict[str, Any]:
    """Read and return ``index.json`` for a task directory."""
    return json.loads((Path(task_dir) / "index.json").read_text())


def read_expert_meta(task_dir: str | Path, expert_id: int) -> dict[str, Any]:
    """Read and return the per-expert JSON metadata."""
    return json.loads(
        (Path(task_dir) / "experts" / f"E{expert_id}.json").read_text()
    )


def read_expert_tensors(task_dir: str | Path, expert_id: int) -> dict[str, torch.Tensor]:
    """Read and return the per-expert tensor dict."""
    return torch.load(
        Path(task_dir) / "experts" / f"E{expert_id}.pth",
        map_location="cpu",
        weights_only=True,
    )


def is_task_dir(path: Path) -> bool:
    """True if ``path`` is a directory containing an ``index.json`` we wrote."""
    if not path.is_dir():
        return False
    idx = path / "index.json"
    if not idx.exists():
        return False
    try:
        head = json.loads(idx.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(head, dict) and "task_type" in head and "experts_by_view" in head


def iter_task_dirs(root: str | Path) -> Iterator[Path]:
    """Yield every probing-task directory under ``root`` (anywhere in the tree)."""
    root = Path(root)
    if not root.exists():
        return
    for idx in root.rglob("index.json"):
        parent = idx.parent
        if is_task_dir(parent):
            yield parent
