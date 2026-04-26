#!/usr/bin/env python3
"""Build static browser assets for Netlify/Vercel deployment.

Generates:
  browser/tasks.json                   — task index
  browser/results/**/experts/E*.bin     — binary tensor blobs
  browser/colorscales/*.json           — discrete + continuous color samples

Run as a Netlify build command, or locally before deploying::

    uv run python scripts/build_static.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
BROWSER_DIR = ROOT / "browser"
COLORS_DIR = BROWSER_DIR / "colorscales"
RESULTS_BROWSER = BROWSER_DIR / "results"


def scan_tasks() -> list[dict]:
    """Yield task metadata dicts from every index.json in results/."""
    tasks = []
    for idx_path in RESULTS_DIR.rglob("index.json"):
        if "/model/" in str(idx_path):
            continue
        rel = str(idx_path.parent.relative_to(RESULTS_DIR))
        with idx_path.open() as fh:
            data = json.load(fh)
        tasks.append({
            "rel": rel,
            "dataset_name": data.get("dataset_name", rel.split("/")[-1]),
            "experiment_id": data.get("experiment_id", rel.split("/")[0]),
            "title": data.get("title", rel),
        })
    return tasks


def write_tasks_json(tasks: list[dict]) -> None:
    """Write browser/tasks.json."""
    COLORS_DIR.parent.mkdir(parents=True, exist_ok=True)
    out = BROWSER_DIR / "tasks.json"
    with out.open("w") as fh:
        json.dump({"tasks": tasks}, fh, indent=2)
    print(f"Wrote {out}")


def _collect_colorscale_needs() -> set[tuple]:
    """Collect (scale_name, n, skip_endpoints) triples from all index.json files."""
    needs: set[tuple] = set()
    for idx_path in RESULTS_DIR.rglob("index.json"):
        if "/model/" in str(idx_path):
            continue
        try:
            with idx_path.open() as fh:
                data = json.load(fh)
        except Exception:
            continue
        color = data.get("color", {})
        name = color.get("scale")
        if not name:
            continue
        n = len(color.get("color_map") or [])
        if n == 0:
            label_names = data.get("label_names")
            n = len(label_names) if label_names else 0
            if n == 0:
                n = 7
        skip = color.get("skip_endpoints", True)
        needs.add((name, n, skip))
        needs.add((name, 64, False))
        needs.add((name, 64, True))
    return needs


def _sample_scale(name: str, n: int, skip: bool) -> list[str]:
    """Sample n colors from a named Plotly scale, return rgb(...) strings."""
    try:
        from analysis.colors import sample_named_scale_discrete
        return sample_named_scale_discrete(name, n, skip_endpoints=skip)
    except ImportError:
        sys.path.insert(0, str(ROOT / "src"))
        from analysis.colors import sample_named_scale_discrete
        return sample_named_scale_discrete(name, n, skip_endpoints=skip)


def write_colorscales() -> None:
    """Write all required colorscale JSON files."""
    needs = _collect_colorscale_needs()
    COLORS_DIR.mkdir(parents=True, exist_ok=True)
    for name, n, skip in needs:
        skip_str = "True" if skip else "False"
        out = COLORS_DIR / f"{name}_{n}_{skip_str}.json"
        if out.exists():
            print(f"  skip {out.name} (exists)")
            continue
        colors = _sample_scale(name, n, skip)
        with out.open("w") as fh:
            json.dump({"colors": colors}, fh)
        print(f"  wrote {out.relative_to(BROWSER_DIR)}")


def write_expert_bins() -> None:
    """Convert .pth files to flat binary blobs under browser/results/."""
    try:
        sys.path.insert(0, str(ROOT / "src"))
    except Exception:
        pass

    count = 0
    for pth_path in RESULTS_DIR.rglob("experts/*.pth"):
        rel_task = str(pth_path.parent.relative_to(RESULTS_DIR))
        str(pth_path.parent.parent.relative_to(RESULTS_DIR))

        out = RESULTS_BROWSER / rel_task / f"{pth_path.stem}.bin"
        if out.exists() and out.stat().st_mtime > pth_path.stat().st_mtime:
            continue

        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = torch.load(pth_path, map_location="cpu")
        except Exception as e:
            print(f"  ERROR loading {pth_path}: {e}", file=sys.stderr)
            continue

        parts: list[bytes] = []
        pts = data.get("points")
        if pts is not None:
            parts.append(pts.numpy().astype("float32").tobytes())

        lbl = data.get("labels")
        if lbl is not None:
            parts.append(lbl.numpy().astype("int32").tobytes())

        cont = data.get("continuity")
        if cont is not None:
            parts.append(cont.numpy().astype("float32").tobytes())

        with out.open("wb") as fh:
            fh.write(b"".join(parts))

        count += 1
        print(f"  wrote {out.relative_to(BROWSER_DIR)}")

    print(f"Wrote {count} binary expert files")


def main() -> None:
    """Build all static browser assets under browser/."""
    print("Building static browser assets...")
    print("1. Scanning tasks...")
    tasks = scan_tasks()
    write_tasks_json(tasks)
    print(f"   Found {len(tasks)} tasks")

    print("2. Writing colorscale files...")
    write_colorscales()

    print("3. Converting .pth → .bin...")
    write_expert_bins()

    print("Done.")


if __name__ == "__main__":
    main()

