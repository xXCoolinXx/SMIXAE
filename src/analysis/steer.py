"""Steering experiments using SMIXAE expert coordinate substitution.

Loads a trained SMIXAE, identifies the top expert(s) for the hours-of-day
probing task by Fisher discriminant score, and runs two causal interventions
using pre-generated prompt datasets from datasets/steering/:

  Task 1 — Current time (hours_current_time.csv):
      "You glance at the clock and find it is X:00YM. Your friend Z asks
       you for the time, and you respond, saying it is"
      Steer Source_Hour representation → Target_Hour class mean.

  Task 2 — Elapsed time (hours_elapsed_time.csv):
      "You check your phone, and see that the current time is X:00YM.
       You realize that, since Z:00WM, the amount of hours that has passed is"
      Steer Current_Hour representation → (Current_Hour + target_delta_hours).

Steering mechanism (coordinate substitution):
    z        = sae.encode(x_last)                  # bottleneck activations at last token
    contrib  = decode_single_expert(z, expert_id)  # expert's contribution to residual stream
    x_steered = x_last - contrib + decode_single_expert(z_target, expert_id)
    where z_target has only expert_id set to the target class mean.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from typing import Generator, Optional

import pandas as pd
import torch
import typer
from tqdm import tqdm

from analysis.utils import (
    ActivationBatch,
    DatasetConfig,
    ExpertFilterConfig,
    _strip_prefix,
    collect_activations,
    get_sae_activations,
    load_llm,
    load_sae,
)

app = typer.Typer()


# ── Steering primitives ───────────────────────────────────────────────────────


def compute_class_means(expert) -> dict[int, torch.Tensor]:
    """Return per-class mean bottleneck vectors for an Expert.

    Returns:
        dict mapping class_id → ``(d_bottleneck,)`` float32 tensor
    """
    means: dict[int, torch.Tensor] = {}
    for c in range(expert.n_classes):
        mask = expert.labels == c
        if mask.any():
            means[c] = expert.expert_activations[mask].mean(dim=0)
    return means


def decode_single_expert(
    sae,
    bottleneck: torch.Tensor,
    expert_id: int,
) -> torch.Tensor:
    """Decode only expert ``expert_id``'s contribution to the residual stream.

    Args:
        sae:        SMIXAE model (on device).
        bottleneck: ``(batch, n_experts, d_bottleneck)`` tensor.
        expert_id:  Index of the expert to decode.

    Returns:
        ``(batch, d_model)`` reconstruction from expert ``expert_id`` alone.
    """
    z_mask = torch.zeros_like(bottleneck)
    z_mask[:, expert_id, :] = bottleneck[:, expert_id, :]
    return sae.decode(z_mask)


@contextmanager
def steering_hook(
    model,
    hook_point: str,
    sae,
    expert_id: int,
    tgt_means: torch.Tensor,
    device: str,
) -> Generator[None, None, None]:
    """Context manager that installs a coordinate-substitution hook on ``hook_point``.

    ``tgt_means`` is ``(batch, d_bottleneck)`` — one target per prompt in the batch.

    At every forward pass the hook:
    1. Encodes the last-sequence-position activation through the SAE.
    2. Subtracts the expert's current contribution.
    3. Adds the decoded per-item target class mean for that expert.

    The hook is removed on context exit.
    """
    sae_device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype
    tgt = tgt_means.to(device=sae_device, dtype=sae_dtype)  # (batch, d_bottleneck)

    def _hook(_module, _input, output):
        x = output[0] if isinstance(output, tuple) else output  # (batch, seq, d_model)
        x_last = x[:, -1:, :].to(device=sae_device, dtype=sae_dtype)  # (batch, 1, d_model)

        with torch.no_grad():
            z = sae.encode(x_last.squeeze(1))  # (batch, n_experts, d_bottleneck)
            contrib = decode_single_expert(sae, z, expert_id)  # (batch, d_model)

            z_tgt = torch.zeros_like(z)
            z_tgt[:, expert_id, :] = tgt  # (batch, d_bottleneck) — per-item targets
            tgt_contrib = decode_single_expert(sae, z_tgt, expert_id)  # (batch, d_model)

        x_steered = x_last.squeeze(1) - contrib + tgt_contrib
        x[:, -1, :] = x_steered.to(dtype=x.dtype)
        return (x,) + output[1:] if isinstance(output, tuple) else x

    handle = model.get_submodule(hook_point).register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


# ── Hour label utilities ──────────────────────────────────────────────────────


def build_hour_map(label_names: dict[int, str]) -> dict[str, int]:
    """Build a mapping from stripped hour string (e.g. '6PM') to class id."""
    return {_strip_prefix(v): k for k, v in label_names.items()}


def hour_plus_delta(hour_str: str, delta: int, hour_map: dict[str, int]) -> str | None:
    """Return the hour string that is ``delta`` hours after ``hour_str``.

    Uses the sorted hour_map keys as a 24-element cycle.
    Returns ``None`` if the result is not in the map.
    """
    hours_ordered = list(hour_map.keys())  # already in class-id order
    if hour_str not in hour_map:
        return None
    idx = hours_ordered.index(hour_str)
    return hours_ordered[(idx + delta) % len(hours_ordered)]


# ── Generation helpers ────────────────────────────────────────────────────────


def generate_text_batch(
    model, tokenizer, prompts: list[str], max_new_tokens: int, device: str
) -> list[str]:
    """Batched generation. Tokenizer must have padding_side='left'."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(out[i, input_len:], skip_special_tokens=True) for i in range(len(prompts))]


# ── Regex scoring ─────────────────────────────────────────────────────────────

# Matches "3PM", "11AM", "3:00 PM", "11:00am", etc.
_TIME_RE = re.compile(r"\b(\d{1,2})(?::\d{2})?\s*([AaPp][Mm])\b")
# Matches the first integer in the output (for elapsed hours)
_NUM_RE = re.compile(r"\b(\d+)\b")


def _extract_time(text: str) -> str | None:
    """Return normalized hour string (e.g. '3PM') from model output, or None."""
    m = _TIME_RE.search(text)
    if m:
        return f"{int(m.group(1))}{m.group(2).upper()}"
    return None


def _extract_number(text: str) -> int | None:
    """Return first integer found in text, or None."""
    m = _NUM_RE.search(text)
    return int(m.group(1)) if m else None


def _score_record(record: dict) -> dict:
    """Return scoring fields for a single steering record."""
    if record["task"] == "current_time":
        # Baseline should say the source hour; steered output should say the target hour.
        src = record["src_hour"]
        tgt = record["tgt_hour"]
        bp = _extract_time(record["baseline_output"])
        sp = _extract_time(record["steered_output"])
        return dict(
            baseline_pred=bp,
            steered_pred=sp,
            baseline_correct=(bp == src) if bp is not None else False,
            steered_correct=(sp == tgt) if sp is not None else False,
        )
    else:
        # Baseline: should output expected_hours (unsteered elapsed time).
        # Steered: should output steered_expected_hours (elapsed time after shifting current hour).
        baseline_tgt = record.get("expected_hours")
        steered_tgt = record.get("steered_expected_hours")
        bp = _extract_number(record["baseline_output"])
        sp = _extract_number(record["steered_output"])
        return dict(
            baseline_pred=bp,
            steered_pred=sp,
            baseline_correct=(bp == baseline_tgt) if bp is not None else False,
            steered_correct=(sp == steered_tgt) if sp is not None else False,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def main(
    checkpoint_path: str = typer.Option(..., help="Path to SMIXAE model directory"),
    base_model_name: str = typer.Option(..., help="HuggingFace model name"),
    hook_point: str = typer.Option(..., help="Hook point used during SAE training"),
    hours_dataset: str = typer.Option(
        "datasets/probing/hours.csv",
        help="CSV with Sentence/Label columns for expert discovery",
    ),
    task1_dataset: str = typer.Option(
        "datasets/steering/hours_current_time.csv",
        help="Steering prompt CSV for Task 1 (current time)",
    ),
    task2_dataset: str = typer.Option(
        "datasets/steering/hours_elapsed_time.csv",
        help="Steering prompt CSV for Task 2 (elapsed time)",
    ),
    output_dir: str = typer.Option("steer_results", help="Output directory"),
    n_top_experts: int = typer.Option(10, help="Number of top Fisher experts to steer with"),
    n_probing_samples: int = typer.Option(1000, help="Number of samples for the expert-discovery probing pass"),
    target_delta_hours: int = typer.Option(1, help="Task 2: hours to add to Current_Hour to get the steering target"),
    generate_tokens: int = typer.Option(20, help="Max new tokens to generate per prompt"),
    device: str = typer.Option("cuda", help="Device"),
    llm_batch_size: int = typer.Option(16, help="Batch size for probing LLM forward pass"),
    sae_batch_size: int = typer.Option(2048, help="Batch size for SAE encoding"),
    gen_batch_size: int = typer.Option(32, help="Batch size for text generation"),
    active_threshold: float = typer.Option(1e-5, help="L2 norm threshold for expert activity"),
    min_active_fraction: float = typer.Option(0.05, help="Minimum fraction of samples an expert must fire on (0–1)"),
    expert_ids: Optional[str] = typer.Option(
        None,
        help="Comma-separated expert IDs to steer (e.g. '42,137,512'). "
             "Overrides automatic Fisher-based discovery. The probing pass still runs "
             "to compute class means for the specified experts.",
    ),
):
    """Run SMIXAE steering experiments on hours-of-day prompts."""
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load models ────────────────────────────────────────────────
    print(f"Loading {base_model_name}…")
    model, tokenizer = load_llm(base_model_name, device)
    tokenizer.padding_side = "right"
    sae = load_sae(checkpoint_path, device)

    # ── 2. Expert discovery via hours probing ─────────────────────────
    print(f"\nDiscovering experts from {hours_dataset}…")
    hours_cfg = DatasetConfig(
        dataframe_path=hours_dataset,
        text_column="Sentence",
        label_column="Label",
    )
    batch = collect_activations(
        model=model,
        tokenizer=tokenizer,
        hook_name=hook_point,
        max_length=128,
        n_input_samples=n_probing_samples,
        device=device,
        llm_batch_size=llm_batch_size,
        cfg=hours_cfg,
    )
    label_names = batch.label_names

    experts = get_sae_activations(
        sae=sae,
        device=device,
        batch=batch,
        sae_batch_size=sae_batch_size,
        filter_cfg=ExpertFilterConfig(
            active_threshold=active_threshold,
            min_active_fraction=min_active_fraction,
            max_points=0,  # no cap — keep all points for accurate means
        ),
    )
    del batch

    print(f"Scoring Fisher for {len(experts)} active experts…")
    for e in tqdm(experts):
        e.evaluate_fisher()

    if expert_ids is not None:
        id_list = [int(x.strip()) for x in expert_ids.split(",")]
        id_set = set(id_list)
        found_map = {e.expert_id: e for e in experts if e.expert_id in id_set}
        missing = id_set - set(found_map)
        if missing:
            print(f"Warning: the following expert IDs were not active (below threshold or min_active_fraction): {sorted(missing)}")
        top_experts = [found_map[i] for i in id_list if i in found_map]
        print(
            f"\nUsing {len(top_experts)} manually specified expert(s):"
            + "".join(f"\n  Expert {e.expert_id}  fisher={e.fisher_score:.4f}" for e in top_experts)
        )
    else:
        experts.sort(key=lambda e: e.fisher_score or 0.0, reverse=True)
        top_experts = experts[:n_top_experts]
        print(
            f"\nTop {n_top_experts} expert(s) by Fisher score:"
            + "".join(f"\n  Expert {e.expert_id}  fisher={e.fisher_score:.4f}" for e in top_experts)
        )

    assert label_names is not None, "Hours dataset must have labels"
    hour_map = build_hour_map(label_names)  # stripped_hour → class_id

    df1 = pd.read_csv(task1_dataset)
    df2 = pd.read_csv(task2_dataset)
    sae_device = next(sae.parameters()).device

    # ── 3. Pre-filter rows and pre-generate baselines ─────────────────
    # Baselines don't depend on the expert — generate once, reuse for all.
    t1_rows: list[dict] = []
    for _, row in df1.iterrows():
        tgt_id = hour_map.get(row["Target_Hour"])
        if tgt_id is not None:
            t1_rows.append(dict(
                src_hour=row["Source_Hour"], tgt_hour=row["Target_Hour"],
                tgt_id=tgt_id, prompt=row["Prompt"],
            ))

    t2_rows: list[dict] = []
    for _, row in df2.iterrows():
        tgt_hour = hour_plus_delta(row["Current_Hour"], target_delta_hours, hour_map)
        tgt_id = hour_map.get(tgt_hour) if tgt_hour else None
        expected = int(row["Expected_Hours"])
        steered_expected = expected + target_delta_hours
        # Skip rows where steering produces the same answer as baseline (no measurable effect).
        if tgt_id is not None and steered_expected != expected:
            t2_rows.append(dict(
                src_hour=row["Current_Hour"], tgt_hour=tgt_hour, tgt_id=tgt_id,
                start_hour=row["Start_Hour"], expected_hours=expected,
                steered_expected_hours=steered_expected,
                prompt=row["Prompt"],
            ))

    # Switch to left-padding for generation
    tokenizer.padding_side = "left"

    def _gen_chunked(prompts: list[str]) -> list[str]:
        out = []
        for i in tqdm(range(0, len(prompts), gen_batch_size), desc="  batches"):
            out.extend(generate_text_batch(model, tokenizer, prompts[i : i + gen_batch_size], generate_tokens, device))
        return out

    print(f"\nPre-generating baselines ({len(t1_rows)} Task 1 + {len(t2_rows)} Task 2)…")
    t1_baselines = _gen_chunked([r["prompt"] for r in t1_rows])
    t2_baselines = _gen_chunked([r["prompt"] for r in t2_rows])

    # ── 4. Steering loop (batched per expert) ─────────────────────────
    records: list[dict] = []

    for expert in top_experts:
        class_means = compute_class_means(expert)
        print(f"\n── Expert {expert.expert_id} ──")

        # ── Task 1: Current time ──────────────────────────────────────
        t1_valid = [(i, r) for i, r in enumerate(t1_rows) if r["tgt_id"] in class_means]
        t1_tgt_means = torch.stack([class_means[r["tgt_id"]] for _, r in t1_valid]).to(device=sae_device)

        print(f"Task 1: {len(t1_valid)} prompts (batched, gen_batch_size={gen_batch_size})")
        t1_steered: list[str] = []
        for chunk_start in tqdm(range(0, len(t1_valid), gen_batch_size), desc="  Task 1"):
            chunk = t1_valid[chunk_start : chunk_start + gen_batch_size]
            chunk_tgt = t1_tgt_means[chunk_start : chunk_start + gen_batch_size]
            prompts = [r["prompt"] for _, r in chunk]
            with steering_hook(model, hook_point, sae, expert.expert_id, chunk_tgt, device):
                t1_steered.extend(generate_text_batch(model, tokenizer, prompts, generate_tokens, device))

        for (i, r), steered in zip(t1_valid, t1_steered):
            records.append(dict(
                task="current_time", expert_id=expert.expert_id, fisher=expert.fisher_score,
                src_hour=r["src_hour"], tgt_hour=r["tgt_hour"], start_hour=None,
                prompt=r["prompt"], baseline_output=t1_baselines[i], steered_output=steered,
            ))

        # ── Task 2: Elapsed time ──────────────────────────────────────
        t2_valid = [(i, r) for i, r in enumerate(t2_rows) if r["tgt_id"] in class_means]
        t2_tgt_means = torch.stack([class_means[r["tgt_id"]] for _, r in t2_valid]).to(device=sae_device)

        print(f"Task 2: {len(t2_valid)} prompts (delta={target_delta_hours}h)")
        t2_steered: list[str] = []
        for chunk_start in tqdm(range(0, len(t2_valid), gen_batch_size), desc="  Task 2"):
            chunk = t2_valid[chunk_start : chunk_start + gen_batch_size]
            chunk_tgt = t2_tgt_means[chunk_start : chunk_start + gen_batch_size]
            prompts = [r["prompt"] for _, r in chunk]
            with steering_hook(model, hook_point, sae, expert.expert_id, chunk_tgt, device):
                t2_steered.extend(generate_text_batch(model, tokenizer, prompts, generate_tokens, device))

        for (i, r), steered in zip(t2_valid, t2_steered):
            records.append(dict(
                task="elapsed_time", expert_id=expert.expert_id, fisher=expert.fisher_score,
                src_hour=r["src_hour"], tgt_hour=r["tgt_hour"], start_hour=r["start_hour"],
                expected_hours=r["expected_hours"],
                steered_expected_hours=r["steered_expected_hours"],
                prompt=r["prompt"],
                baseline_output=t2_baselines[i], steered_output=steered,
            ))

    # ── 4. Score all records and save per-expert detail files ────────
    df = pd.DataFrame(records)
    scores_dir = os.path.join(output_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    score_cols = ["task", "src_hour", "tgt_hour", "start_hour", "expected_hours", "steered_expected_hours", "prompt"]
    for expert_id, group in df.groupby("expert_id"):
        scored_rows = [dict(row[score_cols]) | _score_record(dict(row)) for _, row in group.iterrows()]
        score_df = pd.DataFrame(scored_rows)
        score_path = os.path.join(scores_dir, f"expert_{expert_id}.csv")
        score_df.to_csv(score_path, index=False)
    print(f"Saved per-expert detail scores ({len(top_experts)} files) → {scores_dir}/")

    # ── 5. Summary report: one row per (expert, task) ─────────────────
    summary_rows = []
    for expert_id, group in df.groupby("expert_id"):
        fisher = group["fisher"].iloc[0]
        for task, tgroup in group.groupby("task"):
            scored = [_score_record(dict(row)) for _, row in tgroup.iterrows()]
            n = len(scored)
            baseline_acc = sum(r["baseline_correct"] for r in scored) / n if n else 0.0
            steered_acc = sum(r["steered_correct"] for r in scored) / n if n else 0.0
            summary_rows.append(dict(
                expert_id=expert_id,
                fisher=fisher,
                task=task,
                n_prompts=n,
                baseline_accuracy=round(baseline_acc, 4),
                steered_accuracy=round(steered_acc, 4),
                delta_accuracy=round(steered_acc - baseline_acc, 4),
            ))
    summary_df = pd.DataFrame(summary_rows).sort_values(["task", "fisher"], ascending=[True, False])
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary ({len(summary_df)} rows) → {summary_path}")


if __name__ == "__main__":
    app()
