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
from contextlib import contextmanager
from typing import Generator

import pandas as pd
import torch
import typer
from tqdm import tqdm

from analysis.utils import (
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
    tgt_mean: torch.Tensor,
    device: str,
) -> Generator[None, None, None]:
    """Context manager that installs a coordinate-substitution hook on ``hook_point``.

    At every forward pass the hook:
    1. Encodes the last-sequence-position activation through the SAE.
    2. Subtracts the expert's current contribution.
    3. Adds the decoded target class mean for that expert.

    The hook is removed on context exit.
    """
    sae_device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype
    tgt = tgt_mean.to(device=sae_device, dtype=sae_dtype)

    def _hook(_module, _input, output):
        x = output[0] if isinstance(output, tuple) else output  # (batch, seq, d_model)
        x_last = x[:, -1:, :].to(device=sae_device, dtype=sae_dtype)  # (batch, 1, d_model)

        with torch.no_grad():
            z = sae.encode(x_last.squeeze(1))  # (batch, n_experts, d_bottleneck)
            contrib = decode_single_expert(sae, z, expert_id)  # (batch, d_model)

            z_tgt = torch.zeros_like(z)
            z_tgt[:, expert_id, :] = tgt.unsqueeze(0).expand(z.shape[0], -1)
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


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_ids = out[0, enc["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


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
    n_top_experts: int = typer.Option(1, help="Number of top Fisher experts to steer with"),
    n_probing_samples: int = typer.Option(1000, help="Number of samples for the expert-discovery probing pass"),
    target_delta_hours: int = typer.Option(1, help="Task 2: hours to add to Current_Hour to get the steering target"),
    generate_tokens: int = typer.Option(20, help="Max new tokens to generate per prompt"),
    device: str = typer.Option("cuda", help="Device"),
    llm_batch_size: int = typer.Option(16, help="Batch size for probing LLM forward pass"),
    sae_batch_size: int = typer.Option(2048, help="Batch size for SAE encoding"),
    active_threshold: float = typer.Option(1e-5, help="L2 norm threshold for expert activity"),
    min_points: int = typer.Option(50, help="Minimum active tokens to keep an expert"),
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
    acts, _str_tokens, labels, label_names, last_positions, n_classes = collect_activations(
        model=model,
        tokenizer=tokenizer,
        hook_name=hook_point,
        max_length=128,
        n_input_samples=n_probing_samples,
        device=device,
        llm_batch_size=llm_batch_size,
        dataframe_path=hours_dataset,
        text_column="Sentence",
        label_column="Label",
    )

    experts = get_sae_activations(
        sae=sae,
        device=device,
        activations=acts,
        sae_batch_size=sae_batch_size,
        active_threshold=active_threshold,
        min_points=min_points,
        max_points=0,  # no cap — keep all points for accurate means
        labels=labels,
        last_token_only=True,
        last_token_positions=last_positions,
        n_classes=n_classes,
    )
    del acts, labels, last_positions

    print(f"Scoring Fisher for {len(experts)} active experts…")
    for e in tqdm(experts):
        e.evaluate_fisher()
    experts.sort(key=lambda e: e.fisher_score or 0.0, reverse=True)

    top_experts = experts[:n_top_experts]
    print(
        f"\nTop {n_top_experts} expert(s):"
        + "".join(f"\n  Expert {e.expert_id}  fisher={e.fisher_score:.4f}" for e in top_experts)
    )

    assert label_names is not None, "Hours dataset must have labels"
    hour_map = build_hour_map(label_names)  # stripped_hour → class_id

    df1 = pd.read_csv(task1_dataset)
    df2 = pd.read_csv(task2_dataset)
    sae_device = next(sae.parameters()).device

    # ── 3. Steering loop ──────────────────────────────────────────────
    records: list[dict] = []

    for expert in top_experts:
        class_means = compute_class_means(expert)
        print(f"\n── Expert {expert.expert_id} ──")

        # ── Task 1: Current time ──────────────────────────────────────
        print(f"Task 1: {len(df1)} prompts from {task1_dataset}")
        for _, row in tqdm(df1.iterrows(), total=len(df1), desc="  Task 1"):
            src_hour = row["Source_Hour"]
            tgt_hour = row["Target_Hour"]
            tgt_id = hour_map.get(tgt_hour)
            if tgt_id is None or tgt_id not in class_means:
                continue

            prompt = row["Prompt"]
            baseline = generate_text(model, tokenizer, prompt, generate_tokens, device)

            tgt_mean = class_means[tgt_id].to(device=sae_device)
            with steering_hook(model, hook_point, sae, expert.expert_id, tgt_mean, device):
                steered = generate_text(model, tokenizer, prompt, generate_tokens, device)

            records.append(
                dict(
                    task="current_time",
                    expert_id=expert.expert_id,
                    fisher=expert.fisher_score,
                    src_hour=src_hour,
                    tgt_hour=tgt_hour,
                    start_hour=None,
                    prompt=prompt,
                    baseline_output=baseline,
                    steered_output=steered,
                )
            )

        # ── Task 2: Elapsed time ──────────────────────────────────────
        # For each row the steering target is Current_Hour + target_delta_hours.
        print(f"Task 2: {len(df2)} prompts from {task2_dataset} (delta={target_delta_hours}h)")
        for _, row in tqdm(df2.iterrows(), total=len(df2), desc="  Task 2"):
            curr_hour = row["Current_Hour"]
            tgt_hour = hour_plus_delta(curr_hour, target_delta_hours, hour_map)
            if tgt_hour is None:
                continue
            tgt_id = hour_map.get(tgt_hour)
            if tgt_id is None or tgt_id not in class_means:
                continue

            prompt = row["Prompt"]
            baseline = generate_text(model, tokenizer, prompt, generate_tokens, device)

            tgt_mean = class_means[tgt_id].to(device=sae_device)
            with steering_hook(model, hook_point, sae, expert.expert_id, tgt_mean, device):
                steered = generate_text(model, tokenizer, prompt, generate_tokens, device)

            records.append(
                dict(
                    task="elapsed_time",
                    expert_id=expert.expert_id,
                    fisher=expert.fisher_score,
                    src_hour=curr_hour,
                    tgt_hour=tgt_hour,
                    start_hour=row["Start_Hour"],
                    expected_hours=row["Expected_Hours"],
                    prompt=prompt,
                    baseline_output=baseline,
                    steered_output=steered,
                )
            )

    # ── 4. Save results ───────────────────────────────────────────────
    out_path = os.path.join(output_dir, "steering_results.csv")
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")


if __name__ == "__main__":
    app()
