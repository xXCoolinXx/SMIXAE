"""Steering experiments using SMIXAE activation patching.

Loads a trained SMIXAE, selects the top experts for the hours-of-day probing
task from the cyc_24h regression hypothesis in results.json, and runs a causal
intervention using the current-time steering prompt dataset.

Steering mechanism (full-sequence activation patching):
    At every token position where expert e is active (norm > 0 after threshold):
        x_steered[pos] = x[pos]
                        - decode(z[pos, e, :])              # subtract expert's current contribution
                        + decode(mean_bottleneck[c_target])  # inject target class representation
    Only positions where the expert fires are patched; inactive positions are untouched.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

import typer

if TYPE_CHECKING:
    import torch

app = typer.Typer()


# ── Expert selection from results.json ─────────────────────────────────────────


def load_experts_from_results(
    results_path: str,
    run_name: str,
    dataset_name: str = "hours",
    hypothesis: str = "cyc_24h",
    n_top: int = 2,
) -> list[tuple[int, float]]:
    """Load top expert IDs from the probing results JSON.

    Navigates to ``data[run_name]["probe"][dataset_name]["hypotheses"][hypothesis]``
    and returns the top ``n_top`` entries as ``(expert_id, score)`` tuples.

    Raises:
        FileNotFoundError: If *results_path* does not exist.
        KeyError: If any navigation key is missing.
    """
    with open(results_path) as f:
        data = json.load(f)

    try:
        run_data = data[run_name]
    except KeyError:
        raise KeyError(f"Run '{run_name}' not found in {results_path}. Available: {list(data.keys())}")

    try:
        probe_data = run_data["probe"][dataset_name]
    except KeyError:
        available = list(run_data.get("probe", {}).keys())
        raise KeyError(f"Dataset '{dataset_name}' not found under probe for run '{run_name}'. Available: {available}")

    try:
        hyp_data = probe_data["hypotheses"][hypothesis]
    except KeyError:
        available = list(probe_data.get("hypotheses", {}).keys())
        raise KeyError(f"Hypothesis '{hypothesis}' not found under {dataset_name}. Available: {available}")

    experts = hyp_data["top10_experts"]
    return [(int(e["expert_id"]), float(e["score"])) for e in experts[:n_top]]


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
    import torch

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
    """Context manager that installs a full-sequence activation-patching hook.

    ``tgt_means`` is ``(batch, d_bottleneck)`` — one target per prompt in the batch.

    At every forward pass the hook:
    1. Encodes ALL token positions through the SAE.
    2. For each position where the expert is active, subtracts its current
       contribution and adds the decoded target class mean.
    3. Inactive positions are left untouched.

    The hook is removed on context exit.
    """
    import torch

    sae_device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype
    tgt = tgt_means.to(device=sae_device, dtype=sae_dtype)  # (batch, d_bottleneck)

    def _hook(_module, _input, output):
        x = output[0] if isinstance(output, tuple) else output  # (batch, seq, d_model)
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D).to(device=sae_device, dtype=sae_dtype)

        with torch.no_grad():
            z = sae.encode(x_flat)  # (B*S, n_experts, d_bottleneck)
            expert_norms = z[:, expert_id, :].norm(dim=-1)  # (B*S,)
            active_mask = expert_norms > 0  # encode() already applies threshold

            # Current contribution of this expert
            contrib = decode_single_expert(sae, z, expert_id)  # (B*S, d_model)

            # Target contribution: expand per-item targets across all positions
            tgt_expanded = tgt.repeat_interleave(S, dim=0)  # (B*S, d_bottleneck)
            z_tgt = torch.zeros_like(z)
            z_tgt[:, expert_id, :] = tgt_expanded
            tgt_contrib = decode_single_expert(sae, z_tgt, expert_id)  # (B*S, d_model)

            # Patch: subtract current, add target, only at active positions
            delta = (-contrib + tgt_contrib) * active_mask.unsqueeze(-1).float()
            x_steered = x_flat + delta

        x.copy_(x_steered.reshape(B, S, D).to(dtype=x.dtype))
        return (x,) + output[1:] if isinstance(output, tuple) else x

    handle = model.get_submodule(hook_point).register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


# ── Hour label utilities ──────────────────────────────────────────────────────


def build_hour_map(label_names: dict[int, str]) -> dict[str, int]:
    """Build a mapping from stripped hour string (e.g. '6PM') to class id."""
    from analysis.utils import _strip_prefix

    return {_strip_prefix(v): k for k, v in label_names.items()}


# ── Generation helpers ────────────────────────────────────────────────────────


def generate_text_batch(
    model, tokenizer, prompts: list[str], max_new_tokens: int, device: str
) -> list[str]:
    """Batched generation. Tokenizer must have padding_side='left'."""
    import torch

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


def _extract_time(text: str) -> str | None:
    """Return normalized hour string (e.g. '3PM') from model output, or None."""
    m = _TIME_RE.search(text)
    if m:
        return f"{int(m.group(1))}{m.group(2).upper()}"
    return None


def _score_record(record: dict) -> dict:
    """Return scoring fields for a single steering record."""
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


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def main(
    checkpoint_path: str = typer.Option(..., help="Path to SMIXAE model directory"),
    base_model_name: str = typer.Option(..., help="HuggingFace model name"),
    hook_point: str = typer.Option(..., help="Hook point used during SAE training"),
    hours_dataset: str = typer.Option(
        "datasets/probing/hours.csv",
        help="CSV with Sentence/Label columns for class-mean computation",
    ),
    task1_dataset: str = typer.Option(
        "datasets/steering/hours_current_time.csv",
        help="Steering prompt CSV (current time task)",
    ),
    output_dir: str = typer.Option("steer_results", help="Output directory"),
    # ── Expert selection ──
    results_json: str = typer.Option(
        "results/results.json",
        help="Path to results.json for hypothesis-based expert selection",
    ),
    run_name: str = typer.Option(
        "",
        help="Run name key in results.json (e.g. gemma_2_9b_l20). Required.",
    ),
    hypothesis: str = typer.Option(
        "cyc_24h",
        help="Regression hypothesis name for expert selection",
    ),
    n_top_experts: int = typer.Option(2, help="Number of top experts from hypothesis"),
    # ── Probing ──
    n_probing_samples: int = typer.Option(1000, help="Number of samples for the probing pass"),
    # ── Generation ──
    generate_tokens: int = typer.Option(20, help="Max new tokens to generate per prompt"),
    device: str = typer.Option("cuda", help="Device"),
    llm_batch_size: int = typer.Option(16, help="Batch size for probing LLM forward pass"),
    sae_batch_size: int = typer.Option(2048, help="Batch size for SAE encoding"),
    gen_batch_size: int = typer.Option(128, help="Batch size for text generation"),
    active_threshold: float = typer.Option(1e-5, help="L2 norm threshold for expert activity"),
    min_active_fraction: float = typer.Option(0.05, help="Minimum fraction of samples an expert must fire on"),
    # ── Cross-layer sweep ──
    sweep_layers: bool = typer.Option(False, help="Sweep steering across multiple layers"),
    layer_start: int = typer.Option(0, help="First layer to sweep (inclusive)"),
    layer_end: int = typer.Option(
        -1,
        help="Last layer to sweep (inclusive). -1 means trained_layer - 1.",
    ),
):
    """Run SMIXAE steering experiments on hours-of-day prompts."""
    import pandas as pd
    import torch
    from tqdm import tqdm

    from analysis.utils import (
        DatasetConfig,
        ExpertFilterConfig,
        collect_activations,
        extract_layer_from_hook,
        get_sae_activations,
        load_llm,
        load_sae,
    )

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load models ────────────────────────────────────────────────
    print(f"Loading {base_model_name}…")
    model, tokenizer = load_llm(base_model_name, device)
    tokenizer.padding_side = "right"
    sae = load_sae(checkpoint_path, device)

    # ── 2. Expert selection ────────────────────────────────────────────
    if not run_name:
        raise typer.BadParameter("--run-name is required.")
    selected_experts = load_experts_from_results(
        results_json, run_name, dataset_name="hours",
        hypothesis=hypothesis, n_top=n_top_experts,
    )

    print(
        f"\nSelected {len(selected_experts)} expert(s):"
        + "".join(f"\n  Expert {eid}  score={score:.4f}" for eid, score in selected_experts)
    )

    # ── 3. Probing pass (computes class means) ────────────────────────
    print(f"\nCollecting activations from {hours_dataset} for class means…")
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

    id_set = {eid for eid, _ in selected_experts}
    found_map = {e.expert_id: e for e in experts if e.expert_id in id_set}
    missing = id_set - set(found_map)
    if missing:
        print(f"Warning: the following expert IDs were not active: {sorted(missing)}")

    top_experts = [found_map[eid] for eid, _ in selected_experts if eid in found_map]
    if not top_experts:
        raise RuntimeError("None of the selected experts are active. Try lowering --active-threshold or --min-active-fraction.")

    assert label_names is not None, "Hours dataset must have labels"
    hour_map = build_hour_map(label_names)  # stripped_hour → class_id

    df1 = pd.read_csv(task1_dataset)
    sae_device = next(sae.parameters()).device

    # ── 4. Pre-filter rows and pre-generate baselines ─────────────────
    t1_rows: list[dict] = []
    for _, row in df1.iterrows():
        tgt_id = hour_map.get(row["Target_Hour"])
        if tgt_id is not None:
            t1_rows.append(dict(
                src_hour=row["Source_Hour"], tgt_hour=row["Target_Hour"],
                tgt_id=tgt_id, prompt=row["Prompt"],
            ))

    # Switch to left-padding for generation
    tokenizer.padding_side = "left"

    def _gen_chunked(prompts: list[str]) -> list[str]:
        out = []
        for i in tqdm(range(0, len(prompts), gen_batch_size), desc="  batches"):
            out.extend(generate_text_batch(model, tokenizer, prompts[i : i + gen_batch_size], generate_tokens, device))
        return out

    print(f"\nPre-generating baselines ({len(t1_rows)} prompts)…")
    t1_baselines = _gen_chunked([r["prompt"] for r in t1_rows])

    # ── 5. Determine layers to steer at ────────────────────────────────
    trained_layer = extract_layer_from_hook(hook_point)
    if sweep_layers:
        if trained_layer is None:
            raise typer.BadParameter(f"Cannot extract layer number from --hook-point '{hook_point}'")
        sweep_end = layer_end if layer_end >= 0 else trained_layer - 1
        layers = list(range(layer_start, sweep_end + 1))
        print(f"\nSweeping {len(layers)} layers: {layers[0]}–{layers[-1]} (trained at layer {trained_layer})")
    else:
        layers = None  # use trained hook_point directly

    # ── 6. Steering loop ───────────────────────────────────────────────
    records: list[dict] = []

    def _steer_at_layer(target_hook: str, layer_label: int | None = None):
        """Run steering for all experts at the given hook point."""
        for expert in top_experts:
            class_means = compute_class_means(expert)
            label_tag = f"layer {layer_label}, " if layer_label is not None else ""
            print(f"\n── {label_tag}Expert {expert.expert_id} ──")

            t1_valid = [(i, r) for i, r in enumerate(t1_rows) if r["tgt_id"] in class_means]
            t1_tgt_means = torch.stack([class_means[r["tgt_id"]] for _, r in t1_valid]).to(device=sae_device)

            print(f"Steering: {len(t1_valid)} prompts (gen_batch_size={gen_batch_size})")
            t1_steered: list[str] = []
            for chunk_start in tqdm(range(0, len(t1_valid), gen_batch_size), desc="  Steering"):
                chunk = t1_valid[chunk_start : chunk_start + gen_batch_size]
                chunk_tgt = t1_tgt_means[chunk_start : chunk_start + gen_batch_size]
                prompts = [r["prompt"] for _, r in chunk]
                with steering_hook(model, target_hook, sae, expert.expert_id, chunk_tgt, device):
                    t1_steered.extend(generate_text_batch(model, tokenizer, prompts, generate_tokens, device))

            for (i, r), steered in zip(t1_valid, t1_steered):
                rec = dict(
                    expert_id=expert.expert_id,
                    src_hour=r["src_hour"], tgt_hour=r["tgt_hour"],
                    prompt=r["prompt"], baseline_output=t1_baselines[i], steered_output=steered,
                )
                if layer_label is not None:
                    rec["layer"] = layer_label
                records.append(rec)

    if sweep_layers:
        for L in layers:
            layer_hook = f"model.layers.{L}"
            print(f"\n{'=' * 60}")
            print(f"Layer {L}  (hook: {layer_hook})")
            print(f"{'=' * 60}")
            _steer_at_layer(layer_hook, layer_label=L)
    else:
        _steer_at_layer(hook_point)

    # ── 7. Score all records and save ──────────────────────────────────
    df = pd.DataFrame(records)
    scores_dir = os.path.join(output_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    score_cols = ["src_hour", "tgt_hour", "prompt"]
    if sweep_layers:
        score_cols.append("layer")

    group_cols = ["expert_id", "layer"] if sweep_layers else ["expert_id"]
    for group_key, group in df.groupby(group_cols):
        scored_rows = [dict(row[score_cols]) | _score_record(dict(row)) for _, row in group.iterrows()]
        score_df = pd.DataFrame(scored_rows)
        if sweep_layers:
            eid = group_key[0] if isinstance(group_key, tuple) else group_key
            lid = group_key[1] if isinstance(group_key, tuple) else 0
            score_path = os.path.join(scores_dir, f"layer_{lid}_expert_{eid}.csv")
        else:
            eid = group_key
            score_path = os.path.join(scores_dir, f"expert_{eid}.csv")
        score_df.to_csv(score_path, index=False)
    print(f"Saved detail scores → {scores_dir}/")

    # ── 8. Summary report ──────────────────────────────────────────────
    summary_rows = []
    summary_group_cols = group_cols
    for group_key, group in df.groupby(summary_group_cols):
        scored = [_score_record(dict(row)) for _, row in group.iterrows()]
        n = len(scored)
        baseline_acc = sum(r["baseline_correct"] for r in scored) / n if n else 0.0
        steered_acc = sum(r["steered_correct"] for r in scored) / n if n else 0.0
        row = dict(
            n_prompts=n,
            baseline_accuracy=round(baseline_acc, 4),
            steered_accuracy=round(steered_acc, 4),
            delta_accuracy=round(steered_acc - baseline_acc, 4),
        )
        if sweep_layers:
            row["layer"] = group_key[1] if isinstance(group_key, tuple) else group_key
            row["expert_id"] = group_key[0] if isinstance(group_key, tuple) else group_key
        else:
            row["expert_id"] = group_key
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    sort_cols = ["layer", "delta_accuracy"] if sweep_layers else ["delta_accuracy"]
    summary_df = summary_df.sort_values(sort_cols, ascending=[True, False] if sweep_layers else [False])
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary ({len(summary_df)} rows) → {summary_path}")


if __name__ == "__main__":
    app()
