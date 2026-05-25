"""Metric computation for the synthetic toy-model benchmark.

Contains:
- _encode_all: batch encoder helper
- compute_restricted_r2: per-manifold R² via co-firing matching + linear OLS
- compute_metrics: overall MSE, effective L0, dead-expert count
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from toy.zoo import EvalData, ManifoldZoo

if TYPE_CHECKING:
    from smixae.smixae import SMIXAETraining


# ── Metrics ────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _encode_all(
    model: "SMIXAETraining",
    x: torch.Tensor,
    device: str,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Run the trained model's encoder on x and return h_bottleneck on CPU.

    Args:
        model: Trained SMIXAETraining on device.
        x: (N, d_in) on CPU.
        device: Model device.
        chunk_size: Encoding chunk size.

    Returns:
        h_bottleneck: (N, n_experts, d_bottleneck) on CPU.
    """
    N = x.shape[0]
    n_exp = model.cfg.n_experts
    d_b   = model.cfg.d_bottleneck
    out   = torch.zeros(N, n_exp, d_b)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        model.encode_with_hidden_pre(x[start:end].to(device))
        out[start:end] = model.h_bottleneck.detach().cpu()
    return out


@torch.no_grad()
def compute_restricted_r2(
    model: "SMIXAETraining",
    zoo: ManifoldZoo,
    eval_data: EvalData,
    device: str = "cpu",
    max_n: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Per-manifold R² via co-firing matching and linear OLS in intrinsic coordinates.

    For each manifold instance i:
    1. Finds the expert with the highest co-firing rate P(expert fires | manifold active).
    2. On rows where both the manifold is active and that expert fires, fits an affine
       map from the expert's bottleneck codes to the manifold's intrinsic coordinates
       via OLS (i.e. finds the optimal full-rank linear read-out).
    3. R²(i) = 1 - ||Z - Ẑ||² / ||Z - Z̄||² for the OLS fit.

    Args:
        model: Trained SMIXAETraining.
        zoo: ManifoldZoo used to generate eval_data.
        eval_data: EvalData on CPU.
        device: Model device.
        max_n: Unused; kept for API compatibility.

    Returns:
        Tuple of:
        - r2: (n_instances,) — linear OLS R² from matched expert's bottleneck codes
          to the manifold's intrinsic coordinates.
        - cofiring_rate: (n_instances,) — P(matched expert fires | manifold active).
        - best_experts: list[int] of length n_instances — matched expert per manifold.
    """
    model.eval()
    n_inst = len(zoo.instances)
    h_bott = _encode_all(model, eval_data.x, device)  # (N, n_exp, d_bott) on CPU

    r2:            torch.Tensor = torch.zeros(n_inst)
    cofiring_rate: torch.Tensor = torch.zeros(n_inst)
    best_experts:  list[int]    = []

    for inst_idx, inst in enumerate(zoo.instances):
        off = inst.atom_offset
        ki  = inst.k_i

        active_mask = eval_data.feature_acts[:, off: off + ki].abs().sum(1) > 0
        active_rows = active_mask.nonzero(as_tuple=True)[0]
        if active_rows.numel() < 10:
            best_experts.append(0)
            continue

        # fired_active[n, e] = True if expert e fired for this active sample
        # (post-BatchTopK: non-firing experts have zero bottleneck norm)
        fired_active = h_bott[active_rows].norm(dim=-1) > 0   # (n_active, n_exp)

        # Match expert by co-firing rate: P(e fires | manifold active)
        cofiring = fired_active.float().mean(0)                # (n_exp,)
        best_e   = int(cofiring.argmax().item())
        cofiring_rate[inst_idx] = cofiring[best_e].item()
        best_experts.append(best_e)

        # OLS on co-active rows only (manifold active AND matched expert fires)
        co_active_mask = fired_active[:, best_e]
        co_active_rows = active_rows[co_active_mask.nonzero(as_tuple=True)[0]]
        if co_active_rows.numel() < 4:
            continue  # r2 stays 0.0 — expert barely fires on this manifold

        H = h_bott[co_active_rows, best_e, :].float()          # (n_co, d_bott)
        Z = eval_data.feature_acts[co_active_rows, off: off + ki].float()  # (n_co, ki)

        # Affine OLS: augment H with a constant column, fit W s.t. [H|1] W ≈ Z
        H_aug  = torch.cat([H, torch.ones(H.shape[0], 1)], dim=1)  # (n_co, d_bott+1)
        W      = torch.linalg.lstsq(H_aug, Z).solution              # (d_bott+1, ki)
        Z_pred = H_aug @ W                                           # (n_co, ki)

        ss_res = (Z - Z_pred).pow(2).sum()
        ss_tot = (Z - Z.mean(0)).pow(2).sum().clamp(min=1e-8)
        r2[inst_idx] = (1.0 - ss_res / ss_tot).item()

    return r2, cofiring_rate, best_experts


@torch.no_grad()
def compute_cofiring_matrix(
    model: "SMIXAETraining",
    zoo: ManifoldZoo,
    eval_data: EvalData,
    device: str = "cpu",
) -> torch.Tensor:
    """Return the full (n_instances, n_experts) co-firing rate matrix.

    Entry [i, e] = P(expert e fires | manifold instance i is active).
    ``compute_restricted_r2`` returns only the max per row; this function
    exposes the full distribution so callers can inspect whether the best
    expert is clearly dominant or multiple experts are tied.

    Args:
        model: Trained SMIXAETraining.
        zoo: ManifoldZoo used to generate eval_data.
        eval_data: EvalData on CPU.
        device: Model device.

    Returns:
        cofiring_matrix: float tensor of shape (n_instances, n_experts).
    """
    model.eval()
    n_inst = len(zoo.instances)
    n_exp  = model.cfg.n_experts
    h_bott = _encode_all(model, eval_data.x, device)  # (N, n_exp, d_bott) on CPU

    matrix = torch.zeros(n_inst, n_exp)

    for inst_idx, inst in enumerate(zoo.instances):
        off = inst.atom_offset
        ki  = inst.k_i

        active_mask = eval_data.feature_acts[:, off: off + ki].abs().sum(1) > 0
        active_rows = active_mask.nonzero(as_tuple=True)[0]
        if active_rows.numel() < 4:
            continue

        fired_active = h_bott[active_rows].norm(dim=-1) > 0   # (n_active, n_exp)
        matrix[inst_idx] = fired_active.float().mean(0)

    return matrix


@torch.no_grad()
def compute_metrics(
    model: "SMIXAETraining",
    zoo: ManifoldZoo,
    eval_data: EvalData,
    device: str = "cpu",
    chunk_size: int = 4096,
) -> dict[str, float]:
    """Compute overall MSE, effective L0, and dead-expert count on the eval set.

    Args:
        model: Trained SMIXAETraining.
        zoo: ManifoldZoo (unused, kept for signature consistency).
        eval_data: EvalData on CPU.
        device: Model device.
        chunk_size: Evaluation chunk size.

    Returns:
        Dict with keys ``mse``, ``effective_l0``, ``dead_experts``.
    """
    model.eval()
    N = eval_data.x.shape[0]
    mse_sum = 0.0
    l0_sum  = 0.0
    ever_fired = torch.zeros(model.cfg.n_experts, dtype=torch.bool)

    for start in range(0, N, chunk_size):
        end     = min(start + chunk_size, N)
        x_chunk = eval_data.x[start:end].to(device)
        model.encode_with_hidden_pre(x_chunk)
        sae_out = model.decode(model.h_bottleneck)

        mse_sum += (sae_out - x_chunk).pow(2).sum(dim=-1).sum().item()
        norms    = model.h_bottleneck.norm(dim=-1)          # (B, n_experts)
        l0_sum  += (norms > 0).float().sum(dim=-1).sum().item()
        ever_fired |= (norms > 0).any(0).cpu()

    return {
        "mse":          mse_sum / N,
        "effective_l0": l0_sum / N,
        "dead_experts": int((~ever_fired).sum().item()),
    }
