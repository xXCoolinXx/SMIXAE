"""SAEBench core evaluation reimplemented using HuggingFace.

Computes the same metrics as SAEBench core without depending on TransformerLens.
Supports both SMIXAE checkpoints and GemmaScope baselines loaded via SAELens.

Metrics computed (matching SAEBench definitions):

  l0                  Mean active features per token (count of non-zero elements).
  mse                 Normalised reconstruction error: mean ||x - x̂||² / ||x||² per token.
  explained_variance  1 - mean(||x - x̂||²) / mean(||x||²).  Normalised-MSE form; less
                      inflated than mean-centred variance when the residual stream has a large DC mean.
  cosine_similarity   Mean cosine similarity between reconstruction and input.
  l2_ratio            Mean ||x̂|| / ||x|| per token.
  ce_loss_without_sae Baseline cross-entropy loss.
  ce_loss_with_sae    CE loss when the hook layer output is replaced by the SAE reconstruction.
  ce_loss_with_ablation CE loss when the hook layer output is zero-ablated.
  ce_loss_score       (ce_ablation - ce_sae) / (ce_ablation - ce_without_sae).

Results are saved to a JSON file with the hierarchy::

    {
      "<model_name>": {
        "layer_<N>": {
          "<human-readable SAE name>": { ...metrics... },
          ...
        }
      }
    }
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

import smixae as _smixae_register  # noqa: F401  — registers "smixae" with SAELens
from analysis.utils import load_sae

CORE_EVAL_RESULTS_PATH = Path("results/core_eval_results.json")


@dataclass
class CoreEvalConfig:
    """Configuration for a core evaluation run."""

    dataset: str = "Skylion007/openwebtext"
    context_size: int = 128
    n_reconstruction_batches: int = 200
    n_sparsity_batches: int = 2000
    batch_size: int = 16
    device: str = "cuda"
    dtype: str = "bfloat16"

    def llm_dtype(self) -> torch.dtype:
        """Return the torch dtype corresponding to ``self.dtype``."""
        return getattr(torch, self.dtype)


# ---------------------------------------------------------------------------
# SAE encode/decode closures
# ---------------------------------------------------------------------------

def _make_sae_fns(
    sae: Any,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], int]:
    """Return ``(encode_fn, decode_fn, d_sae)`` closures for *sae*.

    SMIXAE models are detected by the presence of ``n_experts`` on their config
    and get a flatten/unflatten wrapper. Standard SAELens SAEs pass through
    directly.

    Inputs are cast to the SAE's parameter dtype so bfloat16 LLM activations
    work against float32 SAE weights (e.g. GemmaScope defaults).
    """
    if hasattr(sae, "eval"):
        sae.eval()
    sae_device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype

    def _to_sae(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=sae_device, dtype=sae_dtype)

    is_smixae = hasattr(sae.cfg, "n_experts")

    if is_smixae:
        n = sae.cfg.n_experts
        d = sae.cfg.d_bottleneck
        d_sae = n * d

        def encode(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 3:
                b, s, dim = x.shape
                acts = sae.encode(_to_sae(x.reshape(b * s, dim)))
                return acts.flatten(-2, -1).reshape(b, s, d_sae)
            acts = sae.encode(_to_sae(x))
            return acts.flatten(-2, -1)

        def decode(feat: torch.Tensor) -> torch.Tensor:
            if feat.ndim == 3:
                b, s, _ = feat.shape
                out = sae.decode(_to_sae(feat.reshape(b * s, n, d)))
                return out.reshape(b, s, -1)
            return sae.decode(_to_sae(feat.reshape(feat.shape[0], n, d)))
    else:
        d_sae = sae.cfg.d_sae

        def encode(x: torch.Tensor) -> torch.Tensor:
            return sae.encode(_to_sae(x))

        def decode(feat: torch.Tensor) -> torch.Tensor:
            return sae.decode(_to_sae(feat))

    return encode, decode, d_sae


# ---------------------------------------------------------------------------
# Dataset tokenisation
# ---------------------------------------------------------------------------

def _build_token_batches(
    dataset_name: str,
    tokenizer: Any,
    context_size: int,
    n_batches: int,
    batch_size: int,
) -> list[torch.Tensor]:
    """Stream a dataset and pack into fixed-length token batches.

    Matches SAELens' pretokenizer defaults (``begin_batch_token = bos``,
    ``sequence_separator_token = bos``): every window starts with BOS, and
    documents are joined by BOS inside the packed buffer. Without this, eval
    sees a different token distribution than training and inference metrics
    (L0, reconstruction norm) drift.

    Args:
        dataset_name: HuggingFace dataset identifier.
        tokenizer: HuggingFace tokenizer (must have ``encode``).
        context_size: Number of tokens per sequence.
        n_batches: Number of batches to produce.
        batch_size: Sequences per batch.

    Returns:
        List of ``(batch_size, context_size)`` long tensors.
    """
    needed = n_batches * batch_size
    ds = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)

    bos_id = tokenizer.bos_token_id
    payload_size = context_size - 1 if bos_id is not None else context_size

    buffer: list[int] = []
    chunks: list[list[int]] = []

    for example in ds:
        doc_tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        if bos_id is not None:
            buffer.append(bos_id)  # sequence_separator_token = bos
        buffer.extend(doc_tokens)
        while len(buffer) >= payload_size:
            payload = buffer[:payload_size]
            buffer = buffer[payload_size:]
            if bos_id is not None:
                chunks.append([bos_id, *payload])
            else:
                chunks.append(payload)
        if len(chunks) >= needed:
            break

    chunks = chunks[:needed]
    batches: list[torch.Tensor] = []
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        batch = chunks[i : i + batch_size]
        if len(batch) == batch_size:
            batches.append(torch.tensor(batch, dtype=torch.long))

    return batches


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def _get_named_module(model: nn.Module, name: str) -> nn.Module:
    """Return the submodule at ``name`` (dot-separated), e.g. ``"model.layers.11"``."""
    parts = name.split(".")
    mod: nn.Module = model
    for part in parts:
        if part.isdigit():
            mod = mod[int(part)]  # type: ignore[index]
        else:
            mod = getattr(mod, part)
    return mod


def _collect_acts(
    model: nn.Module,
    tokens: torch.Tensor,
    hook_name: str,
) -> torch.Tensor:
    """Run a forward pass and collect the hidden states at ``hook_name``."""
    collected: list[torch.Tensor] = []

    def _hook(module: nn.Module, inp: Any, out: Any) -> None:
        acts = out[0] if isinstance(out, tuple) else out
        collected.append(acts.detach())

    handle = _get_named_module(model, hook_name).register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        handle.remove()

    return collected[0]


def _compute_sparsity_variance_metrics(
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    model: nn.Module,
    token_batches: list[torch.Tensor],
    hook_name: str,
    device: str,
    llm_dtype: torch.dtype,
    verbose: bool = False,
    bos_id: int | None = None,
) -> dict[str, float]:
    r"""Compute sparsity and reconstruction quality metrics.

    Args:
        encode_fn: SAE encode closure from ``_make_sae_fns``.
        decode_fn: SAE decode closure from ``_make_sae_fns``.
        model: HuggingFace LLM.
        token_batches: Tokenised input batches.
        hook_name: Hook point in HuggingFace notation.
        device: Compute device string.
        llm_dtype: Dtype to cast activations to before encoding.
        verbose: Show progress bar.
        bos_id: Token ID to exclude from all metrics (avoids BOS-spike bias).

    Returns:
        Dict with keys ``l0``, ``mse``, ``explained_variance``,
        ``cosine_similarity``, ``l2_ratio``, ``l2_norm_in``, ``l2_norm_out``.
    """
    l0_list: list[torch.Tensor] = []
    mse_list: list[torch.Tensor] = []
    cossim_list: list[torch.Tensor] = []
    l2_in_list: list[torch.Tensor] = []
    l2_out_list: list[torch.Tensor] = []
    l2_ratio_list: list[torch.Tensor] = []

    # FVE accumulators: 1 - mean(||x-x̂||²) / mean(||x||²).
    x_sq_list: list[torch.Tensor] = []
    resid_sq_list: list[torch.Tensor] = []

    batch_iter = tqdm(token_batches, desc="Sparsity/variance", leave=False) if verbose else token_batches

    for tokens in batch_iter:
        tokens = tokens.to(device)
        original = _collect_acts(model, tokens, hook_name).to(llm_dtype)
        b, s, d = original.shape

        if bos_id is not None:
            valid = (tokens != bos_id).reshape(-1)
        else:
            valid = torch.ones(b * s, dtype=torch.bool, device=tokens.device)

        with torch.no_grad():
            feat = encode_fn(original).to(device)
            recon = decode_fn(feat).to(device, llm_dtype)

        # Metric math in float32 — bfloat16 sums over d_model (~3584 dims) lose precision.
        flat_in = original.reshape(-1, d).float()[valid]
        flat_out = recon.reshape(-1, d).float()[valid]
        flat_feat = feat.reshape(-1, feat.shape[-1])[valid]

        l0_list.append((flat_feat != 0).float().sum(-1))

        resid = flat_in - flat_out
        mse_list.append(resid.pow(2).sum(-1) / (flat_in.pow(2).sum(-1) + 1e-8))

        x_sq_list.append(flat_in.pow(2).sum(-1).mean())
        resid_sq_list.append(resid.pow(2).sum(-1).mean())

        cossim_list.append(F.cosine_similarity(flat_in, flat_out, dim=-1))

        l2_in = flat_in.norm(dim=-1)
        l2_out = flat_out.norm(dim=-1)
        l2_in_list.append(l2_in)
        l2_out_list.append(l2_out)
        l2_ratio_list.append(l2_out / (l2_in + 1e-8))

    mean_x_sq = torch.stack(x_sq_list).mean()
    mean_resid_sq = torch.stack(resid_sq_list).mean()
    explained_var = (1.0 - mean_resid_sq / mean_x_sq).item()

    return {
        "l0": torch.cat(l0_list).mean().item(),
        "mse": torch.cat(mse_list).mean().item(),
        "explained_variance": explained_var,
        "cosine_similarity": torch.cat(cossim_list).mean().item(),
        "l2_ratio": torch.cat(l2_ratio_list).mean().item(),
        "l2_norm_in": torch.cat(l2_in_list).mean().item(),
        "l2_norm_out": torch.cat(l2_out_list).mean().item(),
    }


def _compute_ce_loss_metrics(
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    model: nn.Module,
    token_batches: list[torch.Tensor],
    hook_name: str,
    device: str,
    llm_dtype: torch.dtype,
    verbose: bool = False,
    bos_id: int | None = None,
) -> dict[str, float]:
    r"""Compute CE-loss metrics: baseline, with-SAE, and zero-ablation.

    Args:
        encode_fn: SAE encode closure.
        decode_fn: SAE decode closure.
        model: HuggingFace LLM.
        token_batches: Tokenised input batches.
        hook_name: Hook point in HuggingFace notation.
        device: Compute device string.
        llm_dtype: Dtype for activations.
        verbose: Show progress bar.
        bos_id: Token ID to exclude from CE loss (avoids BOS-spike bias).

    Returns:
        Dict with keys ``ce_loss_without_sae``, ``ce_loss_with_sae``,
        ``ce_loss_with_ablation``, ``ce_loss_score``.
    """
    vocab_size = model.config.vocab_size  # type: ignore[union-attr]
    hook_module = _get_named_module(model, hook_name)

    ce_orig_list: list[float] = []
    ce_sae_list: list[float] = []
    ce_abl_list: list[float] = []

    def _ce_from_logits(logits: torch.Tensor, tokens: torch.Tensor) -> float:
        shift_logits = logits[:, :-1].contiguous().float()
        shift_labels = tokens[:, 1:].contiguous().clone()
        if bos_id is not None:
            bos_mask = (tokens[:, :-1] == bos_id) | (tokens[:, 1:] == bos_id)
            shift_labels[bos_mask] = -100
        return F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            ignore_index=-100,
        ).item()

    def _run_with_hook(tokens: torch.Tensor, hook_fn: Any | None) -> torch.Tensor:
        handle = hook_module.register_forward_hook(hook_fn) if hook_fn is not None else None
        try:
            with torch.no_grad():
                out = model(tokens)
        finally:
            if handle is not None:
                handle.remove()
        return out.logits  # type: ignore[union-attr]

    def _replacement_hook(module: nn.Module, inp: Any, out: Any) -> Any:
        acts = out[0] if isinstance(out, tuple) else out
        recon = decode_fn(encode_fn(acts.to(llm_dtype))).to(acts.device, acts.dtype)
        return (recon,) + out[1:] if isinstance(out, tuple) else recon

    def _zero_ablation_hook(module: nn.Module, inp: Any, out: Any) -> Any:
        acts = out[0] if isinstance(out, tuple) else out
        zero = torch.zeros_like(acts)
        return (zero,) + out[1:] if isinstance(out, tuple) else zero

    batch_iter = tqdm(token_batches, desc="CE loss", leave=False) if verbose else token_batches

    for tokens in batch_iter:
        tokens = tokens.to(device)
        logits_orig = _run_with_hook(tokens, None)
        logits_sae = _run_with_hook(tokens, _replacement_hook)
        logits_abl = _run_with_hook(tokens, _zero_ablation_hook)

        ce_orig_list.append(_ce_from_logits(logits_orig, tokens))
        ce_sae_list.append(_ce_from_logits(logits_sae, tokens))
        ce_abl_list.append(_ce_from_logits(logits_abl, tokens))

    ce_orig = sum(ce_orig_list) / len(ce_orig_list)
    ce_sae = sum(ce_sae_list) / len(ce_sae_list)
    ce_abl = sum(ce_abl_list) / len(ce_abl_list)
    ce_score = (ce_abl - ce_sae) / (ce_abl - ce_orig + 1e-8)

    return {
        "ce_loss_without_sae": ce_orig,
        "ce_loss_with_sae": ce_sae,
        "ce_loss_with_ablation": ce_abl,
        "ce_loss_score": ce_score,
    }


# ---------------------------------------------------------------------------
# Top-level evaluation entry points
# ---------------------------------------------------------------------------

def run_core_eval(
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    model: nn.Module,
    tokenizer: Any,
    hook_name: str,
    cfg: CoreEvalConfig,
    verbose: bool = False,
) -> dict[str, float]:
    """Run the full core evaluation for a single SAE.

    Args:
        encode_fn: SAE encode closure from ``_make_sae_fns``.
        decode_fn: SAE decode closure from ``_make_sae_fns``.
        model: HuggingFace LLM (already on ``cfg.device``).
        tokenizer: HuggingFace tokenizer.
        hook_name: HuggingFace hook point, e.g. ``"model.layers.11"``.
        cfg: Evaluation configuration.
        verbose: Enable tqdm progress bars.

    Returns:
        Dict of scalar metric values.
    """
    llm_dtype = cfg.llm_dtype()
    bos_id = getattr(tokenizer, "bos_token_id", None)
    n_total = cfg.n_reconstruction_batches + cfg.n_sparsity_batches
    logger.info(f"Tokenising {n_total} batches × {cfg.batch_size} seqs × {cfg.context_size} tokens …")
    all_batches = _build_token_batches(
        cfg.dataset, tokenizer, cfg.context_size, n_total, cfg.batch_size
    )
    recon_batches = all_batches[: cfg.n_reconstruction_batches]
    sparsity_batches = all_batches[cfg.n_reconstruction_batches :]

    logger.info("Computing sparsity/variance metrics …")
    metrics = _compute_sparsity_variance_metrics(
        encode_fn, decode_fn, model, sparsity_batches, hook_name, cfg.device, llm_dtype, verbose=verbose, bos_id=bos_id
    )

    logger.info("Computing CE-loss metrics …")
    metrics.update(
        _compute_ce_loss_metrics(
            encode_fn, decode_fn, model, recon_batches, hook_name, cfg.device, llm_dtype, verbose=verbose, bos_id=bos_id
        )
    )

    return metrics


def run_single_eval(
    sae: Any,
    model: nn.Module,
    tokenizer: Any,
    hook_name: str,
    cfg: CoreEvalConfig,
    verbose: bool = False,
) -> dict[str, float]:
    """Run core eval for a single SAE (SMIXAE or GemmaScope).

    Args:
        sae: A loaded SAE model (SMIXAE or SAELens SAE).
        model: HuggingFace LLM (already on ``cfg.device``).
        tokenizer: HuggingFace tokenizer.
        hook_name: HuggingFace hook point, e.g. ``"model.layers.11"``.
        cfg: Evaluation config.
        verbose: Enable progress bars.

    Returns:
        Dict of scalar metric values.
    """
    enc, dec, _ = _make_sae_fns(sae)
    metrics = run_core_eval(enc, dec, model, tokenizer, hook_name, cfg, verbose=verbose)
    metrics.update(_extract_model_info(sae))
    return metrics


def _extract_model_info(sae: Any) -> dict[str, Any]:
    """Extract architecture stats from a loaded SAE for inclusion in results.

    Returns:
        Dict with ``total_params`` and, when detectable, ``width`` (flattened
        latent dimension: ``n_experts * d_bottleneck`` for SMIXAE, ``d_sae``
        for standard SAELens SAEs).
    """
    info: dict[str, Any] = {"total_params": sum(p.numel() for p in sae.parameters())}
    cfg = getattr(sae, "cfg", None)
    if cfg is not None:
        if hasattr(cfg, "n_experts"):
            info["width"] = cfg.n_experts * cfg.d_bottleneck
        elif hasattr(cfg, "d_sae"):
            info["width"] = cfg.d_sae
    return info


def load_sae_from_path(path: str, device: str) -> Any:
    """Load an SAE from a local checkpoint directory."""
    return load_sae(path, device=device)


def load_sae_from_hf(release: str, sae_id: str, device: str) -> Any:
    """Load an SAE from HuggingFace via SAELens ``from_pretrained``."""
    from sae_lens import SAE as SAELensModel

    sae, _, _ = SAELensModel.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae


def load_llm_for_eval(model_name: str, cfg: CoreEvalConfig) -> tuple[Any, Any]:
    """Load and return ``(llm, tokenizer)`` in eval mode."""
    from analysis.utils import load_llm

    llm, tokenizer = load_llm(model_name, device=cfg.device, dtype=cfg.llm_dtype())
    llm.eval()
    return llm, tokenizer


# ---------------------------------------------------------------------------
# Results JSON helpers
# ---------------------------------------------------------------------------

def load_core_eval_results(path: Path = CORE_EVAL_RESULTS_PATH) -> dict[str, Any]:
    """Load existing core eval results JSON, returning an empty dict if absent."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_core_eval_results(results: dict[str, Any], path: Path = CORE_EVAL_RESULTS_PATH) -> None:
    """Merge ``results`` into the on-disk JSON and write.

    Args:
        results: New or updated results to merge.
        path: Path to the JSON file.
    """
    existing = load_core_eval_results(path)
    for model_name, layers in results.items():
        existing.setdefault(model_name, {})
        for layer_key, saes in layers.items():
            existing[model_name].setdefault(layer_key, {})
            existing[model_name][layer_key].update(saes)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"Core eval results saved to {path}")
