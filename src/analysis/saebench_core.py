"""SAEBench core evaluation reimplemented using HuggingFace.

Computes the same metrics as SAEBench core without depending on TransformerLens.
Supports both SMIXAE checkpoints and GemmaScope baselines loaded via SAELens.

Metrics computed (matching SAEBench definitions):

  l0                  Mean active features per token (count of non-zero elements).
  mse                 Normalised reconstruction error: mean ||x - x̂||² / ||x||² per token.
  explained_variance  1 - residual_var / total_var  (correct formula, not legacy).
  cosine_similarity   Mean cosine similarity between reconstruction and input.
  l2_ratio            Mean ||x̂|| / ||x|| per token.
  ce_loss_without_sae Baseline cross-entropy loss.
  ce_loss_with_sae    CE loss when the hook layer output is replaced by the SAE reconstruction.
  ce_loss_with_ablation CE loss when the hook layer output is zero-ablated.
  ce_loss_score       (ce_ablation - ce_sae) / (ce_ablation - ce_without_sae).

Results are saved to ``results/saebench_results.json`` with the hierarchy::

    {
      "<model_name>": {
        "layer_<N>": {
          "<human-readable SAE name>": { ...metrics... },
          ...
        }
      }
    }

GemmaScope baseline paths are specified in ``GEMMASCOPE_BASELINES`` and can be
overridden at runtime via CLI flags. For width-16k SAEs the available average L0
buckets for each layer are documented in the GemmaScope paper; the defaults here
use the bucket closest to 192 (= 3 × k_experts with k_experts=64).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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

# ---------------------------------------------------------------------------
# GemmaScope baseline SAE paths (SAELens release / sae_id pairs).
# Each value is a dict mapping layer index (int) to (release, sae_id).
# Override via --gemmascope-sae-id at runtime if your target L0 differs.
# ---------------------------------------------------------------------------
GEMMASCOPE_BASELINES: dict[str, dict[int, tuple[str, str]]] = {
    "google/gemma-2-2b": {
        12: (
            "gemma-scope-2b-pt-res",
            "layer_12/width_16k/average_l0_176",
        ),
    },
    "google/gemma-2-9b": {
        11: (
            "gemma-scope-9b-pt-res",
            "layer_11/width_16k/average_l0_131",
        ),
        20: (
            "gemma-scope-9b-pt-res",
            "layer_20/width_16k/average_l0_131",
        ),
    },
}

# Human-readable display names for GemmaScope baselines in the results JSON.
GEMMASCOPE_DISPLAY_NAMES: dict[str, dict[int, str]] = {
    "google/gemma-2-2b": {
        12: "GemmaScope 2B 16k (L0≈176)",
    },
    "google/gemma-2-9b": {
        11: "GemmaScope 9B 16k (L0≈131)",
        20: "GemmaScope 9B 16k (L0≈131)",
    },
}

SAEBENCH_RESULTS_PATH = Path("results/saebench_results.json")


@dataclass
class SAEBenchCoreConfig:
    """Configuration for a SAEBench core evaluation run."""

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


@dataclass
class SAEBenchRunSpec:
    """Specifies a single (experiment, SAE) pair to evaluate."""

    experiment_name: str
    model_name: str
    hook_name: str  # HuggingFace style, e.g. "model.layers.11"
    hook_layer: int
    checkpoint_path: str | None = None  # Path to SMIXAE checkpoint; None for GemmaScope
    gemmascope_release: str | None = None
    gemmascope_sae_id: str | None = None
    display_name: str = ""  # Human-readable name in the results JSON
    baselines: list[SAEBenchRunSpec] = field(default_factory=list)  # GemmaScope comparisons


# ---------------------------------------------------------------------------
# SAE wrappers
# ---------------------------------------------------------------------------

class SMIXAEBenchWrapper(nn.Module):
    """Wraps SMIXAE for SAEBench: flattens (batch, seq, n_experts, d_bottleneck) to (batch, seq, d_sae).

    SMIXAE's ``encode()`` returns a 3-D bottleneck tensor.  SAEBench and our
    core eval expect a flat 2-D feature vector.  This wrapper reshapes in both
    directions so the rest of the evaluation code sees a standard interface.
    """

    def __init__(self, smixae_model: Any) -> None:
        super().__init__()
        self._sae = smixae_model
        self._n = smixae_model.cfg.n_experts
        self._d = smixae_model.cfg.d_bottleneck
        self.d_sae = self._n * self._d

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to flat feature space.

        Args:
            x: ``(batch, seq, d_model)`` or ``(batch, d_model)``.

        Returns:
            ``(batch, seq, d_sae)`` or ``(batch, d_sae)`` float tensor.
        """
        if x.ndim == 3:
            b, s, d = x.shape
            flat = x.reshape(b * s, d)
            acts = self._sae.encode(flat.to(self._sae.device))  # (b*s, n, d_b)
            return acts.flatten(-2, -1).reshape(b, s, self.d_sae)
        acts = self._sae.encode(x.to(self._sae.device))  # (b, n, d_b)
        return acts.flatten(-2, -1)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode flat features back to residual stream.

        Args:
            feature_acts: ``(batch, seq, d_sae)`` or ``(batch, d_sae)``.

        Returns:
            Same leading dims with ``d_model`` as last dimension.
        """
        if feature_acts.ndim == 3:
            b, s, _ = feature_acts.shape
            unflat = feature_acts.reshape(b * s, self._n, self._d)
            out = self._sae.decode(unflat.to(self._sae.device))
            return out.reshape(b, s, -1)
        unflat = feature_acts.reshape(feature_acts.shape[0], self._n, self._d)
        return self._sae.decode(unflat.to(self._sae.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode."""
        return self.decode(self.encode(x))

    @property
    def device(self) -> torch.device:
        """Device of the underlying SAE."""
        return self._sae.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the underlying SAE."""
        return self._sae.dtype


class GemmaScopeBenchWrapper(nn.Module):
    """Thin wrapper around a SAELens SAE for use in the core eval.

    SAELens SAEs already expose a standard ``encode`` / ``decode`` interface.
    This wrapper just ensures consistent device placement.
    """

    def __init__(self, sae_lens_sae: Any) -> None:
        super().__init__()
        self._sae = sae_lens_sae
        self.d_sae: int = sae_lens_sae.cfg.d_sae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations."""
        return self._sae.encode(x.to(self._sae.device))

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode feature activations."""
        return self._sae.decode(feature_acts.to(self._sae.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode."""
        return self.decode(self.encode(x))

    @property
    def device(self) -> torch.device:
        """Device of the underlying SAE."""
        return self._sae.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the underlying SAE."""
        return self._sae.dtype


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
    """Stream OpenWebText and pack into fixed-length token batches.

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

    buffer: list[int] = []
    chunks: list[list[int]] = []

    for example in ds:
        buffer.extend(tokenizer.encode(example["text"], add_special_tokens=False))
        while len(buffer) >= context_size:
            chunks.append(buffer[:context_size])
            buffer = buffer[context_size:]
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
    """Run a forward pass and collect the hidden states at ``hook_name``.

    Args:
        model: HuggingFace language model.
        tokens: ``(batch, seq)`` input token IDs.
        hook_name: Dot-separated submodule path, e.g. ``"model.layers.11"``.

    Returns:
        Hidden states tensor ``(batch, seq, d_model)``.
    """
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
    sae: nn.Module,
    model: nn.Module,
    token_batches: list[torch.Tensor],
    hook_name: str,
    device: str,
    llm_dtype: torch.dtype,
    verbose: bool = False,
) -> dict[str, float]:
    r"""Compute sparsity and reconstruction quality metrics.

    Args:
        sae: Wrapped SAE (``SMIXAEBenchWrapper`` or ``GemmaScopeBenchWrapper``).
        model: HuggingFace LLM.
        token_batches: Tokenised input batches.
        hook_name: Hook point in HuggingFace notation.
        device: Compute device string.
        llm_dtype: Dtype to cast activations to before encoding.
        verbose: Show progress bar.

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

    # For the correct explained-variance formula.
    sum_sq_list: list[torch.Tensor] = []
    mean_per_dim_list: list[torch.Tensor] = []
    resid_sq_list: list[torch.Tensor] = []

    batch_iter = tqdm(token_batches, desc="Sparsity/variance", leave=False) if verbose else token_batches

    for tokens in batch_iter:
        tokens = tokens.to(device)
        original = _collect_acts(model, tokens, hook_name).to(llm_dtype)
        b, s, d = original.shape

        with torch.no_grad():
            feat = sae.encode(original).to(device)           # (b, s, d_sae)
            recon = sae.decode(feat).to(device, llm_dtype)   # (b, s, d_model)

        flat_in = original.reshape(-1, d)
        flat_out = recon.reshape(-1, d)
        flat_feat = feat.reshape(-1, feat.shape[-1])

        # L0: count non-zero feature activations per token.
        l0_list.append((flat_feat != 0).float().sum(-1))

        # Normalised MSE: per-token ||x - x̂||² / ||x||².
        resid = flat_in - flat_out
        mse_list.append(resid.pow(2).sum(-1) / (flat_in.pow(2).sum(-1) + 1e-8))

        # Explained variance accumulators.
        sum_sq_list.append(flat_in.pow(2).sum(-1).mean())
        mean_per_dim_list.append(flat_in.pow(2).mean(0))
        resid_sq_list.append(resid.pow(2).sum(-1).mean())

        # Cosine similarity.
        cossim_list.append(F.cosine_similarity(flat_in, flat_out, dim=-1))

        # L2 norms and ratio.
        l2_in = flat_in.norm(dim=-1)
        l2_out = flat_out.norm(dim=-1)
        l2_in_list.append(l2_in)
        l2_out_list.append(l2_out)
        l2_ratio_list.append(l2_out / (l2_in + 1e-8))

    # Aggregate.
    mean_sum_sq = torch.stack(sum_sq_list).mean()
    mean_per_dim = torch.stack(mean_per_dim_list).mean(0)
    total_var = mean_sum_sq - mean_per_dim.pow(2).sum()
    mean_resid_sq = torch.stack(resid_sq_list).mean()
    explained_var = (1.0 - mean_resid_sq / total_var).item()

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
    sae: nn.Module,
    model: nn.Module,
    token_batches: list[torch.Tensor],
    hook_name: str,
    device: str,
    llm_dtype: torch.dtype,
    verbose: bool = False,
) -> dict[str, float]:
    r"""Compute CE-loss metrics: baseline, with-SAE, and zero-ablation.

    Args:
        sae: Wrapped SAE.
        model: HuggingFace LLM.
        token_batches: Tokenised input batches.
        hook_name: Hook point in HuggingFace notation.
        device: Compute device string.
        llm_dtype: Dtype for activations.
        verbose: Show progress bar.

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
        shift_labels = tokens[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.reshape(-1, vocab_size),
            shift_labels.reshape(-1),
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
        recon = sae.decode(sae.encode(acts.to(llm_dtype))).to(acts.device, acts.dtype)
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
    # Higher = better; 1.0 means perfect recovery, 0.0 = as bad as zero ablation.
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
    sae: nn.Module,
    model: nn.Module,
    tokenizer: Any,
    hook_name: str,
    cfg: SAEBenchCoreConfig,
    verbose: bool = False,
) -> dict[str, float]:
    """Run the full SAEBench core evaluation for a single SAE.

    Args:
        sae: Wrapped SAE with flat ``encode`` / ``decode``.
        model: HuggingFace LLM (already on ``cfg.device``).
        tokenizer: HuggingFace tokenizer.
        hook_name: HuggingFace hook point, e.g. ``"model.layers.11"``.
        cfg: Evaluation configuration.
        verbose: Enable tqdm progress bars.

    Returns:
        Dict of scalar metric values.
    """
    llm_dtype = cfg.llm_dtype()
    n_total = cfg.n_reconstruction_batches + cfg.n_sparsity_batches
    logger.info(f"Tokenising {n_total} batches × {cfg.batch_size} seqs × {cfg.context_size} tokens …")
    all_batches = _build_token_batches(
        cfg.dataset, tokenizer, cfg.context_size, n_total, cfg.batch_size
    )
    recon_batches = all_batches[: cfg.n_reconstruction_batches]
    sparsity_batches = all_batches[cfg.n_reconstruction_batches :]

    logger.info("Computing sparsity/variance metrics …")
    metrics = _compute_sparsity_variance_metrics(
        sae, model, sparsity_batches, hook_name, cfg.device, llm_dtype, verbose=verbose
    )

    logger.info("Computing CE-loss metrics …")
    metrics.update(
        _compute_ce_loss_metrics(
            sae, model, recon_batches, hook_name, cfg.device, llm_dtype, verbose=verbose
        )
    )

    return metrics


def _load_smixae_wrapper(checkpoint_path: str, device: str) -> SMIXAEBenchWrapper:
    """Load a SMIXAE checkpoint and wrap it for SAEBench evaluation."""
    sae = load_sae(checkpoint_path, device=device)
    return SMIXAEBenchWrapper(sae)


def _load_gemmascope_wrapper(
    release: str,
    sae_id: str,
    device: str,
) -> GemmaScopeBenchWrapper:
    """Load a GemmaScope SAE via SAELens and wrap it for SAEBench evaluation."""
    from sae_lens import SAE as SAELensModel  # imported here to avoid hard dep at module load

    sae, _, _ = SAELensModel.from_pretrained(release=release, sae_id=sae_id, device=device)
    return GemmaScopeBenchWrapper(sae)


def evaluate_experiment(
    experiment_name: str,
    model_name: str,
    hook_name: str,
    checkpoint_path: str,
    cfg: SAEBenchCoreConfig,
    baseline_sae_ids: dict[str, tuple[str, str]] | None = None,
    verbose: bool = False,
) -> dict[str, dict[str, float]]:
    """Run core eval on a SMIXAE checkpoint and optional GemmaScope baselines.

    Args:
        experiment_name: Human-readable label, e.g. ``"gemma_2_9b_l11"``.
        model_name: HuggingFace model identifier.
        hook_name: Hook point, e.g. ``"model.layers.11"``.
        checkpoint_path: Path to the SMIXAE ``model/`` directory.
        cfg: Evaluation config.
        baseline_sae_ids: Optional dict mapping display name to (release, sae_id)
            for GemmaScope baselines.  Defaults to ``GEMMASCOPE_BASELINES``.
        verbose: Enable progress bars.

    Returns:
        Dict mapping display name → metrics dict.
    """
    from analysis.utils import load_llm  # avoid circular import at module level

    logger.info(f"Loading LLM {model_name} …")
    llm, tokenizer = load_llm(model_name, device=cfg.device, dtype=cfg.llm_dtype())
    llm.eval()

    layer = int(hook_name.split(".")[-1])

    results: dict[str, dict[str, float]] = {}

    # --- SMIXAE ---
    logger.info(f"Evaluating SMIXAE from {checkpoint_path} …")
    smixae_wrapper = _load_smixae_wrapper(checkpoint_path, cfg.device)
    results["SMIXAE"] = run_core_eval(smixae_wrapper, llm, tokenizer, hook_name, cfg, verbose=verbose)
    logger.info(f"SMIXAE: {results['SMIXAE']}")

    # --- GemmaScope baselines ---
    if baseline_sae_ids is None:
        defaults = GEMMASCOPE_BASELINES.get(model_name, {})
        display_defaults = GEMMASCOPE_DISPLAY_NAMES.get(model_name, {})
        baseline_sae_ids = {
            display_defaults.get(lyr, "GemmaScope 16k"): (rel, sid)
            for lyr, (rel, sid) in defaults.items()
            if lyr == layer
        }

    for display_name, (release, sae_id) in baseline_sae_ids.items():
        logger.info(f"Loading GemmaScope baseline: {release} / {sae_id} …")
        try:
            gs_wrapper = _load_gemmascope_wrapper(release, sae_id, cfg.device)
            results[display_name] = run_core_eval(
                gs_wrapper, llm, tokenizer, hook_name, cfg, verbose=verbose
            )
            logger.info(f"{display_name}: {results[display_name]}")
        except Exception as exc:
            logger.error(f"Failed to load/eval GemmaScope SAE {release}/{sae_id}: {exc}")
            results[display_name] = {"error": str(exc)}

    return results


# ---------------------------------------------------------------------------
# Results JSON helpers
# ---------------------------------------------------------------------------

def load_saebench_results(path: Path = SAEBENCH_RESULTS_PATH) -> dict[str, Any]:
    """Load existing SAEBench results JSON, returning an empty dict if absent."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_saebench_results(results: dict[str, Any], path: Path = SAEBENCH_RESULTS_PATH) -> None:
    """Merge ``results`` into the on-disk JSON and write atomically.

    Args:
        results: New or updated results to merge.
        path: Path to the JSON file.
    """
    existing = load_saebench_results(path)
    # Deep-merge: model → layer → sae_name
    for model_name, layers in results.items():
        existing.setdefault(model_name, {})
        for layer_key, saes in layers.items():
            existing[model_name].setdefault(layer_key, {})
            existing[model_name][layer_key].update(saes)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"SAEBench results saved to {path}")


def run_all_from_results_json(
    results_json_path: Path,
    checkpoint_base_dir: Path,
    cfg: SAEBenchCoreConfig,
    output_path: Path = SAEBENCH_RESULTS_PATH,
    force_rerun: bool = False,
    verbose: bool = False,
    gemmascope_overrides: dict[str, dict[int, tuple[str, str, str]]] | None = None,
) -> dict[str, Any]:
    """Evaluate all experiments listed in ``results.json``.

    Reads ``model_name`` and ``hook_name`` from ``results.json``, locates the
    SMIXAE checkpoint at ``<checkpoint_base_dir>/<experiment_name>/model``, and
    runs core eval on each.  GemmaScope baselines are loaded from
    ``GEMMASCOPE_BASELINES`` unless overridden by ``gemmascope_overrides``.

    Args:
        results_json_path: Path to existing ``results.json``.
        checkpoint_base_dir: Root directory that contains per-experiment subdirs.
        cfg: Evaluation config.
        output_path: Where to write ``saebench_results.json``.
        force_rerun: Re-evaluate even if results already exist.
        verbose: Enable tqdm progress bars.
        gemmascope_overrides: Optional ``{model_name: {layer: (release, sae_id, display_name)}}``
            to override the default GemmaScope comparison SAEs.

    Returns:
        The full merged saebench_results dict.
    """
    with open(results_json_path) as f:
        experiments: dict[str, Any] = json.load(f)

    existing = load_saebench_results(output_path)
    all_results: dict[str, Any] = {}

    for exp_name, exp_data in experiments.items():
        model_name: str = exp_data["model_name"]
        hook_name: str = exp_data["hook_name"]
        layer: int = int(hook_name.split(".")[-1])
        layer_key = f"layer_{layer}"
        checkpoint_path = checkpoint_base_dir / exp_name / "model"

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found for {exp_name}: {checkpoint_path} — skipping")
            continue

        # Skip if already evaluated and not forcing rerun.
        if not force_rerun:
            already_done = (
                model_name in existing
                and layer_key in existing[model_name]
                and "SMIXAE" in existing[model_name][layer_key]
            )
            if already_done:
                logger.info(f"Skipping {exp_name} (already in {output_path})")
                all_results.setdefault(model_name, {}).setdefault(layer_key, {}).update(
                    existing[model_name][layer_key]
                )
                continue

        # Build baseline_sae_ids for this layer.
        baseline_sae_ids: dict[str, tuple[str, str]] = {}
        overrides_for_model = (gemmascope_overrides or {}).get(model_name, {})
        if layer in overrides_for_model:
            rel, sid, dname = overrides_for_model[layer]
            baseline_sae_ids[dname] = (rel, sid)
        else:
            defaults = GEMMASCOPE_BASELINES.get(model_name, {})
            display_defaults = GEMMASCOPE_DISPLAY_NAMES.get(model_name, {})
            if layer in defaults:
                rel, sid = defaults[layer]
                dname = display_defaults.get(layer, "GemmaScope 16k")
                baseline_sae_ids[dname] = (rel, sid)

        logger.info(f"=== Evaluating {exp_name} ({model_name} {hook_name}) ===")
        layer_results = evaluate_experiment(
            experiment_name=exp_name,
            model_name=model_name,
            hook_name=hook_name,
            checkpoint_path=str(checkpoint_path),
            cfg=cfg,
            baseline_sae_ids=baseline_sae_ids,
            verbose=verbose,
        )

        all_results.setdefault(model_name, {}).setdefault(layer_key, {}).update(layer_results)

        # Save incrementally after each experiment.
        save_saebench_results({model_name: {layer_key: layer_results}}, output_path)

    return all_results
