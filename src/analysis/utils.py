"""Shared utilities for SMIXAE analysis scripts."""
from __future__ import annotations

import gc
from typing import Any, Iterable

import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from loguru import logger
from plotly.graph_objects import Figure
from sae_lens import SAE
from smixae import SMIXAE
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


# ── GPU memory ────────────────────────────────────────────────────────────────


def gpu_mem_mb() -> str:
    if not torch.cuda.is_available():
        return ""
    cur = torch.cuda.memory_allocated() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    return f"[GPU {cur:.0f}/{peak:.0f} MB]"


def flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ── Model / SAE loading ───────────────────────────────────────────────────────


def load_llm(
    model_name: str,
    device: str,
    dtype: torch.dtype | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load an AutoModelForCausalLM and its tokenizer.

    Sets ``pad_token = eos_token`` when no pad token is defined.
    Padding side is left to the caller to configure.
    """
    if dtype is None:
        dtype = torch.float32
    logger.info(f"Loading {model_name}  {gpu_mem_mb()}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device).eval()
    logger.info(f"Loaded {model_name}  {gpu_mem_mb()}")
    return model, tokenizer


def load_sae(checkpoint_path: str, device: str) -> SMIXAE:
    """Load a SMIXAE checkpoint from disk."""
    logger.info(f"Loading SAE from {checkpoint_path}  {gpu_mem_mb()}")
    sae = SAE.load_from_disk(path=checkpoint_path, device=device)
    logger.info(f"SAE threshold: {sae.threshold}  {gpu_mem_mb()}")
    return sae  # type: ignore


# ── Activation collection ─────────────────────────────────────────────────────


def collect_hook_activations(
    model: PreTrainedModel,
    hook_name: str,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: str | torch.device,
    desc: str = "Collecting activations",
) -> list[torch.Tensor]:
    """Register a forward hook at ``hook_name`` and collect hidden states over batches.

    Args:
        model:     HuggingFace causal LM.
        hook_name: Dotted submodule path, e.g. ``"model.layers.20"``.
        batches:   Iterable of ``(input_ids, attention_mask)`` tensors on any device;
                   moved to ``device`` internally.
        device:    Target device for inference.
        desc:      tqdm description string.

    Returns:
        One ``(batch_size, seq_len, d_model)`` CPU tensor per input batch.
    """
    _store: dict[str, torch.Tensor] = {}

    def _hook(_module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        _store["h"] = h.detach().cpu()

    handle = model.get_submodule(hook_name).register_forward_hook(_hook)
    results: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for ids, mask in tqdm(batches, desc=desc):
                ids = ids.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                model(input_ids=ids, attention_mask=mask)
                results.append(_store["h"])
                del ids, mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        handle.remove()
    return results


# ── SAE encoding ──────────────────────────────────────────────────────────────


def encode_sae_batched(
    sae: SMIXAE,
    hiddens: torch.Tensor,
    batch_size: int = 4096,
    desc: str = "SAE encode",
) -> torch.Tensor:
    """Encode a flat ``(N, d_model)`` tensor through the SAE in batches.

    Args:
        sae:        SMIXAE model (on any device).
        hiddens:    ``(N, d_model)`` CPU tensor of LLM hidden states.
        batch_size: Number of tokens per SAE forward pass.
        desc:       tqdm description string.

    Returns:
        ``(N, n_experts, d_bottleneck)`` CPU float32 tensor.
    """
    device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype
    chunks: list[torch.Tensor] = []
    sae.eval()
    with torch.no_grad():
        for b in tqdm(hiddens.split(batch_size), desc=desc):
            b = b.to(device=device, dtype=sae_dtype)
            chunks.append(sae.encode(b).float().cpu())
            del b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


# ── Expert ────────────────────────────────────────────────────────────────────


class Expert:
    def __init__(
        self,
        active_mask: torch.Tensor,
        expert_id: int,
        llm_activations: torch.Tensor | None,
        expert_activations: torch.Tensor,
        labels: torch.Tensor | None = None,
        seq_positions: torch.Tensor | None = None,
        n_classes: int = 0,
    ):
        self.expert_activations = expert_activations[active_mask].float().cpu()
        self.llm_activations = (
            llm_activations[active_mask].float().cpu()
            if llm_activations is not None
            else None
        )
        self.labels = labels[active_mask].long().cpu() if labels is not None else None
        self.seq_positions = seq_positions
        self.n_classes = n_classes

        self.expert_id = expert_id
        self.active_indices = active_mask.nonzero().cpu().tolist()

        # Metrics
        self.local_continuity_scores: torch.Tensor | None = None
        self.fisher_score: float | None = None

    # ── accessors ─────────────────────────────────────────────────────
    @property
    def mean_continuity(self) -> float | None:
        if self.local_continuity_scores is None:
            return None
        return float(self.local_continuity_scores.mean().item())

    @property
    def n_unique_labels(self) -> int | None:
        if self.labels is None:
            return None
        return int(self.labels.unique().numel())

    @property
    def adjusted_fisher_score(self) -> float | None:
        if (
            self.fisher_score is None
            or self.n_unique_labels is None
            or self.n_classes == 0
        ):
            return None
        return self.fisher_score * (self.n_unique_labels / self.n_classes)

    def sort_key(self, sort_by: str) -> float:
        mapping: dict[str, float | None] = {
            "fisher": self.fisher_score,
            "adjusted_fisher": self.adjusted_fisher_score,
            "continuity": self.mean_continuity,
        }
        if sort_by not in mapping:
            raise ValueError(
                f"Unknown sort_by: {sort_by}. Options: {', '.join(mapping.keys())}"
            )
        v = mapping[sort_by]
        return v if v is not None else float("-inf")

    # ── continuity (unlabelled) ───────────────────────────────────────
    def evaluate_manifold(
        self, k_neighbors: int = 10, device: str = "cuda"
    ) -> torch.Tensor:
        if self.llm_activations is None:
            raise ValueError(
                "evaluate_manifold requires llm_activations, which was not provided."
            )
        expert_acts_gpu = self.expert_activations.to(device)
        llm_acts_gpu = self.llm_activations.to(device)

        dists = torch.cdist(expert_acts_gpu, expert_acts_gpu)
        _, indices = torch.topk(dists, k=k_neighbors + 1, largest=False)
        indices = indices[:, 1:]
        del dists

        llm_normed = F.normalize(llm_acts_gpu, p=2, dim=-1)
        llm_neighbors = llm_normed[indices]
        center = llm_normed.unsqueeze(1)
        sims = (center * llm_neighbors).sum(dim=-1)
        self.local_continuity_scores = sims.mean(dim=-1).cpu()

        del expert_acts_gpu, llm_acts_gpu, llm_normed, llm_neighbors, center, sims
        torch.cuda.empty_cache()

        return self.local_continuity_scores

    # ── multivariate fisher score (labelled) ──────────────────────────
    def evaluate_fisher(self) -> float:
        """
        Multivariate Fisher discriminant ratio: tr(S_W^{-1} S_B)

        Measures how well the 3D bottleneck separates classes.
        Works for clusters, ordered clusters, and rings — anything
        where different labels occupy different regions of the space.
        """
        if self.labels is None:
            self.fisher_score = 0.0
            return 0.0

        X = self.expert_activations  # (N, D)
        y = self.labels  # (N,)
        _, D = X.shape

        overall_mean = X.mean(dim=0)  # (D,)

        S_W = torch.zeros(D, D)
        S_B = torch.zeros(D, D)

        classes = y.unique()

        for c in classes:
            mask = y == c
            X_c = X[mask]
            n_c = X_c.shape[0]

            if n_c < 2:
                continue

            mean_c = X_c.mean(dim=0)

            # Within-class scatter
            diff_w = X_c - mean_c  # (n_c, D)
            S_W += diff_w.T @ diff_w

            # Between-class scatter
            diff_b = (mean_c - overall_mean).unsqueeze(1)  # (D, 1)
            S_B += n_c * (diff_b @ diff_b.T)

        # Regularise S_W for numerical stability
        S_W += 1e-6 * torch.eye(D)

        try:
            S_W_inv = torch.linalg.inv(S_W)
            self.fisher_score = float(torch.trace(S_W_inv @ S_B).item())
        except torch.linalg.LinAlgError:
            # Fallback: ratio of traces
            tr_w = torch.trace(S_W).item()
            tr_b = torch.trace(S_B).item()
            self.fisher_score = tr_b / tr_w if tr_w > 1e-10 else 0.0

        return self.fisher_score

    # ── context windows ───────────────────────────────────────────────
    def get_context_windows(
        self, str_tokens: list[list[str]], context_window: int = 10
    ) -> list[str]:
        contexts = []
        for batch_idx, seq_idx in self.active_indices:
            seq = str_tokens[batch_idx]
            if self.seq_positions is not None:
                seq_idx = int(self.seq_positions[batch_idx].item())
            start = max(0, seq_idx - context_window)
            end = min(len(seq), seq_idx + context_window + 1)
            window = list(seq[start:end])
            target_rel_idx = seq_idx - start
            window[target_rel_idx] = f"<b>[{window[target_rel_idx]}]</b>"
            contexts.append("".join(window).replace("\n", "<br>"))
        return contexts

    # ── plotting ──────────────────────────────────────────────────────
    def _make_title(self) -> str:
        parts = [f"Expert {self.expert_id}  (n={self.expert_activations.shape[0]}"]
        if self.n_unique_labels is not None:
            parts.append(f"labels={self.n_unique_labels}")
        if self.fisher_score is not None:
            parts.append(f"fisher={self.fisher_score:.3f}")
        if self.adjusted_fisher_score is not None:
            parts.append(f"adj_fisher={self.adjusted_fisher_score:.3f}")
        if self.mean_continuity is not None:
            parts.append(f"cont={self.mean_continuity:.3f}")
        return ", ".join(parts) + ")"

    def get_plot(
        self,
        str_tokens: list[list[str]],
        k_neighbors: int = 10,
        context_window: int = 10,
        device: str = "cuda",
        label_names: dict[int, str] | None = None,
        continuous_color: bool = False,
        color_scale: str = "Plasma",
    ) -> Figure:
        # Lazy-evaluate continuity
        if self.local_continuity_scores is None:
            self.evaluate_manifold(k_neighbors=k_neighbors, device=device)

        contexts = self.get_context_windows(str_tokens, context_window=context_window)
        pts = self.expert_activations.numpy()

        hover_extra: dict[str, list[float] | bool] = {"Context": True}

        cont_list: list[float] = []
        if self.local_continuity_scores is not None:
            cont_list = self.local_continuity_scores.numpy().tolist()

        if self.labels is not None and label_names is not None:
            label_strs = [
                label_names.get(int(l.item()), str(l.item())) for l in self.labels
            ]
            label_ids = self.labels.tolist()
            df_dict: dict[str, Any] = {
                "x": pts[:, 0].tolist(),
                "y": pts[:, 1].tolist(),
                "z": pts[:, 2].tolist(),
                "Label": label_strs,
                "LabelId": label_ids,
                "Context": contexts,
            }
            if self.local_continuity_scores is not None:
                df_dict["Continuity"] = cont_list
                hover_extra["Continuity"] = True
            hover_extra.update({"x": False, "y": False, "z": False, "LabelId": False})

            df = pd.DataFrame(df_dict)
            if continuous_color:
                hover_extra["Label"] = True
                fig = px.scatter_3d(
                    df,
                    x="x",
                    y="y",
                    z="z",
                    color="LabelId",
                    color_continuous_scale=color_scale,
                    hover_data=hover_extra,
                    title=self._make_title(),
                    opacity=0.8,
                )
                n = len(label_names)
                fig.update_coloraxes(
                    colorbar=dict(
                        tickvals=list(range(n)),
                        ticktext=[label_names[i] for i in range(n)],
                    )
                )
            else:
                sorted_label_names = sorted(label_names.values())
                fig = px.scatter_3d(
                    df,
                    x="x",
                    y="y",
                    z="z",
                    color="Label",
                    category_orders={"Label": sorted_label_names},
                    hover_data=hover_extra,
                    title=self._make_title(),
                    opacity=0.8,
                )
                fig.update_layout(showlegend=True)
        else:
            color_col = "Continuity"
            color_vals: list[float] = (
                cont_list
                if self.local_continuity_scores is not None
                else [0.0] * pts.shape[0]
            )

            df_dict = {
                "x": pts[:, 0].tolist(),
                "y": pts[:, 1].tolist(),
                "z": pts[:, 2].tolist(),
                color_col: color_vals,
                "Context": contexts,
            }
            hover_extra.update({"x": False, "y": False, "z": False})

            df = pd.DataFrame(df_dict)
            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                color=color_col,
                color_continuous_scale="Viridis",
                hover_data=hover_extra,
                title=self._make_title(),
                opacity=0.8,
            )

        fig.update_traces(marker=dict(size=4))
        return fig

    def get_mean_plot(
        self,
        label_names: dict[int, str] | None = None,
        color_scale: str = "Plasma",
        continuous_color: bool = False,
    ) -> Figure | None:
        if self.labels is None or label_names is None:
            return None

        pts = self.expert_activations.numpy()
        xs, ys, zs, label_strs, label_ids, counts = [], [], [], [], [], []
        for c in range(self.n_classes):
            mask = (self.labels == c).numpy()
            if not mask.any():
                continue
            centroid = pts[mask].mean(axis=0)
            xs.append(float(centroid[0]))
            ys.append(float(centroid[1]))
            zs.append(float(centroid[2]))
            label_strs.append(label_names[c])
            label_ids.append(c)
            counts.append(int(mask.sum()))

        df = pd.DataFrame(
            {
                "x": xs,
                "y": ys,
                "z": zs,
                "Label": label_strs,
                "LabelId": label_ids,
                "Count": counts,
            }
        )

        if continuous_color:
            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                color="LabelId",
                color_continuous_scale=color_scale,
                size="Count",
                size_max=30,
                text="Label",
                hover_data={
                    "x": False,
                    "y": False,
                    "z": False,
                    "Label": True,
                    "Count": True,
                    "LabelId": False,
                },
                title=self._make_title() + " [class means]",
                opacity=0.9,
            )
            n = len(label_names)
            fig.update_coloraxes(
                colorbar=dict(
                    tickvals=list(range(n)),
                    ticktext=[label_names[i] for i in range(n)],
                )
            )
        else:
            sorted_label_names = sorted(label_names.values())
            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                color="Label",
                category_orders={"Label": sorted_label_names},
                size="Count",
                size_max=30,
                text="Label",
                hover_data={
                    "x": False,
                    "y": False,
                    "z": False,
                    "Label": True,
                    "Count": True,
                },
                title=self._make_title() + " [class means]",
                opacity=0.9,
            )
            fig.update_layout(showlegend=True)

        fig.update_traces(
            marker=dict(line=dict(width=1, color="DarkSlateGrey")),
            textposition="top center",
        )
        return fig
