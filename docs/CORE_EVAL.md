# SMIXAE Core SAE Evaluation (SAEBench Reimplementation)

Implemented in `src/analysis/core_eval.py` and exposed as `smixae core`. Reimplements SAEBench core metrics using HuggingFace directly — **no TransformerLens dependency**. One invocation evaluates a single SAE (SMIXAE checkpoint or any SAELens-compatible SAE loadable via `SAE.from_pretrained`) and merges results into `results/core_eval_results.json`.

---

## Metrics Computed

| Metric | Definition |
|--------|-----------|
| `l0` | Mean active features per token (count of non-zero elements in flat feature vector) |
| `fraction_alive` | Fraction of feature dimensions whose firing density (fraction of tokens where activation > 0) exceeds `min_firing_density` (default 1e-6) |
| `mse` | Normalised MSE: mean `‖x − x̂‖² / ‖x‖²` per token |
| `explained_variance` | `1 − mean(‖x−x̂‖²) / var(x)` (standard mean-centred form; denominator uses global per-feature mean) |
| `cosine_similarity` | Mean cosine similarity between reconstruction and input |
| `l2_ratio` | Mean `‖x̂‖ / ‖x‖` per token |
| `ce_loss_score` | `(CE_ablation − CE_SAE) / (CE_ablation − CE_orig)`, higher = better |
| `ce_loss_without_sae` | Baseline CE loss |
| `ce_loss_with_sae` | CE loss with SAE reconstruction patched in at hook point |
| `ce_loss_with_ablation` | CE loss with zero-ablation at hook point |

---

## SMIXAE Encode/Decode Adapter

SMIXAE's `encode()` returns `(batch, n_experts, d_bottleneck)`. The closures built by `_make_sae_fns` in `core_eval.py` flatten this to `(batch, n_experts × d_bottleneck)` for metric computation and unflatten before calling `decode()`. Inputs are cast to the SAE's parameter dtype so bfloat16 LLM activations work against float32 SAE weights (GemmaScope default). The effective L0 for SMIXAE is `k_experts × d_bottleneck` per token (e.g. 64 × 3 = 192 with default settings).

---

## GemmaScope Baselines

Loaded via SAELens `SAE.from_pretrained()`. Typical width-16k comparison paths:

| Model | Layer | SAELens release | sae_id |
|-------|-------|-----------------|--------|
| Gemma 2 2B | 12 | `gemma-scope-2b-pt-res` | `layer_12/width_16k/average_l0_176` |
| Gemma 2 9B | 11 | `gemma-scope-9b-pt-res` | `layer_11/width_16k/average_l0_118` |
| Gemma 2 9B | 20 | `gemma-scope-9b-pt-res` | `layer_20/width_16k/average_l0_138` |

Passed as `--hf-release` / `--hf-sae-id` to `smixae core`. See `experiments/core_eval.sh` for the full set of invocations used in the paper.

---

## Results JSON Structure

`results/core_eval_results.json` has the hierarchy **model → layer → SAE name → metrics**. Human-readable SAE names are used as keys (e.g. `"SMIXAE"`, `"GemmaScope 9B 16k (L0=118)"`). Re-running an evaluation with the same name overwrites the metrics block; different names coexist under the same layer.

```json
{
  "google/gemma-2-9b": {
    "layer_11": {
      "SMIXAE": { "l0": 191.2, "ce_loss_score": 0.89, ... },
      "GemmaScope 9B 16k (L0=118)": { "l0": 118.3, "ce_loss_score": 0.92, ... }
    }
  }
}
```

---

## `src/analysis/core_eval.py`

- **`CoreEvalConfig`**: Dataclass — `dataset`, `context_size`, `n_reconstruction_batches` (default 200), `n_sparsity_batches` (default 500), `batch_size`, `device`, `dtype`, `min_firing_density` (default 1e-6).
- **`_make_sae_fns(sae)`**: Returns `(encode, decode, d_sae)` closures; SMIXAE detected via `hasattr(sae.cfg, "n_experts")` and flattened automatically.
- **`_make_act_store(...)`**: Tokenises a streaming HuggingFace dataset on the fly (BOS prepended, short docs filtered), wraps it as a pretokenized ``IterableDataset``, and creates an SAELens ``ActivationsStore``. Call ``get_batch_tokens(n)`` to stream ``(n, context_size)`` batches. No TransformerLens adapter code needed.
- **`_compute_sparsity_variance_metrics(...)`**: L0, MSE, explained variance, cosine similarity, L2 ratio/norms. Metric math runs in float32 for precision over `d_model` dims.
- **`_compute_ce_loss_metrics(...)`**: `ce_loss_without_sae`, `ce_loss_with_sae` (SAE reconstruction patched in via forward hook), `ce_loss_with_ablation` (zero-ablation at hook point), and the derived `ce_loss_score`.
- **`run_core_eval(...)`**: End-to-end metrics for one SAE; called by `run_single_eval(sae, ...)` after building the closures.
- **`load_sae_from_path(path, device)`** / **`load_sae_from_hf(release, sae_id, device)`**: Source-selector helpers.
- **`save_core_eval_results(results, path)`**: Deep-merges and writes `core_eval_results.json`.

---

## `src/cli/core_eval.py`

Exposes `smixae core` with a single positional `checkpoint_path` argument (mutually exclusive with `--hf-release` + `--hf-sae-id`). Always writes to `results/core_eval_results.json` unless `--output-json` is overridden, keyed by `base_model_name` / derived `layer_{N}` / `display-name`.

---

## `experiments/core_eval.sh`

Self-contained batch script that runs `smixae core` for each trained experiment plus its GemmaScope 16k baseline. Chain with `bash experiments/core_eval.sh --verbose` to enable per-batch progress bars.
