# Synthetic Toy-Model Benchmark for SMIXAE

## Context

Reviewers will want to see SMIXAE evaluated on a controlled synthetic benchmark where the
ground-truth manifold structure is known. A recent (unpublished) paper proposes such a
benchmark — see `plans/toymodel.md` — built from a sparse mixture of low-dimensional
manifolds embedded in a high-dimensional ambient space, then evaluated by per-manifold
reconstruction quality.

Goal: implement the data pipeline (using `sae_lens.synthetic.FeatureDictionary` and
`ActivationGenerator`), train SMIXAE across a sweep of expert budgets on this synthetic
data, and emit a single bundle of artifacts (CSV, metric plot, best-model bottleneck
visualizations) that supports both an in-paper figure and an in-paper table.

Because SMIXAE's bottleneck is 3D, manifolds whose ambient embedding dimension `k_i > 3`
are excluded from the zoo. From Table 4 in `plans/toymodel.md`, this excludes the **Torus**
(Clifford embedding requires `k=4`); all other 7 types are kept.

---

## Manifold Zoo (7 types × 6 variants = 42 instances)

| Type       | d_i | k_i | Notes |
|------------|-----|-----|-------|
| Circle     | 1   | 2   | `r ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 3.0}` |
| Sphere     | 2   | 3   | same r set |
| Möbius     | 2   | 3   | width `w ∈ {0.2, 0.3, 0.5, 0.7, 1.0, 1.5}` |
| Swiss roll | 2   | 3   | `(θ_max, h_max)` set per paper |
| Helix      | 1   | 3   | `α ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}` over `θ ∈ [0, 4π]` |
| Flat disk  | 2   | 2   | `R ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 3.0}` (sample `r ~ R√U(0,1)`) |
| Segment    | 1   | 1   | `length ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 3.0}` |

Total atoms across all instances: `Σ k_i = 6·(2+3+3+3+3+2+1) = 102`.

**Per-instance normalization**: draw 50,000 calibration points from raw embedding `γ_i(θ)`,
compute `μ_i` and `σ_i = √E[‖γ - μ‖²]`, store the normalized embedding
`γ̃_i(θ) = (γ_i(θ) - μ_i) / σ_i` (unit RMS norm).

**Ambient embedding `V_i ∈ R^{k_i × d}`** (with `d = 128`): sample a `d × k_i` Gaussian
matrix and take the Q-factor of its QR decomposition (transposed) — orthonormal rows per
instance. Concatenate all 42 `V_i` matrices into a single
`feature_vectors ∈ R^{102 × 128}` matrix used as the `FeatureDictionary`.

---

## Data Pipeline (`src/analysis/synthetic.py`)

A single new module with all manifold + sampling logic:

1. **`build_manifold_zoo(d_in=128, seed=...) -> ManifoldZoo`** — constructs the 42 manifold
   instances with their normalized samplers `γ̃_i: θ → R^{k_i}`, then builds a
   `FeatureDictionary(num_features=102, hidden_dim=d_in, initializer=None)` and overwrites
   `feature_vectors.data` with the concatenated per-instance QR-orthonormal `V_i` rows
   (paper-faithful: `V_i` rows are orthonormal within instance, **not** jointly across
   instances). Records `atom_offset[i]` and `atom_count[i] = k_i` so we know which atom
   slots belong to manifold `i`.

2. **`SparseManifoldSampler(zoo, l0=4, sigma_eps=1e-5, device, mode="exact")`**
   — uses `ActivationGenerator(num_features=42, firing_probabilities=l0/42, std_firing=0,
   mean_firing=1)` as the *manifold-instance selection* layer (provides the configurable
   firing structure SAELens already exposes; supports correlation matrices for free if a
   future variant wants it). Two sampling modes:
   - `"expected"`: use the AG output directly → expected L0 = `l0`, variable per-sample.
   - `"exact"` (default for both train and eval to match the paper's `|S| = L0`): start
     from AG mask but resample-then-truncate to enforce exactly `l0` active manifolds per
     row via a `modify_activations` callback that calls `torch.multinomial`.

   For each sample, the active-manifold mask is converted to `feature_acts ∈ R^{B × 102}`:
   for each active manifold, draw `θ` uniformly from its parameter range, compute
   `z = γ̃_i(θ) ∈ R^{k_i}`, and write `z` into the corresponding atom slots. Pass
   `feature_acts` through the `FeatureDictionary` to obtain `x = feature_acts @ V`. Add
   `ε ~ N(0, σ_eps² I_d)` with `σ_eps = 1e-5`.

3. **`generate_eval_set(zoo, n_samples=200_000, l0=4, seed=...)` → dict** with:
   - `x ∈ R^{N × 128}` (the inputs)
   - `active_mask ∈ {0,1}^{N × 42}` (which manifolds fired per sample)
   - `m ∈ R^{N × 42 × 128}` (per-manifold ground-truth contributions `γ̃_i V_i`, zero where
     inactive). Stored sparsely / lazily because dense storage is heavy — concretely we
     store only the active rows: a flat `(row_idx, manifold_idx, contribution)` table.

   Eval set size kept at 200k by default (not the paper's 1M) to fit a single-machine
   bundled run; CLI flag `--eval-samples` lets the user override.

---

## Training Pipeline (`src/cli/synthetic.py`)

SMIXAE training already exists in `src/smixae/smixae.py:194` (`SMIXAETraining`). The
existing `smixae train` CLI consumes activations through SAELens `ActivationsStore` (LLM
forward) and is unsuitable for raw `[B, 128]` synthetic batches. We write a small
hand-rolled loop instead:

```python
cfg = SMIXAETrainingConfig(
    architecture="smixae", d_in=128, d_sae=n_experts*d_expert,  # set automatically
    n_experts=n_experts, d_expert=d_expert, d_bottleneck=3,
    k_experts=k_experts, normalize_activations="none",
    apply_b_dec_to_input=False,
    metadata=SAEMetadata(model_name="synthetic_toy", hook_name="ambient", ...),
)
model = SMIXAETraining(cfg).to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
for step in range(n_steps):
    x = sampler.sample(batch_size)       # [B, 128]
    out = model.training_forward_pass(TrainStepInput(
        sae_in=x, coefficients={}, dead_neuron_mask=None,
        n_training_steps=step, is_logging_step=(step % 100 == 0),
    ))
    optim.zero_grad(); out.loss.backward(); optim.step()
```

After training: `model.save_inference_model(out_dir/"model")` writes the inference SMIXAE
in standard SAELens format, loadable later via `SAE.load_from_disk`. Per the paper, lr
defaults to `3e-3` and batch size to `1024`; default `n_steps = 20_000` (≈ 2M tokens, 10
epochs).

**Sweep loop** lives in the same CLI command: `n_experts` is fixed at a reasonable default
(see CLI Surface), and the loop iterates over `k_experts` values (the active-expert budget
per sample, the SMIXAE analogue of TopK SAE's `k`). For each `k_experts`, train one SMIXAE,
evaluate on the shared eval set, and append one row per (k_experts, manifold_instance, n)
to the result table.

---

## Evaluation Metrics

For each trained model and each manifold instance `i`:

1. **Restricted R²(i, n)** — SMIXAE analogue of paper Eq. 14. The paper's "select n decoder
   atoms" maps cleanly to "select n experts" because each SMIXAE expert *is* a unit of
   capacity (a 3D bottleneck with its own decoder block). Procedure:
   - Encode the rows where manifold `i` is active to get bottleneck codes
     `B_i ∈ R^{n_i × n_experts × 3}` (and the corresponding per-manifold ground-truth
     contributions `M_i ∈ R^{n_i × 128}`).
   - Greedy expert selection: at each of `n` steps, pick the single expert whose contributed
     reconstruction (decoded on rows for `i`, with all other experts zeroed out and `b_dec`
     subtracted) most reduces the residual variance of `M_i`. Use the cumulative selected
     subset's reconstruction at step `n` to compute R²(i, n).
   - Report R²(i, n) for `n ∈ {1, 2, 3}` (since each SMIXAE expert already supplies up to
     3 decoder dims, the range `1..3` brackets full subspace capture for the kept manifolds
     where `k_i ∈ {1, 2, 3}`).
2. **MSE** — mean over the eval set of `‖x - x̂‖²` (overall reconstruction loss).
3. **Per-manifold MSE** — mean over rows where `i` is active of `‖m_i - m̂_i‖²`, where
   `m̂_i` is decoded from the **single best expert** for `i` (greedy `n=1`).
4. **Effective L0** — mean count of active experts per sample on the eval set.
5. **Dead expert count** — experts whose bottleneck-norm exceeds 0 on no eval samples.
6. **Best-expert assignment table** — per manifold instance, the index of its top-1 greedy
   expert (used for the bottleneck reconstruction figure).

All metrics aggregated into a single CSV row per `(k_experts, manifold_type, variant_idx,
n)` (with `n_experts` recorded as a constant column).

---

## Outputs (single `results/synthetic/{run_id}/` bundle)

- `results.csv` — all metric rows for all configs in the sweep.
- `summary.json` — config sweep + aggregate metrics (mean R²(n=1) per config, mean MSE,
  best config).
- `metrics.html` — Plotly figure (using `analysis.scatter3d` Plotly idioms): one panel per
  metric (mean R²(n=1..3), MSE, dead-expert count) plotted against `k_experts`, with one
  line per manifold type (and an "all" aggregate).
- `best_model_bottlenecks.html` — grid of 3D scatter plots (one per manifold instance)
  showing the assigned expert's bottleneck code coloured by intrinsic parameter `θ`,
  reusing `analysis.scatter3d.plot_3d_scatter`.
- `models/k_experts={k}/model/` — saved inference SMIXAE per swept config (only the best
  by mean R²(n=1) by default; `--keep-all-models` flag keeps every checkpoint).

---

## CLI Surface

New module `src/cli/synthetic.py` exporting a `synthetic()` Typer command, registered in
`src/cli/cli.py:27` next to `core_eval`:

```bash
smixae synthetic \
    --d-in 128 --l0 4 --sigma-eps 1e-5 \
    --n-experts 42 --d-expert 16 --d-bottleneck 3 \
    --k-experts-list 2,4,6,8,12,16 \
    --train-steps 20000 --batch-size 1024 --lr 3e-3 \
    --eval-samples 200000 \
    --output-dir results/synthetic/{run_id} \
    --seed 0
```

`n_experts` is fixed at 42 by default — exactly matching the ground-truth manifold instance
count. This pins capacity to the oracle budget so the sweep cleanly isolates the effect of
the active-expert budget `k_experts` (the SMIXAE analogue of TopK SAE's `k`).

Shell wrapper `experiments/synthetic_toy.sh` runs the canonical default sweep and writes
to `results/synthetic/$(date +%Y%m%d_%H%M%S)/`.

---

## Critical files

- **Create** `src/analysis/synthetic.py` — manifold zoo, `FeatureDictionary` build,
  `SparseManifoldSampler`, eval-set generator, R²/MSE metrics, plotting helpers.
- **Create** `src/cli/synthetic.py` — Typer command, training loop, sweep orchestration.
- **Edit** `src/cli/cli.py` — register `app.command(name="synthetic", ...)(synthetic)`
  alongside the existing `train` and `core` registrations (line ~27).
- **Create** `experiments/synthetic_toy.sh` — wrapper invoking the canonical sweep.
- **Edit** `AGENTS.md` and `docs/REFERENCE.md` — add brief entries documenting the new
  subcommand and module per the project's documentation policy.
- **Reuse** `src/analysis/scatter3d.py:plot_3d_scatter` for bottleneck figures and
  `src/analysis/colors.py` for colour maps.
- **Reuse** `sae_lens.synthetic.FeatureDictionary` (atom dictionary) and
  `sae_lens.synthetic.ActivationGenerator` (manifold-instance firing).
- **Reuse** `src/smixae/smixae.py:SMIXAETraining` directly with custom loop;
  `model.save_inference_model(...)` for serialization.

---

## Verification

End-to-end smoke test (small/fast configuration):

```bash
uv run smixae synthetic \
    --n-experts-list 32,64 --train-steps 1000 \
    --eval-samples 5000 --output-dir results/synthetic/_smoke
```

Then check:

1. `results/synthetic/_smoke/results.csv` — has rows for both configs × all 42 manifolds ×
   `n ∈ {1, 2, 3}`.
2. `results/synthetic/_smoke/metrics.html` — opens in a browser, shows monotone trend in R²
   with `n_experts`.
3. `results/synthetic/_smoke/best_model_bottlenecks.html` — each panel shows visibly
   manifold-shaped 3D scatter (circles, helices, etc.) coloured by intrinsic θ.
4. `uv run python -c "from sae_lens import SAE; SAE.load_from_disk('results/synthetic/_smoke/models/n_experts=64,k_experts=4/model')"` — loads cleanly, confirming the saved SMIXAE is round-trippable.
5. `uv run ruff check src/analysis/synthetic.py src/cli/synthetic.py src/cli/cli.py` —
   passes lint.

Larger validation (canonical sweep): `bash experiments/synthetic_toy.sh` — should finish in
roughly 20–40 minutes on a single GPU and produce `metrics.html` showing R²(n=1) climbing
toward 1.0 for `n_experts ≥ 256` on most manifolds.
