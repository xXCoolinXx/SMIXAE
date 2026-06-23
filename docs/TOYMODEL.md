# Synthetic Toy-Model Benchmark

The toy-model benchmark provides a controlled ground-truth environment for evaluating
how well SMIXAE (and SAEs) recover nonlinear manifold structure from sparse superpositions.
Because the generative process is known exactly, recovery quality can be measured directly
rather than inferred from downstream tasks.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/toy/manifolds.py` | Type registries, `ManifoldInstance`, all `_sample_*` functions |
| `src/toy/zoo.py` | `ManifoldZoo`, `EvalData`, `ManifoldActivationGenerator`, `build_manifold_zoo`, `generate_eval_set`, `optimize_subspaces` |
| `src/toy/metrics.py` | `compute_restricted_r2` (co-firing + linear OLS R²), `compute_metrics` |
| `src/toy/plot.py` | `plot_metrics_vs_k_experts`, `plot_bottlenecks`, `plot_all_experts_with_originals` |
| `src/cli/toy.py` | CLI: `generate`, `train`, `eval`, `plot`, `pipeline` |

---

## Generative Model

Observations follow a sparse mixture of low-dimensional manifolds embedded in a
high-dimensional ambient space:

$$x = \sum_{i \in S} \tilde{\gamma}_i(\theta_i) V_i + b_{\text{global}} + \varepsilon$$

- **S** — active set, drawn uniformly without replacement: `|S| = L₀ = 4`
- **γ̃ᵢ(θᵢ)** — normalised manifold coordinates (zero mean, unit RMS norm)
- **Vᵢ ∈ ℝ^{kᵢ × d}** — orthonormal ambient embedding for instance i (Grassmannian-optimised)
- **b_global** — fixed random unit direction scaled to `sigma_bias` (global DC offset)
- **ε ~ N(0, σ²ε I)** — small noise (`σ_ε = 10⁻⁵`)

### Why a global bias?

Real LLM residual streams have a significant DC offset due to RMSNorm.
A single fixed random direction `b_global` (always present in every sample) simulates this.
It is stored as the `FeatureDictionary.bias` so it is automatically included in every
call to `zoo.feature_dict(feature_acts)` — no changes needed in the training loop.

The previous implementation used per-instance bias atoms (one per manifold, activated only
when that manifold was active), which incorrectly modelled the offset as feature-dependent.

### Why Grassmannian optimisation?

The classical QR-based embedding draws each subspace Vᵢ independently, giving no control
over mutual coherence between subspaces. For 2–3-dimensional manifolds, high coherence
makes recovery trivially easy or provably impossible in ways unrelated to the SAE's quality.

`optimize_subspaces` minimises the smooth max-coherence surrogate

$$\mathcal{L} = \frac{1}{T} \log \sum_{i < j} \exp\!\bigl(T \cdot \|V_i^T V_j\|_F^2\bigr)$$

via Adam + QR retraction (projected gradient on the Stiefel manifold), giving a set of
subspaces with demonstrably lower pairwise coherence than random QR.

---

## Manifold Zoo

8 types × 6 variants = **48 instances**, **120 manifold atoms** total.

| Type | dᵢ | kᵢ | Parametric embedding γᵢ(θ) | Variant parameters |
|------|----|----|-----------------------------|-------------------|
| Circle | 1 | 2 | (r cos θ, r sin θ) | r ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 3.0} |
| Sphere | 2 | 3 | (r sin φ cos θ, r sin φ sin θ, r cos φ) | r ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 3.0} |
| Torus | 2 | 3 | ((R+r cos φ) cos θ, (R+r cos φ) sin θ, r sin φ) | (R,r) ∈ {(2,0.5),(2,1),(3,1),(3,1.5),(4,1),(4,2)} |
| Möbius | 2 | 3 | ((1 + t cos φ/2) cos φ, ..., t sin φ/2) | w ∈ {0.2, 0.3, 0.5, 0.7, 1.0, 1.5} |
| Swiss Roll | 2 | 3 | (θ cos θ, h, θ sin θ) | (θ_max, h_max) ∈ {(2π,1.5) ... (4.5π,6)} |
| Helix | 1 | 3 | (cos θ, sin θ, α θ) | α ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6} |
| Flat Disk | 2 | 2 | (r cos φ, r sin φ), r ~ R√U(0,1) | R ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 3.0} |
| Segment | 1 | 1 | (t) | length ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 3.0} |

**Normalisation.** Each instance is calibrated on 50,000 points: mean μᵢ and RMS norm σᵢ
are computed in local coordinates, then `γ̃ᵢ(θ) = (γᵢ(θ) − μᵢ) / σᵢ`.
This gives every instance unit RMS norm regardless of type or variant.

**Note on Torus.** The paper's Table 4 uses a 4-D Clifford embedding (kᵢ=4).  This
benchmark uses the standard 3-D embedding (kᵢ=3) so SMIXAE's 3-D bottleneck can
faithfully capture the torus geometry.

---

## CLI Workflow

### Generate

```bash
uv run smixae toy generate --seed 0 --d-in 128 --l0 4 --sigma-bias 3.0
```

Builds the zoo (including Grassmannian optimisation, ~minutes on CPU) and generates
200,000 eval samples.  Saves to `toy_data/seed0_d128_l4_b3/`.

Add `--skip-grassmannian` for a fast random-QR fallback during debugging.

### Train

```bash
uv run smixae toy train --seed 0 --k-experts-list 2,4,6,8,12,16
```

Loads the saved dataset, sweeps `k_experts`, and writes results to
`toy_data/seed0_d128_l4_b3/results/`.

### Plot

```bash
uv run smixae toy plot --seed 0
```

Loads the saved dataset and training results; regenerates all HTML plots to
`toy_data/seed0_d128_l4_b3/plots/` without retraining.

### Eval (re-run metrics on saved checkpoints)

```bash
uv run smixae toy eval --seed 0
```

Scans `results/k*/model/` checkpoints and re-runs `compute_restricted_r2` +
`compute_metrics` without retraining.  Overwrites `results/results.csv` and
`results/summary.json`.  Use this when the metric definition changes and you want
updated numbers without paying for another full training sweep.

### Pipeline (all in one)

```bash
uv run smixae toy pipeline --seed 0
```

Runs generate → train → plot in sequence.  Generation is skipped if the dataset
directory already exists for the given seed/d_in/l0/sigma_bias (use `--force-generate`
to override).

---

## Notebook Hyperparameter Search (Optuna)

`notebooks/toy_smixae.ipynb` exposes an **optional** Optuna search over the
GrumpReLU sparsity path. It is opt-in via a single switch in the top config cell:

```python
RUN_HYPEROPT      = False   # flip to True to search before the final run
HYPEROPT_N_TRIALS = 20
```

A shared `hparams` dict is the single source of truth for the tunable values, and
the final model is always built from it via `build_model(hparams)`. When the
search is **off**, `hparams` keeps its baseline defaults; when **on**, the best
trial's parameters overwrite `hparams` before the final training + reporting
cells run — so both paths train and report identically apart from where the
values come from.

Each trial trains a fresh model on the full budget and is scored by the **mean
co-firing-matched R²** (`compute_restricted_r2(...).r2.mean()` — see Evaluation
Metrics below), which Optuna maximises. Tuned parameters and ranges:

| Param | Range | Goes to |
|-------|-------|---------|
| `bandwidth` | `[0.5, 5.0]` | `GrumpReLULayerConfig` |
| `init_threshold` | `[1e-3, 1e-1]` (log) | `GrumpReLULayerConfig` |
| `hardness_coefficient` | `[1.0, 16.0]` (log) | `GrumpReLULayerConfig` |
| `sparsity_coefficient` | `[0.1, 10.0]` (log) | `SMIXAEV2Config` |
| `sparsity_warm_up_steps` | `[0, training_samples // batch_size]` | `SMIXAEV2Config` |

Requires the `optuna` dependency (declared in `pyproject.toml`; run `uv sync`).
The import is lazy, so the notebook still loads without it when `RUN_HYPEROPT` is
`False`.

---

## Output Layout

```
toy_data/
└── seed{seed}_d{d_in}_l{l0}_b{sigma_bias}/
    ├── manifest.json          Scalar hyperparameters
    ├── zoo.pt                 Serialised ManifoldZoo (V_i matrices, b_global)
    ├── eval.pt                Serialised EvalData (x, feature_acts, color_param)
    ├── results/               Populated by `toy train`
    │   ├── results.csv        Per-instance r2_linear + cofiring_rate for each k_experts
    │   ├── summary.json       Aggregate metrics and best k_experts
    │   └── k{k}/model/        Inference-ready SMIXAE checkpoint
    └── plots/                 Populated by `toy plot`
        ├── metrics.html           R², MSE, effective L0, dead experts vs k_experts
        ├── bottlenecks_k{k}.html  3-D bottleneck codes per manifold type, one per k
        └── all_experts_k{k}.html  All 48 instances: original (blue) vs learned (red), one per k
```

---

## Evaluation Metrics

### Linear OLS R² and co-firing rate

For each manifold instance i and trained SMIXAE at sparsity k, two metrics are computed:

**Co-firing rate** — routing quality:

P(expert fires | manifold active) for the best-matched expert.  The best expert is
chosen as the one with the highest co-firing rate (argmax over all experts).  A value
near 1 means the model reliably routes samples from manifold i to the same expert.

**Linear OLS R²** — representation quality:

Given the matched expert and the co-active rows (manifold i active AND expert fires):

1. H = expert's 3-D bottleneck codes on co-active rows.
2. Z = manifold's intrinsic coordinates on co-active rows.
3. Fit an affine map W via OLS: [H | 1] W ≈ Z.
4. R² = 1 − ||Z − Ẑ||² / ||Z − Z̄||²

R² ≈ 1 means the bottleneck geometrically captures the manifold; co-firing ≈ 1 means
the routing is reliable.  Low co-firing with high R² indicates good representation
but poor routing; low R² with high co-firing indicates the expert fires reliably but
the bottleneck does not reflect the manifold's geometry.

### MSE / Effective L0 / Dead Experts

Overall reconstruction MSE, mean active experts per sample, and number of experts
that never fire on the eval set — computed by `compute_metrics`.

---

## Python API

```python
from toy.zoo import ManifoldZoo, EvalData, build_manifold_zoo, generate_eval_set
from toy.metrics import compute_restricted_r2, compute_metrics

# Build the zoo (slow first time due to Grassmannian optimisation)
zoo = build_manifold_zoo(d_in=128, seed=0, sigma_bias=3.0)

# Access the global bias
print(zoo.b_global.norm())  # ≈ 3.0

# Generate eval data
eval_data = generate_eval_set(zoo, n_samples=200_000, l0=4, seed=1)

# Serialise / load
zoo.save("zoo.pt")
zoo2 = ManifoldZoo.load("zoo.pt", device="cpu")

eval_data.save("eval.pt")
eval_data2 = EvalData.load("eval.pt")

# Evaluate a trained model
# Returns: r2 (n_instances,), cofiring (n_instances,), best_experts (n_instances,)
r2, cofiring, best_experts = compute_restricted_r2(model, zoo, eval_data, device="cuda")
metrics = compute_metrics(model, zoo, eval_data, device="cuda")
```
