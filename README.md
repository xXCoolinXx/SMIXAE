# SMIXAE — Sparse Mixture of Autoencoders

SMIXAE is an interpretability architecture for large language models that replaces
the standard Sparse Autoencoder (SAE) linear decomposition with a mixture of experts,
each governed by a low-dimensional nonlinear bottleneck.

A standard SAE decomposes residual stream activations into a sparse sum of independent
linear directions. SMIXAE instead routes each activation to a sparse set of **experts**,
each with a **3-D bottleneck** that can represent arbitrary geometry — rings, spirals,
helices, clusters, or arbitrary manifolds. This allows the model to capture features
whose natural representation is nonlinear.

See the visualization of the experts [here](https://dainty-sawine-dc149c.netlify.app/).

---

## Architecture at a glance

```
x (batch, d_model)
  │
  ├─ W_enc + b_enc + LeakyReLU  →  (batch, n_experts, d_expert)    expert activations
  │
  ├─ W_bottleneck                →  (batch, n_experts, d_bottleneck) bottleneck
  │
  ├─ BatchTopK over L_2 mask     →  keep k experts per token
  │
  ├─ W_latent_dec                →  (batch, n_experts, d_expert)    back to expert space
  │
  └─ W_dec + b_dec               →  (batch, d_model)                reconstruction
```

---

## Quick start

```bash
# Install (requires Python ≥ 3.14 and uv)
uv sync

# Train on Gemma 2-9B layer 11
smixae train \
    --model-name google/gemma-2-9b \
    --hook-name model.layers.11 \
    --training-tokens 500000000 \
    --n-experts 2048 \
    --d-in 3584 \
    --d-expert 8 \
    --k-experts 128 \
    --output-path results/my_run/model \
    --checkpoint-path results/my_run/checkpoints

# Or run the full experiment (train → probe → newline analysis):
bash experiments/gemma_2_9b_l11.sh

# Generate probing datasets
smixae generate-probing-data generate

# Probe a single dataset against a checkpoint
smixae probe single \
    --checkpoint-path results/my_run/model \
    --base-model-name google/gemma-2-9b \
    --hook-point model.layers.11 \
    --dataframe-path datasets/probing/weekdays.csv \
    --label-column Label

# Newline-position manifold analysis
smixae newline main \
    --smixae-path results/my_run/model \
    --model-name google/gemma-2-9b \
    --hook-name model.layers.11 \
    --output-path results/my_run/newline
```

---

## CLI overview

```
smixae
├── train                        # Train a SMIXAE
├── generate-probing-data
│   └── generate                 # Generate all probing datasets → datasets/probing/
├── generate-steering-data
│   └── generate                 # Generate steering prompt datasets → datasets/steering/
├── probe
│   ├── single                   # Analyze one labeled dataset against a checkpoint
│   └── all-datasets             # Batch over a JSON config of datasets
├── newline
│   └── main                     # Newline-position manifold analysis
└── steer
    └── main                     # Steering experiments (coordinate substitution)
```

Run `smixae --help` or `smixae <subcommand> --help` for all flags.

---

## Results

Analysis outputs land in `results/{experiment_name}/`:

| Directory | Contents |
|-----------|----------|
| `model/` | Final inference-ready SMIXAE checkpoint |
| `checkpoints/` | Intermediate training checkpoints |
| `probe/` | `top_experts.html` + per-expert `dim_analysis_expert*.html` |
| `newline/` | Newline-position regression results |

The HTML files contain interactive Plotly 3-D scatter plots of each expert's bottleneck
activations, colored by label. Open them in a browser — no server required.

---

## Reproducing experiments

Each script in `experiments/` is self-contained: set the variables at the top and run it.
To add a new experiment, copy `experiments/gemma_2_9b_l11.sh` and adjust the model name,
hook point, and output paths.

---

## Contributing

See [AGENTS.md](AGENTS.md) for the full developer guide: architecture details, coding
conventions, analysis patterns, and what to avoid.
