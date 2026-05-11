# SMIXAE — Sparse Mixture of Autoencoders

SMIXAE is an interpretability architecture for large language models that replaces
the standard Sparse Autoencoder (SAE) linear decomposition with a mixture of experts,
each governed by a low-dimensional nonlinear bottleneck.

A standard SAE decomposes residual stream activations into a sparse sum of independent
linear directions. SMIXAE instead routes each activation to a sparse set of **experts**,
each with a **3-D bottleneck** that can represent arbitrary geometry — rings, spirals,
helices, clusters, or arbitrary manifolds. This allows the model to capture features
whose natural representation is nonlinear.

---

## Pretrained models & visualization

| Resource | Link |
|----------|------|
| Pretrained checkpoints (HuggingFace) | [Link](https://huggingface.co/xXCoolinXx/SMIXAE) |
| Interactive expert visualization | [dainty-sawine-dc149c.netlify.app](https://dainty-sawine-dc149c.netlify.app/) |

The visualization site is a static SPA — no server required. Browse experts, rotate 3-D bottleneck scatter plots, and filter by label.

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
bash experiments/gemma_2_9b_l11_newline.sh

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

# Evaluate a trained checkpoint on core SAE metrics (L0, MSE, CE score)
smixae core \
    --sae-path results/my_run/model \
    --model-name google/gemma-2-9b \
    --hook-name model.layers.11
```

---

## CLI overview

```
smixae
├── train                        # Train a SMIXAE
├── core                         # Core SAE eval metrics (L0, MSE, CE score)
├── generate-probing-data
│   └── generate                 # Generate all probing datasets → datasets/probing/
├── generate-steering-data
│   └── generate                 # Generate steering prompt datasets → datasets/steering/
├── pretokenize
│   └── pretokenize              # Tokenize a HuggingFace dataset for fast training
├── probe
│   ├── single                   # Analyze one labeled dataset against a checkpoint
│   └── all-datasets             # Batch over a JSON config of datasets
├── newline
│   └── main                     # Newline-position manifold analysis
├── steer
│   └── main                     # Steering experiments (coordinate substitution)
└── latex
    ├── tables                   # Generate LaTeX tables from results JSON
    ├── figures                  # Assemble camera-ready PNGs into LaTeX figure files
    └── save-server              # Start local HTTP server (port 7788) for figure collection
```

Run `smixae --help` or `smixae <subcommand> --help` for all flags.

---

## Results

Analysis outputs land in `results/{experiment_name}/`:

| Path | Contents |
|------|----------|
| `results/core_eval_results.json` | Core eval metrics across all experiments (written by `smixae core`) |
| `{experiment_name}/model/` | Final inference-ready SMIXAE checkpoint |
| `{experiment_name}/checkpoints/` | Intermediate training checkpoints |
| `{experiment_name}/probe/` | `top_experts.html` + per-expert `dim_analysis_expert*.html` |
| `{experiment_name}/newline/` | Newline-position regression results |

The HTML files contain interactive Plotly 3-D scatter plots of each expert's bottleneck
activations, colored by label. Open them in a browser — no server required.

---

## Reproducing experiments

Each script in `experiments/` is self-contained: set the variables at the top and run it.

| Script | Purpose |
|--------|---------|
| `run.sh` | Generic runner — pass `--model`, `--hook`, `--experiment-name`, `--steps` |
| `core_eval.sh` | Core SAE evaluation for all experiments + GemmaScope baselines |
| `gemma_2_2b_l12.sh` | Gemma 2-2B layer 12 (thin wrapper over `run.sh`) |
| `gemma_2_9b_l11_newline.sh` | Gemma 2-9B layer 11 — probe + newline analysis |
| `gemma_2_9b_l20_general.sh` | Gemma 2-9B layer 20 (thin wrapper over `run.sh`) |

To add a new experiment, copy the closest existing script and adjust the model name,
hook point, and output paths.

---

## Contributing

See [AGENTS.md](AGENTS.md) for the full developer guide: architecture details, coding
conventions, analysis patterns, and what to avoid.

--- 

## Known Issues

In the spirit of improving this architecture, I've compiled a list of several important issues 
- While this architecture does represent multidimensional features more succinctly, there are still a few issues with the featurization design
  - The bottleneck dimension is fixed at $3$ dimensions. This is not ideal, for there are some features which are higher dimensional (newline counting manifold is 6D) and some which are lower dimensional (standard 1D directions). In practice, this leads to experts describing multiple low-rank manifolds, or higher dimensional manifolds being shattered across multiple experts. Neither of these is ideal
  - It seems that there is still some degree of feature splitting going on for those higher dimensional manifolds, though I do not demonstrate this rigorously in the paper. I think this can be sufficiently mitigated by fixing the above issue and exploring more expressive (higher depth) encoders, but I also think incorporating some minimality penalty (see for example MDL-SAEs or the recent VPD paper) would fully mitigate this issue. Traditional SAEs are suffering from feature splitting both from manifolds and from getting unlucky with multiple directions happening to model the exact same thing. SMIXAE (and its future versions) reduce the first issue, but still likely suffer from the second.
- The rescaling by decoder norm is a weird hack that helps prevent the encoder norm from growing progressively during training. This should be fixed in a future variant
- BatchTopK is very unsatisfying to use, and some hyper-scaler ought to figure out the optimal hyperparameters for a JumpReLU version of this architecture
- Scaling properties of this architecture have not yet been considered
  - I get the sense that earlier claims that there are potentially millions of features per layer is quite wrong, and I'd suspect that the real number is closer to $O(d_{\text{model}})$ or at least polynomial in $d_{\text{model}}$
- Making good toy models. I tried the most obvious toy models (see the paper Do SAEs Capture Concept Manifolds?), and I could not get them to work in SMIXAE.
  - I omitted this failure from the paper because I did not try to test these super rigorously, and because toy models should never be considered greater evidence than the results from the real thing you are studying (contrary to what most people in this field seem to think)
  - I did get better results when I added random affine shifts to manifolds, so that they were no longer origin-centered when they were summed together, but I was still unable to match SMIXAE's performance on language model activations. There is probably something additional going on (and maybe SMIXAE is benefitting from scale and just sucks on toy examples?)
 
