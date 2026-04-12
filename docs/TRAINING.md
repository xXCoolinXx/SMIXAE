# SMIXAE Training

## Full Training Command (annotated)

```bash
smixae train \
    # ── Model ─────────────────────────────────────────────────────────────────
    --model-name google/gemma-2-9b \      # HuggingFace model ID
    --hook-name model.layers.11 \         # Hook point (layer to extract activations from)
    --d-in 3584 \                         # Residual stream dimension (must match model)

    # ── Data ──────────────────────────────────────────────────────────────────
    --training-tokens 500000000 \         # Total tokens to train on
    --dataset-path monology/pile-uncopyrighted \
    --context-size 128 \                  # Token sequence length

    # ── SAE Architecture ──────────────────────────────────────────────────────
    --n-experts 2048 \                    # Number of experts
    --d-expert 16 \                        # Expert activation dimension
    --d-bottleneck 3 \                    # Bottleneck dimension
    --k-experts 128 \                     # Active experts per forward pass

    # ── Training ──────────────────────────────────────────────────────────────
    --lr 2e-4 \
    --factor 4 \                          # Scales batch size, LR, warmup, dead-window together

    # ── Output ────────────────────────────────────────────────────────────────
    --output-path results/my_run/model \         # Final model (fixed path, used by analysis)
    --checkpoint-path results/my_run/checkpoints # Intermediate checkpoints (run_id subdirs)
```

Run `smixae train --help` for the complete flag reference grouped by category.

---

## Hyperparameter Guide

### `n_experts`

Paper experiments use **2048**. Higher values (e.g., 4096) are possible but proportionally increase GPU memory because `W_enc` and `W_dec` scale as `n_experts * d_expert`.

Higher dictionary values also seem to proportionally need higher `k_experts`. As such, it is important to tune properly to avoid risks of absorption/splitting. The primary reason **2048** was chosen, despite the smaller size, was to avoid this issue. 

Trade-off: more experts → more fine-grained decomposition, but also more dead experts to recover and a higher risk of expert underutilisation.

### `k_experts`

Controls sparsity. **64–128 is a good operating range.**

- Too low (< 32): most activations are zeroed, reconstruction loss suffers and gradients are sparse. Missing features.
- Too high (> 256): expert utilisation decreases and the model effectively learns a dense autoencoder, losing interpretability.
- `k_experts / n_experts ≈ 0.05–0.10` (5–10% active) is a reasonable target.

### `d_expert`

Scales expert capacity. **8-16** is fine for most experiments. 

Increasing `d_expert` gives each expert more capacity to capture variations within a concept but also increases `W_bottleneck` and `W_latent_dec` parameters quadratically (since the bottleneck projection is `d_expert × d_bottleneck`).

Note that some features have more complex geometry (e.g. helices) that aren't well approximated by a single ReLU layer. This work focuses on using a single additional encoder layer to preserve interpretability, but future would should certainly explore increasing the encoder complexity to multiple layers to be able to properly model helices.

### `d_bottleneck`

**Use 3 for all visualisation experiments.** Increasing it requires a minimum-dimensionality penalty (not in this repo) to prevent the bottleneck from using only 1–2 effective dimensions.

If you add the regularization, monitor the effective rank of the bottleneck activations per expert in W&B to verify it's working.

### `--factor` multiplier

`--factor N` scales four quantities together:
- Batch size × N
- Learning rate x N
- LR warmup steps / N
- `dead_after_n_passes` / N

Use `--factor 4` for a 4× batch size with proportionally larger LR and warmup. This is the recommended way to scale training budget without manually tuning each hyperparameter.

---

## W&B Monitoring

Key metrics to watch during training:

| Metric | What to watch for |
|--------|-------------------|
| `l2_loss` | Should decrease steadily. Plateau early → increase LR or `n_experts`. |
| `aux_loss` | High aux loss = many dead experts. Lower `dead_after_n_passes` or check that `k_experts` isn't too small. |
| `n_dead_experts` | Should converge to 0. Chronic dead experts → increase `aux_loss_coefficient` or decrease `dead_after_n_passes`. |
| `act_threshold` | Should converge to a stable value ~5–50% into training. |
| `experts_above_threshold` | Should stabilise near `k_experts`. |
| `expert_norm_mean` | Monitor for collapse (near 0). |

---

We caution that `l2_loss` and `FVE` are not especially meaningful for interpretability. They should be considered vibe checks to make sure the model didn't collapse, rather than indications that your run performed "better". You should essentially treat `l2_loss` as a flow that provides pressure on the latent representation to be meaningful - you shouldn't add any additional flows from e.g. the residual to "increase performance" - this will wreck your SAE/SMIXAE and I will read your paper and laugh at you.

## Resuming Training

Intermediate checkpoints are saved under `--checkpoint-path/{run_id}/{step}/` where `run_id` is the W&B run ID (non-deterministic). To resume:

1. Find the checkpoint path from the W&B run page or by inspecting `--checkpoint-path/`.
2. Pass `--checkpoint-path <specific_checkpoint_dir>` to the next run command.

The final model at `--output-path` is inference-ready and does **not** contain optimizer state — it cannot be resumed from directly.

---

## Adding a New Experiment

1. Copy `experiments/gemma_2_9b_l11.sh` to a new file, e.g. `experiments/my_model_l8.sh`.
2. Update the variables at the top:
   ```bash
   MODEL="google/my-model"
   HOOK="model.layers.8"
   D_IN=2048                    # residual stream dimension of the new model
   EXPERIMENT_NAME="my_model_l8"
   ```
3. Adjust `--training-tokens` if needed (scale roughly with model size).
4. Run: `bash experiments/my_model_l8.sh`

All outputs land in `results/my_model_l8/` following the same directory structure as the existing experiments.
