# SMIXAE Steering

## Activation Patching

Steering in SMIXAE works by **full-sequence activation patching** at the SAE level: at every token position where the target expert fires, replace its bottleneck contribution with the decoded mean of a target class. 

It is currently a work in progress - we suspect that the correct intervention layer for many of the manifolds we found is different from the layers we trained on.

We also expect that one may neeed to steer along multiple manifolds due to the many ways the model can interact with a single concept. This is future work. But here is a start if you have some better ideas than what we did.

For a target expert `e` and target class `c`:

```
# At every position where expert e is active:
z = sae.encode(x)                                    # (batch*seq, n_experts, d_bottleneck)
active_mask = z[:, e, :].norm(dim=-1) > 0            # threshold-masked by encode()

# Current and target contributions:
contrib_current = decode(z[:, e, :])                  # (batch*seq, d_model)
contrib_target  = decode(mean_bottleneck_c)           # (batch*seq, d_model) — same for all active positions

# Patch only active positions:
x_steered = x + (-contrib_current + contrib_target) * active_mask
```

Inactive positions are left untouched. The intervention is applied at the model's hook point (same as the SAE training hook) during generation, so the model produces text from the modified residual stream.

**Key assumption**: the expert's bottleneck encodes the concept of interest, and the class means in bottleneck space are separable. If the expert doesn't encode the concept cleanly, steering will have no effect or will produce incoherent outputs.

---

## Expert Selection

Experts are selected from the probing results stored in `results/results.json`. The `cyc_24h` regression hypothesis (24-hour cyclical ring) identifies experts whose bottleneck activations best model circular hour-of-day structure — these are the most natural targets for steering.

The top 2 experts by `cyc_24h` R² score are used by default. Use `--n-top-experts` to change how many are selected.

---

## Workflow

1. **Probe** to populate `results/results.json` with regression hypothesis scores:
   ```bash
   smixae probe all-datasets \
       --checkpoint-path results/my_run/model \
       --base-model-name google/gemma-2-9b \
       --hook-point model.layers.20 \
       --datasets-config datasets/probing/dataset_config.json \
       --output-dir results/my_run/probe \
       --results-json results/results.json
   ```

2. **Run steering**:
   ```bash
   smixae steer main \
       --checkpoint-path results/my_run/model \
       --base-model-name google/gemma-2-9b \
       --hook-point model.layers.20 \
       --run-name gemma_2_9b_l20 \
       --output-dir results/my_run/steer
   ```

3. **Inspect** `results/my_run/steer/summary.csv` for per-expert accuracy, and `results/my_run/steer/scores/` for per-prompt detail.

---

## Current-Time Steering

**Prompt**: `"You glance at the clock and find it is {src_hour}. Your friend {name} asks you for the time, and you respond, saying it is"`

**Goal**: Steer the model to answer as if the current time were `tgt_hour` instead of `src_hour`.

**Mechanism**: The top `cyc_24h` experts' bottleneck contributions are replaced with the decoded mean for class `tgt_hour` at every position where each expert fires. If the expert cleanly encodes hour-of-day, the model's answer shifts accordingly.

**What success looks like**: The `steered_output` mentions `tgt_hour`, while `baseline_output` mentions `src_hour`.

---

## Cross-Layer Sweep

When steering at the trained layer doesn't produce observable effects, you can sweep the same SMIXAE patching across multiple layers to find where the intervention is causally effective.

**Idea**: The SMIXAE was trained at layer N, but the same encode/decode/patch operation can be applied at any layer. The class means are still computed from the trained layer's activations, but the hook is installed on a different layer.

```bash
smixae steer main \
    --checkpoint-path results/my_run/model \
    --base-model-name google/gemma-2-9b \
    --hook-point model.layers.20 \
    --run-name gemma_2_9b_l20 \
    --sweep-layers \
    --layer-start 0 \
    --layer-end -1 \
    --output-dir results/my_run/steer_sweep
```

`--layer-end -1` means "trained layer minus 1" (all layers below the trained layer). The output `summary.csv` includes a `layer` column so you can compare steering effectiveness across depths.

---

## CLI Reference

```
smixae steer main [OPTIONS]

Required:
  --checkpoint-path PATH   Path to SMIXAE model directory
  --base-model-name TEXT   HuggingFace model name
  --hook-point TEXT        Hook point used during SAE training

Expert selection:
  --results-json PATH      Path to results.json (default: results/results.json)
  --run-name TEXT          Run name key in results.json (required)
  --hypothesis TEXT        Regression hypothesis name (default: cyc_24h)
  --n-top-experts INT      Number of top experts (default: 2)

Cross-layer sweep:
  --sweep-layers           Sweep steering across multiple layers
  --layer-start INT        First layer to sweep (inclusive, default: 0)
  --layer-end INT          Last layer to sweep (inclusive, default: trained_layer - 1)

Data:
  --hours-dataset PATH     Probing CSV for class means (default: datasets/probing/hours.csv)
  --task1-dataset PATH     Steering prompt CSV (default: datasets/steering/hours_current_time.csv)

Generation:
  --generate-tokens INT    Max new tokens per prompt (default: 20)
  --gen-batch-size INT     Batch size for generation (default: 32)

Probing:
  --n-probing-samples INT  Samples for class-mean computation (default: 1000)
  --llm-batch-size INT     LLM forward pass batch size (default: 16)
  --sae-batch-size INT     SAE encoding batch size (default: 2048)

Output:
  --output-dir PATH        Output directory (default: steer_results)
```
