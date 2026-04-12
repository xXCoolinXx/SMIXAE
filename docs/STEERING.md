# SMIXAE Steering

## Coordinate Substitution

Steering in SMIXAE works by **coordinate substitution**: replace the expert's current bottleneck contribution to the residual stream with the decoded mean of a target class.

For a target expert `e` and target class `c`:

```
# Current contribution of expert e to the residual:
contribution_current = decode(h_bottleneck[:, e, :])   # (batch, d_model) — only expert e is non-zero

# Mean bottleneck for class c:
mean_bottleneck_c = h_bottleneck[labels == c, e, :].mean(dim=0)   # (d_bottleneck,)

# Steering intervention at the hook point:
x_steered = x_original
           - contribution_current          # remove current expert output
           + decode(mean_bottleneck_c)     # inject class c representation
```

The intervention is applied at the model's hook point (same as the SAE training hook). The model then generates text from the modified residual stream.

**Key assumption**: the expert's bottleneck encodes the concept of interest, and the class means in bottleneck space are separable (i.e., the expert scores high on Fisher or continuity). If the expert doesn't encode the concept cleanly, the steering will have no effect or will produce incoherent outputs.

**Warning** This code isn't finalized yet, and doesn't currently work.

---

## Workflow

1. **Probe** to find the expert with the highest Fisher score for the target concept:
   ```bash
   smixae probe single \
       --checkpoint-path results/my_run/model \
       --base-model-name google/gemma-2-9b \
       --hook-point model.layers.11 \
       --dataframe-path datasets/probing/hours.csv \
       --label-column Label
   ```
   Open `results/my_run/probe/top_experts.html` and note the top expert ID.

2. **Run steering**:
   ```bash
   smixae steer main \
       --checkpoint-path results/my_run/model \
       --base-model-name google/gemma-2-9b \
       --hook-point model.layers.11 \
       --output-dir results/my_run/steer
   ```

3. **Inspect** `results/my_run/steer/steering_results.csv`. Columns: `task`, `expert_id`, `src_hour`, `tgt_hour`, `start_hour`, `prompt`, `baseline_output`, `steered_output`.

---

## Task 1: Current-Time Steering

**Prompt**: `"Right now it is {src_hour}. What time is it?"`

**Goal**: Steer the model to answer as if the current time were `tgt_hour` instead of `src_hour`.

**Mechanism**: The expert that scores highest for `hours.csv` (Fisher discriminant) is used. The intervention replaces the expert's bottleneck contribution with the decoded mean for class `tgt_hour`. If the expert cleanly encodes hour-of-day, the model's answer shifts accordingly.

**What success looks like**: The `steered_output` mentions `tgt_hour`, while `baseline_output` mentions `src_hour`.

---

## Task 2: Elapsed-Time Steering

> ⚠ **In progress / design not finalised.** The implementation exists but the exact intervention point and evaluation metric are still being determined.

**Prompt**: `"Right now it is {curr_hour}. How much time has it been since {start_hour}?"`

**Goal**: Steer the model's representation of `curr_hour` by `+target_delta_hours`, changing the perceived elapsed time from `curr_hour - start_hour` to `(curr_hour + target_delta) - start_hour`.

**Current state**: The code in `steer.py` constructs the intervention and generates both a baseline and a steered response. However:
- The choice of which token position to apply the intervention to (the `curr_hour` token vs. the end of the prompt) is not yet settled.
- Whether to steer on the `hours.csv` expert or a different expert specialised for elapsed-time reasoning is an open question.
- The evaluation metric (beyond looking at the raw text output) is not defined.

**What is known**: Task 1 works for in-distribution source hours. Task 2 is more complex because the model must reason about the *difference* between two time representations, not just report one.
