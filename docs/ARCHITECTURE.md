# SMIXAE Architecture

## SMIXAE vs. AffineSMIXAE

SMIXAE has two implementations in this codebase:

| | SMIXAE | AffineSMIXAE |
|---|---|---|
| **Routing** | Bottleneck norm — top-k experts by `‖W_bottleneck · h_expert‖` | Cosine similarity — top-k experts by `cos(x, W_directions[i])` |
| **Extra parameters** | None beyond base | `W_directions: (n_experts, d_in)` |
| **Status** | Active, used in all current experiments | **Not actively used or maintained** (historical reference only) |
| **Source** | `src/smixae/smixae.py` | `src/smixae/affine_smixae.py` |

AffineSMIXAE was explored as an alternative where each expert "owns" a direction in the original input space, and the input is routed based on how well it aligns with that direction. This decouples routing from the encoder output, which changes the inductive bias. It was not found to offer consistent advantages and is kept only for reference.

---

## Encoding Pipeline (step-by-step with shapes)

Assume `x` has shape `(B, D)` where `B = batch size`, `D = d_model`.

### Step 1: Linear projection

```
hidden = x @ W_enc + b_enc        # (B, D) × (D, n_experts * d_expert) = (B, n_experts * d_expert)
```

`W_enc` shape: `(d_model, n_experts * d_expert)` — a single shared linear map from input space into the stacked expert space.

### Step 2: LeakyReLU

```
h = LeakyReLU(hidden, slope=1e-4)  # (B, n_experts * d_expert)
```

Slope `1e-4` avoids dead neurons (all-zero gradients) while keeping negative activations small enough that they don't dominate expert norms.

### Step 3: Reshape

```
h_experts = h.unflatten(-1, (n_experts, d_expert))  # (B, n_experts, d_expert)
```

Each expert now has its own `d_expert`-dimensional activation vector.

### Step 4: Bottleneck projection (rescaled)

```
h_bottleneck = einsum("bne,ned->bnd", h_experts, W_bottleneck)  # (B, n_experts, d_bottleneck)
```

`W_bottleneck` shape: `(n_experts, d_expert, d_bottleneck)`.

If `rescale_acts_by_decoder_norm=True` (default), each expert's bottleneck is multiplied by the Frobenius norm of the effective `(d_bottleneck → d_model)` projection:

```
effective_norm[i] = ‖W_latent_dec[i] @ W_dec_reshaped[i]‖_F    # scalar per expert
h_bottleneck *= effective_norm.unsqueeze(-1)                      # (B, n_experts, d_bottleneck)
```

This rescaling prevents experts with large decoder norms from dominating the reconstruction loss simply due to scale, encouraging the model to use all experts rather than concentrating capacity in a few.

### Step 5: BatchTopK masking

```
expert_norms = h_bottleneck.norm(dim=-1)   # (B, n_experts)
mask = BatchTopK(k)(expert_norms)          # (B, n_experts) — 1 for top-k per item, 0 otherwise
h_active = h_bottleneck * mask.unsqueeze(-1)  # (B, n_experts, d_bottleneck)
```

`BatchTopK` guarantees `k` active experts on average per item in the batch (no threshold tuning required during training).

### Step 6: Threshold masking (inference only)

During inference, a learned threshold `θ` (maintained by `update_threshold()`) adds a second gate:

```
h_active = h_bottleneck * (expert_norms > θ).float().unsqueeze(-1)
```

The threshold is updated via EMA on the minimum positive bottleneck norm seen during training.

---

## Decoding Pipeline

```
# Bottleneck → expert space
h_expert_dec = einsum("bnd,nde->bne", h_active, W_latent_dec)  # (B, n_experts, d_expert)

# Flatten and project to residual stream
sae_out = h_expert_dec.flatten(-2, -1) @ W_dec + b_dec          # (B, d_model)
```

`W_latent_dec` shape: `(n_experts, d_bottleneck, d_expert)` — note the transpose of `W_bottleneck`.

The bias `b_dec` is applied **only here** (at the output), not inside the bottleneck. Adding it earlier would shift the origin of the bottleneck manifold, collapsing the geometric structure that makes the visualizations meaningful.

---

## Bottleneck and Why 3-D

`d_bottleneck=3` is chosen so each expert's active samples can be directly plotted as a 3-D scatter in Plotly — no dimensionality reduction required, and the geometry is exactly what the model learned.

Three dimensions is the minimum that supports rich manifold geometry: a ring, helix, or non-planar spiral all require at least 3 coordinates. Two dimensions would collapse helices, and one dimension would force everything linear.

**Tuning `d_bottleneck`**: Increasing it is possible but requires a minimum-dimensionality penalty to prevent the model from using only 1–2 effective dimensions while nominally having a larger bottleneck. This regularization is not implemented in this repo. If you add it, expect to also add a dimensionality monitoring metric to W&B.

---

## Dead Expert Recovery

An expert is "dead" if it hasn't fired (had a non-zero bottleneck norm for any sample in the batch) in `dead_after_n_passes` consecutive passes (default `1000`).

The current recovery strategy (`calculate_pre_act_aux_loss`) is a **norm-pressure approach**:

```
shortfall = relu(θ - expert_norms[:, dead_mask])          # how far below threshold
aux_loss = coeff * (shortfall * decoder_norm_dead).sum()   # weighted by decoder norm
```

This pushes dead experts' pre-activation norms toward the threshold without peeking at the residual. The `effective_decoder_norm` weighting ensures experts with larger decoders get more pressure — they have more capacity and are more valuable to recover.

An alternative (`calculate_topk_aux_loss`) selects `k_aux` dead experts and directly minimises residual reconstruction error. This is approach is not preferred because it causes feature splitting, by routing gradient from the residual error. This leads to a situation where an expert that is still undergoing training has its residual error stolen by an expert being revived, leading to the aformetioned feature splitting. This is likely also an issue in SAEs

---

## Threshold Semantics

The `threshold` buffer has two different roles:

| Phase | Role |
|-------|------|
| **Training** | Used only by `calculate_pre_act_aux_loss` to measure shortfall; **does not gate the forward pass** |
| **Inference** | Gates `h_active`: experts below threshold are zeroed |

`update_threshold()` tracks the EMA of the minimum positive bottleneck norm across the batch. This means the threshold approximates "what a barely-active expert looks like," which is the right threshold for separating genuine signal from noise at inference time.

The threshold is stored in `float64` to avoid numerical instability when it becomes very small.
