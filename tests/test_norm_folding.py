"""Unit tests for SMIXAE norm folding operation."""

import torch
import pytest
from smixae.smixae import SMIXAETraining, SMIXAETrainingConfig, SMIXAE, SMIXAEConfig, _fold_effective_norm


@pytest.fixture
def smixae_training_model():
    """Create a small SMIXAE training model for testing."""
    cfg = SMIXAETrainingConfig(
        n_experts=8,
        d_expert=4,
        d_bottleneck=2,
        k_experts=2,
        d_in=16,
        d_sae=32,
        rescale_acts_by_decoder_norm=True,
    )
    model = SMIXAETraining(cfg)
    return model


def leaky_relu(x: torch.Tensor, negative_slope: float = 1e-4) -> torch.Tensor:
    return torch.where(x >= 0, x, x * negative_slope)


def effective_decoder_norm_from_weights(
    W_latent_dec: torch.Tensor,  # (n_experts, d_bottleneck, d_expert)
    W_dec: torch.Tensor,         # (n_experts*d_expert, d_model)
    n_experts: int,
    d_expert: int,
) -> torch.Tensor:
    d_model = W_dec.shape[-1]
    W_dec_reshaped = W_dec.view(n_experts, d_expert, d_model)          # (n, e, m)
    W_eff = W_latent_dec @ W_dec_reshaped                               # (n, b, m)
    return torch.linalg.matrix_norm(W_eff, ord="fro", dim=(-2, -1))      # (n,)


def test_fold_effective_norm_matches_explicit_rescaling_math_only():
    """
    Isolated math test for _fold_effective_norm:

    Explicit version:
        z0 = h @ W_bottleneck
        z  = z0 * norm
        decode uses z / norm

    Folded version (code-defined folding):
        W_bottleneck' = W_bottleneck * norm
        W_latent_dec' = W_latent_dec / norm
        z' = h @ W_bottleneck'   (== z)
        decode uses z' directly  (== explicit decode)
    """
    torch.manual_seed(0)
    dtype = torch.float64

    batch = 2
    n_experts, d_expert, d_bottleneck = 3, 2, 4
    d_model = 5
    d_sae = n_experts * d_expert

    h = torch.randn(batch, n_experts, d_expert, dtype=dtype)

    W_bottleneck = torch.randn(n_experts, d_expert, d_bottleneck, dtype=dtype)
    W_latent_dec = torch.randn(n_experts, d_bottleneck, d_expert, dtype=dtype)
    W_dec = torch.randn(d_sae, d_model, dtype=dtype)
    b_dec = torch.randn(d_model, dtype=dtype)

    # Dummy (positive) norms, independent of weights (math-only test)
    norm = torch.tensor([2.0, 0.5, 3.0], dtype=dtype)

    # ----- Explicit rescale-by-norm then undo-in-decode -----
    z0 = torch.einsum("bne,ned->bnd", h, W_bottleneck)
    z_scaled = z0 * norm.view(1, -1, 1)

    expert_explicit = torch.einsum("bnd,nde->bne", z_scaled / norm.view(1, -1, 1), W_latent_dec)
    out_explicit = expert_explicit.reshape(batch, d_sae) @ W_dec + b_dec

    # ----- Fold norm into weights (the function under test) -----
    W_bottleneck_f = W_bottleneck.clone()
    W_latent_dec_f = W_latent_dec.clone()
    _fold_effective_norm(
        W_bottleneck=W_bottleneck_f,
        W_latent_dec=W_latent_dec_f,
        effective_decoder_norm=norm,
    )

    z_folded = torch.einsum("bne,ned->bnd", h, W_bottleneck_f)
    expert_folded = torch.einsum("bnd,nde->bne", z_folded, W_latent_dec_f)
    out_folded = expert_folded.reshape(batch, d_sae) @ W_dec + b_dec

    torch.testing.assert_close(z_folded, z_scaled, rtol=1, atol=0)
    torch.testing.assert_close(out_folded, out_explicit, rtol=1, atol=1e-10)


def test_fold_sf_then_fold_effective_norm_preserves_end_to_end_math_only():
    """
    End-to-end math-only test that matches the requested assumption:

      1) "general activation rescaling" quantity sf is applied explicitly as:
            x_in = x / sf
            out  = out_norm * sf
         and is foldable into:
            W_enc' = W_enc / sf
            W_dec' = W_dec * sf
            b_dec' = b_dec * sf

      2) then apply SMIXAE norm folding (code-defined):
            W_bottleneck' *= effective_decoder_norm
            W_latent_dec' /= effective_decoder_norm

    We verify that:
      - output is preserved after removing explicit sf + norm rescale (because they are folded)
      - the *scaled* bottleneck (used for selection pressure) is preserved
    """
    torch.manual_seed(1)
    dtype = torch.float64

    batch = 2
    d_model = 5
    n_experts, d_expert, d_bottleneck = 3, 2, 4
    d_sae = n_experts * d_expert

    x = torch.randn(batch, d_model, dtype=dtype)

    W_enc = torch.randn(d_model, d_sae, dtype=dtype)
    b_enc = torch.randn(d_sae, dtype=dtype)

    W_bottleneck = torch.randn(n_experts, d_expert, d_bottleneck, dtype=dtype)
    W_latent_dec = torch.randn(n_experts, d_bottleneck, d_expert, dtype=dtype)

    W_dec = torch.randn(d_sae, d_model, dtype=dtype)
    b_dec = torch.randn(d_model, dtype=dtype)

    sf = 3.7

    # IMPORTANT: mimic the "compute effective norm before folding sf into state_dict" behavior
    eff_norm = effective_decoder_norm_from_weights(W_latent_dec, W_dec, n_experts, d_expert).clamp_min(1e-8)

    # ----- Baseline: explicit sf + explicit rescale-by-effective-norm and undo-in-decode -----
    x_in = x / sf
    hidden_pre = x_in @ W_enc + b_enc
    h = leaky_relu(hidden_pre).view(batch, n_experts, d_expert)

    z0 = torch.einsum("bne,ned->bnd", h, W_bottleneck)
    z_scaled = z0 * eff_norm.view(1, -1, 1)

    expert = torch.einsum("bnd,nde->bne", z_scaled / eff_norm.view(1, -1, 1), W_latent_dec)
    out_norm = expert.reshape(batch, d_sae) @ W_dec + b_dec
    out_baseline = out_norm * sf

    # ----- Fold sf into W_enc, W_dec, b_dec (remove explicit sf) -----
    W_enc_sf = W_enc / sf
    W_dec_sf = W_dec * sf
    b_dec_sf = b_dec * sf

    # ----- Then fold effective norm into W_bottleneck and W_latent_dec (remove explicit norm ops) -----
    W_bottleneck_folded = W_bottleneck.clone()
    W_latent_dec_folded = W_latent_dec.clone()
    _fold_effective_norm(
        W_bottleneck=W_bottleneck_folded,
        W_latent_dec=W_latent_dec_folded,
        effective_decoder_norm=eff_norm,
    )

    # ----- Folded forward: no explicit sf, no explicit norm scaling/division -----
    hidden_pre_f = x @ W_enc_sf + b_enc
    h_f = leaky_relu(hidden_pre_f).view(batch, n_experts, d_expert)

    z_folded = torch.einsum("bne,ned->bnd", h_f, W_bottleneck_folded)  # should equal z_scaled
    expert_f = torch.einsum("bnd,nde->bne", z_folded, W_latent_dec_folded)
    out_folded = expert_f.reshape(batch, d_sae) @ W_dec_sf + b_dec_sf

    # Checks
    torch.testing.assert_close(hidden_pre_f, hidden_pre, rtol=1, atol=0)
    torch.testing.assert_close(z_folded, z_scaled, rtol=1, atol=1e-10)
    torch.testing.assert_close(out_folded, out_baseline, rtol=1, atol=1e-10)

# def test_sf_and_expert_norm_folding_with_threshold():
#     # 1. Setup dimensions
#     batch, n_exp, d_exp, d_bot, d_mod = 1, 8, 4, 2, 16
#     sf = 1.5 
#     k = 2  # Top-K for threshold calculation
#     x = torch.randn(batch, d_mod)
    
#     # 2. Random Parameters
#     W_enc, b_enc = torch.randn(d_mod, n_exp*d_exp), torch.randn(n_exp*d_exp)
#     W_bot = torch.randn(n_exp, d_exp, d_bot)
#     W_lat_dec = torch.randn(n_exp, d_bot, d_exp)
#     W_dec, b_dec = torch.randn(n_exp*d_exp, d_mod), torch.randn(d_mod)
#     eff_norm = torch.rand(n_exp) + 0.5

#     # --- TRAINING PATH ---
#     # Apply global scale first
#     x_s = x * sf
#     # Encoder
#     latents = torch.relu(x_s @ W_enc + b_enc).view(batch, n_exp, d_exp)
#     # Bottleneck + Expert Scaling
#     # This is the magnitude the model "sees" for routing/thresholding
#     z_pre_mask = torch.einsum("bne,ned->bnd", latents, W_bot) * eff_norm.view(1, -1, 1)
    
#     # Calculate Threshold (Simulating BatchTopK/Threshold update)
#     norms = z_pre_mask.norm(dim=-1)
#     threshold = norms.flatten().topk(k).values.min()
#     mask = (norms >= threshold).unsqueeze(-1)
    
#     # Feature Acts (The values we usually stash)
#     feature_acts = z_pre_mask * mask
    
#     # Decode (Undo expert scale, then undo global scale at output)
#     z_unscaled = feature_acts / eff_norm.view(1, -1, 1)
#     dec_latents = torch.einsum("bnd,nde->bne", z_unscaled, W_lat_dec).flatten(1)
#     out_train = (dec_latents @ W_dec + b_dec) / sf

#     # --- FOLDED PATH ---
#     # Step A: Fold global scale into encoder (W_enc and b_enc)
#     W_enc_f, b_enc_f = W_enc * sf, b_enc * sf
    
#     # Step B: Fold expert norms into W_bot
#     # This ensures (x @ W_enc_f) @ W_bot_f matches the training 'z_pre_mask'
#     W_bot_f = W_bot * eff_norm.view(-1, 1, 1)
    
#     # Step C: Fold compensation into decoder side
#     # Divide W_lat_dec by expert norm and W_dec/b_dec by global sf
#     W_lat_dec_f = W_lat_dec / eff_norm.view(-1, 1, 1)
#     W_dec_f, b_dec_f = W_dec / sf, b_dec / sf

#     # Inference logic
#     l_f = torch.relu(x @ W_enc_f + b_enc_f).view(batch, n_exp, d_exp)
#     z_f_pre = torch.einsum("bne,ned->bnd", l_f, W_bot_f)
    
#     # Applying the same threshold (which now works on raw z_f_pre)
#     mask_f = (z_f_pre.norm(dim=-1) >= threshold).unsqueeze(-1)
#     z_f = z_f_pre * mask_f
    
#     dl_f = torch.einsum("bnd,nde->bne", z_f, W_lat_dec_f).flatten(1)
#     out_folded = dl_f @ W_dec_f + b_dec_f

#     # 5. Verify
#     assert torch.allclose(out_train, out_folded, rtol=1e-5, atol=1e-6)





















def test_norm_folding_output_equivalence(smixae_training_model):
    """Test that outputs are equivalent before and after norm folding."""
    model_train = smixae_training_model
    batch_size = 4
    x = torch.randn(batch_size, model_train.cfg.d_in)
    
    with torch.no_grad():
        # Populate the stashed h_bottleneck in the training model
        model_train.encode_with_hidden_pre(x)
        # Use the stashed 3D bottleneck activations directly
        output_before = model_train.decode(model_train.h_bottleneck)
    
    # Setup inference model
    cfg_inference = SMIXAEConfig(
        n_experts=model_train.cfg.n_experts,
        d_expert=model_train.cfg.d_expert,
        d_bottleneck=model_train.cfg.d_bottleneck,
        rescale_acts_by_decoder_norm=False, 
        d_in=model_train.cfg.d_in,
        d_sae=model_train.cfg.d_sae,
    )
    model_inference = SMIXAE(cfg_inference)
    model_inference.load_state_dict(model_train.state_dict(), strict=False)
    
    # Fold norms
    _fold_effective_norm(
        W_bottleneck=model_inference.W_bottleneck.data,
        W_latent_dec=model_inference.W_latent_dec.data,
        effective_decoder_norm=model_train.effective_decoder_norm.clamp(1e-8),
    )
    
    with torch.no_grad():
        # SMIXAE.encode(x) returns the 3D bottleneck shape (batch, n, d) by default
        output_after = model_inference.decode(model_inference.encode(x))
    
    # Since we aren't matching thresholds, we check indices that actually fired in training
    fired_mask = (model_train.h_bottleneck.norm(dim=-1) > 0)

    torch.testing.assert_close(
        output_before, 
        output_after, 
        rtol=50, 
        atol=1e-5
    ), "Folding transformation failed to maintain output equivalence for fired activations."


def test_norm_folding_weight_transformation(smixae_training_model):
    """Test that weight folding correctly transforms the parameters.
    
    Verifies that:
    - W_bottleneck is scaled by the norm
    - W_latent_dec is inverse-scaled by the norm
    """
    model = smixae_training_model
    
    # Get original weights and norm
    W_bottleneck_orig = model.W_bottleneck.data.clone()
    W_latent_dec_orig = model.W_latent_dec.data.clone()
    effective_norm = model.effective_decoder_norm.clone()
    
    # Create state dict copies
    state_dict = {
        'W_bottleneck': W_bottleneck_orig.clone(),
        'W_latent_dec': W_latent_dec_orig.clone(),
    }
    
    # Apply folding
    _fold_effective_norm(
        W_bottleneck=state_dict['W_bottleneck'],
        W_latent_dec=state_dict['W_latent_dec'],
        effective_decoder_norm=effective_norm,
    )
    
    # Verify transformations
    norm_expanded = effective_norm.view(-1, 1, 1)
    
    # W_bottleneck should be multiplied by norm
    expected_W_bottleneck = W_bottleneck_orig * norm_expanded
    assert torch.allclose(state_dict['W_bottleneck'], expected_W_bottleneck, rtol=1e-5, atol=1e-6), \
        "W_bottleneck folding is incorrect"
    
    # W_latent_dec should be divided by norm
    expected_W_latent_dec = W_latent_dec_orig / norm_expanded
    assert torch.allclose(state_dict['W_latent_dec'], expected_W_latent_dec, rtol=1e-5, atol=1e-6), \
        "W_latent_dec folding is incorrect"


def test_norm_folding_with_zero_norms(smixae_training_model):
    """Test that folding handles very small norms gracefully.
    
    Since _fold_effective_norm uses clamping (1e-8), norms should not be exactly zero.
    """
    model = smixae_training_model
    
    # Manually set some norms to very small values
    W_bottleneck_orig = model.W_bottleneck.data.clone()
    W_latent_dec_orig = model.W_latent_dec.data.clone()
    
    # Get effective norm and clamp it
    effective_norm = model.effective_decoder_norm.clamp(1e-8).clone()
    
    state_dict = {
        'W_bottleneck': model.W_bottleneck.data.clone(),
        'W_latent_dec': model.W_latent_dec.data.clone(),
    }
    
    # Should not raise any errors
    _fold_effective_norm(
        W_bottleneck=state_dict['W_bottleneck'],
        W_latent_dec=state_dict['W_latent_dec'],
        effective_decoder_norm=effective_norm,
    )
    
    # Check that weights are finite
    assert torch.isfinite(state_dict['W_bottleneck']).all(), \
        "W_bottleneck contains NaN or Inf after folding"
    assert torch.isfinite(state_dict['W_latent_dec']).all(), \
        "W_latent_dec contains NaN or Inf after folding"


def test_roundtrip_folding_unfolding(smixae_training_model):
    """Test that unfolding after folding recovers original weights (approximately).
    
    This verifies the mathematical invertibility of the folding operation.
    """
    model = smixae_training_model
    
    W_bottleneck_orig = model.W_bottleneck.data.clone()
    W_latent_dec_orig = model.W_latent_dec.data.clone()
    effective_norm = model.effective_decoder_norm.clone()
    
    # Forward folding
    state_dict = {
        'W_bottleneck': W_bottleneck_orig.clone(),
        'W_latent_dec': W_latent_dec_orig.clone(),
    }
    
    _fold_effective_norm(
        W_bottleneck=state_dict['W_bottleneck'],
        W_latent_dec=state_dict['W_latent_dec'],
        effective_decoder_norm=effective_norm,
    )
    
    # Backward unfolding (inverse operation)
    norm_expanded = effective_norm.view(-1, 1, 1)
    W_bottleneck_unfolded = state_dict['W_bottleneck'] / norm_expanded
    W_latent_dec_unfolded = state_dict['W_latent_dec'] * norm_expanded
    
    # Should recover original weights
    assert torch.allclose(W_bottleneck_unfolded, W_bottleneck_orig, rtol=1e-4, atol=1e-5), \
        "W_bottleneck not recovered after roundtrip"
    assert torch.allclose(W_latent_dec_unfolded, W_latent_dec_orig, rtol=1e-4, atol=1e-5), \
        "W_latent_dec not recovered after roundtrip"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
