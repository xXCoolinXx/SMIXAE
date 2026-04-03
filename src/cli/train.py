"""Training CLI for SMIXAE. Exposes all LanguageModelSAERunnerConfig and
SMIXAETrainingConfig options. Run-specific args (model, hook, architecture
scale, token budget) are required; all others default to the standard
experiment settings."""

import re as _re
from typing import Optional

import sae_lens.training.activations_store as _acts_store
import torch
import typer
from datasets import load_from_disk as _load_from_disk
from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, LoggingConfig

from smixae import AffineSMIXAETrainingConfig, SMIXAETrainingConfig  # also registers architecture via __init__

# Patch ActivationsStore to auto-detect datasets saved with save_to_disk (state.json sentinel)
# and redirect to load_from_disk, so callers don't need to distinguish loading methods.

_orig_load_dataset = _acts_store.load_dataset


def _auto_load_dataset(path, *args, **kwargs):
    if isinstance(path, str):
        try:
            return _load_from_disk(path)
        except Exception:
            pass
    return _orig_load_dataset(path, *args, **kwargs)


_acts_store.load_dataset = _auto_load_dataset

# Patch tokenizer validation to support local save_to_disk datasets.
# SAELens only catches HfHubHTTPError, but local paths trigger HFValidationError.
# We instead read sae_lens.json directly from the local directory when present.
_orig_validate = _acts_store.validate_pretokenized_dataset_tokenizer


def _smart_validate(dataset_path: str, model_tokenizer) -> None:  # type: ignore[type-arg]
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    local_cfg = Path(dataset_path) / "sae_lens.json"
    if local_cfg.exists():
        tokenizer_name = json.loads(local_cfg.read_text())["tokenizer_name"]
        try:
            ds_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if ds_tokenizer.get_vocab() != model_tokenizer.get_vocab():
                raise ValueError(
                    f"Dataset tokenizer '{tokenizer_name}' does not match model tokenizer."
                )
        except Exception:
            pass  # Can't load tokenizer to compare — skip check
        return
    _orig_validate(dataset_path, model_tokenizer)


_acts_store.validate_pretokenized_dataset_tokenizer = _smart_validate

# Patch stop_at_layer extraction to support HuggingFace-style hook names like
# "model.layers.11" (no trailing dot). SAELens regex r"\.(\d+)\." requires a dot
# after the layer number, so it returns None for these names → full forward pass
# including lm_head → OOM on large vocab (Gemma 2: 256k tokens).
_orig_extract_stop = _acts_store.extract_stop_at_layer_from_tlens_hook_name


def _smart_extract_stop(hook_name: str) -> int | None:
    result = _orig_extract_stop(hook_name)
    if result is not None:
        return result
    # HuggingFace format ends with ".N" (no trailing dot)
    match = _re.search(r"\.(\d+)$", hook_name)
    return None if match is None else int(match.group(1)) + 1


_acts_store.extract_stop_at_layer_from_tlens_hook_name = _smart_extract_stop

app = typer.Typer()


@app.command()
def train(
    # ------------------------------------------------------------------ #
    # Required: run-specific settings (no defaults)                       #
    # ------------------------------------------------------------------ #
    model_name: str = typer.Option(..., help="HuggingFace model name.", rich_help_panel="Model"),
    hook_name: str = typer.Option(..., help="Hook point for activation collection.", rich_help_panel="Model"),
    training_tokens: int = typer.Option(..., help="Total number of training tokens.", rich_help_panel="Training"),
    n_experts: int = typer.Option(..., help="Number of experts.", rich_help_panel="SAE Architecture"),
    d_in: int = typer.Option(
        ..., help="Input dimensionality (d_model of the LLM).", rich_help_panel="SAE Architecture"
    ),
    d_expert: int = typer.Option(..., help="Dimensionality of each expert.", rich_help_panel="SAE Architecture"),
    k_experts: int = typer.Option(
        ..., help="Active experts per forward pass (BatchTopK).", rich_help_panel="SAE Architecture"
    ),
    # ------------------------------------------------------------------ #
    # Convenience multiplier                                              #
    # ------------------------------------------------------------------ #
    factor: int = typer.Option(
        1,
        help=(
            "Batch-size multiplier. Scales train_batch_size_tokens (×8192), lr (×5e-4), "
            "lr_warm_up_steps (÷500), lr_decay_steps, and dead_after_n_passes (÷500) together. "
            "Individual options below override the factor-derived values when set."
        ),
        rich_help_panel="Training",
    ),
    # ------------------------------------------------------------------ #
    # SAE architecture                                                    #
    # ------------------------------------------------------------------ #
    d_bottleneck: int = typer.Option(
        3, help="Bottleneck dimensionality (3 for 3D visualization).", rich_help_panel="SAE Architecture"
    ),
    aux_loss_coefficient: float = typer.Option(
        3 * 3e-6, help="Auxiliary loss coefficient for dead-expert recovery. Heuristic: d_bottleneck / 32 for topk aux or d_bottleneck * 3e-6 for pre_act loss", rich_help_panel="SAE Architecture"
    ),
    rescale_acts_by_decoder_norm: bool = typer.Option(
        True, help="Rescale bottleneck activations by decoder norm.", rich_help_panel="SAE Architecture"
    ),
    threshold_lr: float = typer.Option(
        1e-1, help="Learning rate for the inference threshold update.", rich_help_panel="SAE Architecture"
    ),
    dead_after_n_passes: Optional[int] = typer.Option(
        None,
        help="Passes without firing before an expert is considered dead. Defaults to 500 // factor.",
        rich_help_panel="SAE Architecture",
    ),
    normalize_activations: str = typer.Option(
        "expected_average_only_in",
        help="Activation normalization mode: none | expected_average_only_in | layer_norm.",
        rich_help_panel="SAE Architecture",
    ),
    decoder_init_norm: Optional[float] = typer.Option(
        0.1, help="Initial decoder weight norm.", rich_help_panel="SAE Architecture"
    ),
    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #
    train_batch_size_tokens: Optional[int] = typer.Option(
        None, help="Token batch size. Defaults to 8192 * factor.", rich_help_panel="Training"
    ),
    lr: Optional[float] = typer.Option(
        None, help="Learning rate. Defaults to 5e-4 * factor.", rich_help_panel="Training"
    ),
    lr_warm_up_steps: Optional[int] = typer.Option(
        None, help="LR warm-up steps. Defaults to 500 // factor.", rich_help_panel="Training"
    ),
    lr_decay_steps: Optional[int] = typer.Option(
        None, help="LR decay steps. Defaults to 20% of total training steps.", rich_help_panel="Training"
    ),
    lr_scheduler_name: str = typer.Option(
        "constant", help="LR scheduler: constant | cosine | linear.", rich_help_panel="Training"
    ),
    lr_end: Optional[float] = typer.Option(
        None, help="Final LR (used with cosine/linear schedulers).", rich_help_panel="Training"
    ),
    n_restart_cycles: int = typer.Option(1, help="Number of cosine-restart cycles.", rich_help_panel="Training"),
    adam_beta1: float = typer.Option(0.9, help="Adam β₁.", rich_help_panel="Training"),
    adam_beta2: float = typer.Option(0.999, help="Adam β₂.", rich_help_panel="Training"),
    dead_feature_window: int = typer.Option(
        1000, help="Window size for dead-feature detection (SAELens).", rich_help_panel="Training"
    ),
    feature_sampling_window: int = typer.Option(
        2000, help="Feature-sampling window (SAELens).", rich_help_panel="Training"
    ),
    dead_feature_threshold: float = typer.Option(
        1e-8, help="Activation frequency below which a feature is dead.", rich_help_panel="Training"
    ),
    seed: int = typer.Option(42, help="Random seed.", rich_help_panel="Training"),
    resume_from_checkpoint: Optional[str] = typer.Option(
        None, help="Path to a checkpoint to resume training from.", rich_help_panel="Training"
    ),
    # ------------------------------------------------------------------ #
    # Model / device                                                      #
    # ------------------------------------------------------------------ #
    model_class_name: str = typer.Option(
        "AutoModelForCausalLM", help="Model class used by SAELens.", rich_help_panel="Model"
    ),
    hook_head_index: Optional[int] = typer.Option(
        None, help="Attention head index (None for residual-stream hooks).", rich_help_panel="Model"
    ),
    device: str = typer.Option("cuda", help="Torch device for training.", rich_help_panel="Model"),
    act_store_device: str = typer.Option("cpu", help="Device for the activation store.", rich_help_panel="Model"),
    dtype: str = typer.Option("bfloat16", help="Training dtype.", rich_help_panel="Model"),
    autocast: bool = typer.Option(True, help="Use torch autocast for the SAE.", rich_help_panel="Model"),
    autocast_lm: bool = typer.Option(True, help="Use torch autocast for the LLM.", rich_help_panel="Model"),
    compile_llm: bool = typer.Option(True, help="torch.compile the LLM.", rich_help_panel="Model"),
    compile_sae: bool = typer.Option(True, help="torch.compile the SAE.", rich_help_panel="Model"),
    llm_compilation_mode: Optional[str] = typer.Option(
        None, help="Compilation mode for the LLM (None = default).", rich_help_panel="Model"
    ),
    sae_compilation_mode: Optional[str] = typer.Option(
        None, help="Compilation mode for the SAE (None = default).", rich_help_panel="Model"
    ),
    prepend_bos: bool = typer.Option(True, help="Prepend BOS token.", rich_help_panel="Model"),
    # ------------------------------------------------------------------ #
    # Data                                                                #
    # ------------------------------------------------------------------ #
    dataset_path: str = typer.Option(
        "monology/pile-uncopyrighted", help="HuggingFace dataset path.", rich_help_panel="Data"
    ),
    is_dataset_tokenized: bool = typer.Option(
        False, help="Whether the dataset is pre-tokenized.", rich_help_panel="Data"
    ),
    context_size: int = typer.Option(128, help="Context size in tokens.", rich_help_panel="Data"),
    n_batches_in_buffer: int = typer.Option(1024, help="Activation buffer size in batches.", rich_help_panel="Data"),
    store_batch_size_prompts: int = typer.Option(
        128, help="Prompt batch size for the activation store.", rich_help_panel="Data"
    ),
    n_batches_for_norm_estimate: int = typer.Option(
        100, help="Batches used to estimate activation norms.", rich_help_panel="Data"
    ),
    disable_concat_sequences: bool = typer.Option(
        False, help="Disable sequence concatenation in the store.", rich_help_panel="Data"
    ),
    dataset_trust_remote_code: bool = typer.Option(
        True, help="Trust remote code when loading the dataset.", rich_help_panel="Data"
    ),
    streaming: bool = typer.Option(True, help="Stream the dataset.", rich_help_panel="Data"),
    use_chat_formatting: bool = typer.Option(False, help="Apply chat formatting to prompts.", rich_help_panel="Data"),
    # ------------------------------------------------------------------ #
    # Eval                                                                #
    # ------------------------------------------------------------------ #
    n_eval_batches: int = typer.Option(10, help="Number of batches for evaluation.", rich_help_panel="Eval"),
    eval_batch_size_prompts: Optional[int] = typer.Option(
        None, help="Prompt batch size for evaluation (None = same as store).", rich_help_panel="Eval"
    ),
    # ------------------------------------------------------------------ #
    # Checkpointing                                                       #
    # ------------------------------------------------------------------ #
    checkpoint_path: str = typer.Option(
        "checkpoints", help="Directory for checkpoint output.", rich_help_panel="Checkpointing"
    ),
    n_checkpoints: int = typer.Option(3, help="Number of intermediate checkpoints.", rich_help_panel="Checkpointing"),
    save_final_checkpoint: bool = typer.Option(
        True, help="Save an additional checkpoint at end of training.", rich_help_panel="Checkpointing"
    ),
    output_path: str = typer.Option(
        "output", help="Output directory for other artifacts.", rich_help_panel="Checkpointing"
    ),
    verbose: bool = typer.Option(True, help="Verbose SAELens runner output.", rich_help_panel="Checkpointing"),
    # ------------------------------------------------------------------ #
    # Logging (W&B)                                                       #
    # ------------------------------------------------------------------ #
    log_to_wandb: bool = typer.Option(True, help="Log to Weights & Biases.", rich_help_panel="Logging"),
    wandb_project: str = typer.Option(
        "SMIXAE on Gemma 2-9B, Batch Top K", help="W&B project name.", rich_help_panel="Logging"
    ),
    wandb_id: Optional[str] = typer.Option(None, help="W&B run ID (for resuming).", rich_help_panel="Logging"),
    run_name: Optional[str] = typer.Option(None, help="W&B run name.", rich_help_panel="Logging"),
    wandb_entity: Optional[str] = typer.Option(None, help="W&B entity/team.", rich_help_panel="Logging"),
    wandb_log_frequency: int = typer.Option(30, help="Log every N training steps.", rich_help_panel="Logging"),
    eval_every_n_wandb_logs: int = typer.Option(
        5_000_000, help="Run evals every N W&B logs.", rich_help_panel="Logging"
    ),
    log_activations_store_to_wandb: bool = typer.Option(
        False, help="Log activation store stats to W&B.", rich_help_panel="Logging"
    ),
    log_optimizer_state_to_wandb: bool = typer.Option(
        False, help="Log optimizer state to W&B.", rich_help_panel="Logging"
    ),
    log_weights_to_wandb: bool = typer.Option(True, help="Log model weights to W&B.", rich_help_panel="Logging"),
    use_affine_smixae : str = typer.Option("false", help="Whether to use affine smixae or not, defaults false (bool)", rich_help_panel="Model"),
) -> None:
    """Train a SMIXAE on a language model using SAELens."""
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True  # silence graph-break warning

    # Resolve factor-derived values (explicit option overrides factor)
    actual_batch_size = train_batch_size_tokens if train_batch_size_tokens is not None else 8192 * factor
    actual_lr = lr if lr is not None else 5e-4 * factor
    total_training_steps = training_tokens // actual_batch_size
    actual_lr_warm_up_steps = lr_warm_up_steps if lr_warm_up_steps is not None else 500 // factor
    actual_lr_decay_steps = lr_decay_steps if lr_decay_steps is not None else total_training_steps // 5
    actual_dead_after_n_passes = dead_after_n_passes if dead_after_n_passes is not None else 1000 // factor

    config_type = SMIXAETrainingConfig if not use_affine_smixae.lower() == 'true' else AffineSMIXAETrainingConfig
    cfg = LanguageModelSAERunnerConfig(
        sae=config_type(
            d_in=d_in,
            d_sae=d_expert * n_experts,  # derived; overridden in SMIXAETraining.__init__
            n_experts=n_experts,
            d_expert=d_expert,
            d_bottleneck=d_bottleneck,
            k_experts=k_experts,
            aux_loss_coefficient=aux_loss_coefficient,
            rescale_acts_by_decoder_norm=rescale_acts_by_decoder_norm,
            normalize_activations=normalize_activations,
            dead_after_n_passes=actual_dead_after_n_passes,
            threshold_lr=threshold_lr,
            decoder_init_norm=decoder_init_norm,
        ),
        # Model
        model_name=model_name,
        model_class_name=model_class_name,
        hook_name=hook_name,
        hook_head_index=hook_head_index,
        device=device,
        act_store_device=act_store_device,
        dtype=dtype,
        autocast=autocast,
        autocast_lm=autocast_lm,
        compile_llm=compile_llm,
        compile_sae=compile_sae,
        llm_compilation_mode=llm_compilation_mode,
        sae_compilation_mode=sae_compilation_mode,
        prepend_bos=prepend_bos,
        # Data
        dataset_path=dataset_path,
        is_dataset_tokenized=is_dataset_tokenized,
        context_size=context_size,
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=store_batch_size_prompts,
        n_batches_for_norm_estimate=n_batches_for_norm_estimate,
        disable_concat_sequences=disable_concat_sequences,
        dataset_trust_remote_code=dataset_trust_remote_code,
        streaming=streaming,
        use_chat_formatting=use_chat_formatting,
        # Training
        train_batch_size_tokens=actual_batch_size,
        lr=actual_lr,
        lr_warm_up_steps=actual_lr_warm_up_steps,
        lr_decay_steps=actual_lr_decay_steps,
        lr_scheduler_name=lr_scheduler_name,
        lr_end=lr_end,
        n_restart_cycles=n_restart_cycles,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        training_tokens=training_tokens,
        dead_feature_window=dead_feature_window,
        feature_sampling_window=feature_sampling_window,
        dead_feature_threshold=dead_feature_threshold,
        seed=seed,
        resume_from_checkpoint=resume_from_checkpoint,
        # Eval
        n_eval_batches=n_eval_batches,
        eval_batch_size_prompts=eval_batch_size_prompts,
        # Checkpointing
        checkpoint_path=checkpoint_path,
        n_checkpoints=n_checkpoints,
        save_final_checkpoint=save_final_checkpoint,
        output_path=output_path,
        verbose=verbose,
        # Logging
        logger=LoggingConfig(
            log_to_wandb=log_to_wandb,
            wandb_project=wandb_project,
            wandb_id=wandb_id,
            run_name=run_name,
            wandb_entity=wandb_entity,
            wandb_log_frequency=wandb_log_frequency,
            eval_every_n_wandb_logs=eval_every_n_wandb_logs,
            log_activations_store_to_wandb=log_activations_store_to_wandb,
            log_optimizer_state_to_wandb=log_optimizer_state_to_wandb,
            log_weights_to_wandb=log_weights_to_wandb,
        ),
    )

    LanguageModelSAETrainingRunner(cfg).run()  # type: ignore
