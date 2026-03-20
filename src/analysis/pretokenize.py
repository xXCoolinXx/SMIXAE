"""Pretokenize a HuggingFace dataset and save to disk via SAELens.

Delegates entirely to ``PretokenizeRunner`` / ``PretokenizeRunnerConfig`` from
SAELens, which handles downloading, parallel tokenization (via ``dataset.map``
with ``num_proc``), chunking, and saving in the format expected by SAELens
training (``is_dataset_tokenized=True``).
"""
from __future__ import annotations

import os

import typer
from sae_lens.config import PretokenizeRunnerConfig
from sae_lens.pretokenize_runner import PretokenizeRunner

app = typer.Typer()


@app.command()
def pretokenize(
    model_name: str = typer.Option(..., help="HuggingFace model whose tokenizer to use."),
    output_path: str = typer.Option(..., help="Directory to save the tokenized dataset."),
    dataset_path: str = typer.Option("monology/pile-uncopyrighted", help="Source HuggingFace dataset."),
    split: str = typer.Option("train", help="Dataset split to use."),
    context_size: int = typer.Option(128, help="Sequence length in tokens (tokens per row)."),
    text_column: str = typer.Option("text", help="Column name for raw text in the source dataset."),
    batch_size: int = typer.Option(1000, help="Documents per tokenization batch."),
    num_proc: int = typer.Option(
        os.cpu_count() or 8,
        help="Parallel workers for tokenization and saving.",
    ),
    shuffle: bool = typer.Option(True, help="Shuffle the tokenized dataset."),
    seed: int = typer.Option(42, help="Random seed for shuffling."),
) -> None:
    """Tokenize a dataset and save to disk for fast SAELens training."""
    cfg = PretokenizeRunnerConfig(
        tokenizer_name=model_name,
        dataset_path=dataset_path,
        split=split,
        context_size=context_size,
        column_name=text_column,
        pretokenize_batch_size=batch_size,
        num_proc=num_proc,
        shuffle=shuffle,
        seed=seed,
        streaming=False,  # allows num_proc > 1 for parallel tokenization
        save_path=output_path,
    )
    PretokenizeRunner(cfg).run()


if __name__ == "__main__":
    app()
