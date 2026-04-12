"""Pretokenize a HuggingFace dataset and save to disk via SAELens.

Delegates entirely to ``PretokenizeRunner`` / ``PretokenizeRunnerConfig`` from
SAELens, which handles downloading, parallel tokenization (via ``dataset.map``
with ``num_proc``), chunking, and saving in the format expected by SAELens
training (``is_dataset_tokenized=True``).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import typer
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from sae_lens.config import PretokenizeRunnerConfig
from sae_lens.pretokenize_runner import (
    PretokenizeRunner,
    metadata_from_config,
    pretokenize_dataset,
)
from transformers import AutoTokenizer

app = typer.Typer()


@dataclass
class LimitedPretokenizeRunnerConfig(PretokenizeRunnerConfig):
    """Extends PretokenizeRunnerConfig with a token-count cap for partial dataset tokenization."""

    n_tokens: int = 1_000_000_000


class LimitedPretokenizeRunner(PretokenizeRunner):
    """PretokenizeRunner variant that streams and materialises only ``n_tokens`` tokens."""

    cfg: LimitedPretokenizeRunnerConfig

    def run(self) -> Dataset:
        """Run the limited pretokenization pipeline and return (and optionally save) the result.

        Streams the source dataset, takes the minimum number of source documents needed
        to hit ``cfg.n_tokens``, tokenizes, truncates to exactly ``n_seqs`` sequences, and
        saves to ``cfg.save_path`` if set.

        Returns:
            A :class:`datasets.Dataset` with exactly ``n_seqs`` rows of ``context_size`` tokens,
            or fewer if the source dataset was exhausted.
        """
        n_seqs = self.cfg.n_tokens // self.cfg.context_size
        n_docs = n_seqs

        # Stream with .take() so we never index the full dataset, then materialise
        # the small subset into a regular Dataset for full num_proc tokenization.
        iterable_ds = hf_load_dataset(
            self.cfg.dataset_path, name=self.cfg.dataset_name, data_dir=self.cfg.data_dir,
            data_files=self.cfg.data_files, split=self.cfg.split, streaming=True,
        )
        print(f"Taking {n_docs:,} source documents (target: {n_seqs:,} sequences).")
        source = Dataset.from_generator(
            lambda: iterable_ds.take(n_docs),
            features=iterable_ds.features,
        )
        print(f"Materialised {len(source):,} source documents.")

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
        tokenizer.model_max_length = sys.maxsize

        tokenized = pretokenize_dataset(cast(Dataset, source), tokenizer, self.cfg)

        if len(tokenized) < n_seqs:
            print(f"Warning: only produced {len(tokenized):,} sequences (target {n_seqs:,}). "
                  "Saving what we have. Raise --avg-doc-tokens to get more next time.")
        else:
            tokenized = tokenized.select(range(n_seqs))

        if self.cfg.shuffle:
            tokenized = tokenized.shuffle(seed=self.cfg.seed)

        if self.cfg.save_path is not None:
            tokenized.save_to_disk(self.cfg.save_path, num_proc=self.cfg.num_proc)
            with open(Path(self.cfg.save_path) / "sae_lens.json", "w") as f:
                json.dump(metadata_from_config(self.cfg).__dict__, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(tokenized):,} sequences → {self.cfg.save_path}")

        return tokenized  # type: ignore[return-value]


@app.command()
def pretokenize(
    model_name: str = typer.Option(..., help="HuggingFace model whose tokenizer to use."),
    output_path: str = typer.Option(..., help="Directory to save the tokenized dataset."),
    dataset_path: str = typer.Option("monology/pile-uncopyrighted", help="Source HuggingFace dataset."),
    split: str = typer.Option("train", help="Dataset split to use."),
    n_tokens: int = typer.Option(1_000_000_000, help="Target number of tokens to produce."),
    context_size: int = typer.Option(128, help="Sequence length in tokens (tokens per row)."),
    text_column: str = typer.Option("text", help="Column name for raw text in the source dataset."),
    batch_size: int = typer.Option(1000, help="Documents per tokenization batch."),
    num_proc: int = typer.Option(
        int(os.environ.get("PBS_NCPUS", os.cpu_count() or 8)),
        help="Parallel workers for tokenization and saving. Defaults to $PBS_NCPUS if set.",
    ),
    shuffle: bool = typer.Option(True, help="Shuffle the tokenized dataset."),
    seed: int = typer.Option(42, help="Random seed for shuffling."),
) -> None:
    """Tokenize a dataset and save to disk for fast SAELens training."""
    cfg = LimitedPretokenizeRunnerConfig(
        tokenizer_name=model_name,
        dataset_path=dataset_path,
        split=split,
        n_tokens=n_tokens,
        context_size=context_size,
        column_name=text_column,
        pretokenize_batch_size=batch_size,
        num_proc=num_proc,
        shuffle=shuffle,
        seed=seed,
        streaming=False,
        save_path=output_path,

    )
    LimitedPretokenizeRunner(cfg).run()


if __name__ == "__main__":
    app()
