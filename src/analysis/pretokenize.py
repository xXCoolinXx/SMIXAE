"""Pretokenize a streaming HuggingFace dataset and save to disk.

Streams text, tokenizes with the specified model's tokenizer, concatenates all
tokens, chunks into fixed-length sequences, and saves as a HuggingFace Dataset
with a ``tokens`` column — ready for SAELens with ``is_dataset_tokenized=True``.
"""
from __future__ import annotations

from collections.abc import Generator

import typer
from datasets import Dataset
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

app = typer.Typer()


def _generate_sequences(
    tokenizer,
    raw_dataset,
    n_seqs: int,
    context_size: int,
    text_column: str,
    batch_size: int,
) -> Generator[dict, None, None]:
    """Yield ``{"tokens": list[int]}`` dicts of length ``context_size``."""
    overflow: list[int] = []
    yielded = 0
    with tqdm(total=n_seqs * context_size, desc="Tokenizing", unit="tok", unit_scale=True) as pbar:
        for batch in raw_dataset.iter(batch_size=batch_size):
            encoded = tokenizer(
                batch[text_column],
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            for doc_ids in encoded:
                overflow.extend(doc_ids)
            while len(overflow) >= context_size:
                yield {"tokens": overflow[:context_size]}
                overflow = overflow[context_size:]
                pbar.update(context_size)
                yielded += 1
                if yielded >= n_seqs:
                    return


@app.command()
def pretokenize(
    model_name: str = typer.Option(..., help="HuggingFace model whose tokenizer to use."),
    output_path: str = typer.Option(..., help="Directory to save the tokenized dataset."),
    dataset_path: str = typer.Option("monology/pile-uncopyrighted", help="Source HuggingFace dataset."),
    n_tokens: int = typer.Option(1_000_000_000, help="Target number of tokens to produce."),
    context_size: int = typer.Option(128, help="Sequence length in tokens (tokens per row)."),
    text_column: str = typer.Option("text", help="Column name for raw text in the source dataset."),
    batch_size: int = typer.Option(512, help="Documents per tokenization batch."),
    split: str = typer.Option("train", help="Dataset split to use."),
) -> None:
    """Tokenize a dataset and save to disk for fast SAELens training."""
    n_seqs = n_tokens // context_size

    print(f"Loading tokenizer from {model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Streaming {dataset_path} ({split})…")
    raw = hf_load_dataset(
        dataset_path,
        streaming=True,
        split=split,
        trust_remote_code=True,
    )

    print(
        f"Tokenizing — target {n_seqs:,} × {context_size}-token sequences"
        f" ({n_tokens:,} tokens total)…"
    )
    ds = Dataset.from_generator(
        _generate_sequences,
        gen_kwargs=dict(
            tokenizer=tokenizer,
            raw_dataset=raw,
            n_seqs=n_seqs,
            context_size=context_size,
            text_column=text_column,
            batch_size=batch_size,
        ),
    )

    print(f"Saving {len(ds):,} sequences → {output_path}")
    ds.save_to_disk(output_path)
    print("Done.")


if __name__ == "__main__":
    app()
