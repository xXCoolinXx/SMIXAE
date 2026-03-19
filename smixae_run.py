"""Thin shim for PBS / direct invocation. Calls `smixae train` with the
standard Gemma 2-9B experiment settings. For full control use the CLI:

    smixae train --help
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cli.cli import app

sys.argv = [
    "smixae", "train",
    "--model-name", "google/gemma-2-9b",
    "--hook-name", "model.layers.11",
    "--training-tokens", "500000000",
    "--n-experts", "4096",
    "--d-in", "3584",
    "--d-expert", "8",
    "--k-experts", "128",
]

app()
