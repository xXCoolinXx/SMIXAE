"""Centralised CLI entry point for the ``smixae`` command.

Assembles all sub-apps (train, probe, newline, steer, generate-probing-data,
generate-steering-data, pretokenize, latex) into a single Typer application
registered as the ``smixae`` console script in ``pyproject.toml``.
"""
import typer

from analysis.anthropic_newline import app as newline
from analysis.categorize_all import app as probe
from analysis.generate_probing_data import app as generate_probing_data
from analysis.generate_steering_data import app as generate_steering_data
from analysis.pretokenize import app as pretokenize
from analysis.steer import app as steer
from cli.saebench import app as saebench
from cli.train import train
from latex.camera_ready import figures as latex_figures
from latex.save_server import save_server as latex_save_server
from latex.tables import generate as latex_tables
from latex.tables import generate_saebench as latex_saebench_table

app = typer.Typer(name="smixae", help="SMIXAE research toolkit.")

app.command(name="train", help="Train a SMIXAE.")(train)
app.add_typer(probe, name="probe", help="Expert probing and visualization.")
app.add_typer(newline, name="newline", help="Newline-position manifold analysis.")
app.add_typer(steer, name="steer", help="Steering experiments.")
app.add_typer(saebench, name="saebench", help="SAEBench core evaluation (no TransformerLens).")
app.add_typer(generate_probing_data, name="generate-probing-data", help="Generate labeled probing datasets.")
app.add_typer(generate_steering_data, name="generate-steering-data", help="Generate steering prompt datasets.")
app.add_typer(pretokenize, name="pretokenize", help="Pretokenize a dataset for fast SAELens training.")

latex_app = typer.Typer(help="LaTeX output utilities (tables and camera-ready figures).")
latex_app.command(name="tables", help="Generate probing and newline LaTeX tables.")(latex_tables)
latex_app.command(name="saebench-table", help="Generate SAEBench core metrics LaTeX table.")(latex_saebench_table)
latex_app.command(name="figures", help="Assemble camera-ready PNGs into LaTeX figure files.")(latex_figures)
latex_app.command(name="save-server", help="Start the local save server for camera-ready figure collection.")(latex_save_server)
app.add_typer(latex_app, name="latex", help="LaTeX output utilities.")


if __name__ == "__main__":
    app()
