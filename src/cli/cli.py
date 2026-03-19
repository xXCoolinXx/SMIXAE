import typer

from analysis.categorize_all import app as probe
from analysis.anthropic_newline import app as newline
from analysis.generate_probing_data import app as generate_probing_data

app = typer.Typer(name="smixae", help="SMIXAE research toolkit.")

app.add_typer(probe, name="probe", help="Expert probing and visualization.")
app.add_typer(newline, name="newline", help="Newline-position manifold analysis.")
app.add_typer(generate_probing_data, name="generate-probing-data", help="Generate labeled probing datasets.")


if __name__ == "__main__":
    app()
