import typer

from analysis.categorize_all import app as probe
from analysis.anthropic_newline import app as newline

app = typer.Typer(name="smixae", help="SMIXAE research toolkit.")

app.add_typer(probe, name="probe", help="Expert probing and visualization.")
app.add_typer(newline, name="newline", help="Newline-position manifold analysis.")


@app.command()
def generate_probing_data():
    """Generate all probing datasets and write them to datasets/probing/."""
    from analysis.generate_data import main
    main()


if __name__ == "__main__":
    app()
