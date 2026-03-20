"""Generate steering experiment datasets.

Writes prompt CSVs to datasets/steering/:

  hours_current_time.csv
      Task 1: "Right now it is {src_hour}. What time is it?"
      Columns: Prompt, Source_Hour, Target_Hour
      One row per (src_hour, tgt_hour) pair (24 × 23 = 552 rows).

  hours_elapsed_time.csv
      Task 2: "Right now it is {curr_hour}. How much time has it been since {start_hour}?"
      Columns: Prompt, Current_Hour, Start_Hour, Expected_Hours
      One row per (curr_hour, delta) where delta = 1 … max_delta.
"""
from pathlib import Path

import pandas as pd
import typer

# Canonical 24-hour ordering — index determines the ordering prefix in probing labels.
HOURS = [
    "1AM", "2AM", "3AM", "4AM", "5AM", "6AM",
    "7AM", "8AM", "9AM", "10AM", "11AM", "12PM",
    "1PM", "2PM", "3PM", "4PM", "5PM", "6PM",
    "7PM", "8PM", "9PM", "10PM", "11PM", "12AM",
]

app = typer.Typer()


def generate_hours_current_time() -> pd.DataFrame:
    """Task 1: all (src_hour, tgt_hour) pairs — 24 × 23 = 552 rows."""
    rows = []
    for src in HOURS:
        for tgt in HOURS:
            if src == tgt:
                continue
            rows.append(
                dict(
                    Prompt=f"Right now it is {src}. What time is it?",
                    Source_Hour=src,
                    Target_Hour=tgt,
                )
            )
    return pd.DataFrame(rows)


def generate_hours_elapsed_time(max_delta: int = 12) -> pd.DataFrame:
    """Task 2: (curr_hour, start_hour) pairs for delta = 1 … max_delta.

    The baseline answer is always ``delta`` hours.
    The steered target shifts the perceived current time by +delta, so the
    model's answer should become 0 (or some other value depending on the
    steering direction).

    Args:
        max_delta: Maximum elapsed-time delta to generate (default 12 hours).
    """
    n = len(HOURS)
    rows = []
    for curr_idx, curr in enumerate(HOURS):
        for delta in range(1, max_delta + 1):
            start_idx = (curr_idx - delta) % n
            start = HOURS[start_idx]
            rows.append(
                dict(
                    Prompt=(
                        f"Right now it is {curr}. "
                        f"How much time has it been since {start}?"
                    ),
                    Current_Hour=curr,
                    Start_Hour=start,
                    Expected_Hours=delta,
                )
            )
    return pd.DataFrame(rows)


@app.command()
def generate(
    output_dir: str = typer.Option(
        "datasets/steering", help="Directory to write CSV files into."
    ),
    max_delta: int = typer.Option(
        12, help="Maximum elapsed-time delta for Task 2 (hours)."
    ),
):
    """Generate all steering prompt datasets and write them to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("hours_current_time.csv", generate_hours_current_time()),
        ("hours_elapsed_time.csv", generate_hours_elapsed_time(max_delta)),
    ]

    for filename, df in datasets:
        path = out / filename
        df.to_csv(path, index=False)
        print(f"Wrote {len(df):>5} rows → {path}")


if __name__ == "__main__":
    app()
