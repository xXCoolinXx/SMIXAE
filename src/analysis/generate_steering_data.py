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

import re
from pathlib import Path

import pandas as pd
import typer

# Canonical 24-hour ordering — index determines the ordering prefix in probing labels.
HOURS = [
    "1AM",
    "2AM",
    "3AM",
    "4AM",
    "5AM",
    "6AM",
    "7AM",
    "8AM",
    "9AM",
    "10AM",
    "11AM",
    "12PM",
    "1PM",
    "2PM",
    "3PM",
    "4PM",
    "5PM",
    "6PM",
    "7PM",
    "8PM",
    "9PM",
    "10PM",
    "11PM",
    "12AM",
]

_FRIENDS = ["Alex", "Sam", "Jordan", "Taylor", "Morgan"]

app = typer.Typer()


def fmt_hour(h: str) -> str:
    """Convert short hour string to clock format: '1AM' → '1:00AM', '12PM' → '12:00PM'."""
    return re.sub(r"(AM|PM)$", r":00\1", h)


def generate_hours_current_time() -> pd.DataFrame:
    """Task 1: all (src_hour, tgt_hour) pairs — 24 × 23 = 552 rows.

    Prompt: "You glance at the clock and find it is X:00YM. Your friend Z asks
    you for the time, and you respond, saying it is"
    """
    rows = []
    i = 0
    for src in HOURS:
        for tgt in HOURS:
            if src == tgt:
                continue
            friend = _FRIENDS[i % len(_FRIENDS)]
            rows.append(
                dict(
                    Prompt=(
                        f"You glance at the clock and find it is {fmt_hour(src)}. "
                        f"Your friend {friend} asks you for the time, "
                        f"and you respond, saying it is"
                    ),
                    Source_Hour=src,
                    Target_Hour=tgt,
                    Friend=friend,
                )
            )
            i += 1
    return pd.DataFrame(rows)


def generate_hours_elapsed_time(max_delta: int = 12) -> pd.DataFrame:
    """Task 2: (curr_hour, start_hour) pairs for delta = 1 … max_delta.

    Prompt: "You check your phone, and see that the current time is X:00YM.
    You realize that, since Z:00WM, the amount of hours that has passed is"

    The baseline answer is always ``delta`` hours. Steering shifts the
    perceived current time by +delta hours.

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
                        f"You check your phone, and see that the current time is "
                        f"{fmt_hour(curr)}. You realize that, since {fmt_hour(start)}, "
                        f"the amount of hours that has passed is"
                    ),
                    Current_Hour=curr,
                    Start_Hour=start,
                    Expected_Hours=delta,
                )
            )
    return pd.DataFrame(rows)


@app.command()
def generate(
    output_dir: str = typer.Option("datasets/steering", help="Directory to write CSV files into."),
    max_delta: int = typer.Option(12, help="Maximum elapsed-time delta for Task 2 (hours)."),
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
