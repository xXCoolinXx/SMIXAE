"""Generate LaTeX tables from results.json.

Produces two tables:
  1. Probing table: all models × all tasks × hypotheses (excluding ordinal unless sole hypothesis)
  2. Newline table: Gemma 2 9B models only, periodic gain by line length

CLI usage:
    smixae latex tables
    smixae latex tables --probing-output results/table_probing.tex \\
                        --newline-output results/table_newline.tex

Required LaTeX packages: booktabs, multirow
"""

from pathlib import Path

import typer

app = typer.Typer()

RESULTS_PATH = Path("results/results.json")
DATASET_CONFIG_PATH = Path("datasets/probing/dataset_config.json")
DEFAULT_PROBING_OUTPUT = Path("results/table_probing.tex")
DEFAULT_NEWLINE_OUTPUT = Path("results/table_newline.tex")

# ── Display names ──────────────────────────────────────────────────────────────

MODEL_NAMES: dict[str, str] = {
    "gemma_2_9b_l11": "9B, Layer 11",
    "gemma_2_9b_l20": "9B, Layer 20",
    "gemma_2_2b_l12": "2B, Layer 12",
}

MODEL_FAMILY = "Gemma 2"

DATASET_NAMES: dict[str, str] = {
    "weekdays":     "Weekdays",
    "hours":        "Hours",
    "temperatures": "Temperature",
    "time_units":   "Time Units",
    "body_parts":   "Body Parts",
    "living_things":"Living Things",
    "months":       "Months",
    "colors":       "Colors",
    "emotions":     "Emotions",
    "continuity":   "Continuity",
}

SKIP_DATASETS: set[str] = {"body_parts", "continuity"}

SCORE_LABEL: dict[str, str] = {
    "linear":      r"$R^2$",
    "logistic":    r"Acc.",
    "multinomial": r"Acc.",
    "ridge":       r"$R^2$",
}

REGRESSION_LABEL: dict[str, str] = {
    "linear":      "Linear",
    "logistic":    "Logistic",
    "multinomial": "Multinomial",
    "ridge":       "Ridge",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt(v: float | None, decimals: int = 3) -> str:
    if v is None:
        return "--"
    return f"{v:.{decimals}f}"


def esc(s: str) -> str:
    """Escape LaTeX special characters in a plain string."""
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def load_data(results_path: Path, dataset_config_path: Path) -> tuple[dict, list]:
    import json
    with open(results_path) as f:
        results = json.load(f)
    with open(dataset_config_path) as f:
        dataset_config = json.load(f)
    return results, dataset_config


def build_hypothesis_map(dataset_config: list) -> dict[str, dict]:
    """Return {dataset_stem: {hyp_name: {description, regression_type}}}."""
    hyp_map: dict[str, dict] = {}
    for entry in dataset_config:
        ds_name = Path(entry["dataframe_path"]).stem
        hyp_map[ds_name] = {
            h["name"]: {
                "description":     h["description"],
                "regression_type": h["regression_type"],
            }
            for h in entry.get("regression_hypotheses", [])
        }
    return hyp_map


def hypotheses_to_show(ds_name: str, hyp_map: dict) -> dict:
    """Return hypotheses for a dataset, omitting 'ordinal' unless it is the sole entry."""
    all_hyps = hyp_map.get(ds_name, {})
    if not all_hyps:
        return {}
    non_ordinal = {k: v for k, v in all_hyps.items() if k != "ordinal"}
    return non_ordinal if non_ordinal else all_hyps


# ── Probing table ──────────────────────────────────────────────────────────────

def build_probing_table(results: dict, hyp_map: dict) -> str:
    models = list(results.keys())
    n_models = len(models)

    col_spec = "ll ll " + " ".join(["rr"] * n_models)

    rows: list[str] = []

    rows.append(r"% Required packages: booktabs, multirow")
    rows.append(r"\begin{table*}[htbp]")
    rows.append(r"\centering")
    rows.append(r"\small")
    rows.append(
        r"\caption{Probing results across tasks and models. "
        r"For each hypothesis the top-1 score and mean over the top-5 experts are reported. "
        r"$R^2$ = coefficient of determination; Acc.\ = classification accuracy.}"
    )
    rows.append(r"\label{tab:probing}")
    rows.append(r"\resizebox{\textwidth}{!}{%")
    rows.append(r"\begin{tabular}{" + col_spec + "}")
    rows.append(r"\toprule")

    n_header_cols  = 4
    n_data_cols    = n_models * 2
    first_data_col = n_header_cols + 1

    h1_parts = [rf"\multicolumn{{{n_header_cols}}}{{l}}{{{esc(MODEL_FAMILY)}}}"]
    for mk in models:
        name = esc(MODEL_NAMES.get(mk, mk))
        h1_parts.append(r"\multicolumn{2}{c}{" + name + "}")
    rows.append(" & ".join(h1_parts) + r" \\")

    cmidrules = []
    for i in range(n_models):
        c0 = first_data_col + i * 2
        c1 = c0 + 1
        cmidrules.append(rf"\cmidrule(lr){{{c0}-{c1}}}")
    rows.append(" ".join(cmidrules))

    h2_parts = ["Task", "Hypothesis", "Regression", "Score"]
    for _ in models:
        h2_parts += ["Top-1", r"Top-5$_{\mu}$"]
    rows.append(" & ".join(h2_parts) + r" \\")
    rows.append(r"\midrule")

    all_datasets: list[str] = list(next(iter(results.values()))["probe"].keys())
    last_dataset_with_data: str | None = None

    for ds_name in all_datasets:
        if ds_name in SKIP_DATASETS:
            continue
        hyps = hypotheses_to_show(ds_name, hyp_map)
        if not hyps:
            continue
        last_dataset_with_data = ds_name

    separator_added = False
    for ds_name in all_datasets:
        if ds_name in SKIP_DATASETS:
            continue
        hyps = hypotheses_to_show(ds_name, hyp_map)
        if not hyps:
            continue

        ds_display = esc(DATASET_NAMES.get(ds_name, ds_name.title()))
        hyp_list = list(hyps.items())
        n_hyps = len(hyp_list)

        for i, (hyp_name, hyp_info) in enumerate(hyp_list):
            row: list[str] = []

            if i == 0:
                cell = (
                    rf"\multirow{{{n_hyps}}}{{*}}{{{ds_display}}}"
                    if n_hyps > 1
                    else ds_display
                )
                row.append(cell)
            else:
                row.append("")

            row.append(esc(hyp_info["description"]))
            row.append(REGRESSION_LABEL.get(hyp_info["regression_type"], "?"))
            row.append(SCORE_LABEL.get(hyp_info["regression_type"], "?"))

            for mk in models:
                hyp_data = (
                    results.get(mk, {})
                    .get("probe", {})
                    .get(ds_name, {})
                    .get("hypotheses", {})
                    .get(hyp_name, {})
                )
                top1 = (
                    hyp_data["top10_experts"][0]["score"]
                    if hyp_data and hyp_data.get("top10_experts")
                    else None
                )
                top5_mean = hyp_data.get("top5_mean") if hyp_data else None
                row += [fmt(top1), fmt(top5_mean)]

            rows.append(" & ".join(row) + r" \\")

        if ds_name == last_dataset_with_data:
            rows.append(r"\bottomrule")
            separator_added = True
        else:
            rows.append(r"\midrule")

    if not separator_added:
        rows.append(r"\bottomrule")

    rows.append(r"\end{tabular}")
    rows.append(r"}% end resizebox")
    rows.append(r"\end{table*}")

    return "\n".join(rows)


# ── Newline table ──────────────────────────────────────────────────────────────

def build_newline_table(results: dict) -> str:
    nine_b_models = [mk for mk in results if "9b" in mk]
    n_models = len(nine_b_models)
    line_length_keys = ["newline_80", "newline_150"]

    col_spec = "l " + " ".join(["cc"] * n_models)

    rows: list[str] = []
    rows.append(r"\begin{table}[htbp]")
    rows.append(r"\centering")
    rows.append(r"\small")
    rows.append(
        r"\caption{Newline position encoding results (Gemma 2 9B). "
        r"$\Delta R^2_{\text{periodic}} = R^2_{\text{periodic}} - R^2_{\text{linear}}$ on the bottleneck; "
        r"positive values indicate ring or spiral geometry. "
        r"Top-1 and mean over top-5 experts reported.}"
    )
    rows.append(r"\label{tab:newline}")
    rows.append(r"\begin{tabular}{" + col_spec + "}")
    rows.append(r"\toprule")

    h1_parts = [r"\multicolumn{1}{l}{Gemma 2 9B}"]
    for mk in nine_b_models:
        short = esc(MODEL_NAMES.get(mk, mk)).split(", ", 1)[-1]
        h1_parts.append(r"\multicolumn{2}{c}{" + short + "}")
    rows.append(" & ".join(h1_parts) + r" \\")

    cmidrules = []
    for i in range(n_models):
        c0 = 2 + i * 2
        c1 = c0 + 1
        cmidrules.append(rf"\cmidrule(lr){{{c0}-{c1}}}")
    rows.append(" ".join(cmidrules))

    h2_parts = ["Line length"]
    for _ in nine_b_models:
        h2_parts += [r"Top-1 $\Delta R^2_{\text{per.}}$", r"Top-5$_{\mu}$ $\Delta R^2_{\text{per.}}$"]
    rows.append(" & ".join(h2_parts) + r" \\")
    rows.append(r"\midrule")

    for ll_key in line_length_keys:
        ll_display = ll_key.replace("newline_", "") + " chars"
        row: list[str] = [ll_display]
        for mk in nine_b_models:
            nl = results.get(mk, {}).get("newline", {}).get(ll_key, {})
            row += [
                fmt(nl.get("top1_periodic_gain")),
                fmt(nl.get("top5_mean_periodic_gain")),
            ]
        rows.append(" & ".join(row) + r" \\")

    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")

    return "\n".join(rows)


# ── CLI ────────────────────────────────────────────────────────────────────────

@app.command()
def generate(
    results_path: Path = typer.Option(RESULTS_PATH, help="Path to results.json"),
    dataset_config_path: Path = typer.Option(DATASET_CONFIG_PATH, help="Path to dataset_config.json"),
    probing_output: Path = typer.Option(DEFAULT_PROBING_OUTPUT, help="Output path for the probing table .tex file"),
    newline_output: Path = typer.Option(DEFAULT_NEWLINE_OUTPUT, help="Output path for the newline table .tex file"),
) -> None:
    """Generate LaTeX tables (probing + newline) from results.json."""
    results, dataset_config = load_data(results_path, dataset_config_path)
    hyp_map = build_hypothesis_map(dataset_config)

    probing = build_probing_table(results, hyp_map)
    newline = build_newline_table(results)

    probing_output.parent.mkdir(parents=True, exist_ok=True)
    probing_output.write_text(probing)
    typer.echo(f"Probing table written to {probing_output}")

    newline_output.parent.mkdir(parents=True, exist_ok=True)
    newline_output.write_text(newline)
    typer.echo(f"Newline table written to {newline_output}")
