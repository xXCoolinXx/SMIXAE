"""Generate LaTeX tables from results.json.

Produces four tables, all written under ``<output-dir>/paper/``:
  1. table_probing.tex          — all models × tasks × hypotheses (summary: top-1 ± std, top-5μ ± mean-std)
  2. table_newline.tex          — Gemma 2 9B only, periodic gain summary (top-1, top-5μ)
  3. table_probing_appendix.tex — per-model detail: all 10 experts per hypothesis with score ± std
  4. table_newline_appendix.tex — per-model detail: all 10 experts per line length with periodic gain

CLI usage:
    smixae latex tables
    smixae latex tables --output-dir results/my_run

Required LaTeX packages: booktabs, multirow
"""

from pathlib import Path

import typer

app = typer.Typer()

RESULTS_PATH = Path("results/results.json")
DATASET_CONFIG_PATH = Path("datasets/probing/dataset_config.json")
DEFAULT_OUTPUT_DIR = Path("results/")

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


def fmt_with_std(v: float | None, std: float | None = None, decimals: int = 3) -> str:
    if v is None:
        return "--"
    s = f"{v:.{decimals}f}"
    if std is not None:
        s += r" $\pm$ " + f"{std:.{decimals}f}"
    return s


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
        r"$\pm$ values show the per-expert cross-validation standard deviation (top-1) "
        r"and the mean of CV standard deviations across the top-5 experts (Top-5$_{\mu}$). "
        r"Where $R^2$ coefficient of determination and Acc. is classification accuracy.}"
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
                top_experts = hyp_data.get("top10_experts") if hyp_data else None
                top1 = top_experts[0]["score"] if top_experts else None
                top1_std = top_experts[0].get("score_std") if top_experts else None
                top5_mean = hyp_data.get("top5_mean") if hyp_data else None
                top5_mean_std = hyp_data.get("top5_mean_std") if hyp_data else None
                row += [fmt_with_std(top1, top1_std), fmt_with_std(top5_mean, top5_mean_std)]

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


# ── Probing appendix table ─────────────────────────────────────────────────────

def build_probing_appendix_tables(results: dict, hyp_map: dict) -> str:
    """One table* per model listing all 10 experts for each dataset × hypothesis."""
    parts: list[str] = []

    for mk, run_data in results.items():
        model_display = esc(MODEL_NAMES.get(mk, mk))
        col_spec = "llll rr"

        rows: list[str] = []
        rows.append(r"% Required packages: booktabs, multirow")
        rows.append(r"\begin{table*}[htbp]")
        rows.append(r"\centering")
        rows.append(r"\small")
        rows.append(
            r"\caption{Probing expert detail --- "
            + model_display
            + r". All top-10 experts per hypothesis. "
            r"Score $\pm$ cross-validation standard deviation.}"
        )
        rows.append(rf"\label{{tab:probing_appendix_{mk}}}")
        rows.append(r"\begin{tabular}{" + col_spec + "}")
        rows.append(r"\toprule")
        rows.append(r"Task & Hypothesis & Regression & Score & Rank & Expert ID & Score $\pm$ Std \\")
        rows.append(r"\midrule")

        all_datasets = list(run_data.get("probe", {}).keys())
        last_ds = None
        for ds_name in all_datasets:
            if ds_name in SKIP_DATASETS:
                continue
            hyps = hypotheses_to_show(ds_name, hyp_map)
            if hyps:
                last_ds = ds_name

        first_ds = True
        for ds_name in all_datasets:
            if ds_name in SKIP_DATASETS:
                continue
            hyps = hypotheses_to_show(ds_name, hyp_map)
            if not hyps:
                continue

            if not first_ds:
                rows.append(r"\midrule")
            first_ds = False

            ds_display = esc(DATASET_NAMES.get(ds_name, ds_name.title()))
            hyp_list = list(hyps.items())
            n_hyps = len(hyp_list)
            total_rows = sum(
                len(
                    run_data.get("probe", {})
                    .get(ds_name, {})
                    .get("hypotheses", {})
                    .get(hyp_name, {})
                    .get("top10_experts", [])
                )
                for hyp_name, _ in hyp_list
            )

            ds_cell_written = False
            for i, (hyp_name, hyp_info) in enumerate(hyp_list):
                hyp_data = (
                    run_data.get("probe", {})
                    .get(ds_name, {})
                    .get("hypotheses", {})
                    .get(hyp_name, {})
                )
                experts = hyp_data.get("top10_experts", []) if hyp_data else []
                n_experts = len(experts)
                if not n_experts:
                    continue

                hyp_display = esc(hyp_info["description"])
                reg_label = REGRESSION_LABEL.get(hyp_info["regression_type"], "?")
                score_label = SCORE_LABEL.get(hyp_info["regression_type"], "?")

                for j, entry in enumerate(experts):
                    row: list[str] = []

                    if j == 0 and not ds_cell_written:
                        row.append(
                            rf"\multirow{{{total_rows}}}{{*}}{{{ds_display}}}"
                            if total_rows > 1
                            else ds_display
                        )
                        ds_cell_written = True
                    else:
                        row.append("")

                    if j == 0:
                        row.append(
                            rf"\multirow{{{n_experts}}}{{*}}{{{hyp_display}}}"
                            if n_experts > 1
                            else hyp_display
                        )
                        row.append(
                            rf"\multirow{{{n_experts}}}{{*}}{{{reg_label}}}"
                            if n_experts > 1
                            else reg_label
                        )
                        row.append(
                            rf"\multirow{{{n_experts}}}{{*}}{{{score_label}}}"
                            if n_experts > 1
                            else score_label
                        )
                    else:
                        row += ["", "", ""]

                    row.append(str(entry["rank"]))
                    row.append(str(entry["expert_id"]))
                    row.append(fmt_with_std(entry.get("score"), entry.get("score_std")))
                    rows.append(" & ".join(row) + r" \\")

        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}")
        rows.append(r"\end{table*}")
        parts.append("\n".join(rows))

    return "\n\n".join(parts)


# ── Newline appendix table ──────────────────────────────────────────────────────

def build_newline_appendix_tables(results: dict) -> str:
    """One table per 9B model listing all 10 experts for each line length."""
    nine_b_models = [mk for mk in results if "9b" in mk]
    line_length_keys = ["newline_80", "newline_150"]
    parts: list[str] = []

    for mk in nine_b_models:
        model_display = esc(MODEL_NAMES.get(mk, mk))
        col_spec = "l rr"

        rows: list[str] = []
        rows.append(r"% Required packages: booktabs, multirow")
        rows.append(r"\begin{table}[htbp]")
        rows.append(r"\centering")
        rows.append(r"\small")
        rows.append(
            r"\caption{Newline position expert detail --- "
            + model_display
            + r". All top-10 experts per line length, ranked by "
            r"$\Delta R^2_{\text{periodic}}$.}"
        )
        rows.append(rf"\label{{tab:newline_appendix_{mk}}}")
        rows.append(r"\begin{tabular}{" + col_spec + "}")
        rows.append(r"\toprule")
        rows.append(r"Line length & Rank & Expert ID & $\Delta R^2_{\text{per.}}$ \\")
        rows.append(r"\midrule")

        run_newline = results.get(mk, {}).get("newline", {})
        for ll_idx, ll_key in enumerate(line_length_keys):
            if ll_idx > 0:
                rows.append(r"\midrule")
            ll_display = ll_key.replace("newline_", "") + " chars"
            experts = run_newline.get(ll_key, {}).get("top10_experts", [])
            n_experts = len(experts)
            for j, entry in enumerate(experts):
                row: list[str] = []
                if j == 0:
                    row.append(
                        rf"\multirow{{{n_experts}}}{{*}}{{{ll_display}}}"
                        if n_experts > 1
                        else ll_display
                    )
                else:
                    row.append("")
                row.append(str(entry["rank"]))
                row.append(str(entry["expert_id"]))
                row.append(fmt(entry.get("periodic_gain")))
                rows.append(" & ".join(row) + r" \\")

        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}")
        rows.append(r"\end{table}")
        parts.append("\n".join(rows))

    return "\n\n".join(parts)


# ── CLI ────────────────────────────────────────────────────────────────────────

@app.command()
def generate(
    results_path: Path = typer.Option(RESULTS_PATH, help="Path to results.json"),
    dataset_config_path: Path = typer.Option(DATASET_CONFIG_PATH, help="Path to dataset_config.json"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Parent directory; all tables are written to <output-dir>/paper/"),
) -> None:
    """Generate LaTeX tables (probing + newline, summary + appendix) from results.json."""
    results, dataset_config = load_data(results_path, dataset_config_path)
    hyp_map = build_hypothesis_map(dataset_config)
    paper_dir = output_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        ("table_probing.tex",          build_probing_table(results, hyp_map)),
        ("table_newline.tex",          build_newline_table(results)),
        ("table_probing_appendix.tex", build_probing_appendix_tables(results, hyp_map)),
        ("table_newline_appendix.tex", build_newline_appendix_tables(results)),
    ]
    for filename, content in outputs:
        out_path = paper_dir / filename
        out_path.write_text(content)
        typer.echo(f"Written: {out_path}")
