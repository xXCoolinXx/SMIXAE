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

import re
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

SKIP_DATASETS: set[str] = {"body_parts", "continuity", "emotions"}

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
    """Format a float to ``decimals`` places, or return ``"--"`` if ``v`` is None."""
    if v is None:
        return "--"
    return f"{v:.{decimals}f}"


def fmt_with_std(v: float | None, std: float | None = None, decimals: int = 3) -> str:
    r"""Format ``v`` with optional ``$\pm$ std`` suffix; returns ``"--"`` if ``v`` is None."""
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
    """Load ``results.json`` and ``dataset_config.json`` and return them as ``(results, dataset_config)``."""
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
    """Build the summary probing table (best-expert regression score per model × hypothesis)."""
    models = list(results.keys())
    n_models = len(models)

    col_spec = "ll ll " + " ".join(["rr"] * n_models)

    rows: list[str] = []

    rows.append(r"% Required packages: booktabs, multirow")
    rows.append(r"\begin{table*}[htbp]")
    rows.append(r"\centering")
    rows.append(r"\small")
    rows.append(
        r"\caption{Probing results for SMIXAE experts across several tasks. "
        r"Each hypothesis targets a structured property that may be geometrically encoded in a 3-D expert bottleneck. "
        r"For each hypothesis we fit a regression or classifier directly to the bottleneck activations "
        r"of the top-performing experts and report the score of the single best expert (Top-1) "
        r"and the mean over the top-5 experts (Top-5$_{\mu}$). "
        r"$\pm$ values are cross-validation standard deviations; for Top-5$_{\mu}$, the standard deviation is averaged across the top-5 experts. "
        r"$R^2$ is the coefficient of determination (linear and ridge regression); "
        r"Acc.\ is classification accuracy (logistic and multinomial regression).}"
    )
    rows.append(r"\label{tab:probing}")
    rows.append(r"\resizebox{\textwidth}{!}{%")
    rows.append(r"\begin{tabular}{" + col_spec + "}")
    rows.append(r"\toprule")

    n_header_cols  = 4
    # n_data_cols    = n_models * 2
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
    """Build the summary newline table (periodic gain of top expert per model × line length)."""
    nine_b_models = [mk for mk in results if "9b" in mk]
    n_models = len(nine_b_models)
    line_length_keys = ["newline_80", "newline_150"]

    col_spec = "l " + " ".join(["cc"] * n_models)

    rows: list[str] = []
    rows.append(r"\begin{table}[htbp]")
    rows.append(r"\centering")
    rows.append(r"\small")
    rows.append(
        r"\caption{Newline position encoding in SMIXAE experts at layers 11 and 20 of Gemma 2 9B, "
        r"evaluated at two nominal line lengths. "
        r"We fit both a linear and a periodic (ring or spiral) model to each expert's 3-D bottleneck "
        r"activations using the number of characters since the previous newline as the target. "
        r"$\Delta R^2_{\text{periodic}} = R^2_{\text{periodic}} - R^2_{\text{linear}}$ measures the "
        r"additional variance explained by curved geometry beyond a linear fit: values near zero indicate "
        r"a linear arrangement, while large positive values indicate ring or helical structure in the bottleneck. "
        r"Top-1 is the score of the single best expert; Top-5$_{\mu}$ is the mean across the top-5 experts.}"
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
    """One table* per model: one row per hypothesis, one column per rank (1–10)."""
    n_ranks = 10
    rank_headers = " & ".join(str(r) for r in range(1, n_ranks + 1))
    parts: list[str] = []

    for mk, run_data in results.items():
        model_display = esc(MODEL_NAMES.get(mk, mk))
        col_spec = "ll ll " + "r" * n_ranks

        rows: list[str] = []
        rows.append(r"% Required packages: booktabs, multirow")
        rows.append(r"\begin{table*}[htbp]")
        rows.append(r"\centering")
        rows.append(r"\small")
        rows.append(
            r"\caption{Complete probing scores for all top-10 experts in "
            + model_display
            + r", listed per task and hypothesis. "
            r"Columns 1--10 are rank positions; each cell shows score $\pm$ cross-validation standard deviation. "
            r"The same expert may appear under multiple hypotheses if it encodes more than one concept.}"
        )
        rows.append(rf"\label{{tab:probing_appendix_{mk}}}")
        rows.append(r"\resizebox{\linewidth}{!}{%")
        rows.append(r"\begin{tabular}{" + col_spec + "}")
        rows.append(r"\toprule")
        rows.append(r" & & & & \multicolumn{" + str(n_ranks) + r"}{c}{Expert rank} \\")
        rows.append(r"\cmidrule(lr){5-" + str(4 + n_ranks) + "}")
        rows.append(r"Task & Hypothesis & Regression & Score & " + rank_headers + r" \\")
        rows.append(r"\midrule")

        all_datasets = list(run_data.get("probe", {}).keys())

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

            ds_cell_written = False
            for i, (hyp_name, hyp_info) in enumerate(hyp_list):
                hyp_data = (
                    run_data.get("probe", {})
                    .get(ds_name, {})
                    .get("hypotheses", {})
                    .get(hyp_name, {})
                )
                experts = hyp_data.get("top10_experts", []) if hyp_data else []
                if not experts:
                    continue

                hyp_display = esc(hyp_info["description"])
                reg_label = REGRESSION_LABEL.get(hyp_info["regression_type"], "?")
                score_label = SCORE_LABEL.get(hyp_info["regression_type"], "?")

                row: list[str] = []
                if not ds_cell_written:
                    row.append(
                        rf"\multirow{{{n_hyps}}}{{*}}{{{ds_display}}}"
                        if n_hyps > 1
                        else ds_display
                    )
                    ds_cell_written = True
                else:
                    row.append("")

                row += [hyp_display, reg_label, score_label]

                for rank in range(n_ranks):
                    if rank < len(experts):
                        e = experts[rank]
                        row.append(fmt_with_std(e.get("score"), e.get("score_std")))
                    else:
                        row.append("--")

                rows.append(" & ".join(row) + r" \\")

        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}}% end resizebox")
        rows.append(r"\end{table*}")
        parts.append("\n".join(rows))

    return "\n\n".join(parts)


# ── Newline appendix table ──────────────────────────────────────────────────────

def build_newline_appendix_tables(results: dict) -> str:
    """One table per 9B model: one row per line length, one column per rank (1–10)."""
    nine_b_models = [mk for mk in results if "9b" in mk]
    line_length_keys = ["newline_80", "newline_150"]
    n_ranks = 10
    rank_headers = " & ".join(str(r) for r in range(1, n_ranks + 1))
    parts: list[str] = []

    for mk in nine_b_models:
        model_display = esc(MODEL_NAMES.get(mk, mk))
        col_spec = "l " + "r " * n_ranks

        rows: list[str] = []
        rows.append(r"% Required packages: booktabs, multirow")
        rows.append(r"\begin{table}[htbp]")
        rows.append(r"\centering")
        rows.append(r"\small")
        rows.append(
            r"\caption{Complete newline-position probing scores for all top-10 experts in "
            + model_display
            + r" at each line length, ranked by $\Delta R^2_{\text{periodic}}$. "
            r"This table supports Table\ref{tab:newline}.}"
        )
        rows.append(rf"\label{{tab:newline_appendix_{mk}}}")
        rows.append(r"\resizebox{\linewidth}{!}{%")
        rows.append(r"\begin{tabular}{" + col_spec.strip() + "}")
        rows.append(r"\toprule")
        rows.append(r" & \multicolumn{" + str(n_ranks) + r"}{c}{Expert rank} \\")
        rows.append(r"\cmidrule(lr){2-" + str(1 + n_ranks) + "}")
        rows.append(r"Line length & " + rank_headers + r" \\")
        rows.append(r"\midrule")

        run_newline = results.get(mk, {}).get("newline", {})
        for ll_idx, ll_key in enumerate(line_length_keys):
            if ll_idx > 0:
                rows.append(r"\midrule")
            ll_display = ll_key.replace("newline_", "") + " chars"
            experts = run_newline.get(ll_key, {}).get("top10_experts", [])
            row: list[str] = [ll_display]
            for rank in range(n_ranks):
                if rank < len(experts):
                    row.append(fmt(experts[rank].get("periodic_gain")))
                else:
                    row.append("--")
            rows.append(" & ".join(row) + r" \\")

        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}}% end resizebox")
        rows.append(r"\end{table}")
        parts.append("\n".join(rows))

    return "\n\n".join(parts)


# ── SAEBench table ─────────────────────────────────────────────────────────────

CORE_EVAL_RESULTS_PATH = Path("results/core_eval_results.json")

CORE_EVAL_METRIC_LABELS: dict[str, str] = {
    "width":               "Width",
    "total_params":        "Params",
    "l0":                  "L0",
    "mse":                 "MSE (norm.)",
    "explained_variance":  "Expl. Var.",
    "cosine_similarity":   "Cos. Sim.",
    "l2_ratio":            r"$\|$recon$\|/\|$in$\|$",
    "ce_loss_score":       "CE Score",
    "ce_loss_without_sae": "CE (orig.)",
    "ce_loss_with_sae":    "CE (SAE)",
    "ce_loss_with_ablation": "CE (ablation)",
}

CORE_EVAL_SUMMARY_METRICS = [
    "width",
    "total_params",
    "l0",
    "explained_variance",
    "ce_loss_score",
    "mse",
    "cosine_similarity",
]

# Human-readable model names reused from existing table helpers.
CORE_EVAL_MODEL_DISPLAY: dict[str, str] = {
    "google/gemma-2-2b": "Gemma 2 2B",
    "google/gemma-2-9b": "Gemma 2 9B",
}


def _fmt_core_eval(v: int | float | str | None, decimals: int = 3) -> str:
    """Format a core eval metric value for LaTeX output."""
    if v is None:
        return "--"
    if isinstance(v, str):
        return esc(v[:20])  # truncate error messages
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:.{decimals}f}"


def build_core_eval_table(core_eval_results: dict) -> str:
    r"""Build a core eval metrics table.

    Produces one block per model, with rows = (layer, SAE) and columns = metrics.
    Requires ``booktabs`` LaTeX package.

    Args:
        core_eval_results: Contents of ``core_eval_results.json``.

    Returns:
        LaTeX string for a ``table*`` environment.
    """
    rows: list[str] = []
    rows.append(r"% Required packages: booktabs, multirow")
    rows.append(r"\begin{table*}[htbp]")
    rows.append(r"\centering")
    rows.append(r"\small")
    rows.append(
        r"\caption{Core evaluation metrics comparing SMIXAE against GemmaScope SAE baselines, "
        r"evaluated on OpenWebText (128-token context windows). "
        r"L0 and width are unflattened numbers for SMIXAE.}"
    )
    rows.append(r"\label{tab:saebench}")
    rows.append(r"\resizebox{\textwidth}{!}{%")

    n_metrics = len(CORE_EVAL_SUMMARY_METRICS)
    col_spec = "lll " + "r " * n_metrics
    rows.append(r"\begin{tabular}{" + col_spec.strip() + "}")
    rows.append(r"\toprule")

    header = ["Model", "Layer", "SAE"] + [CORE_EVAL_METRIC_LABELS[m] for m in CORE_EVAL_SUMMARY_METRICS]
    rows.append(" & ".join(header) + r" \\")
    rows.append(r"\midrule")

    model_names = sorted(core_eval_results.keys())

    for model_idx, model_name in enumerate(model_names):
        if model_idx > 0:
            rows.append(r"\midrule")
        model_display = esc(CORE_EVAL_MODEL_DISPLAY.get(model_name, model_name))
        layers = core_eval_results[model_name]
        layer_keys = sorted(layers.keys(), key=lambda lk: int(lk.split("_")[-1]))

        # Count total data rows for this model to span the model cell.
        total_model_rows = sum(len(layers[lk]) for lk in layer_keys)
        model_cell_written = False

        for layer_key in layer_keys:
            layer_display = str(int(layer_key.split("_")[-1]))
            sae_names = list(layers[layer_key].keys())
            n_saes = len(sae_names)

            for sae_idx, sae_name in enumerate(sae_names):
                row: list[str] = []

                # Model column (spans all rows for this model).
                if not model_cell_written:
                    row.append(
                        rf"\multirow{{{total_model_rows}}}{{*}}{{{model_display}}}"
                        if total_model_rows > 1
                        else model_display
                    )
                    model_cell_written = True
                else:
                    row.append("")

                # Layer column (spans SAE rows within this layer).
                if sae_idx == 0:
                    row.append(
                        rf"\multirow{{{n_saes}}}{{*}}{{{layer_display}}}"
                        if n_saes > 1
                        else layer_display
                    )
                else:
                    row.append("")

                sae_display = re.sub(r"\s*\(L0=[^)]*\)", "", sae_name).strip()
                row.append(esc(sae_display))

                metrics = layers[layer_key][sae_name]
                for metric in CORE_EVAL_SUMMARY_METRICS:
                    row.append(_fmt_core_eval(metrics.get(metric)))

                rows.append(" & ".join(row) + r" \\")

    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"}% end resizebox")
    rows.append(r"\end{table*}")

    return "\n".join(rows)


# ── CLI ────────────────────────────────────────────────────────────────────────

@app.command()
def generate(
    results_path: Path = typer.Option(RESULTS_PATH, help="Path to results.json"),
    dataset_config_path: Path = typer.Option(DATASET_CONFIG_PATH, help="Path to dataset_config.json"),
    core_eval_results_path: Path = typer.Option(CORE_EVAL_RESULTS_PATH, help="Path to core_eval_results.json"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Parent directory; all tables are written to <output-dir>/paper/"),
) -> None:
    """Generate all LaTeX tables from results.json and core_eval_results.json."""
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

    if core_eval_results_path.exists():
        import json

        with open(core_eval_results_path) as f:
            core_eval_results = json.load(f)
        out_path = paper_dir / "table_core_eval.tex"
        out_path.write_text(build_core_eval_table(core_eval_results))
        typer.echo(f"Written: {out_path}")
    else:
        typer.echo(f"Skipping core eval table ({core_eval_results_path} not found)")
