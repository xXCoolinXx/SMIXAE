# SMIXAE Figures & Tables Design Specification

This document specifies the exact output format of every LaTeX artifact produced by
`src/latex/camera_ready.py` (figure `.tex` files + legend PNGs) and `src/latex/tables.py`
(table `.tex` files). It is the **source of truth for human-readable text** — captions,
display names, column labels, and ordering. The code is the source of truth for data
processing logic.

**Edit workflow**: change a caption, rename a task, reorder columns, or adjust a layout
constant here, then run the reconcile agent pointing at this file. The reconciliation notes
in [§5](#5-reconciliation-notes) tell the agent exactly which dict, constant, or string
literal in the source code to update.

All outputs land under `<output-dir>/paper/` (default `results/paper/`). Copy the `paper/`
folder next to your main `.tex` and use `\input{paper/...}`.

---

## Contents

1. [Shared Configuration](#1-shared-configuration)
2. [Hypothesis Filtering Rules](#2-hypothesis-filtering-rules)
3. [tables.py — Table Specifications](#3-tablespy--table-specifications)
4. [camera_ready.py — Figure Specifications](#4-camera_readypy--figure-specifications)
5. [Reconciliation Notes](#5-reconciliation-notes)

---

## 1. Shared Configuration

The lookup tables below control human-readable text in **both** figures and tables.
Editing a row here and reconciling will update both.

### 1a. Task Display Names

| Internal key       | Figures (`_TASK_DISPLAY`)      | Tables (`DATASET_NAMES`)  |
|--------------------|-------------------------------|---------------------------|
| `weekdays`         | Weekdays                      | Weekdays                  |
| `hours`            | Hours                         | Hours                     |
| `months`           | Months                        | Months                    |
| `temperatures`     | Temperature                   | Temperature               |
| `time_units`       | Time Units                    | Time Units                |
| `body_parts`       | Body Parts                    | Body Parts                |
| `living_things`    | Living Things                 | Living Things             |
| `colors`           | Colors                        | Colors                    |
| `emotions`         | Emotions                      | Emotions                  |
| `pile-uncopyrighted` | Newline Position            | *(not in tables)*         |
| `continuity`       | Random Experts (Unlabeled)    | Continuity                |

> **Note**: `body_parts`, `continuity`, and `emotions` are excluded from the main probing
> table and appendix (see `SKIP_DATASETS` in §1e). They still appear in figure `.tex` files.

### 1b. Hypothesis Display Names

These are used in **figure captions** only (`_HYP_DISPLAY` in `camera_ready.py`).
Table cells use the `"description"` field from `datasets/probing/dataset_config.json` directly.

| `hyp_name` key    | Figure caption display        | `dataset_config.json` description (tables) |
|-------------------|-------------------------------|---------------------------------------------|
| `cyc_7d`          | 7-Day Ring                    | 7-Day Ring                                  |
| `weekday_weekend` | Weekday vs Weekend            | Weekday vs Weekend                          |
| `cyc_24h`         | 24-Hour Ring                  | 24-Hour Ring                                |
| `cyc_12h`         | 12-Hour Ring                  | 12-Hour Ring                                |
| `am_pm`           | AM vs PM                      | AM vs PM                                    |
| `linear_f`        | Fahrenheit          | Linear Fahrenheit                           |
| `log_f`           | $\log$ Fahrenheit             | $\log$ Fahrenheit                |
| `log_duration`    | $\log_{10}$ Duration                  | $\log_{10}$ Duration                        |
| `plant_animal`    | Plant vs Animal               | Plant vs Animal                             |
| `taxonomy`        | Taxonomy                      | Taxonomic Group                             |
| `cyc_12m`         | 12-Month Ring                 | 12-Month Ring                               |
| `season`          | Season                        | Season                                      |
| `hue_wheel`       | Hue Ring                      | Hue Ring                                    |
| `rgb`             | RGB                           | Normalized RGB                              |
| `warm_nat_cool`   | Warm/Natural/Cool                     | Warm / Natural / Cool                       |
| `valence_arousal` | Valence-Arousal               | Valence-Arousal Circumplex                  |
| `quadrant`        | Quadrant                      | Affective Quadrant                          |
| `periodic_gain`   | Periodic Gain                 | *(newline figures only)*                    |
| `random`          | Random Sample                 | *(not used in tables)*                      |
| `continuity`      | Continuity                    | *(not used in tables)*                      |

> **Discrepancy**: `valence_arousal` and `quadrant` use shorter display names in figure captions
> than in table cells. To unify them, update either `_HYP_DISPLAY` in `camera_ready.py` or the
> `"description"` fields in `dataset_config.json`.

### 1c. Score-Type Labels

**In figures** (`_SCORE_LABEL`, keyed by `score_type` from the PNG filename):

| `score_type` | Rendered label                         |
|--------------|----------------------------------------|
| `r2`         | `$R^2$`                                |
| `acc`        | `Accuracy`                             |
| `score`      | `Score`                                |
| `per_gain`   | `$\Delta R^2_{\mathrm{per}}$`          |
| `cont`       | `Cont.`                                |

**In tables** (`SCORE_LABEL`, keyed by `regression_type` from `dataset_config.json`):

| `regression_type` | Rendered label |
|-------------------|----------------|
| `linear`          | `$R^2$`        |
| `logistic`        | `Acc.`         |
| `multinomial`     | `Acc.`         |
| `ridge`           | `$R^2$`        |

### 1d. Model Display Names

| Experiment ID      | Figures (`_format_experiment_id`) | Tables (`MODEL_NAMES`)  |
|--------------------|-----------------------------------|-------------------------|
| `gemma_2_9b_l11`  | Gemma 2 9B, Layer 11             | 9B, Layer 11            |
| `gemma_2_9b_l20`  | Gemma 2 9B, Layer 20             | 9B, Layer 20            |
| `gemma_2_2b_l12`  | Gemma 2 2B, Layer 12             | 2B, Layer 12            |

The figures use the full `"Gemma 2 9B, Layer 11"` form; the tables use the shortened
`"9B, Layer 11"` form under a shared `"Gemma 2"` family header.

### 1e. Skipped Datasets (Tables Only)

The following datasets are excluded from `table_probing.tex` and `table_probing_appendix.tex`:

```python
SKIP_DATASETS = {"body_parts", "continuity", "emotions"}
```

They still appear in figure `.tex` files (body parts and emotions as probe figures,
continuity as a random-sample figure). Body parts and emotions are excluded due to issues with those probing tasks. Continuity is excluded because there currently isn't a meaningful metric to assess how "interpretable" the random sample is other than visualization. 

---

## 2. Hypothesis Filtering Rules

Two filtering rules apply before any hypothesis appears in a table or figure:

1. **Ordinal hypotheses are suppressed** unless they are the only hypothesis for a dataset.
   This hides `cyc_7d`'s partner `ordinal` for weekdays, `ordinal` for hours and months, etc. This is because the ordinal task is much lower quality for probing than the specific hypotheses.
   Rule lives in `hypotheses_to_show()` in `tables.py`.

2. **`hypothesis_color_overrides`** in `dataset_config.json` cause a task's entries to be split
   into two `LegendGroup` objects in `camera_ready.py`: one per overridden hypothesis (with its
   own discrete color map) and one for all remaining hypotheses (with the default color
   scale). Currently only `living_things` uses this, for the `plant_animal` hypothesis. This is usually not rendered (since the model does not perform well on either task), but this capability should be preserved.

---

## 3. `tables.py` — Table Specifications

All five files are written to `<output-dir>/paper/`. Required LaTeX packages: `booktabs`,
`multirow`.

---

### 3.1 `table_probing.tex`

**Environment**: `table*` (spans both columns in a two-column layout)
**Column spec**: `ll ll rr rr rr`  (4 label columns + 2 score columns per model)
**Packages**: `booktabs`, `multirow`, `\resizebox{\textwidth}{!}{...}`

#### Caption (verbatim)

```latex
\caption{Probing results for SMIXAE experts across several tasks.
Each hypothesis targets a structured property that may be geometrically encoded in a 3-D expert bottleneck.
For each hypothesis we fit a regression or classifier directly to the bottleneck activations
of the top-performing experts and report the score of the single best expert (Top-1)
and the mean over the top-5 experts (Top-5$_{\mu}$).
$\pm$ values are cross-validation standard deviations; for Top-5$_{\mu}$, the standard deviation is averaged across the top-5 experts.
$R^2$ is the coefficient of determination (linear and ridge regression);
Acc.\ is classification accuracy (logistic and multinomial regression).}
```

#### Header Structure

```
Row 1:  Gemma 2 (cols 1-4)  |  9B, Layer 11 (cols 5-6)  |  9B, Layer 20 (cols 7-8)  |  2B, Layer 12 (cols 9-10)
                                \cmidrule(lr){5-6}            \cmidrule(lr){7-8}           \cmidrule(lr){9-10}
Row 2:  Task  Hypothesis  Regression  Score  |  Top-1  Top-5μ  |  Top-1  Top-5μ  |  Top-1  Top-5μ
\midrule
```

#### Full ASCII Example (real data, 3 decimal places)

```
\begin{table*}[htbp]
\centering\small
\caption{...}
\label{tab:probing}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll ll rr rr rr}
\toprule
Gemma 2          &                            &            &       & \multicolumn{2}{c}{9B, Layer 11} & \multicolumn{2}{c}{9B, Layer 20} & \multicolumn{2}{c}{2B, Layer 12} \\
                                                                    \cmidrule(lr){5-6}                  \cmidrule(lr){7-8}                  \cmidrule(lr){9-10}
Task             & Hypothesis                 & Regression & Score & Top-1        & Top-5μ       & Top-1        & Top-5μ       & Top-1        & Top-5μ       \\
\midrule
\multirow{2}{*}{Weekdays}
                 & 7-Day Ring                 & Linear     & $R^2$ & 0.855 ± 0.012 & 0.751 ± 0.014 & 0.724 ± 0.025 & 0.594 ± 0.019 & 0.579 ± 0.028 & 0.487 ± 0.023 \\
                 & Weekday vs Weekend         & Logistic   & Acc.  & 1.000 ± 0.000 & 0.994 ± 0.005 & 0.998 ± 0.003 & 0.944 ± 0.010 & 0.973 ± 0.015 & 0.935 ± 0.020 \\
\midrule
\multirow{3}{*}{Hours}
                 & 24-Hour Ring               & Linear     & $R^2$ & 0.671 ± 0.023 & 0.593 ± 0.026 & 0.590 ± 0.027 & 0.494 ± 0.028 & 0.681 ± 0.016 & 0.491 ± 0.028 \\
                 & 12-Hour Ring               & Linear     & $R^2$ & 0.657 ± 0.011 & 0.539 ± 0.024 & 0.397 ± 0.027 & 0.266 ± 0.025 & 0.425 ± 0.036 & 0.333 ± 0.037 \\
                 & AM vs PM                   & Logistic   & Acc.  & 1.000 ± 0.000 & 0.984 ± 0.009 & 1.000 ± 0.000 & 0.911 ± 0.013 & 1.000 ± 0.000 & 0.939 ± 0.016 \\
\midrule
\multirow{2}{*}{Temperature}
                 & Linear Fahrenheit          & Linear     & $R^2$ & 0.916 ± 0.007 & 0.610 ± 0.024 & 0.875 ± 0.015 & 0.531 ± 0.038 & 0.663 ± 0.014 & 0.559 ± 0.034 \\
                 & $\log$-Compressed Fahrenheit & Linear   & $R^2$ & 0.950 ± 0.007 & 0.629 ± 0.036 & 0.887 ± 0.022 & 0.510 ± 0.043 & 0.682 ± 0.038 & 0.509 ± 0.056 \\
\midrule
Time Units       & $\log_{10}$ Duration       & Linear     & $R^2$ & 0.894 ± 0.008 & 0.737 ± 0.021 & 0.856 ± 0.016 & 0.672 ± 0.021 & 0.695 ± 0.028 & 0.617 ± 0.029 \\
\midrule
\multirow{2}{*}{Living Things}
                 & Plant vs Animal            & Logistic   & Acc.  & 0.963 ± 0.008 & 0.946 ± 0.011 & 0.914 ± 0.010 & 0.886 ± 0.016 & 0.998 ± 0.003 & 0.965 ± 0.010 \\
                 & Taxonomic Group            & Multinomial & Acc. & 0.926 ± 0.020 & 0.822 ± 0.022 & 0.780 ± 0.048 & 0.666 ± 0.046 & 0.911 ± 0.019 & 0.781 ± 0.030 \\
\midrule
\multirow{2}{*}{Months}
                 & 12-Month Ring              & Linear     & $R^2$ & 0.659 ± 0.024 & 0.411 ± 0.021 & 0.469 ± 0.022 & 0.215 ± 0.020 & 0.487 ± 0.040 & 0.201 ± 0.028 \\
                 & Season                     & Multinomial & Acc. & 0.838 ± 0.036 & 0.704 ± 0.035 & 0.872 ± 0.021 & 0.531 ± 0.032 & 0.674 ± 0.029 & 0.464 ± 0.045 \\
\midrule
\multirow{3}{*}{Colors}
                 & Hue Ring                   & Linear     & $R^2$ & 0.827 ± 0.018 & 0.775 ± 0.026 & 0.575 ± 0.016 & 0.441 ± 0.032 & 0.655 ± 0.024 & 0.546 ± 0.020 \\
                 & Normalized RGB             & Linear     & $R^2$ & 0.789 ± 0.028 & 0.689 ± 0.027 & 0.452 ± 0.017 & 0.400 ± 0.027 & 0.596 ± 0.020 & 0.491 ± 0.023 \\
                 & Warm / Natural / Cool      & Multinomial & Acc. & 0.995 ± 0.004 & 0.947 ± 0.012 & 0.888 ± 0.017 & 0.806 ± 0.024 & 0.865 ± 0.034 & 0.844 ± 0.032 \\
\bottomrule
\end{tabular}}
\end{table*}
```

**Row ordering**: follows the key order in `results.json["<first_model>"]["probe"]`.
With the current results file this is: Weekdays → Hours → Temperature → Time Units →
*(Body Parts — skipped)* → Living Things → Months → Colors → *(Emotions — skipped)* → *(Continuity — skipped)*.

**`\multirow` spans**: the Task cell spans all hypothesis rows for that dataset.
`Time Units` (1 hypothesis) does not get a `\multirow`.

---

### 3.2 `table_newline.tex`

**Environment**: `table` (single column)
**Column spec**: `l cc cc`  (line-length column + 2 score columns per model)
**Packages**: `booktabs`

Only 9B models appear (those with `"9b"` in the experiment ID key).

#### Caption (verbatim)

```latex
\caption{Newline position encoding in SMIXAE experts at layers 11 and 20 of Gemma 2 9B,
evaluated at two nominal line lengths.
We fit both a linear and a periodic (ring or spiral) model to each expert's 3-D bottleneck
activations using the number of characters since the previous newline as the target.
$\Delta R^2_{\text{periodic}} = R^2_{\text{periodic}} - R^2_{\text{linear}}$ measures the
additional variance explained by curved geometry beyond a linear fit: values near zero indicate
a linear arrangement, while large positive values indicate ring or helical structure in the
bottleneck.
Top-1 is the score of the single best expert; Top-5$_{\mu}$ is the mean across the top-5 experts.}
```

#### Full ASCII Example (real data)

```
\begin{table}[htbp]
\centering\small
\caption{...}
\label{tab:newline}
\begin{tabular}{l cc cc}
\toprule
Gemma 2 9B   & \multicolumn{2}{c}{Layer 11}                              & \multicolumn{2}{c}{Layer 20}                              \\
               \cmidrule(lr){2-3}                                           \cmidrule(lr){4-5}
Line length  & Top-1 $\Delta R^2_{\text{per.}}$ & Top-5μ $\Delta R^2_{\text{per.}}$ & Top-1 $\Delta R^2_{\text{per.}}$ & Top-5μ $\Delta R^2_{\text{per.}}$ \\
\midrule
80 chars     & 0.558  &  0.290  &  0.507  &  0.178 \\
150 chars    & 0.548  &  0.287  &  0.320  &  0.135 \\
\bottomrule
\end{tabular}
\end{table}
```

**Header**: the first row shows the family header `"Gemma 2 9B"` merged across all columns
(`\multicolumn{1}{l}{Gemma 2 9B}`), then each model's short name extracted as the part after
`", "` in `MODEL_NAMES` (e.g., `"9B, Layer 11"` → `"Layer 11"`).

---

### 3.3 `table_probing_appendix.tex`

**Environment**: one `table*` block per model (concatenated in a single file)
**Column spec**: `llll rr`  *(6 columns; header row has 7 cells — known discrepancy)*
**Packages**: `booktabs`, `multirow`

#### Caption Template (verbatim, `{MODEL}` filled per model)

```latex
\caption{Complete probing scores for all top-10 experts in {MODEL}, listed per task and
hypothesis.
Experts are ranked by their cross-validated score on each hypothesis independently,
so the same expert may appear under multiple hypotheses if it encodes more than one concept.
Score $\pm$ standard deviation reports the cross-validated score and its standard deviation across folds for each expert.}
```

Concrete examples:
- `\caption{Complete probing scores for all top-10 experts in 9B, Layer 11, listed per task and hypothesis. ...}`
- `\caption{Complete probing scores for all top-10 experts in 9B, Layer 20, ...}`
- `\caption{Complete probing scores for all top-10 experts in 2B, Layer 12, ...}`

#### Labels (verbatim)

```latex
\label{tab:probing_appendix_gemma_2_9b_l11}
\label{tab:probing_appendix_gemma_2_9b_l20}
\label{tab:probing_appendix_gemma_2_2b_l12}
```

#### Full ASCII Example (Gemma 2 9B, Layer 11 — Weekdays section)

```
\begin{table*}[htbp]
\centering\small
\caption{Probing expert detail --- 9B, Layer 11. All top-10 experts per hypothesis.
Score $\pm$ cross-validation standard deviation.}
\label{tab:probing_appendix_gemma_2_9b_l11}
\begin{tabular}{llll rr}
\toprule
Task & Hypothesis & Regression & Score & Rank & Expert ID & Score $\pm$ Std \\
\midrule
\multirow{20}{*}{Weekdays}
  & \multirow{10}{*}{7-Day Ring}        & \multirow{10}{*}{Linear} & \multirow{10}{*}{$R^2$} &  1 &   76 & 0.855 $\pm$ 0.012 \\
  &                                     &                          &                         &  2 & 1997 & 0.742 $\pm$ 0.033 \\
  &                                     &                          &                         &  3 &  732 & 0.742 $\pm$ 0.013 \\
  &                                     &                          &                         &  4 &  247 & 0.709 $\pm$ 0.008 \\
  &                                     &                          &                         &  5 & 1787 & 0.708 $\pm$ 0.006 \\
  &                                     &                          &                         &  6 &  131 & 0.698 $\pm$ 0.021 \\
  &                                     &                          &                         &  7 & 1365 & 0.671 $\pm$ 0.015 \\
  &                                     &                          &                         &  8 & 1304 & 0.619 $\pm$ 0.026 \\
  &                                     &                          &                         &  9 & 1574 & 0.567 $\pm$ 0.020 \\
  &                                     &                          &                         & 10 &  179 & 0.523 $\pm$ 0.014 \\
  & \multirow{10}{*}{Weekday vs Weekend}& \multirow{10}{*}{Logistic}& \multirow{10}{*}{Acc.} &  1 & 1029 & 1.000 $\pm$ 0.000 \\
  &                                     &                          &                         &  2 & 1304 & 0.999 $\pm$ 0.001 \\
  &                                     &                          &                         &  3 & 1884 & 0.999 $\pm$ 0.002 \\
  &                                     &                          &                         &  4 &  329 & 0.988 $\pm$ 0.008 \\
  &                                     &                          &                         &  5 &  179 & 0.982 $\pm$ 0.011 \\
  &                                     &                          &                         &  6 & 1365 & 0.981 $\pm$ 0.013 \\
  &                                     &                          &                         &  7 &  247 & 0.976 $\pm$ 0.007 \\
  &                                     &                          &                         &  8 & 1997 & 0.964 $\pm$ 0.011 \\
  &                                     &                          &                         &  9 & 1574 & 0.960 $\pm$ 0.019 \\
  &                                     &                          &                         & 10 & 1614 & 0.921 $\pm$ 0.015 \\
\midrule
[Hours section — same structure: 10 experts × 3 hypotheses]
\midrule
[Temperature section — 10 experts × 2 hypotheses]
\midrule
...
\bottomrule
\end{tabular}
\end{table*}

[Second table* block for 9B, Layer 20]
[Third table* block for 2B, Layer 12]
```

**`\multirow` spans**: the Task cell spans the total number of expert rows across all
hypotheses for that dataset. For Weekdays with 2 hypotheses × 10 experts = 20 rows.
The Hypothesis, Regression, and Score cells each span 10 rows (one per expert).

**Dataset ordering**: same as `table_probing.tex` — follows key order in `results.json`.

---

### 3.4 `table_newline_appendix.tex`

**Environment**: one `table` block per 9B model (concatenated in a single file)
**Column spec**: `l rr`
**Packages**: `booktabs`, `multirow`

#### Caption Template (verbatim)

```latex
\caption{Complete newline-position probing scores for all top-10 experts in {MODEL} at each
line length, ranked by $\Delta R^2_{\text{periodic}}$.
This table supports Table\ref{tab:newline}.}
```

#### Labels

```latex
\label{tab:newline_appendix_gemma_2_9b_l11}
\label{tab:newline_appendix_gemma_2_9b_l20}
```

#### Full ASCII Example (Gemma 2 9B, Layer 11)

```
\begin{table}[htbp]
\centering\small
\caption{Newline position expert detail --- 9B, Layer 11. All top-10 experts per line length,
ranked by $\Delta R^2_{\text{periodic}}$.}
\label{tab:newline_appendix_gemma_2_9b_l11}
\begin{tabular}{l rr}
\toprule
Line length & Rank & Expert ID & $\Delta R^2_{\text{per.}}$ \\
\midrule
\multirow{10}{*}{80 chars}
            &  1 &  541 & 0.558 \\
            &  2 & 1521 & 0.430 \\
            &  3 & 1204 & 0.392 \\
            &  4 & 2039 & 0.039 \\
            &  5 & 1224 & 0.031 \\
            &  6 & 1131 & 0.026 \\
            &  7 &  865 & 0.025 \\
            &  8 & 1646 & 0.019 \\
            &  9 & 1470 & 0.018 \\
            & 10 &  753 & 0.015 \\
\midrule
\multirow{10}{*}{150 chars}
            &  1 &  541 & 0.548 \\
            &  2 & 1521 & 0.500 \\
            &  3 & 1204 & 0.317 \\
            &  4 & 1224 & 0.041 \\
            &  5 & 2039 & 0.028 \\
            &  6 & 1131 & 0.028 \\
            &  7 &  865 & 0.027 \\
            &  8 & 1470 & 0.015 \\
            &  9 & 1646 & 0.010 \\
            & 10 & 1166 & 0.009 \\
\bottomrule
\end{tabular}
\end{table}

[Second table block for 9B, Layer 20 — same structure with expert IDs 1520, 1749, ...]
```

**Gemma 2 9B, Layer 20 — top-2 experts per line length for reference**:
- 80 chars: rank 1 = expert 1520 (0.507), rank 2 = expert 1749 (0.303)
- 150 chars: rank 1 = expert 1749 (0.320), rank 2 = expert 1520 (0.280)

---

### 3.5 `table_core_eval.tex`

**Environment**: `table*` (spans both columns)
**Column spec**: `ll rrrrrrrrrrr`  (model + layer label + 11 metric columns)
**Packages**: `booktabs`, `multirow`, `\resizebox{\textwidth}{!}{...}`
**Generated only if** `core_eval_results.json` exists.

#### Caption (verbatim)

```latex
\caption{Core evaluation metrics comparing SMIXAE against GemmaScope SAE baselines, evaluated on OpenWebText (128-token context windows). L0 and width are unflattened numbers for SMIXAE.}
```

#### Metric Column Order and Labels

```python
CORE_EVAL_SUMMARY_METRICS = [
    "width",               # Width
    "total_params",        # Params
    "l0",                  # L0
    "explained_variance",  # Expl. Var.
    "ce_loss_score",       # CE Score
    "mse",                 # MSE (norm.)
    "cosine_similarity",   # Cos. Sim.
]
```

Integers (`width`, `total_params`) are formatted with comma separators. Floats use 3 decimal
places. Missing values render as `--`.

#### Full ASCII Example (real data)

```
\begin{table*}[htbp]
\centering\small
\caption{...}
\label{tab:saebench}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll rrrrrrr}
\toprule
Model / Layer  & SAE                          & Width  & Params        &    L0 & Expl. Var. & CE Score &   MSE & Cos. Sim. \\
\midrule
\multirow{4}{*}{Gemma 2 2B}
  & SMIXAE                       &  6,144 &   151,226,624 & 230.778 &      0.752 &    0.984 & 0.188 &     0.901 \\
  & GemmaScope 2B 16k (L0=176)   & 16,384 &    75,532,544 & 184.535 &      0.841 &    0.994 & 0.121 &     0.938 \\
\midrule
\multirow{4}{*}{Gemma 2 9B}
  & [Layer 11] SMIXAE             &  6,144 &   235,113,984 & 229.389 &      0.655 &    0.985 & 0.231 &     0.876 \\
  & [Layer 11] GemmaScope 9B 16k  & 16,384 &   117,476,864 & 130.275 &      0.774 &    0.995 & 0.150 &     0.922 \\
  & [Layer 20] SMIXAE             &  6,144 &   235,113,984 & 212.189 &      0.736 &    0.978 & 0.201 &     0.895 \\
  & [Layer 20] GemmaScope 9B 16k  & 16,384 &   117,476,864 & 139.555 &      0.819 &    0.991 & 0.137 &     0.929 \\
\bottomrule
\end{tabular}}
\end{table*}
```

> **Note on row grouping**: the Model column uses `\multirow{N}` spanning all SAE rows for
> that model across all layers. The layer distinction currently appears in the SAE name string
> (e.g., `"GemmaScope 9B 16k (L0=118)"`) rather than in a separate layer column — the actual
> layer grouping uses a separate column with `\multirow`. The ASCII above shows the
> logical grouping; exact column arrangement is controlled by `build_core_eval_table()`.

---

## 4. `camera_ready.py` — Figure Specifications

### 4.1 PNG Filename Convention

Files must match this exact pattern (double-underscore `__` delimiters):

```
{experiment_id}__{task}__{E}{expert_id}__{hyp_name}__{score_type}__{score}__{figure_type}.png
```

| Field           | Type   | Example          | Notes                                         |
|-----------------|--------|------------------|-----------------------------------------------|
| `experiment_id` | string | `gemma_2_9b_l11` | No `__` allowed inside                        |
| `task`          | string | `weekdays`       | May contain `_—_` suffix, stripped by code    |
| `expert_id`     | int    | `76`             | Preceded by literal `E`                       |
| `hyp_name`      | string | `cyc_7d`         |                                               |
| `score_type`    | string | `r2`             | One of: `r2`, `acc`, `score`, `per_gain`, `cont` |
| `score`         | float  | `0.855`          | Digits and `.` only                           |
| `figure_type`   | string | `scatter`        | Exactly `scatter` or `means`                  |

**Concrete examples**:
```
gemma_2_9b_l11__weekdays__E76__cyc_7d__r2__0.855__scatter.png
gemma_2_9b_l11__weekdays__E76__cyc_7d__r2__0.855__means.png
gemma_2_9b_l11__pile-uncopyrighted__E541__periodic_gain__per_gain__0.548__scatter.png
gemma_2_9b_l20__living_things__E42__plant_animal__acc__0.963__scatter.png
```

**Files that don't match are silently skipped** (logged to stderr with `[skip]`).

**Task suffix stripping**: `_canonical_task()` strips everything from `_—_` onward.
So `"weekdays_—_expert_analysis"` → `"weekdays"` for grouping and routing.

---

### 4.2 Legend Types

Three rendering paths, selected by the color configuration for each task:

| `dataset_config.json` fields              | Legend type          | Renderer                          |
|-------------------------------------------|----------------------|-----------------------------------|
| `color_map` present                       | Discrete swatches    | `render_discrete_legend_png()`    |
| `color_scale` + `continuous_color: true`  | Continuous colorbar  | `render_continuous_colorbar_png()`|
| `color_scale` + `continuous_color: false` | Discrete from scale  | `render_discrete_from_scale_png()`|
| `labels` only (fallback)                  | Discrete (tab10)     | `render_discrete_legend_png()`    |

**Discrete swatch legend** (example: Weekdays, color_map with 7 colors):
```
+-------------------+
| █ Monday          |
| █ Tuesday         |
| █ Wednesday       |
| █ Thursday        |
| █ Friday          |
| █ Saturday        |
| █ Sunday          |
+-------------------+
```
Labels are sorted before rendering: numerically if all-numeric, by integer prefix if all
prefixed (e.g., `01_Sunday`, `02_Monday` → sorted by prefix, displayed without prefix),
otherwise lexicographically.

**Continuous colorbar** (example: Newline Position, Viridis, 150-char line length):
Viridis maps low values → dark purple, high values → bright yellow.
Ticks are placed every `colorbar_tick_increment` units (20 for newline, per `newline_config.json`).
```
+----------+
| chars    |
| since \n |
|          |
| 140 ──── |░  ← bright yellow (high)
| 120 ──── |▒
| 100 ──── |▒
|  80 ──── |▓
|  60 ──── |▓
|  40 ──── |▓
|  20 ──── |█
|   1 ──── |█  ← dark purple (low)
+----------+
```

**Discrete-from-scale legend** (example: Hours, `phase` colorscale, 24 discrete classes):
24 colored swatches sampled at positions `(i+1)/(n+1)` to avoid endpoint collision on
circular scales (so hour 0 and hour 23 are visually distinct).

---

### 4.3 Layout Constants

All five constants live at the top of `camera_ready.py` (lines 59–63).

| Constant        | Current value      | Controls                                               |
|-----------------|--------------------|--------------------------------------------------------|
| `_USABLE_FRAC`  | `0.98`             | Fraction of `\linewidth` used for all content          |
| `_LEGEND_SCALE` | `0.35`             | Legend slot width as a fraction of one plot slot width |
| `_BLOCK_GAP`    | `\hspace{2.5mm}`   | Horizontal gap between adjacent task blocks in a row   |
| `_ROW_VSPACE`   | `\vspace{5pt}`     | Vertical gap between physical rows                     |
| `_PANEL_HEIGHT` | `4.0cm`            | Fixed minipage height for plots and legends            |

---

### 4.4 Figure Layout — Compact Row

Used when each task in the row has ≤ `--cols` plots (default 3). Multiple tasks can share
the same physical row if their combined plot count ≤ `--cols`.

**Width formula**:
```
plot_w = min(0.98 / (G + 0.35 * L),  0.98 / (cols + 0.35))
```
where `G` = total plots in the row, `L` = number of blocks that have a legend.
The cap `0.98 / (cols + 0.35)` prevents a single-plot row from blowing up.

**ASCII — two tasks sharing a row (cols=3, task A has 2 plots, task B has 1 plot)**:
```
\noindent\makebox[\linewidth][c]{%
┌─────────────────────────────────────────────────────────────────┐
│ [Task A minipage]                                               │
│  ┌──────────────┐ ┌──────────────┐ ┌────────┐                  │
│  │ (a)          │ │ (b)          │ │  key   │                  │
│  │  scatter     │ │   means      │ │        │                  │
│  │  [4.0cm]     │ │  [4.0cm]     │ │        │                  │
│  └──────────────┘ └──────────────┘ └────────┘                  │
│       Task A  (italic footnotesize)                             │
└─────────────────────────────────────────────────────────────────┘
\hspace{2.5mm}
┌─────────────────────────────────────────────────────────────────┐
│ [Task B minipage]                                               │
│  ┌──────────────┐ ┌────────┐                                   │
│  │ (c)          │ │  key   │                                   │
│  │  scatter     │ │        │                                   │
│  │  [4.0cm]     │ │        │                                   │
│  └──────────────┘ └────────┘                                   │
│       Task B  (italic footnotesize)                             │
└─────────────────────────────────────────────────────────────────┘
}%
```

Panels use `tikzpicture` to overlay the letter `(a)` at the top-left corner.

---

### 4.5 Figure Layout — Multi-Row Block

Used when a single task has more plots than `--cols`. The task gets its own dedicated block
that wraps internally. The legend slot is reserved on every internal row but **rendered only
on the final row**.

**Width formula** (legend always reserved):
```
plot_w = 0.98 / (cols + 0.35)
```

**ASCII — Emotions task with 6 scatter+means pairs = 12 plots, cols=3**:
```
\noindent\makebox[\linewidth][c]{%
\begin{minipage}[t]{\linewidth}%
\vspace{0pt}\centering%
┌──────────────────────────────────────────────────────────────────────┐
│ Internal row 0:  (a)  (b)  (c)  [legend slot: empty]                │
├──────────────────────────────────────────────────────────────────────┤
│ Internal row 1:  (d)  (e)  (f)  [legend slot: empty]                │
├──────────────────────────────────────────────────────────────────────┤
│ Internal row 2:  (g)  (h)  (i)  [legend slot: empty]                │
├──────────────────────────────────────────────────────────────────────┤
│ Internal row 3:  (j)  (k)  (l)  [legend: rendered here]            │
└──────────────────────────────────────────────────────────────────────┘
  Emotions  (italic footnotesize)
\end{minipage}}%
```

Empty slots in the final row (when `len(units) % cols != 0`) are rendered as
`\vspace{0pt}%` inside a same-sized minipage.

---

### 4.6 Panel Lettering

Panels are assigned letters in order of appearance across all groups in the figure:
A, B, C, …, Z, AA, AB, …, AZ, BA, …

Within each group, scatter and means plots are paired: the scatter comes first, followed
immediately by its partner means plot for the same expert and hypothesis. Groups are sorted
by score descending (highest score gets panel A).

Letters are rendered as `(a)` (lowercase) via a tikz overlay node anchored to the
top-left of the image.

---

### 4.7 Output .tex Files

Three file types are generated per experiment ID:

| File                      | Tasks included                       | `is_newline` | Caption suffix                                                |
|---------------------------|--------------------------------------|:------------:|---------------------------------------------------------------|
| `probe_{exp_id}.tex`      | All except `pile-uncopyrighted`, `continuity` | No | "Each plot shows..." (see Caption Format below) |
| `newline_{exp_id}.tex`    | `pile-uncopyrighted` only            | Yes          | "Points represent..." (see Caption Format below) |
| `random_{exp_id}.tex`     | `continuity` only                    | No           | "Each plot shows..." (see Caption Format below) |

#### Caption Format — Probe Figures

```
{Model}.  \textbf{Task1}: (a) Expert {id}, rank {rank}, {Hypothesis} ({score_label}\,=\,{score}). (b) Expert {id}, rank {rank}, {Hypothesis} ({score_label}\,=\,{score}).  \textbf{Task2}: ...  Each plot shows the 3-D bottleneck activations of a single SMIXAE expert; small points are individual token activations colored by ground-truth label, and larger points mark per-class means.
```

**Concrete example** (Weekdays, 2 panels):
```
Gemma 2 9B, Layer 11.  \textbf{Weekdays}: (a) Expert 76, rank 1, 7-Day Ring ($R^2$\,=\,0.855). (b) Expert 76, rank 1, 7-Day Ring ($R^2$\,=\,0.855).  Each plot shows the 3-D bottleneck activations of a single SMIXAE expert; small points are individual token activations colored by ground-truth label, and larger points mark per-class means.
```

The `rank N` field is omitted if `--results-dir` was not passed to `smixae latex figures`.

#### Caption Format — Newline Figures

```
Newline Position ({N} chars) --- {Model}.  \textbf{Newline Position}: (a) Expert {id}, rank {rank}, Periodic Gain ($\Delta R^2_{\mathrm{per}}$\,=\,{score}).  Points represent individual token activations in the bottleneck space, colored by distance since the last newline.
```

**Concrete example** (150-char line length):
```
Newline Position (150 chars) --- Gemma 2 9B, Layer 11.  \textbf{Newline Position}: (a) Expert 541, rank 1, Periodic Gain ($\Delta R^2_{\mathrm{per}}$\,=\,0.548).  Points represent individual token activations in the bottleneck space, colored by distance since the last newline.
```

---

### 4.8 Task and Figure Ordering

**Probe task order** within `probe_{exp_id}.tex` (controlled by `_TASK_ORDER`):

```
weekdays → hours → months → temperatures → time_units → body_parts → living_things → colors → emotions
```

Tasks not in this list sort after with key 999, then alphabetically by their full group key.

**Newline wrap order** within `newline_{exp_id}.tex`: sorted by wrap length ascending
(80 chars before 150 chars).

**Within each task**: groups with `hypothesis_color_overrides` are placed before the
default-color group. Within a group, scatter+means pairs are ordered by score descending.

---

## 5. Reconciliation Notes

When editing this spec, use the table below to find the exact code location to update.
Line numbers refer to the state at the time this spec was written and may drift slightly.

| Spec section | What you edited                      | Code location                                   | How to reconcile                                  |
|--------------|--------------------------------------|-------------------------------------------------|---------------------------------------------------|
| §1a          | Task display name (figures)          | `_TASK_DISPLAY`, `camera_ready.py:550`          | Update matching key in dict                       |
| §1a          | Task display name (tables)           | `DATASET_NAMES`, `tables.py:36`                 | Update matching key in dict                       |
| §1b          | Hypothesis display name (figures)    | `_HYP_DISPLAY`, `camera_ready.py:564`           | Update matching key in dict                       |
| §1b          | Hypothesis description (tables)      | `"description"` field in `dataset_config.json`  | Edit the JSON description string                  |
| §1c          | Score label (figures)                | `_SCORE_LABEL`, `camera_ready.py:78`            | Update matching key in dict                       |
| §1c          | Score label (tables)                 | `SCORE_LABEL`, `tables.py:51`                   | Update matching key in dict                       |
| §1d          | Model display name (figures)         | `_format_experiment_id()`, `camera_ready.py:598`| Edit regex or fallback title-case                 |
| §1d          | Model display name (tables)          | `MODEL_NAMES`, `tables.py:28`                   | Update matching key in dict                       |
| §1e          | Skip a dataset from tables           | `SKIP_DATASETS`, `tables.py:49`                 | Add/remove set member                             |
| §3.1         | Probing table caption text           | `build_probing_table()`, `tables.py:138`        | Replace the multi-line string literal             |
| §3.2         | Newline table caption text           | `build_newline_table()`, `tables.py:259`        | Replace the multi-line string literal             |
| §3.3         | Probing appendix caption text        | `build_probing_appendix_tables()`, `tables.py:323` | Replace the string literal (keep `{MODEL}` slot) |
| §3.4         | Newline appendix caption text        | `build_newline_appendix_tables()`, `tables.py:449` | Replace the string literal (keep `{MODEL}` slot) |
| §3.5         | Core eval table caption text         | `build_core_eval_table()`, `tables.py:558`      | Replace the multi-line string literal             |
| §3.5         | Core eval metric column order        | `CORE_EVAL_SUMMARY_METRICS`, `tables.py:509`    | Reorder the list                                  |
| §3.5         | Core eval metric column label        | `CORE_EVAL_METRIC_LABELS`, `tables.py:495`      | Update the matching key                           |
| §4.3         | Any layout constant                  | Module-level constants, `camera_ready.py:59–63` | Edit the constant value directly                  |
| §4.7         | Caption suffix text (probe/random)   | `_caption_text()`, `camera_ready.py:669`        | Edit the `suffix` string (the non-newline branch) |
| §4.7         | Caption suffix text (newline)        | `_caption_text()`, `camera_ready.py:668`        | Edit the `suffix` string (the newline branch)     |
| §4.8         | Task display order in probe figures  | `_TASK_ORDER`, `camera_ready.py:587`            | Reorder the list                                  |
