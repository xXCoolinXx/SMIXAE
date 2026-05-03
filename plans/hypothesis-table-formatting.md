# Plan: Hypothesis Table Formatting Cleanup

## Context

The hypothesis table in the probing results is too wide and visually cluttered:
- `±std` values pad every cell, consuming horizontal space in an already wide table.
- Log-scale hypothesis descriptions use ad-hoc LaTeX like `$\log$-Compressed Fahrenheit` instead of the canonical `$\log(\text{Fahrenheit})$`.
- "Warm / Natural / Cool" has extra spaces around the slashes.
- "Weekday vs Weekend" is verbose; "Mo-Fr vs Sa-Su" conveys the same info in fewer characters.

There is also a pre-existing bug: `esc()` in `tables.py` blindly escapes `_` → `\_` everywhere, which mangles subscripts inside math mode (e.g., `$\log_{10}$` becomes `$\log_{\_10}$`). This bug becomes more visible once the new `\text{}` formatting is in place, so it must be fixed in the same pass.

---

## Files to Change

### 1. `datasets/probing/dataset_config.json`

| Location | Old | New |
|---|---|---|
| `weekdays` → `weekday_weekend` description | `"Weekday vs Weekend"` | `"Mo-Fr vs Sa-Su"` |
| `temperatures` → `log_f` description | `"$\\log$-Compressed Fahrenheit"` | `"$\\log(\\text{Fahrenheit})$"` |
| `time_units` → `log_duration` description | `"$\\log_{10}$ Duration"` | `"$\\log_{10}(\\text{Duration})$"` |
| `colors` → `warm_nat_cool` description | `"Warm / Natural / Cool"` | `"Warm/Natural/Cool"` |

### 2. `src/latex/tables.py`

**a. Fix `esc()` to skip math mode (lines 85–87)**

Replace:
```python
def esc(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
```
With a version that splits on `$` and only escapes even-indexed (plain-text) segments:
```python
def esc(s: str) -> str:
    parts = s.split("$")
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    return "$".join(parts)
```

**b. Remove `±std` from the summary table (line 230)**

```python
# Before
row += [fmt_with_std(top1, top1_std), fmt_with_std(top5_mean, top5_mean_std)]
# After
row += [fmt(top1), fmt(top5_mean)]
```

**c. Remove `±std` from the appendix table (line 398)**

```python
# Before
row.append(fmt_with_std(e.get("score"), e.get("score_std")))
# After
row.append(fmt(e.get("score")))
```

**d. Update summary table caption (lines 144–145)**

Remove the sentence: `r"$\pm$ values are cross-validation standard deviations; for Top-5$_{\mu}$, the standard deviation is averaged across the top-5 experts. "`

**e. Update appendix table caption (lines 336–337)**

Change `r"each cell shows score $\pm$ cross-validation standard deviation."` to `r"each cell shows the regression score."`.

### 3. `src/latex/camera_ready.py` — `_HYP_DISPLAY` dict (lines 564–585)

| Key | Old value | New value |
|---|---|---|
| `"weekday_weekend"` | `"Weekday vs Weekend"` | `"Mo-Fr vs Sa-Su"` |
| `"log_f"` | `r"$\log$ Fahrenheit"` | `r"$\log(\text{Fahrenheit})$"` |
| `"log_duration"` | `r"$\log_{10}$ Duration"` | `r"$\log_{10}(\text{Duration})$"` |

(`"warm_nat_cool"` already reads `"Warm/Natural/Cool"` — no change needed.)

---

## Step-by-Step Execution

1. Edit `datasets/probing/dataset_config.json` — 4 string replacements.
2. Edit `src/latex/tables.py`:
   - Fix `esc()`.
   - Replace two `fmt_with_std(…, std)` call sites with `fmt(…)`.
   - Trim the two captions.
3. Edit `src/latex/camera_ready.py` — 3 string replacements in `_HYP_DISPLAY`.
4. Run `uv run ruff check` and fix any lint errors.
5. Save a copy of this plan to `plans/` in the project root.
6. Regenerate tables: `uv run smixae latex tables` (requires `results/results.json`).

---

## Verification

- Confirm `results/paper/table_probing.tex` no longer contains `\pm` in data cells.
- Confirm hypothesis descriptions render: `$\log(\text{Fahrenheit})$`, `$\log_{10}(\text{Duration})$`, `Warm/Natural/Cool`, `Mo-Fr vs Sa-Su`.
- Confirm `$\log_{10}` subscripts are not double-escaped (`_{\_10}`) in the generated `.tex`.
