# Human evaluation 06

Source workbook: `PZL_3.xlsx`  
Copied workbook: `human_eval_06.xlsx`  
Evaluation sheet rows: 75  
Model: `mistral-small3.2:latest`

## Summary

Each row has two human checks: one for the positive Ollama score and one for the negative Ollama score.

| Metric | Count | Rate |
|---|---:|---:|
| Rows where both axes are `tačno` | 40 / 75 | 53.3% |
| Rows where both axes are `tačno` or `blizu` | 54 / 75 | 72.0% |
| Rows with at least one `netačno` axis | 21 / 75 | 28.0% |
| Individual axis judgements that are `tačno` | 113 / 150 | 75.3% |
| Individual axis judgements that are `tačno` or `blizu` | 129 / 150 | 86.0% |

## Axis detail

| Axis | `tačno` | `blizu` | `netačno` | `tačno` rate | `tačno` + `blizu` rate |
|---|---:|---:|---:|---:|---:|
| Positive | 52 | 8 | 15 | 69.3% | 80.0% |
| Negative | 61 | 8 | 6 | 81.3% | 92.0% |

## Effective human value distribution

| Positive value | Count |
|---|---:|
| `nije pozitivan` | 38 |
| `slabo pozitivan` | 3 |
| `pozitivan` | 28 |
| `veoma pozitivan` | 5 |
| `ekstremno pozitivan` | 1 |

| Negative value | Count |
|---|---:|
| `ekstremno negativan` | 2 |
| `negativan` | 11 |
| `slabo negativan` | 5 |
| `veoma negativan` | 3 |
| `nije negativan` | 54 |


Full machine-readable stats are in `human_eval_06_stats.json`; the extracted `Evaluation` sheet is in `human_eval_06.csv`.
