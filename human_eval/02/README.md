# Human evaluation 02

Source workbook: `PZL_treći zadatak(1).xlsx`  
Copied workbook: `human_eval_02.xlsx`  
Evaluation sheet rows: 75  
Model: `mistral-small3.2:latest`

## Summary

Each row has two human checks: one for the positive Ollama score and one for the negative Ollama score.

| Metric | Count | Rate |
|---|---:|---:|
| Rows where both axes are `tačno` | 61 / 75 | 81.3% |
| Rows where both axes are `tačno` or `blizu` | 64 / 75 | 85.3% |
| Rows with at least one `netačno` axis | 11 / 75 | 14.7% |
| Individual axis judgements that are `tačno` | 135 / 150 | 90.0% |
| Individual axis judgements that are `tačno` or `blizu` | 138 / 150 | 92.0% |

## Axis detail

| Axis | `tačno` | `blizu` | `netačno` | `tačno` rate | `tačno` + `blizu` rate |
|---|---:|---:|---:|---:|---:|
| Positive | 68 | 2 | 5 | 90.7% | 93.3% |
| Negative | 67 | 1 | 7 | 89.3% | 90.7% |

## Effective human value distribution

| Positive value | Count |
|---|---:|
| `nije pozitivan` | 51 |
| `slabo pozitivan` | 3 |
| `pozitivan` | 18 |
| `veoma pozitivan` | 2 |
| `ekstremno pozitivan` | 1 |

| Negative value | Count |
|---|---:|
| `ekstremno negativan` | 5 |
| `slabo negativan` | 7 |
| `negativan` | 4 |
| `nije negativan` | 55 |
| `veoma negativan` | 4 |


Full machine-readable stats are in `human_eval_02_stats.json`; the extracted `Evaluation` sheet is in `human_eval_02.csv`.
