# Human evaluation 03

Source workbook: `PZL_treći zadatak_Sofija.xlsx`  
Copied workbook: `human_eval_03.xlsx`  
Evaluation sheet rows: 75  
Model: `mistral-small3.2:latest`

## Summary

Each row has two human checks: one for the positive Ollama score and one for the negative Ollama score.

| Metric | Count | Rate |
|---|---:|---:|
| Rows where both axes are `tačno` | 55 / 75 | 73.3% |
| Rows where both axes are `tačno` or `blizu` | 69 / 75 | 92.0% |
| Rows with at least one `netačno` axis | 6 / 75 | 8.0% |
| Individual axis judgements that are `tačno` | 130 / 150 | 86.7% |
| Individual axis judgements that are `tačno` or `blizu` | 144 / 150 | 96.0% |

## Axis detail

| Axis | `tačno` | `blizu` | `netačno` | `tačno` rate | `tačno` + `blizu` rate |
|---|---:|---:|---:|---:|---:|
| Positive | 61 | 9 | 5 | 81.3% | 93.3% |
| Negative | 69 | 5 | 1 | 92.0% | 98.7% |

## Effective human value distribution

| Positive value | Count |
|---|---:|
| `nije pozitivan` | 48 |
| `pozitivan` | 17 |
| `slabo pozitivan` | 2 |
| `veoma pozitivan` | 7 |
| `ekstremno pozitivan` | 1 |

| Negative value | Count |
|---|---:|
| `ekstremno negativan` | 4 |
| `slabo negativan` | 6 |
| `negativan` | 2 |
| `veoma negativan` | 7 |
| `nije negativan` | 56 |


Full machine-readable stats are in `human_eval_03_stats.json`; the extracted `Evaluation` sheet is in `human_eval_03.csv`.
