# Human evaluation 05

Source workbook: `PZL_treći zadatak Katarina Kužet.xlsx`  
Copied workbook: `human_eval_05.xlsx`  
Evaluation sheet rows: 75  
Model: `mistral-small3.2:latest`

## Summary

Each row has two human checks: one for the positive Ollama score and one for the negative Ollama score.

| Metric | Count | Rate |
|---|---:|---:|
| Rows where both axes are `tačno` | 22 / 75 | 29.3% |
| Rows where both axes are `tačno` or `blizu` | 42 / 75 | 56.0% |
| Rows with at least one `netačno` axis | 33 / 75 | 44.0% |
| Individual axis judgements that are `tačno` | 96 / 150 | 64.0% |
| Individual axis judgements that are `tačno` or `blizu` | 116 / 150 | 77.3% |

## Axis detail

| Axis | `tačno` | `blizu` | `netačno` | `tačno` rate | `tačno` + `blizu` rate |
|---|---:|---:|---:|---:|---:|
| Positive | 39 | 9 | 27 | 52.0% | 64.0% |
| Negative | 57 | 11 | 7 | 76.0% | 90.7% |

## Effective human value distribution

| Positive value | Count |
|---|---:|
| `nije pozitivan` | 26 |
| `slabo pozitivan` | 2 |
| `veoma pozitivan` | 16 |
| `pozitivan` | 31 |

| Negative value | Count |
|---|---:|
| `ekstremno negativan` | 4 |
| `veoma negativan` | 8 |
| `slabo negativan` | 10 |
| `negativan` | 3 |
| `nije negativan` | 50 |


Full machine-readable stats are in `human_eval_05_stats.json`; the extracted `Evaluation` sheet is in `human_eval_05.csv`.
