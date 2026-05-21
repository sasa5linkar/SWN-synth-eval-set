# Human evaluation 01

Source workbook: `PZL_treći zadatak.xlsx`  
Copied workbook: `human_eval_01.xlsx`  
Evaluation sheet rows: 75  
Model: `mistral-small3.2:latest`

## Summary

Each row has two human checks: one for the positive Ollama score and one for the negative Ollama score.

| Metric | Count | Rate |
|---|---:|---:|
| Rows where both axes are `tačno` | 49 / 75 | 65.3% |
| Rows where both axes are `tačno` or `blizu` | 61 / 75 | 81.3% |
| Rows with at least one `netačno` axis | 14 / 75 | 18.7% |
| Individual axis judgements that are `tačno` | 122 / 150 | 81.3% |
| Individual axis judgements that are `tačno` or `blizu` | 134 / 150 | 89.3% |

## Axis detail

| Axis | `tačno` | `blizu` | `netačno` | `tačno` rate | `tačno` + `blizu` rate |
|---|---:|---:|---:|---:|---:|
| Positive | 63 | 5 | 7 | 84.0% | 90.7% |
| Negative | 59 | 7 | 9 | 78.7% | 88.0% |

## Effective human value distribution

| Positive value | Count |
|---|---:|
| `nije pozitivan` | 49 |
| `pozitivan` | 19 |
| `veoma pozitivan` | 5 |
| `ekstremno pozitivan` | 1 |
| `slabo pozitivan` | 1 |

| Negative value | Count |
|---|---:|
| `ekstremno negativan` | 4 |
| `negativan` | 8 |
| `veoma negativan` | 4 |
| `slabo negativan` | 10 |
| `nije negativan` | 49 |

## Quality flags

- `correction_status_but_same_value` on `negative`: ENG30-07175241-n


Full machine-readable stats are in `human_eval_01_stats.json`; the extracted `Evaluation` sheet is in `human_eval_01.csv`.
