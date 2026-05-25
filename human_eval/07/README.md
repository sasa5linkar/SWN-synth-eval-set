# Human evaluation 07

Source workbook: `PZL_treći zadatak_Dunja Bajčetić.xlsx`  
Copied workbook: `human_eval_07.xlsx`  
Evaluation sheet rows: 75  
Model: `mistral-small3.2:latest`

## Summary

Each row has two human checks: one for the positive Ollama score and one for the negative Ollama score.

| Metric | Count | Rate |
|---|---:|---:|
| Rows where both axes are `tačno` | 52 / 75 | 69.3% |
| Rows where both axes are `tačno` or `blizu` | 61 / 75 | 81.3% |
| Rows with at least one `netačno` axis | 14 / 75 | 18.7% |
| Individual axis judgements that are `tačno` | 126 / 150 | 84.0% |
| Individual axis judgements that are `tačno` or `blizu` | 135 / 150 | 90.0% |

## Axis detail

| Axis | `tačno` | `blizu` | `netačno` | `tačno` rate | `tačno` + `blizu` rate |
|---|---:|---:|---:|---:|---:|
| Positive | 72 | 0 | 3 | 96.0% | 96.0% |
| Negative | 54 | 9 | 11 | 72.0% | 84.0% |

## Effective human value distribution

| Positive value | Count |
|---|---:|
| `nije pozitivan` | 52 |
| `pozitivan` | 15 |
| `veoma pozitivan` | 3 |
| `slabo pozitivan` | 4 |
| `ekstremno pozitivan` | 1 |

| Negative value | Count |
|---|---:|
| `ekstremno negativan` | 1 |
| `negativan` | 9 |
| `slabo negativan` | 11 |
| `veoma negativan` | 6 |
| `nije negativan` | 48 |

## Quality flags

- `unknown_human_eval_status` on `negative`: ENG30-04831727-n (: nije negativan -> )
- `human_value_conflicts_with_exact_status` on `negative`: ENG30-01099592-v (: negativan -> slabo negativan)


Full machine-readable stats are in `human_eval_07_stats.json`; the extracted `Evaluation` sheet is in `human_eval_07.csv`.
