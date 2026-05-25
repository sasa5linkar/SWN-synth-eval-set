# Human evaluation 04

Source workbook: `Izabela_Mladenovic_PZL_treći zadatak .xlsx`  
Copied workbook: `human_eval_04.xlsx`  
Evaluation sheet rows: 75  
Model: `mistral-small3.2:latest`

## Summary

Each row has two human checks: one for the positive Ollama score and one for the negative Ollama score.

| Metric | Count | Rate |
|---|---:|---:|
| Rows where both axes are `tačno` | 47 / 75 | 62.7% |
| Rows where both axes are `tačno` or `blizu` | 59 / 75 | 78.7% |
| Rows with at least one `netačno` axis | 16 / 75 | 21.3% |
| Individual axis judgements that are `tačno` | 120 / 150 | 80.0% |
| Individual axis judgements that are `tačno` or `blizu` | 133 / 150 | 88.7% |

## Axis detail

| Axis | `tačno` | `blizu` | `netačno` | `tačno` rate | `tačno` + `blizu` rate |
|---|---:|---:|---:|---:|---:|
| Positive | 58 | 8 | 9 | 77.3% | 88.0% |
| Negative | 62 | 5 | 8 | 82.7% | 89.3% |

## Effective human value distribution

| Positive value | Count |
|---|---:|
| `nije pozitivan` | 42 |
| `slabo pozitivan` | 4 |
| `veoma pozitivan` | 7 |
| `pozitivan` | 17 |
| `ekstremno pozitivan` | 5 |

| Negative value | Count |
|---|---:|
| `nije negativan` | 52 |
| `negativan` | 5 |
| `ekstremno negativan` | 6 |
| `veoma negativan` | 6 |
| `slabo negativan` | 6 |

## Quality flags

- `human_value_conflicts_with_exact_status` on `negative`: BILI-00000941 (: negativan -> nije negativan)


Full machine-readable stats are in `human_eval_04_stats.json`; the extracted `Evaluation` sheet is in `human_eval_04.csv`.
