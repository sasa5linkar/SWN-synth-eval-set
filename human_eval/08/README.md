# Human evaluation 08

Source workbook: `PZL_treći zadatak (Teodora Živojinović).xlsx`  
Copied workbook: `human_eval_08.xlsx`  
Evaluation sheet rows: 75  
Model: `mistral-small3.2:latest`

## Summary

Each row has two human checks: one for the positive Ollama score and one for the negative Ollama score.

| Metric | Count | Rate |
|---|---:|---:|
| Rows where both axes are `tačno` | 0 / 75 | 0.0% |
| Rows where both axes are `tačno` or `blizu` | 0 / 75 | 0.0% |
| Rows with at least one `netačno` axis | 4 / 75 | 5.3% |
| Individual axis judgements that are `tačno` | 58 / 150 | 38.7% |
| Individual axis judgements that are `tačno` or `blizu` | 71 / 150 | 47.3% |

## Axis detail

| Axis | `tačno` | `blizu` | `netačno` | `tačno` rate | `tačno` + `blizu` rate |
|---|---:|---:|---:|---:|---:|
| Positive | 15 | 6 | 0 | 20.0% | 28.0% |
| Negative | 43 | 7 | 4 | 57.3% | 66.7% |

## Effective human value distribution

| Positive value | Count |
|---|---:|
| `nije pozitivan` | 54 |
| `pozitivan` | 9 |
| `slabo pozitivan` | 4 |
| `veoma pozitivan` | 6 |
| `ekstremno pozitivan` | 2 |

| Negative value | Count |
|---|---:|
| `ekstremno negativan` | 4 |
| `nije negativan` | 53 |
| `negativan` | 6 |
| `veoma negativan` | 4 |
| `slabo negativan` | 5 |
| `pozitivan` | 3 |

## Quality flags

- `unknown_human_eval_status` on `positive`: ENG30-01569181-v (: nije pozitivan -> ); ENG30-01989562-v (: nije pozitivan -> ); ENG30-01099592-v (: nije pozitivan -> ); ENG30-00953559-n (: nije pozitivan -> ); ENG30-03600806-n (: nije pozitivan -> ); ENG30-01220336-n (: nije pozitivan -> ); ENG30-01468576-v (: nije pozitivan -> ); ENG30-02585489-v (: nije pozitivan -> ); ENG30-06720371-n (: nije pozitivan -> ); ENG30-00200863-v (: nije pozitivan -> ); ENG30-00824767-v (: nije pozitivan -> ); ENG30-07157273-n (: nije pozitivan -> ); ENG30-01623268-a (: nije pozitivan -> ); ENG30-00421002-a (: nije pozitivan -> ); ENG30-01234345-n (: nije pozitivan -> ); ENG30-00058743-n (: nije pozitivan -> ); ENG30-07436475-n (: nije pozitivan -> ); ENG30-05207130-n (: nije pozitivan -> ); ENG30-02505358-v (: nije pozitivan -> ); ENG30-01215137-v (: nije pozitivan -> ); ENG30-00613683-v (: nije pozitivan -> ); ENG30-02574205-v (: nije pozitivan -> ); BILI-00000941 (: nije pozitivan -> ); ENG30-14440623-n (: nije pozitivan -> ); ENG30-00095121-n (: nije pozitivan -> ); ENG30-02226429-n (: nije pozitivan -> ); ENG30-09426788-n (: nije pozitivan -> ); ENG30-12345280-n (: nije pozitivan -> ); ENG30-02180898-v (: nije pozitivan -> ); ENG30-06828389-n (: nije pozitivan -> ); ENG30-07175241-n (: nije pozitivan -> ); ENG30-05868272-n (: nije pozitivan -> ); ENG30-10402824-n (: nije pozitivan -> ); ENG30-12392549-n (: nije pozitivan -> ); ENG30-00180962-n (: nije pozitivan -> ); ENG30-00892861-n (: nije pozitivan -> ); ENG30-07132729-n (: nije pozitivan -> ); ENG30-00602112-v (: nije pozitivan -> ); ENG30-10000158-n (: nije pozitivan -> ); ENG30-10016103-n (: nije pozitivan -> ); ENG30-01482330-n (: nije pozitivan -> ); ENG30-04525038-n (: nije pozitivan -> ); ENG30-01193721-v (: nije pozitivan -> ); ENG30-01485513-v (: nije pozitivan -> ); ENG30-09398076-n (: nije pozitivan -> ); ENG30-02686625-v (: nije pozitivan -> ); ENG30-02390258-n (: nije pozitivan -> ); ENG30-09362316-n (: nije pozitivan -> ); ENG30-07164546-n (: nije pozitivan -> ); ENG30-00654885-n (: nije pozitivan -> ); ENG30-00913551-a (: nije pozitivan -> ); ENG30-01425892-v (: nije pozitivan -> ); ENG30-00309647-n (: nije pozitivan -> ); ENG30-07497797-n (: nije pozitivan -> )
- `unknown_human_eval_status` on `negative`: ENG30-10557854-n (: nije negativan -> ); ENG30-03016953-n (: nije negativan -> ); ENG30-07592094-n (: nije negativan -> ); ENG30-07453638-n (: nije negativan -> ); ENG30-14437976-n (: nije negativan -> ); ENG30-04840405-n (: nije negativan -> ); ENG30-01375831-a (: nije negativan -> ); ENG30-13254805-n (: nije negativan -> ); ENG30-02490877-v (: nije negativan -> ); ENG30-00183090-b (: nije negativan -> ); ENG30-01983162-a (: nije negativan -> ); ENG30-14438898-n (: nije negativan -> ); ENG30-04829282-n (: nije negativan -> ); ENG30-10407310-n (: nije negativan -> ); ENG30-04191595-n (: nije negativan -> ); ENG30-01041349-n (: nije negativan -> ); ENG30-00169955-a (: nije negativan -> ); ENG30-03135152-n (: nije negativan -> ); ENG30-00429048-n (: nije negativan -> ); ENG30-04831727-n (: nije negativan -> ); ENG30-00719231-v (: nije negativan -> )


Full machine-readable stats are in `human_eval_08_stats.json`; the extracted `Evaluation` sheet is in `human_eval_08.csv`.
