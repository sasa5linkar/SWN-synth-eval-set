# Human evaluation comparison 02 vs 04

Compared rows: 75

## Recommended agreement statistics

- Exact agreement: easiest to read; counts identical human labels.
- Cohen's kappa: corrects exact agreement for chance agreement.
- Quadratic weighted kappa: useful because `tačno`, `blizu`, `netačno` and sentiment intensity values are ordered.
- Binary acceptable agreement: collapses `tačno` + `blizu` vs `netačno` for the practical question "is the synthetic label usable?".
- Effective value agreement: compares the final human sentiment value after applying corrections from `human_*_value`.

## Row-level agreement

| Metric | Count | Rate |
|---|---:|---:|
| Both axes have same status | 51 / 75 | 68.0% |
| Both axes have same effective value | 44 / 75 | 58.7% |
| Both axes agree on acceptable/not acceptable | 61 / 75 | 81.3% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 81.3% | 0.3066 | 0.3396 |
| Positive | 81.3% | 0.3519 | 0.3412 |
| Negative | 81.3% | 0.2553 | 0.3403 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 88.7% | 0.3531 |
| Positive | 89.3% | 0.375 |
| Negative | 88.0% | 0.3337 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 74.7% | 84.0% | 0.1133 | 0.6829 |
| Negative | 78.7% | 88.0% | 0.11 | 0.6327 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 43 / 75.
Rows with any disagreement: 32 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 14 |
| Negative status | 14 |
| Positive acceptable/not acceptable | 8 |
| Negative acceptable/not acceptable | 9 |
| Positive effective value | 19 |
| Negative effective value | 16 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-00058743-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-07175241-n, ENG30-10016103-n, ENG30-01193721-v, ENG30-10407310-n, ENG30-07497797-n
- Negative: ENG30-01569181-v, ENG30-03600806-n, ENG30-00200863-v, ENG30-07436475-n, ENG30-01215137-v, ENG30-10016103-n, ENG30-09398076-n, ENG30-02390258-n, ENG30-10407310-n

`comparison_02_04.csv` lists every aligned row. `disagreements_02_04.csv` lists rows where the annotators differ by status, acceptability, or effective value.
