# Human evaluation comparison 01 vs 02

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
| Both axes have same effective value | 48 / 75 | 64.0% |
| Both axes agree on acceptable/not acceptable | 58 / 75 | 77.3% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 82.0% | 0.302 | 0.3253 |
| Positive | 85.3% | 0.3634 | 0.4072 |
| Negative | 78.7% | 0.2509 | 0.2581 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 86.7% | 0.2138 |
| Positive | 92.0% | 0.4578 |
| Negative | 81.3% | 0.0223 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 81.3% | 92.0% | 0.07 | 0.7839 |
| Negative | 80.0% | 92.0% | 0.0767 | 0.789 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 47 / 75.
Rows with any disagreement: 28 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 11 |
| Negative status | 16 |
| Positive acceptable/not acceptable | 6 |
| Negative acceptable/not acceptable | 14 |
| Positive effective value | 14 |
| Negative effective value | 15 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-01215137-v, ENG30-07175241-n, ENG30-01193721-v, ENG30-10407310-n, ENG30-01041349-n, ENG30-01425892-v
- Negative: ENG30-01989562-v, ENG30-00953559-n, ENG30-03600806-n, ENG30-00200863-v, ENG30-07157273-n, ENG30-07436475-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-02226429-n, ENG30-07175241-n, ENG30-10016103-n, ENG30-09398076-n, ENG30-02390258-n, ENG30-10407310-n

`comparison_01_02.csv` lists every aligned row. `disagreements_01_02.csv` lists rows where the annotators differ by status, acceptability, or effective value.
