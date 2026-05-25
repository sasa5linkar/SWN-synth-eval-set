# Human evaluation comparison 01 vs 07

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
| Both axes have same status | 48 / 75 | 64.0% |
| Both axes have same effective value | 50 / 75 | 66.7% |
| Both axes agree on acceptable/not acceptable | 59 / 75 | 78.7% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 80.0% | 0.3378 | 0.5033 |
| Positive | 82.7% | 0.0871 | 0.1182 |
| Negative | 77.3% | 0.4401 | 0.7057 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 87.3% | 0.3165 |
| Positive | 89.3% | 0.1525 |
| Negative | 85.3% | 0.3929 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 82.7% | 92.0% | 0.0633 | 0.8201 |
| Negative | 82.7% | 96.0% | 0.0567 | 0.8545 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 46 / 75.
Rows with any disagreement: 29 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 13 |
| Negative status | 17 |
| Positive acceptable/not acceptable | 8 |
| Negative acceptable/not acceptable | 11 |
| Positive effective value | 13 |
| Negative effective value | 13 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-01215137-v, ENG30-07175241-n, ENG30-00602112-v, ENG30-01193721-v, ENG30-01041349-n, ENG30-01425892-v, ENG30-04831727-n, ENG30-07497797-n
- Negative: ENG30-01989562-v, ENG30-00953559-n, ENG30-03600806-n, ENG30-00200863-v, ENG30-01234345-n, ENG30-07436475-n, ENG30-01215137-v, ENG30-00613683-v, ENG30-00095121-n, ENG30-07175241-n, ENG30-04831727-n

`comparison_01_07.csv` lists every aligned row. `disagreements_01_07.csv` lists rows where the annotators differ by status, acceptability, or effective value.
