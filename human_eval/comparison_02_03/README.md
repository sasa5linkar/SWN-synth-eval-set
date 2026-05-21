# Human evaluation comparison 02 vs 03

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
| Both axes have same status | 57 / 75 | 76.0% |
| Both axes have same effective value | 57 / 75 | 76.0% |
| Both axes agree on acceptable/not acceptable | 66 / 75 | 88.0% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 87.3% | 0.4107 | 0.5354 |
| Positive | 86.7% | 0.477 | 0.577 |
| Negative | 88.0% | 0.3182 | 0.4886 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 93.3% | 0.4131 |
| Positive | 94.7% | 0.5714 |
| Negative | 92.0% | 0.2321 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 85.3% | 96.0% | 0.0467 | 0.8878 |
| Negative | 89.3% | 94.7% | 0.0533 | 0.801 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 55 / 75.
Rows with any disagreement: 20 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 10 |
| Negative status | 9 |
| Positive acceptable/not acceptable | 4 |
| Negative acceptable/not acceptable | 6 |
| Positive effective value | 11 |
| Negative effective value | 8 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-07157273-n, ENG30-01234345-n, ENG30-00602112-v, ENG30-10407310-n
- Negative: ENG30-01989562-v, ENG30-03600806-n, ENG30-00200863-v, ENG30-05207130-n, ENG30-10016103-n, ENG30-10407310-n

`comparison_02_03.csv` lists every aligned row. `disagreements_02_03.csv` lists rows where the annotators differ by status, acceptability, or effective value.
