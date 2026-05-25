# Human evaluation comparison 02 vs 07

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
| Both axes have same status | 54 / 75 | 72.0% |
| Both axes have same effective value | 50 / 75 | 66.7% |
| Both axes agree on acceptable/not acceptable | 64 / 75 | 85.3% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 84.7% | 0.3484 | 0.4333 |
| Positive | 92.0% | 0.3697 | 0.4389 |
| Negative | 77.3% | 0.3363 | 0.4221 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 91.3% | 0.4715 |
| Positive | 94.7% | 0.4737 |
| Negative | 88.0% | 0.463 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 90.7% | 97.3% | 0.0333 | 0.8836 |
| Negative | 74.7% | 93.3% | 0.0867 | 0.7605 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 50 / 75.
Rows with any disagreement: 25 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 6 |
| Negative status | 17 |
| Positive acceptable/not acceptable | 4 |
| Negative acceptable/not acceptable | 9 |
| Positive effective value | 7 |
| Negative effective value | 19 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-00602112-v, ENG30-10407310-n, ENG30-04831727-n, ENG30-07497797-n
- Negative: ENG30-07157273-n, ENG30-01234345-n, ENG30-00613683-v, ENG30-02226429-n, ENG30-10016103-n, ENG30-09398076-n, ENG30-02390258-n, ENG30-10407310-n, ENG30-04831727-n

`comparison_02_07.csv` lists every aligned row. `disagreements_02_07.csv` lists rows where the annotators differ by status, acceptability, or effective value.
