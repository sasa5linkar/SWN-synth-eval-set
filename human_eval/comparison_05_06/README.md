# Human evaluation comparison 05 vs 06

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
| Both axes have same status | 34 / 75 | 45.3% |
| Both axes have same effective value | 23 / 75 | 30.7% |
| Both axes agree on acceptable/not acceptable | 52 / 75 | 69.3% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 70.7% | 0.3784 | 0.4467 |
| Positive | 61.3% | 0.3029 | 0.3645 |
| Negative | 80.0% | 0.4425 | 0.5318 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 84.7% | 0.4943 |
| Positive | 76.0% | 0.4231 |
| Negative | 93.3% | 0.5791 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 53.3% | 80.0% | 0.1667 | 0.6245 |
| Negative | 72.0% | 89.3% | 0.1 | 0.7504 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 22 / 75.
Rows with any disagreement: 53 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 29 |
| Negative status | 15 |
| Positive acceptable/not acceptable | 18 |
| Negative acceptable/not acceptable | 5 |
| Positive effective value | 35 |
| Negative effective value | 21 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-02180898-v, ENG30-06828389-n, ENG30-05868272-n, ENG30-10402824-n, ENG30-12392549-n, ENG30-00180962-n, ENG30-07132729-n, ENG30-10000158-n, ENG30-10016103-n, ENG30-01482330-n, ENG30-01485513-v, ENG30-02686625-v, ENG30-09362316-n, ENG30-07164546-n, ENG30-00913551-a, ENG30-01375831-a, ENG30-14438898-n, ENG30-00309647-n
- Negative: ENG30-00824767-v, ENG30-00421002-a, ENG30-00058743-n, ENG30-01215137-v, ENG30-09398076-n

`comparison_05_06.csv` lists every aligned row. `disagreements_05_06.csv` lists rows where the annotators differ by status, acceptability, or effective value.
