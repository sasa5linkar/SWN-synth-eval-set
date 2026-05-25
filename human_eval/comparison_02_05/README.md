# Human evaluation comparison 02 vs 05

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
| Both axes have same status | 28 / 75 | 37.3% |
| Both axes have same effective value | 22 / 75 | 29.3% |
| Both axes agree on acceptable/not acceptable | 46 / 75 | 61.3% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 66.0% | 0.1567 | 0.1615 |
| Positive | 58.7% | 0.1755 | 0.1333 |
| Negative | 73.3% | 0.1409 | 0.2414 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 78.7% | 0.211 |
| Positive | 68.0% | 0.1549 |
| Negative | 89.3% | 0.3697 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 54.7% | 66.7% | 0.2167 | 0.4025 |
| Negative | 69.3% | 85.3% | 0.1367 | 0.5723 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 22 / 75.
Rows with any disagreement: 53 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 31 |
| Negative status | 20 |
| Positive acceptable/not acceptable | 24 |
| Negative acceptable/not acceptable | 8 |
| Positive effective value | 34 |
| Negative effective value | 23 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-01215137-v, ENG30-00095121-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-02180898-v, ENG30-06828389-n, ENG30-07175241-n, ENG30-05868272-n, ENG30-10402824-n, ENG30-12392549-n, ENG30-00180962-n, ENG30-00892861-n, ENG30-07132729-n, ENG30-10016103-n, ENG30-01482330-n, ENG30-04525038-n, ENG30-01193721-v, ENG30-01485513-v, ENG30-02686625-v, ENG30-09362316-n, ENG30-07164546-n, ENG30-00913551-a, ENG30-10407310-n, ENG30-00309647-n
- Negative: ENG30-03600806-n, ENG30-00200863-v, ENG30-00824767-v, ENG30-07436475-n, ENG30-01215137-v, ENG30-10016103-n, ENG30-09398076-n, ENG30-10407310-n

`comparison_02_05.csv` lists every aligned row. `disagreements_02_05.csv` lists rows where the annotators differ by status, acceptability, or effective value.
