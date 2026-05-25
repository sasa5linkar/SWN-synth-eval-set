# Human evaluation comparison 02 vs 06

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
| Both axes have same status | 44 / 75 | 58.7% |
| Both axes have same effective value | 42 / 75 | 56.0% |
| Both axes agree on acceptable/not acceptable | 57 / 75 | 76.0% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 77.3% | 0.2657 | 0.3929 |
| Positive | 73.3% | 0.2492 | 0.3577 |
| Negative | 81.3% | 0.2944 | 0.4528 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 87.3% | 0.359 |
| Positive | 84.0% | 0.3333 |
| Negative | 90.7% | 0.4108 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 72.0% | 86.7% | 0.1033 | 0.7214 |
| Negative | 80.0% | 92.0% | 0.0767 | 0.7654 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 40 / 75.
Rows with any disagreement: 35 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 20 |
| Negative status | 14 |
| Positive acceptable/not acceptable | 12 |
| Negative acceptable/not acceptable | 7 |
| Positive effective value | 21 |
| Negative effective value | 15 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-01215137-v, ENG30-00095121-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-07175241-n, ENG30-00892861-n, ENG30-10000158-n, ENG30-04525038-n, ENG30-01193721-v, ENG30-01375831-a, ENG30-14438898-n, ENG30-10407310-n
- Negative: ENG30-03600806-n, ENG30-00200863-v, ENG30-00421002-a, ENG30-00058743-n, ENG30-07436475-n, ENG30-10016103-n, ENG30-10407310-n

`comparison_02_06.csv` lists every aligned row. `disagreements_02_06.csv` lists rows where the annotators differ by status, acceptability, or effective value.
