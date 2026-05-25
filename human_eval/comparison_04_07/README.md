# Human evaluation comparison 04 vs 07

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
| Both axes have same status | 47 / 75 | 62.7% |
| Both axes have same effective value | 45 / 75 | 60.0% |
| Both axes agree on acceptable/not acceptable | 59 / 75 | 78.7% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 79.3% | 0.3381 | 0.4602 |
| Positive | 80.0% | 0.2089 | 0.3056 |
| Negative | 78.7% | 0.4403 | 0.5842 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 88.0% | 0.3706 |
| Positive | 89.3% | 0.2908 |
| Negative | 86.7% | 0.4266 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 78.7% | 81.3% | 0.1067 | 0.695 |
| Negative | 78.7% | 88.0% | 0.0967 | 0.6964 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 42 / 75.
Rows with any disagreement: 33 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 15 |
| Negative status | 16 |
| Positive acceptable/not acceptable | 8 |
| Negative acceptable/not acceptable | 10 |
| Positive effective value | 16 |
| Negative effective value | 16 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-00058743-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-07175241-n, ENG30-00602112-v, ENG30-10016103-n, ENG30-01193721-v, ENG30-04831727-n
- Negative: ENG30-01569181-v, ENG30-03600806-n, ENG30-00200863-v, ENG30-07157273-n, ENG30-01234345-n, ENG30-07436475-n, ENG30-01215137-v, ENG30-00613683-v, ENG30-02226429-n, ENG30-04831727-n

`comparison_04_07.csv` lists every aligned row. `disagreements_04_07.csv` lists rows where the annotators differ by status, acceptability, or effective value.
