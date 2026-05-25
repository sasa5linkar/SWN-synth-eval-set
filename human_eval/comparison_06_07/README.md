# Human evaluation comparison 06 vs 07

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
| Both axes have same status | 40 / 75 | 53.3% |
| Both axes have same effective value | 41 / 75 | 54.7% |
| Both axes agree on acceptable/not acceptable | 50 / 75 | 66.7% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 74.0% | 0.2523 | 0.3 |
| Positive | 72.0% | 0.1422 | 0.1883 |
| Negative | 76.0% | 0.3844 | 0.4856 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 82.7% | 0.1824 |
| Positive | 81.3% | 0.1667 |
| Negative | 84.0% | 0.2537 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 72.0% | 84.0% | 0.11 | 0.6923 |
| Negative | 80.0% | 97.3% | 0.0567 | 0.8729 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 39 / 75.
Rows with any disagreement: 36 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 21 |
| Negative status | 18 |
| Positive acceptable/not acceptable | 14 |
| Negative acceptable/not acceptable | 12 |
| Positive effective value | 21 |
| Negative effective value | 15 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-01215137-v, ENG30-00095121-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-07175241-n, ENG30-00892861-n, ENG30-00602112-v, ENG30-10000158-n, ENG30-04525038-n, ENG30-01193721-v, ENG30-01375831-a, ENG30-14438898-n, ENG30-04831727-n, ENG30-07497797-n
- Negative: ENG30-03600806-n, ENG30-00200863-v, ENG30-07157273-n, ENG30-00421002-a, ENG30-01234345-n, ENG30-00058743-n, ENG30-07436475-n, ENG30-00613683-v, ENG30-02226429-n, ENG30-09398076-n, ENG30-02390258-n, ENG30-04831727-n

`comparison_06_07.csv` lists every aligned row. `disagreements_06_07.csv` lists rows where the annotators differ by status, acceptability, or effective value.
