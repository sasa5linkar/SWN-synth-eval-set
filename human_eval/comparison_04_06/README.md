# Human evaluation comparison 04 vs 06

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
| Both axes have same status | 41 / 75 | 54.7% |
| Both axes have same effective value | 37 / 75 | 49.3% |
| Both axes agree on acceptable/not acceptable | 60 / 75 | 80.0% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 76.0% | 0.3552 | 0.4815 |
| Positive | 70.7% | 0.3154 | 0.463 |
| Negative | 81.3% | 0.4017 | 0.4983 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 89.3% | 0.5187 |
| Positive | 86.7% | 0.5098 |
| Negative | 92.0% | 0.5283 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 68.0% | 85.3% | 0.1167 | 0.7485 |
| Negative | 77.3% | 86.7% | 0.1 | 0.7036 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 36 / 75.
Rows with any disagreement: 39 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 22 |
| Negative status | 14 |
| Positive acceptable/not acceptable | 10 |
| Negative acceptable/not acceptable | 6 |
| Positive effective value | 24 |
| Negative effective value | 17 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-00058743-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-00892861-n, ENG30-10000158-n, ENG30-10016103-n, ENG30-04525038-n, ENG30-01375831-a, ENG30-14438898-n, ENG30-07497797-n
- Negative: ENG30-01569181-v, ENG30-00421002-a, ENG30-00058743-n, ENG30-01215137-v, ENG30-09398076-n, ENG30-02390258-n

`comparison_04_06.csv` lists every aligned row. `disagreements_04_06.csv` lists rows where the annotators differ by status, acceptability, or effective value.
