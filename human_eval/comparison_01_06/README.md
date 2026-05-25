# Human evaluation comparison 01 vs 06

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
| Both axes have same effective value | 44 / 75 | 58.7% |
| Both axes agree on acceptable/not acceptable | 55 / 75 | 73.3% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 76.0% | 0.3403 | 0.4083 |
| Positive | 77.3% | 0.4215 | 0.4985 |
| Negative | 74.7% | 0.2563 | 0.3027 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 86.0% | 0.3542 |
| Positive | 86.7% | 0.4792 |
| Negative | 85.3% | 0.1888 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 77.3% | 89.3% | 0.0867 | 0.7621 |
| Negative | 78.7% | 92.0% | 0.0733 | 0.8178 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 40 / 75.
Rows with any disagreement: 35 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 17 |
| Negative status | 19 |
| Positive acceptable/not acceptable | 10 |
| Negative acceptable/not acceptable | 11 |
| Positive effective value | 17 |
| Negative effective value | 16 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-00095121-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-00892861-n, ENG30-10000158-n, ENG30-04525038-n, ENG30-01375831-a, ENG30-14438898-n, ENG30-01041349-n, ENG30-01425892-v
- Negative: ENG30-01989562-v, ENG30-00953559-n, ENG30-07157273-n, ENG30-00421002-a, ENG30-00058743-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-02226429-n, ENG30-07175241-n, ENG30-09398076-n, ENG30-02390258-n

`comparison_01_06.csv` lists every aligned row. `disagreements_01_06.csv` lists rows where the annotators differ by status, acceptability, or effective value.
