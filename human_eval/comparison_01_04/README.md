# Human evaluation comparison 01 vs 04

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
| Both axes have same effective value | 43 / 75 | 57.3% |
| Both axes agree on acceptable/not acceptable | 63 / 75 | 84.0% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 80.0% | 0.3945 | 0.5132 |
| Positive | 77.3% | 0.3175 | 0.4516 |
| Negative | 82.7% | 0.4758 | 0.5758 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 91.3% | 0.5574 |
| Positive | 92.0% | 0.581 |
| Negative | 90.7% | 0.5358 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 73.3% | 86.7% | 0.1033 | 0.7617 |
| Negative | 80.0% | 93.3% | 0.0733 | 0.8209 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 41 / 75.
Rows with any disagreement: 34 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 17 |
| Negative status | 13 |
| Positive acceptable/not acceptable | 6 |
| Negative acceptable/not acceptable | 7 |
| Positive effective value | 20 |
| Negative effective value | 15 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-00058743-n, ENG30-00095121-n, ENG30-10016103-n, ENG30-01041349-n, ENG30-01425892-v, ENG30-07497797-n
- Negative: ENG30-01569181-v, ENG30-01989562-v, ENG30-00953559-n, ENG30-07157273-n, ENG30-00095121-n, ENG30-02226429-n, ENG30-07175241-n

`comparison_01_04.csv` lists every aligned row. `disagreements_01_04.csv` lists rows where the annotators differ by status, acceptability, or effective value.
