# Human evaluation comparison 01 vs 03

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
| Both axes have same status | 50 / 75 | 66.7% |
| Both axes have same effective value | 47 / 75 | 62.7% |
| Both axes agree on acceptable/not acceptable | 60 / 75 | 80.0% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 80.7% | 0.3178 | 0.2506 |
| Positive | 81.3% | 0.3831 | 0.2744 |
| Negative | 80.0% | 0.255 | 0.2394 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 88.0% | 0.1313 |
| Positive | 89.3% | 0.2771 |
| Negative | 86.7% | -0.0246 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 77.3% | 89.3% | 0.0833 | 0.7872 |
| Negative | 81.3% | 92.0% | 0.07 | 0.8282 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 46 / 75.
Rows with any disagreement: 29 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 14 |
| Negative status | 15 |
| Positive acceptable/not acceptable | 8 |
| Negative acceptable/not acceptable | 10 |
| Positive effective value | 17 |
| Negative effective value | 14 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-07157273-n, ENG30-01234345-n, ENG30-01215137-v, ENG30-07175241-n, ENG30-00602112-v, ENG30-01193721-v, ENG30-01041349-n, ENG30-01425892-v
- Negative: ENG30-00953559-n, ENG30-07157273-n, ENG30-07436475-n, ENG30-05207130-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-02226429-n, ENG30-07175241-n, ENG30-09398076-n, ENG30-02390258-n

`comparison_01_03.csv` lists every aligned row. `disagreements_01_03.csv` lists rows where the annotators differ by status, acceptability, or effective value.
