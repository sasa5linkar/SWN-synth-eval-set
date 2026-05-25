# Human evaluation comparison 03 vs 06

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
| Both axes have same effective value | 39 / 75 | 52.0% |
| Both axes agree on acceptable/not acceptable | 56 / 75 | 74.7% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 75.3% | 0.256 | 0.3045 |
| Positive | 69.3% | 0.252 | 0.2411 |
| Negative | 81.3% | 0.2336 | 0.3828 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 87.3% | 0.2496 |
| Positive | 81.3% | 0.2222 |
| Negative | 93.3% | 0.269 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 69.3% | 84.0% | 0.1167 | 0.7022 |
| Negative | 78.7% | 93.3% | 0.07 | 0.8401 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 37 / 75.
Rows with any disagreement: 38 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 23 |
| Negative status | 14 |
| Positive acceptable/not acceptable | 14 |
| Negative acceptable/not acceptable | 5 |
| Positive effective value | 23 |
| Negative effective value | 16 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-07157273-n, ENG30-01234345-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-07175241-n, ENG30-00892861-n, ENG30-00602112-v, ENG30-10000158-n, ENG30-04525038-n, ENG30-01193721-v, ENG30-01375831-a, ENG30-14438898-n
- Negative: ENG30-01989562-v, ENG30-00421002-a, ENG30-00058743-n, ENG30-07436475-n, ENG30-05207130-n

`comparison_03_06.csv` lists every aligned row. `disagreements_03_06.csv` lists rows where the annotators differ by status, acceptability, or effective value.
