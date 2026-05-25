# Human evaluation comparison 03 vs 04

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
| Both axes have same status | 46 / 75 | 61.3% |
| Both axes have same effective value | 40 / 75 | 53.3% |
| Both axes agree on acceptable/not acceptable | 59 / 75 | 78.7% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 79.3% | 0.2972 | 0.2892 |
| Positive | 74.7% | 0.2766 | 0.181 |
| Negative | 84.0% | 0.3151 | 0.4297 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 88.7% | 0.2144 |
| Positive | 86.7% | 0.2188 |
| Negative | 90.7% | 0.2033 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 68.0% | 84.0% | 0.1267 | 0.7028 |
| Negative | 82.7% | 92.0% | 0.0833 | 0.7421 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 40 / 75.
Rows with any disagreement: 35 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 19 |
| Negative status | 12 |
| Positive acceptable/not acceptable | 10 |
| Negative acceptable/not acceptable | 7 |
| Positive effective value | 24 |
| Negative effective value | 13 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-07157273-n, ENG30-01234345-n, ENG30-00058743-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-07175241-n, ENG30-00602112-v, ENG30-10016103-n, ENG30-01193721-v, ENG30-07497797-n
- Negative: ENG30-01569181-v, ENG30-01989562-v, ENG30-07436475-n, ENG30-05207130-n, ENG30-01215137-v, ENG30-09398076-n, ENG30-02390258-n

`comparison_03_04.csv` lists every aligned row. `disagreements_03_04.csv` lists rows where the annotators differ by status, acceptability, or effective value.
