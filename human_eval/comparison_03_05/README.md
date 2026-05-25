# Human evaluation comparison 03 vs 05

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
| Both axes have same status | 26 / 75 | 34.7% |
| Both axes have same effective value | 25 / 75 | 33.3% |
| Both axes agree on acceptable/not acceptable | 44 / 75 | 58.7% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 65.3% | 0.182 | 0.1682 |
| Positive | 53.3% | 0.1337 | 0.0358 |
| Negative | 77.3% | 0.2178 | 0.4 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 78.7% | 0.1416 |
| Positive | 65.3% | 0.0845 |
| Negative | 92.0% | 0.2321 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 52.0% | 66.7% | 0.2233 | 0.4046 |
| Negative | 77.3% | 92.0% | 0.09 | 0.7509 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 24 / 75.
Rows with any disagreement: 51 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 35 |
| Negative status | 17 |
| Positive acceptable/not acceptable | 26 |
| Negative acceptable/not acceptable | 6 |
| Positive effective value | 36 |
| Negative effective value | 17 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-07157273-n, ENG30-01234345-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-02180898-v, ENG30-06828389-n, ENG30-07175241-n, ENG30-05868272-n, ENG30-10402824-n, ENG30-12392549-n, ENG30-00180962-n, ENG30-00892861-n, ENG30-07132729-n, ENG30-00602112-v, ENG30-10016103-n, ENG30-01482330-n, ENG30-04525038-n, ENG30-01193721-v, ENG30-01485513-v, ENG30-02686625-v, ENG30-09362316-n, ENG30-07164546-n, ENG30-00913551-a, ENG30-00309647-n
- Negative: ENG30-01989562-v, ENG30-00824767-v, ENG30-07436475-n, ENG30-05207130-n, ENG30-01215137-v, ENG30-09398076-n

`comparison_03_05.csv` lists every aligned row. `disagreements_03_05.csv` lists rows where the annotators differ by status, acceptability, or effective value.
