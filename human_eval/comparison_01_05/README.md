# Human evaluation comparison 01 vs 05

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
| Both axes have same status | 29 / 75 | 38.7% |
| Both axes have same effective value | 23 / 75 | 30.7% |
| Both axes agree on acceptable/not acceptable | 46 / 75 | 61.3% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 68.0% | 0.2803 | 0.3014 |
| Positive | 60.0% | 0.2331 | 0.187 |
| Negative | 76.0% | 0.3638 | 0.5408 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 80.0% | 0.2982 |
| Positive | 70.7% | 0.2403 |
| Negative | 89.3% | 0.4413 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 53.3% | 70.7% | 0.2067 | 0.4633 |
| Negative | 74.7% | 93.3% | 0.08 | 0.845 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 21 / 75.
Rows with any disagreement: 54 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 30 |
| Negative status | 18 |
| Positive acceptable/not acceptable | 22 |
| Negative acceptable/not acceptable | 8 |
| Positive effective value | 35 |
| Negative effective value | 19 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-00095121-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-02180898-v, ENG30-06828389-n, ENG30-05868272-n, ENG30-10402824-n, ENG30-12392549-n, ENG30-00180962-n, ENG30-00892861-n, ENG30-07132729-n, ENG30-10016103-n, ENG30-01482330-n, ENG30-04525038-n, ENG30-01485513-v, ENG30-02686625-v, ENG30-09362316-n, ENG30-07164546-n, ENG30-00913551-a, ENG30-01041349-n, ENG30-01425892-v, ENG30-00309647-n
- Negative: ENG30-01989562-v, ENG30-00953559-n, ENG30-00824767-v, ENG30-07157273-n, ENG30-00095121-n, ENG30-02226429-n, ENG30-07175241-n, ENG30-02390258-n

`comparison_01_05.csv` lists every aligned row. `disagreements_01_05.csv` lists rows where the annotators differ by status, acceptability, or effective value.
