# Human evaluation comparison 04 vs 05

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
| Both axes have same status | 35 / 75 | 46.7% |
| Both axes have same effective value | 26 / 75 | 34.7% |
| Both axes agree on acceptable/not acceptable | 52 / 75 | 69.3% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 71.3% | 0.364 | 0.4336 |
| Positive | 61.3% | 0.2864 | 0.2925 |
| Negative | 81.3% | 0.4697 | 0.6774 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 84.7% | 0.4687 |
| Positive | 73.3% | 0.3225 |
| Negative | 96.0% | 0.7779 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 52.0% | 74.7% | 0.19 | 0.5864 |
| Negative | 78.7% | 96.0% | 0.0733 | 0.8302 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 25 / 75.
Rows with any disagreement: 50 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 29 |
| Negative status | 14 |
| Positive acceptable/not acceptable | 20 |
| Negative acceptable/not acceptable | 3 |
| Positive effective value | 36 |
| Negative effective value | 16 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-00058743-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-02180898-v, ENG30-06828389-n, ENG30-05868272-n, ENG30-10402824-n, ENG30-12392549-n, ENG30-00180962-n, ENG30-00892861-n, ENG30-07132729-n, ENG30-01482330-n, ENG30-04525038-n, ENG30-01485513-v, ENG30-02686625-v, ENG30-09362316-n, ENG30-07164546-n, ENG30-00913551-a, ENG30-00309647-n, ENG30-07497797-n
- Negative: ENG30-01569181-v, ENG30-00824767-v, ENG30-02390258-n

`comparison_04_05.csv` lists every aligned row. `disagreements_04_05.csv` lists rows where the annotators differ by status, acceptability, or effective value.
