# Human evaluation comparison 03 vs 07

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
| Both axes have same effective value | 46 / 75 | 61.3% |
| Both axes agree on acceptable/not acceptable | 63 / 75 | 84.0% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 80.0% | 0.2386 | 0.3439 |
| Positive | 84.0% | 0.2611 | 0.4468 |
| Negative | 76.0% | 0.2675 | 0.3226 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 90.0% | 0.2424 |
| Positive | 94.7% | 0.4737 |
| Negative | 85.3% | 0.1325 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 82.7% | 96.0% | 0.0533 | 0.8775 |
| Negative | 76.0% | 93.3% | 0.08 | 0.8042 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 44 / 75.
Rows with any disagreement: 31 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 12 |
| Negative status | 18 |
| Positive acceptable/not acceptable | 4 |
| Negative acceptable/not acceptable | 11 |
| Positive effective value | 13 |
| Negative effective value | 18 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-07157273-n, ENG30-01234345-n, ENG30-04831727-n, ENG30-07497797-n
- Negative: ENG30-01989562-v, ENG30-03600806-n, ENG30-00200863-v, ENG30-07157273-n, ENG30-01234345-n, ENG30-05207130-n, ENG30-00613683-v, ENG30-02226429-n, ENG30-09398076-n, ENG30-02390258-n, ENG30-04831727-n

`comparison_03_07.csv` lists every aligned row. `disagreements_03_07.csv` lists rows where the annotators differ by status, acceptability, or effective value.
