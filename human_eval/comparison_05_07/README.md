# Human evaluation comparison 05 vs 07

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
| Both axes have same effective value | 22 / 75 | 29.3% |
| Both axes agree on acceptable/not acceptable | 40 / 75 | 53.3% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 65.3% | 0.1998 | 0.2013 |
| Positive | 53.3% | 0.0406 | 0.0498 |
| Negative | 77.3% | 0.4623 | 0.6046 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 75.3% | 0.1232 |
| Positive | 65.3% | 0.0661 |
| Negative | 85.3% | 0.3437 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 50.7% | 65.3% | 0.23 | 0.3872 |
| Negative | 76.0% | 92.0% | 0.09 | 0.7435 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 18 / 75.
Rows with any disagreement: 57 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 35 |
| Negative status | 17 |
| Positive acceptable/not acceptable | 26 |
| Negative acceptable/not acceptable | 11 |
| Positive effective value | 37 |
| Negative effective value | 18 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-01215137-v, ENG30-00095121-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-02180898-v, ENG30-06828389-n, ENG30-07175241-n, ENG30-05868272-n, ENG30-10402824-n, ENG30-12392549-n, ENG30-00180962-n, ENG30-00892861-n, ENG30-07132729-n, ENG30-00602112-v, ENG30-10016103-n, ENG30-01482330-n, ENG30-04525038-n, ENG30-01193721-v, ENG30-01485513-v, ENG30-02686625-v, ENG30-09362316-n, ENG30-07164546-n, ENG30-00913551-a, ENG30-00309647-n, ENG30-04831727-n, ENG30-07497797-n
- Negative: ENG30-03600806-n, ENG30-00200863-v, ENG30-00824767-v, ENG30-07157273-n, ENG30-01234345-n, ENG30-07436475-n, ENG30-01215137-v, ENG30-00613683-v, ENG30-02226429-n, ENG30-02390258-n, ENG30-04831727-n

`comparison_05_07.csv` lists every aligned row. `disagreements_05_07.csv` lists rows where the annotators differ by status, acceptability, or effective value.
