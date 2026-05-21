# Human evaluation comparison 01_02_03

Compared rows: 75

## Recommended agreement statistics

- Complete agreement: all annotators chose the same label.
- Fleiss' kappa: chance-corrected agreement for all annotators.
- Binary acceptable agreement: collapses `tačno` + `blizu` vs `netačno`.
- Effective value agreement: compares the final human sentiment value after corrections.

## Row-level complete agreement

| Metric | Count | Rate |
|---|---:|---:|
| Both axes have same status across all annotators | 44 / 75 | 58.7% |
| Both axes have same effective value across all annotators | 42 / 75 | 56.0% |
| Both axes agree on acceptable/not acceptable across all annotators | 55 / 75 | 73.3% |

## Status agreement

| Scope | Complete agreement | Majority exists | Fleiss kappa |
|---|---:|---:|---:|
| All axes | 76.0% | 98.0% | 0.3348 |
| Positive | 77.3% | 98.7% | 0.4041 |
| Negative | 74.7% | 97.3% | 0.2588 |

## Acceptable binary agreement

| Scope | Complete agreement | Majority exists | Fleiss kappa |
|---|---:|---:|---:|
| All axes | 84.0% | 100.0% | 0.2364 |
| Positive | 88.0% | 100.0% | 0.4273 |
| Negative | 80.0% | 100.0% | 0.0455 |

## Effective value agreement

| Axis | Complete agreement | All pairs within one step | Mean pairwise distance | Mean pairwise weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 73.3% | 88.0% | 0.0667 | 0.8196 |
| Negative | 77.3% | 89.3% | 0.0667 | 0.8061 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values across all annotators: 41 / 75.
Rows with any disagreement: 34 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 17 |
| Negative status | 19 |
| Positive acceptable/not acceptable | 9 |
| Negative acceptable/not acceptable | 15 |
| Positive effective value | 20 |
| Negative effective value | 17 |

Binary acceptability disagreements:

- Positive: ENG30-07157273-n, ENG30-01234345-n, ENG30-01215137-v, ENG30-07175241-n, ENG30-00602112-v, ENG30-01193721-v, ENG30-10407310-n, ENG30-01041349-n, ENG30-01425892-v
- Negative: ENG30-01989562-v, ENG30-00953559-n, ENG30-03600806-n, ENG30-00200863-v, ENG30-07157273-n, ENG30-07436475-n, ENG30-05207130-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-02226429-n, ENG30-07175241-n, ENG30-10016103-n, ENG30-09398076-n, ENG30-02390258-n, ENG30-10407310-n

`comparison_01_02_03.csv` lists every aligned row. `disagreements_01_02_03.csv` lists rows where at least one agreement check differs.
