# Human evaluation comparison 01_02_03_04

Compared rows: 75

## Recommended agreement statistics

- Complete agreement: all annotators chose the same label.
- Fleiss' kappa: chance-corrected agreement for all annotators.
- Binary acceptable agreement: collapses `tačno` + `blizu` vs `netačno`.
- Effective value agreement: compares the final human sentiment value after corrections.

## Row-level complete agreement

| Metric | Count | Rate |
|---|---:|---:|
| Both axes have same status across all annotators | 36 / 75 | 48.0% |
| Both axes have same effective value across all annotators | 35 / 75 | 46.7% |
| Both axes agree on acceptable/not acceptable across all annotators | 52 / 75 | 69.3% |

## Status agreement

| Scope | Complete agreement | Majority exists | Fleiss kappa |
|---|---:|---:|---:|
| All axes | 68.0% | 88.7% | 0.3346 |
| Positive | 66.7% | 88.0% | 0.3523 |
| Negative | 69.3% | 89.3% | 0.3131 |

## Acceptable binary agreement

| Scope | Complete agreement | Majority exists | Fleiss kappa |
|---|---:|---:|---:|
| All axes | 80.7% | 94.7% | 0.3214 |
| Positive | 82.7% | 96.0% | 0.4104 |
| Negative | 78.7% | 93.3% | 0.2291 |

## Effective value agreement

| Axis | Complete agreement | All pairs within one step | Mean pairwise distance | Mean pairwise weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 64.0% | 76.0% | 0.0906 | 0.7677 |
| Negative | 72.0% | 85.3% | 0.0778 | 0.769 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values across all annotators: 34 / 75.
Rows with any disagreement: 41 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 25 |
| Negative status | 23 |
| Positive acceptable/not acceptable | 13 |
| Negative acceptable/not acceptable | 16 |
| Positive effective value | 27 |
| Negative effective value | 21 |

Binary acceptability disagreements:

- Positive: ENG30-07157273-n, ENG30-01234345-n, ENG30-00058743-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-07175241-n, ENG30-00602112-v, ENG30-10016103-n, ENG30-01193721-v, ENG30-10407310-n, ENG30-01041349-n, ENG30-01425892-v, ENG30-07497797-n
- Negative: ENG30-01569181-v, ENG30-01989562-v, ENG30-00953559-n, ENG30-03600806-n, ENG30-00200863-v, ENG30-07157273-n, ENG30-07436475-n, ENG30-05207130-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-02226429-n, ENG30-07175241-n, ENG30-10016103-n, ENG30-09398076-n, ENG30-02390258-n, ENG30-10407310-n

`comparison_01_02_03_04.csv` lists every aligned row. `disagreements_01_02_03_04.csv` lists rows where at least one agreement check differs. `all_four_same_effective_values.csv` and `all_four_same_correction_vs_llm.csv` provide focused consensus filters.
