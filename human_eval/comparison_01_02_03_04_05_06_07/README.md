# Human evaluation comparison 01_02_03_04_05_06_07

Compared rows: 75

## Recommended agreement statistics

- Complete agreement: all annotators chose the same label.
- Fleiss' kappa: chance-corrected agreement for all annotators.
- Binary acceptable agreement: collapses `tačno` + `blizu` vs `netačno`.
- Effective value agreement: compares the final human sentiment value after corrections.

## Row-level complete agreement

| Metric | Count | Rate |
|---|---:|---:|
| Both axes have same status across all annotators | 9 / 75 | 12.0% |
| Both axes have same effective value across all annotators | 8 / 75 | 10.7% |
| Both axes agree on acceptable/not acceptable across all annotators | 27 / 75 | 36.0% |

## Status agreement

| Scope | Complete agreement | Majority exists | Fleiss kappa |
|---|---:|---:|---:|
| All axes | 45.3% | 95.3% | 0.2935 |
| Positive | 34.7% | 98.7% | 0.2472 |
| Negative | 56.0% | 92.0% | 0.3427 |

## Acceptable binary agreement

| Scope | Complete agreement | Majority exists | Fleiss kappa |
|---|---:|---:|---:|
| All axes | 62.0% | 100.0% | 0.3181 |
| Positive | 53.3% | 100.0% | 0.2888 |
| Negative | 70.7% | 100.0% | 0.3516 |

## Effective value agreement

| Axis | Complete agreement | All pairs within one step | Mean pairwise distance | Mean pairwise weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 33.3% | 50.7% | 0.1222 | 0.6846 |
| Negative | 58.7% | 78.7% | 0.0825 | 0.7725 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values across all annotators: 8 / 75.
Rows with any disagreement: 67 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 49 |
| Negative status | 33 |
| Positive acceptable/not acceptable | 35 |
| Negative acceptable/not acceptable | 22 |
| Positive effective value | 50 |
| Negative effective value | 31 |

Binary acceptability disagreements:

- Positive: ENG30-07157273-n, ENG30-01234345-n, ENG30-00058743-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-02180898-v, ENG30-06828389-n, ENG30-07175241-n, ENG30-05868272-n, ENG30-10402824-n, ENG30-12392549-n, ENG30-00180962-n, ENG30-00892861-n, ENG30-07132729-n, ENG30-00602112-v, ENG30-10000158-n, ENG30-10016103-n, ENG30-01482330-n, ENG30-04525038-n, ENG30-01193721-v, ENG30-01485513-v, ENG30-02686625-v, ENG30-09362316-n, ENG30-07164546-n, ENG30-00913551-a, ENG30-01375831-a, ENG30-14438898-n, ENG30-10407310-n, ENG30-01041349-n, ENG30-01425892-v, ENG30-00309647-n, ENG30-04831727-n, ENG30-07497797-n
- Negative: ENG30-01569181-v, ENG30-01989562-v, ENG30-00953559-n, ENG30-03600806-n, ENG30-00200863-v, ENG30-00824767-v, ENG30-07157273-n, ENG30-00421002-a, ENG30-01234345-n, ENG30-00058743-n, ENG30-07436475-n, ENG30-05207130-n, ENG30-01215137-v, ENG30-00613683-v, ENG30-00095121-n, ENG30-02226429-n, ENG30-07175241-n, ENG30-10016103-n, ENG30-09398076-n, ENG30-02390258-n, ENG30-10407310-n, ENG30-04831727-n

`comparison_01_02_03_04_05_06_07.csv` lists every aligned row. `disagreements_01_02_03_04_05_06_07.csv` lists rows where at least one agreement check differs. `all_seven_same_effective_values.csv` and `all_seven_same_correction_vs_llm.csv` provide focused consensus filters.
