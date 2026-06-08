# Human evaluation comparison 01_02_03_04_05_06_07_08

Compared rows: 75

## Recommended agreement statistics

- Complete agreement: all annotators chose the same label.
- Fleiss' kappa: chance-corrected agreement for all annotators.
- Binary acceptable agreement: collapses `tačno` + `blizu` vs `netačno`.
- Effective value agreement: compares the final human sentiment value after corrections.

## Row-level complete agreement

| Metric | Count | Rate |
|---|---:|---:|
| Both axes have same status across all annotators | 0 / 75 | 0.0% |
| Both axes have same effective value across all annotators | 6 / 75 | 8.0% |
| Both axes agree on acceptable/not acceptable across all annotators | 0 / 75 | 0.0% |

## Status agreement

| Scope | Complete agreement | Majority exists | Fleiss kappa |
|---|---:|---:|---:|
| All axes | 16.0% | 88.0% | 0.1693 |
| Positive | 5.3% | 90.7% | 0.1275 |
| Negative | 26.7% | 85.3% | 0.2158 |

## Acceptable binary agreement

| Scope | Complete agreement | Majority exists | Fleiss kappa |
|---|---:|---:|---:|
| All axes | 31.3% | 98.0% | 0.1634 |
| Positive | 22.7% | 98.7% | 0.1642 |
| Negative | 40.0% | 97.3% | 0.1358 |

## Effective value agreement

| Axis | Complete agreement | All pairs within one step | Mean pairwise distance | Mean pairwise weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 30.7% | 50.7% | 0.119 | 0.6885 |
| Negative | 54.7% | 77.3% | 0.082 | 0.7757 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values across all annotators: 0 / 75.
Rows with any disagreement: 75 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 71 |
| Negative status | 55 |
| Positive acceptable/not acceptable | 58 |
| Negative acceptable/not acceptable | 45 |
| Positive effective value | 52 |
| Negative effective value | 34 |

Binary acceptability disagreements:

- Positive: ENG30-01569181-v, ENG30-01989562-v, ENG30-01099592-v, ENG30-00953559-n, ENG30-03600806-n, ENG30-01220336-n, ENG30-01468576-v, ENG30-02585489-v, ENG30-06720371-n, ENG30-00200863-v, ENG30-00824767-v, ENG30-07157273-n, ENG30-01623268-a, ENG30-00421002-a, ENG30-01234345-n, ENG30-00058743-n, ENG30-07436475-n, ENG30-05207130-n, ENG30-02505358-v, ENG30-01215137-v, ENG30-00613683-v, ENG30-02574205-v, BILI-00000941, ENG30-14440623-n, ENG30-00095121-n, ENG30-02226429-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-02180898-v, ENG30-06828389-n, ENG30-07175241-n, ENG30-05868272-n, ENG30-10402824-n, ENG30-12392549-n, ENG30-00180962-n, ENG30-00892861-n, ENG30-07132729-n, ENG30-00602112-v, ENG30-10000158-n, ENG30-10016103-n, ENG30-01482330-n, ENG30-04525038-n, ENG30-01193721-v, ENG30-01485513-v, ENG30-09398076-n, ENG30-02686625-v, ENG30-02390258-n, ENG30-09362316-n, ENG30-07164546-n, ENG30-00913551-a, ENG30-01375831-a, ENG30-14438898-n, ENG30-10407310-n, ENG30-01041349-n, ENG30-01425892-v, ENG30-00309647-n, ENG30-04831727-n, ENG30-07497797-n
- Negative: ENG30-01569181-v, ENG30-01989562-v, ENG30-00953559-n, ENG30-03600806-n, ENG30-01468576-v, ENG30-00200863-v, ENG30-00824767-v, ENG30-07157273-n, ENG30-00421002-a, ENG30-01234345-n, ENG30-00058743-n, ENG30-07436475-n, ENG30-05207130-n, ENG30-01215137-v, ENG30-00613683-v, ENG30-00095121-n, ENG30-02226429-n, ENG30-07175241-n, ENG30-10016103-n, ENG30-01193721-v, ENG30-09398076-n, ENG30-10557854-n, ENG30-02390258-n, ENG30-03016953-n, ENG30-07592094-n, ENG30-00654885-n, ENG30-07453638-n, ENG30-14437976-n, ENG30-04840405-n, ENG30-01375831-a, ENG30-13254805-n, ENG30-02490877-v, ENG30-00183090-b, ENG30-01983162-a, ENG30-14438898-n, ENG30-04829282-n, ENG30-10407310-n, ENG30-04191595-n, ENG30-01041349-n, ENG30-01425892-v, ENG30-00169955-a, ENG30-03135152-n, ENG30-00429048-n, ENG30-04831727-n, ENG30-00719231-v

`comparison_01_02_03_04_05_06_07_08.csv` lists every aligned row. `disagreements_01_02_03_04_05_06_07_08.csv` lists rows where at least one agreement check differs. `all_eight_same_effective_values.csv` and `all_eight_same_correction_vs_llm.csv` provide focused consensus filters.
