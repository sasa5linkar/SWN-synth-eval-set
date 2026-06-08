# Human evaluation comparison 01 vs 08

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
| Both axes have same status | 0 / 75 | 0.0% |
| Both axes have same effective value | 42 / 75 | 56.0% |
| Both axes agree on acceptable/not acceptable | 2 / 75 | 2.7% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 30.7% | -0.026 | -0.1887 |
| Positive | 16.0% | -0.0161 | 0.2623 |
| Negative | 45.3% | -0.024 | -0.2739 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 44.7% | -0.0621 |
| Positive | 34.7% | 0.0377 |
| Negative | 54.7% | -0.2143 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 80.0% | 90.7% | 0.08 | 0.747 |
| Negative | 72.0% | 86.7% | 0.0868 | 0.8055 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 0 / 75.
Rows with any disagreement: 75 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 63 |
| Negative status | 41 |
| Positive acceptable/not acceptable | 49 |
| Negative acceptable/not acceptable | 34 |
| Positive effective value | 15 |
| Negative effective value | 21 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-01569181-v, ENG30-01989562-v, ENG30-01099592-v, ENG30-00953559-n, ENG30-03600806-n, ENG30-01220336-n, ENG30-01468576-v, ENG30-02585489-v, ENG30-06720371-n, ENG30-00200863-v, ENG30-00824767-v, ENG30-07157273-n, ENG30-01623268-a, ENG30-00421002-a, ENG30-01234345-n, ENG30-00058743-n, ENG30-07436475-n, ENG30-05207130-n, ENG30-02505358-v, ENG30-00613683-v, ENG30-02574205-v, BILI-00000941, ENG30-14440623-n, ENG30-00095121-n, ENG30-02226429-n, ENG30-09426788-n, ENG30-12345280-n, ENG30-02180898-v, ENG30-06828389-n, ENG30-05868272-n, ENG30-10402824-n, ENG30-12392549-n, ENG30-00180962-n, ENG30-00892861-n, ENG30-07132729-n, ENG30-10000158-n, ENG30-10016103-n, ENG30-01482330-n, ENG30-04525038-n, ENG30-01485513-v, ENG30-09398076-n, ENG30-02686625-v, ENG30-02390258-n, ENG30-09362316-n, ENG30-07164546-n, ENG30-00913551-a, ENG30-01041349-n, ENG30-01425892-v, ENG30-00309647-n
- Negative: ENG30-01468576-v, ENG30-07157273-n, ENG30-07436475-n, ENG30-05207130-n, ENG30-01215137-v, ENG30-00095121-n, ENG30-02226429-n, ENG30-07175241-n, ENG30-01193721-v, ENG30-09398076-n, ENG30-10557854-n, ENG30-02390258-n, ENG30-03016953-n, ENG30-07592094-n, ENG30-00654885-n, ENG30-07453638-n, ENG30-14437976-n, ENG30-04840405-n, ENG30-01375831-a, ENG30-13254805-n, ENG30-02490877-v, ENG30-00183090-b, ENG30-01983162-a, ENG30-14438898-n, ENG30-04829282-n, ENG30-10407310-n, ENG30-04191595-n, ENG30-01041349-n, ENG30-01425892-v, ENG30-00169955-a, ENG30-03135152-n, ENG30-00429048-n, ENG30-04831727-n, ENG30-00719231-v

`comparison_01_08.csv` lists every aligned row. `disagreements_01_08.csv` lists rows where the annotators differ by status, acceptability, or effective value.
