# Human evaluation comparison 05 vs 08

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
| Both axes have same effective value | 19 / 75 | 25.3% |
| Both axes agree on acceptable/not acceptable | 23 / 75 | 30.7% |

## Status agreement

| Scope | Exact agreement | Cohen kappa | Weighted kappa |
|---|---:|---:|---:|
| All axes | 28.0% | 0.0203 | 0.3443 |
| Positive | 12.0% | 0.0072 | 0.5816 |
| Negative | 44.0% | -0.0264 | -0.0224 |

## Acceptable binary agreement

| Scope | Exact agreement | Cohen kappa |
|---|---:|---:|
| All axes | 60.7% | 0.2356 |
| Positive | 64.0% | 0.359 |
| Negative | 57.3% | -0.1707 |

## Effective value agreement

| Axis | Exact agreement | Within one step | Mean distance | Weighted kappa |
|---|---:|---:|---:|---:|
| Positive | 46.7% | 62.7% | 0.2533 | 0.3509 |
| Negative | 72.0% | 88.0% | 0.0938 | 0.7517 |

## Where they differ

Rows with complete agreement on status, binary acceptability, and effective values: 0 / 75.
Rows with any disagreement: 75 / 75.

| Difference type | Count |
|---|---:|
| Positive status | 66 |
| Negative status | 42 |
| Positive acceptable/not acceptable | 27 |
| Negative acceptable/not acceptable | 32 |
| Positive effective value | 40 |
| Negative effective value | 21 |

Binary acceptability disagreements are the most important practical conflicts:

- Positive: ENG30-01569181-v, ENG30-01989562-v, ENG30-01099592-v, ENG30-00953559-n, ENG30-03600806-n, ENG30-01220336-n, ENG30-01468576-v, ENG30-02585489-v, ENG30-06720371-n, ENG30-00200863-v, ENG30-00824767-v, ENG30-07157273-n, ENG30-01623268-a, ENG30-00421002-a, ENG30-01234345-n, ENG30-00058743-n, ENG30-07436475-n, ENG30-05207130-n, ENG30-02505358-v, ENG30-00613683-v, ENG30-02574205-v, BILI-00000941, ENG30-14440623-n, ENG30-02226429-n, ENG30-10000158-n, ENG30-09398076-n, ENG30-02390258-n
- Negative: ENG30-01989562-v, ENG30-00953559-n, ENG30-01468576-v, ENG30-00824767-v, ENG30-07436475-n, ENG30-05207130-n, ENG30-01215137-v, ENG30-01193721-v, ENG30-09398076-n, ENG30-10557854-n, ENG30-03016953-n, ENG30-07592094-n, ENG30-00654885-n, ENG30-07453638-n, ENG30-14437976-n, ENG30-04840405-n, ENG30-01375831-a, ENG30-13254805-n, ENG30-02490877-v, ENG30-00183090-b, ENG30-01983162-a, ENG30-14438898-n, ENG30-04829282-n, ENG30-10407310-n, ENG30-04191595-n, ENG30-01041349-n, ENG30-01425892-v, ENG30-00169955-a, ENG30-03135152-n, ENG30-00429048-n, ENG30-04831727-n, ENG30-00719231-v

`comparison_05_08.csv` lists every aligned row. `disagreements_05_08.csv` lists rows where the annotators differ by status, acceptability, or effective value.
