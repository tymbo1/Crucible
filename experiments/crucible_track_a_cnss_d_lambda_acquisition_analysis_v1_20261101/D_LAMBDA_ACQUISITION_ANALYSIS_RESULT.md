# CNSS Skill D lambda acquisition analysis — result

Classification: `COMPLETED_READ_ONLY_REANALYSIS`.

The existing rank-32 control already tested all four registered lambda values;
no duplicate training was run.

| Updates | Lambda | Train | Test | Displacement | Decoder margin |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 0.1 | 100.00% | 85.94% | 3.284 | 5.697 |
| 2,000 | 0.4 | 100.00% | 85.94% | 2.359 | 6.875 |
| 2,000 | 0.9 | 100.00% | **86.46%** | 2.173 | 7.042 |
| 2,000 | 1.0 | 100.00% | 85.42% | 2.017 | 7.546 |
| 8,000 | 0.1 | 100.00% | **86.46%** | 1.671 | 8.480 |
| 8,000 | 0.4 | 100.00% | 84.90% | 1.519 | 8.753 |
| 8,000 | 0.9 | 100.00% | 84.38% | 1.510 | 8.760 |
| 8,000 | 1.0 | 100.00% | 85.42% | 1.461 | 8.857 |

Alignment strength improves geometry, but not arithmetic generalization
monotonically. The reduced displacement does not by itself produce a better D
skill. No condition reaches the 95% gate, and partner generalization remains
locked.
