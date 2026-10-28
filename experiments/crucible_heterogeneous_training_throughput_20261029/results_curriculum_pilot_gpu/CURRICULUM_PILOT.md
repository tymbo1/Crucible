# Verified-procedure curriculum pilot

Classification: `exploratory_curriculum_learning_pilot`

This exploratory pilot measures whether verified procedure records can enter later GPU batches. It is not Track B and does not reopen the registered benchmark.

| Arm | Functional target success | Parsed | Teacher-forced token accuracy | Mean loss | Verified CPU procedures | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| gpu_no_curriculum | 0/16 | 0/16 | 0.016 | 4.0711 | 0 | 0.954 |
| cpu_work_no_curriculum | 0/16 | 0/16 | 0.016 | 4.0711 | 522 | 8.092 |
| verified_curriculum | 0/16 | 0/16 | 0.581 | 3.1268 | 522 | 8.049 |

Producer result digest matched between CPU-work control and curriculum arm: `True`.

Interpretation: any curriculum-arm improvement is exploratory evidence that verified records affected this small model's training stream. It is not evidence of cumulative learning or general neural procedure induction.

## Boundary

CPU workers received demonstrations and independent verification examples only. The exploratory target ASTs were not sent to workers and were evaluated only after training.
