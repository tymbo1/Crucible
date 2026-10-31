# CNSS λ Sweep v1 — CPU result

Classification: **COMPLETED_EXPLORATORY_CPU_LAMBDA_SWEEP**

The verified frozen workspace and the original registered manifest were reused.
The original CNSS artifacts were not overwritten.

| λ | A test mean | B test mean | A→B mean | A→B minimum seed | B→A mean | displacement L2 | cosine | wall seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 1.0000 | 1.0000 | 0.8385 | 0.7656 | 0.7448 | 5.1604 | 0.9480 | 6.60 |
| 0.2 | 0.9948 | 1.0000 | 0.9740 | 0.9219 | 0.7708 | 4.3752 | 0.9624 | 6.56 |
| 0.3 | 1.0000 | 1.0000 | 0.9844 | 0.9688 | 0.8594 | 3.9181 | 0.9698 | 6.61 |
| 0.4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8698 | 3.8222 | 0.9706 | 6.64 |
| 0.5 | 1.0000 | 1.0000 | 0.9844 | 0.9688 | 0.9531 | 3.7452 | 0.9722 | 6.61 |
| 0.6 | 1.0000 | 1.0000 | 0.9792 | 0.9375 | 0.9635 | 3.0776 | 0.9808 | 6.62 |
| 0.7 | 0.9896 | 1.0000 | 0.9896 | 0.9688 | 0.9792 | 3.5594 | 0.9748 | 6.61 |
| 0.8 | 1.0000 | 1.0000 | 0.9635 | 0.9375 | 0.9896 | 3.3059 | 0.9784 | 6.62 |
| 0.9 | 0.9896 | 1.0000 | 0.9896 | 0.9688 | 0.9948 | 3.0478 | 0.9815 | 6.60 |
| 1.0 | 1.0000 | 1.0000 | 0.9844 | 0.9688 | 0.9531 | 2.9237 | 0.9829 | 6.62 |

Adapter-pair runs: `30`; seeds per λ: `3`; updates per skill: `2000`.

This sweep is exploratory and remains limited to the registered 8-bit mechanism probe.
Promising λ values require independent seeds, a larger held-out domain and longer-chain validation before selection.
