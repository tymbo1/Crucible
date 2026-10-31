# D rank-32 generalization forensics — supplemental pattern report

The 24 final rank-32 checkpoints were audited over all 256 states. The most
useful pattern is in the output bits: held-out accuracy for the least
significant three bits was effectively 100%, while bits 5–7 were substantially
less reliable:

| Output bit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Test bit accuracy | 100.00% | 100.00% | 100.00% | 99.74% | 99.67% | 94.34% | 90.23% | 93.03% |

The test carry strata were highly uneven:

| Carry-event count | States/checkpoint | Mean exact accuracy |
|---:|---:|---:|
| 0 | 1 | 0.00% |
| 1 | 1 | 12.50% |
| 2 | 1 | 4.17% |
| 3 | 12 | 62.50% |
| 4 | 9 | 99.54% |
| 5 | 15 | 98.61% |
| 6 | 17 | 95.59% |
| 7 | 7 | 88.10% |
| 8 | 1 | 95.83% |

The low-count buckets contain only one held-out input each, so they cannot
support a strong monotonic carry claim. The output-bit pattern and the common
error masks nevertheless point toward a high-bit/carry-propagation bottleneck
rather than uniform random failure.

Across all 1,024 one-input-bit edges, transition accuracy was lowest around
input bits 3–6 (approximately 86–88%). This is a full-domain local-sensitivity
measure, not a test-only score.

This remains diagnostic evidence. It does not establish that the frozen
workspace is the sole cause, and it does not qualify D for partner
generalization.
