# Crucible Topology-Control Training-Duration Extension v1

The exact four topology-control arms were retrained from the same initialization and data schedule for 2,000 total updates.

| Condition | Seed | 500-step train | 2,000-step train | Gate | 2,000-step validation | Paired | First recorded gate |
|---|---:|---:|---:|---|---:|---:|---:|
| P12_PRISM | 42 | 139/144 (0.965) | 142/144 (0.986) | PASS | 72/96 (0.750) | 29/48 | 500 |
| P12_PRISM | 1337 | 135/144 (0.938) | 143/144 (0.993) | PASS | 59/96 (0.615) | 23/48 | 750 |
| R12_DIAMETER5 | 42 | 123/144 (0.854) | 141/144 (0.979) | PASS | 62/96 (0.646) | 18/48 | 750 |
| R12_DIAMETER5 | 1337 | 125/144 (0.868) | 143/144 (0.993) | PASS | 51/96 (0.531) | 16/48 | 750 |

## Interpretation

Validation is interpreted only for arms that pass the final 2,000-update training gate. The extension diagnoses acquisition speed versus plateau; it is not a topology-transfer comparison. Any later topology comparison must use a matched duration after both conditions demonstrate competence.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_TWO_CANDIDATE_TOPOLOGY_DURATION_EXTENSION_PROTOCOL_20261027.md)
- [Novelty audit](../../docs/CRUCIBLE_TWO_CANDIDATE_TOPOLOGY_DURATION_EXTENSION_NOVELTY_AUDIT_20261027.json)
- [Source experiment](../../experiments/crucible_two_candidate_topology_control_20261027_v1/REPORT_FINAL.md)
- [Preflight](PREFLIGHT.json)
- [Summary](SUMMARY.json)
