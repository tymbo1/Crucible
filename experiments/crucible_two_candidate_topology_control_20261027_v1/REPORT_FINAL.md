# Crucible Relational Topology Control v1

Matched degree-3, 18-edge graph families on the explicit modular edge-offset composition task.

| Condition | Seed | Train | Gate | Validation value | Paired |
|---|---:|---:|---|---:|---:|
| P12_PRISM | 42 | 139/144 (0.965) | PASS | 59/96 (0.615) | 19/48 |
| P12_PRISM | 1337 | 135/144 (0.938) | FAIL | 56/96 (0.583) | 17/48 |
| R12_DIAMETER5 | 42 | 123/144 (0.854) | FAIL | 55/96 (0.573) | 14/48 |
| R12_DIAMETER5 | 1337 | 125/144 (0.868) | FAIL | 59/96 (0.615) | 22/48 |

## Interpretation boundary

Validation is interpreted only after the 95% training gate. Only `P12_PRISM` seed 42 passed (`139/144`, 96.5%); the other three arms failed the gate. Therefore the topology comparison is **not adjudicated**: the diameter-5 condition has no training-competent arm. The validation figures for failed arms are retained as descriptive diagnostics, not transfer evidence.

The graph controls and path-compositional oracle passed independent post-run checks, including fixed held-out identities, no split contamination, candidate-order grouping, path-sum verification, trace checkpoints and all recorded hashes. The gated `P12_PRISM` arm achieved `59/96` value-correct validation rows and `19/48` paired groups, but there is no matched gated `R12_DIAMETER5` result to compare against.

This run establishes a training-competence bottleneck on the new edge-offset task under the frozen recipe. It does not support or refute a graph-diameter effect, multi-hop neural reasoning, or a general architectural conclusion.

## Artifacts

- [Protocol](../../docs/CRUCIBLE_TWO_CANDIDATE_TOPOLOGY_CONTROL_PROTOCOL_20261027.md)
- [Novelty audit](../../docs/CRUCIBLE_TWO_CANDIDATE_TOPOLOGY_CONTROL_NOVELTY_AUDIT_20261027.json)
- [Preflight](PREFLIGHT.json)
- [Graph manifest](GRAPH_MANIFEST.json)
- [Summary](SUMMARY.json)
- [Post-run audit](POST_RUN_AUDIT.md)
- [Post-run audit data](POST_RUN_AUDIT.json)
