# Post-run audit: relational topology control v1

Audit status: **PASS**

- Independent hash entries verified: 34; mismatches: 0.
- Training gate: 1/4 arms passed at 0.95.
- Topology comparison adjudicated: **no**.
- Sealed test accessed: no.
- Prior checkpoints modified: no.

The graph controls are valid, but a topology comparison is not interpretable under the preregistered gate because the diameter-5 condition has no training-competent arm. The result is therefore a training-competence/optimization finding, not evidence for or against a graph-distance effect.

See `POST_RUN_AUDIT.json` for row-level, oracle, CSV, trace and hash checks.
