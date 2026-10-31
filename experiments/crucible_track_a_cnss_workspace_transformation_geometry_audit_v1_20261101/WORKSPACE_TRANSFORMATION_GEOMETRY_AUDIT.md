# CNSS workspace transformation geometry audit

Classification: `READ_ONLY_WORKSPACE_TRANSFORMATION_GEOMETRY_AUDIT`.

This compares canonical encoder-space transformations for skills A–D across all 256 states. It uses the exact recovered frozen workspace and performs no adapter or optimizer updates.

| Skill | Canonical exact | Mean displacement | Delta rank | Rank-32 delta energy | Pairwise distance correlation | Affine rank-32 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 1.0000 | 12.2681 | 64 | 1.0000 | 0.9451 | 0.1633 |
| B | 1.0000 | 12.7225 | 64 | 1.0000 | 0.8642 | 0.1265 |
| C | 1.0000 | 11.5393 | 64 | 1.0000 | 0.9728 | 0.1824 |
| D | 1.0000 | 13.8551 | 64 | 1.0000 | 0.2951 | 0.2126 |

These measurements are accessibility proxies, not a causal proof that any skill is harder for the registered adapter. D-specific evidence would require a matched training intervention after review.

Full machine-readable results: `WORKSPACE_TRANSFORMATION_GEOMETRY_AUDIT.json`.
