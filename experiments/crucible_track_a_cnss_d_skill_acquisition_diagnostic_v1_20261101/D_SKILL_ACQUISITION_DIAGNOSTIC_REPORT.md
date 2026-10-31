# CNSS D-skill acquisition diagnostic

Classification: READ_ONLY_DIAGNOSTIC_COMPLETED_PARTNER_GATE_REMAINS_INCONCLUSIVE

No optimizer update or GPU operation was performed.

| Skill | Mean final train accuracy | Maximum final train accuracy | Maximum final test accuracy |
| --- | ---: | ---: | ---: |
| A | 1.0000 | 1.0000 | 1.0000 |
| B | 1.0000 | 1.0000 | 1.0000 |
| C | 1.0000 | 1.0000 | 1.0000 |
| D | 0.6732 | 0.8750 | 0.6875 |

Oracle/encoding audit passed: True.
D maximum train accuracy: 0.8750; D maximum test accuracy: 0.6875.

The decisive finding is that D is not training-competent under the existing rank-8 adapter and 2,000-update budget. The partner experiment therefore remains inconclusive; no sender, receiver or pair-specific compatibility conclusion is authorized.

The proposed rank-versus-update corrective control is registered separately and was not run.

Machine-readable report: D_SKILL_ACQUISITION_DIAGNOSTIC_REPORT.json.
