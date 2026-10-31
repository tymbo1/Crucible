# CNSS D rank-32 generalization forensics — result

Classification: COMPLETED_READ_ONLY_GENERALIZATION_FORENSICS.

Audited 132 rank-32 checkpoints (24 final checkpoints) across all 256 input states. No parameters were updated and no GPU was used.

| Updates | Lambda | Mean full-domain | Mean test | Min test | Mean local transition |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 0.1 | 0.9323 | 0.8594 | 0.8594 | 0.8838 |
| 2000 | 0.4 | 0.9362 | 0.8594 | 0.8438 | 0.8916 |
| 2000 | 0.9 | 0.9336 | 0.8646 | 0.8594 | 0.8870 |
| 2000 | 1.0 | 0.9336 | 0.8542 | 0.8438 | 0.8896 |
| 8000 | 0.1 | 0.9336 | 0.8646 | 0.8438 | 0.8926 |
| 8000 | 0.4 | 0.9245 | 0.8490 | 0.8438 | 0.8750 |
| 8000 | 0.9 | 0.9284 | 0.8438 | 0.8125 | 0.8825 |
| 8000 | 1.0 | 0.9349 | 0.8542 | 0.8438 | 0.8936 |

Carry-event stratification is included in `D_GENERALIZATION_FORENSICS_RESULT.json`; detailed per-input traces are retained only in the durable sealed artifact.

The aggregate bit, local-transition and error-mask summary is in
`D_GENERALIZATION_FORENSICS_SUPPLEMENT.json` and `.md`.

The partner-generalization gate remains closed. This audit does not replace the registered 95% individual-competence requirement.
