# Crucible OP foundation pilot

Classification: `exploratory_op_external_scaffold_foundation_pilot`

The model is the existing dense Crucible family. OP is evaluated as the exact bounded external loop from the repository specification; it is not internalized in weights.

| Policy | Accuracy | Mean passes | Mean model tokens |
|---|---:|---:|---:|
| one_pass | 0.0000 (0/64) | 1.00 | 26.5 |
| fixed_depth_2 | 0.0000 (0/64) | 2.00 | 93.3 |
| op_external | 0.0000 (0/64) | 2.00 | 188.9 |

Training seconds: `2.102`; final validation loss: `2.888574655804408`.

The controller's information-gain, self-quality and future-competence terms remain unavailable or disabled because this pilot has no calibrated transition model or later learning intervention.
