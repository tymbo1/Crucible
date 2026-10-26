# Scientific Interpretation

This gate isolates the canonical two-candidate matching task from the prior
mixed 2/3-cardinality experiment. The training and validation rows, labels,
record IDs, candidate metadata and source hashes were frozen before training.

The primary conclusion is the training-competence gate. Validation results are
interpretable only for arms that pass that gate. Neither outcome establishes
variable-cardinality transfer or general authority reasoning.

Both cyclic arms passed training at 144/144, including 48/48 for each target
slot and abstention. Both shuffled arms failed training at 132/144 and
120/144. The result supports schedule sensitivity under this budget, but not a
specific optimization mechanism.

The two passing cyclic arms reached 70/96 and 81/96 on the existing
two-candidate validation rows. Abstention remained 32/32 in both, while paired
conditional correctness was 9/32 and 17/32. The isolated task therefore fixes
the mixed-cardinality training-competence confound without establishing robust
unseen-pair transfer.
