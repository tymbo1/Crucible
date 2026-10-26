# Scientific Interpretation

This is a checkpoint-only diagnostic of the accepted two-candidate matcher. It
crosses familiar versus recombined candidate pairs with compact versus padded
serialization. The original training and validation predictions were
reproduced exactly before derived conditions were interpreted.

The diagnostic is designed to separate two explanations that were confounded
in the original gate: candidate-pair recombination and layout shift. It does
not retrain the model, alter any checkpoint or access sealed test data.

## Supported findings

- Familiar candidate sets were perfect for both seeds in both compact and
  padded serialization: 144/144 rows and 48/48 paired groups.
- Recombined candidate sets remained impaired in compact serialization: seed
  42 reached 75/96 rows and 11/32 paired groups; seed 1337 reached 80/96 rows
  and 16/32 paired groups.
- Padding was not the primary explanation for the gap in this direct
  reserialization control: compact recombined accuracy was close to padded
  recombined accuracy, while familiar accuracy was unchanged.

## Remaining confound and follow-up

The original source populations differ in operand context as well as candidate
set. Training-source rows use six contexts—(1,6), (1,8), (2,7), (2,9), (3,8),
and (4,9)—while validation-source rows use four different contexts—(0,5),
(1,7), (2,8), and (3,9). Therefore this 2x2 audit alone cannot attribute the
full gap to candidate-set recombination. The matched-context probe evaluates
both regimes on each source-context stratum and is the appropriate follow-up.

These results are behavioral and descriptive. They do not establish a
mechanism, an architectural ceiling or variable-cardinality generalization.
The probe did not retrain, alter checkpoints or access sealed test data.
