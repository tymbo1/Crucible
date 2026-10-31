# CNSS skill D acquisition diagnostic v1

## Purpose

Skill D in the partner-generalization pilot was modular addition by 0x37.
It reached at most 68.75% exact test accuracy, while A, B and C reached the
registered individual-skill threshold. The partner result therefore remains
INCONCLUSIVE_PREREQUISITE_FAILURE.

This diagnostic asks why D was not acquired. It is read-only: it uses the
completed checkpoints and the durable answer-bearing backup, performs no
optimizer update, launches no GPU run, and cannot change the original result.

## Checks

- Recompute (x + 0x37) mod 256 exhaustively for all 256 input states.
- Verify the 8-bit input/output convention, bijectivity, partitions and leakage
  boundaries.
- Compare A, B, C and D under the same workspace, adapter rank, four lambda
  values, three seeds and 2,000-update checkpoints.
- Evaluate identity, constant, target-bit-majority, no-carry-XOR and exact
  oracle baselines.
- Replay final and intermediate D checkpoints and measure training, validation
  and test exact accuracy.
- Stratify D errors by carry count, carry-chain length, modular wraparound,
  Hamming weight, output bit and bit-error count.
- Compute final loss and gradient norms without applying an update.

## Corrective control proposal

No corrective training was run. A separate protocol registers a matched
capacity-versus-training-horizon control: adapter ranks 8 and 32 crossed with
2,000 and 8,000 updates, retaining the original workspace, optimizer family,
partitions, seeds, lambda conditions and numerical/checkpoint safeguards.

The partner experiment must not be rerun until D reaches at least 95% exact
test accuracy on genuinely held-out inputs.
