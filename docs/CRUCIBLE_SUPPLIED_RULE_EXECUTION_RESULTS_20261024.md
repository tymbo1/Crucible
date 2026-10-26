# Supplied-Rule Execution Gate — Results

Date: 2026-10-24  
Protocol: `CRUCIBLE_SUPPLIED_RULE_EXECUTION_PROTOCOL_20261024.md`  
Run directory: `experiments/crucible_supplied_rule_execution_20261024/`

## Outcome

The tiny execution control passed its fitting gate, but the full supplied-rule execution gate failed. The model can memorize explicit-rule training records; it has not demonstrated reusable execution on held-out rule instances.

The sealed test split was not evaluated.

## Final exact accuracy

| Arm | Seed | Train | Validation | Gate |
|---|---:|---:|---:|---|
| Tiny | 42 | 100.00% | 7.94% | Train-fit passed |
| Full | 42 | 91.28% | 15.87% | Failed |
| Tiny | 1337 | 100.00% | 7.94% | Train-fit passed |
| Full | 1337 | 92.66% | 17.46% | Failed |
| Full aggregate | — | — | **16.67%** | **Failed; threshold 90%** |

The validation set contains 63 records per seed. The full aggregate is 21 correct predictions out of 126. For a ten-answer digit task, this remains close to the nominal 10% chance level and is far below the preregistered 90% gate.

## Integrity and execution controls

- Two fixed seeds: 42 and 1337.
- Exactly 2,000 updates per arm and seed.
- Explicit rule formulas and parameters were supplied at decision time.
- Training and validation rule instances were disjoint.
- The test partition remained sealed because validation did not pass.
- GPU use was disabled.
- Initial and final checkpoints and per-record manifests were written.
- `git diff --check` passed.

## Interpretation

This result does not test rule induction, temporary adaptation, recursive reasoning or neuromorphic learning. It isolates a prerequisite: whether the current dense model can execute a supplied mathematical rule on new instances.

The observed pattern—near-perfect training fit with poor validation transfer—supports memorization or representation/optimization failure rather than a demonstrated reusable execution procedure. It is therefore premature to add meta-learning loops, local-plasticity rules or hardware-specific mechanisms and interpret their failure as a higher-level learning failure.

## Decision

Keep C2 and hardware experiments blocked. The next software gate should use a deliberately simpler supplied-rule interface and determine whether the failure is caused by representation, arithmetic supervision, feature coverage or optimization. Only after a reliable software adaptation mechanism passes held-out transfer should we add a local-plasticity/eligibility-trace comparison.

The ProjectBrain/neuromorphic direction is retained as a later substrate track:

1. Establish supplied-rule execution and adaptation in software.
2. Compare gradient-based temporary adaptation with a small local-plasticity analogue.
3. Add persistent associative memory and provenance/consolidation safeguards.
4. Only then investigate neuromorphic or analog implementation constraints.

This keeps the ProjectBrain memory principles—associative activation, persistence and continuity—available without treating a hardware substrate as evidence that the learning objective has been solved.
