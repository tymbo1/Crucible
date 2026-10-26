# Crucible braid log — operation-position gate

Date: 2026-10-24
Decision point: primitive competence must be isolated before architectural conclusions

## Provenance

This is a structured research log for the 2026-10-24 Crucible exchange. It is not a verbatim export of the chat platform transcript. No matching raw `braid_exchange` export was found in the local workspace.

## Research context

The preceding frozen-checkpoint diagnostic found:

| Measure | Seed 42 | Seed 1337 |
|---|---:|---:|
| Free-running first-step accuracy | 76.56% | 87.50% |
| Conditional second-step accuracy | 35.94% | 40.63% |
| Free-running end-to-end accuracy | 31.25% | 39.06% |

The operation-position experiment then compared standard `A→B`, isolated `B-only`, and reversed `B→A` conditions across two seeds. Final held-out exact accuracy was:

| Condition | Mean held-out exact |
|---|---:|
| A→B | 3.13% |
| B-only | 4.17% |
| B→A | 3.13% |

All conditions fit training records strongly but generalized poorly.

## Decision

The result does not justify blaming composition, operation position, expert count, architecture size, CMS2, or symbolic critique. Because B-only also fails on held-out data, primitive operation competence has not been established.

The next gate is a primitive-competence isolation audit:

1. Audit predictions, targets, tokenization, answer extraction and random-chance baselines.
2. Identify differences between training and held-out records: operands, numerical ranges, templates, operation identity and position.
3. Establish a deliberately simple, controlled B-only interface.
4. Require primitive generalization before repeating operation-order experiments.

The 90% figure remains an interpretive competence marker, not a universal reasoning threshold.

## Experimental boundary

- Preserve all prior Crucible results and sealed benchmark splits.
- Do not modify the neural architecture.
- Do not add experts, CMS2 retrieval or symbolic critique in response to this result.
- Keep CPU-only evaluation until the task/interface audit is complete.
- Report uncertainty and sample sizes before comparing the 3.13% and 4.17% held-out values.

## Source artifacts

- [Operation-position protocol](CRUCIBLE_OPERATION_POSITION_PROTOCOL_20261024.md)
- [Operation-position results](CRUCIBLE_OPERATION_POSITION_RESULTS_20261024.md)
- [Operation-position summary](../experiments/crucible_operation_position_gate_20261024/SUMMARY.json)
- [Second-step error analysis](CRUCIBLE_SECOND_STEP_ERROR_ANALYSIS_20261024.md)

## Checkpoint status

This exchange marks a research gate: primitive competence is unresolved, and architectural escalation remains deferred.
