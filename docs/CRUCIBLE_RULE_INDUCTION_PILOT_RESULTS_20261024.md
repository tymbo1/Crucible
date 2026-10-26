# Crucible Rule-Induction CPU Pilot Results

Date: 2026-10-24  
Protocol: `CRUCIBLE_RULE_INDUCTION_PILOT_PROTOCOL_20261024.md`  
Benchmark: compact, audited v3  
Status: completed; validation gates failed; test remained sealed

## Final validation results

| Seed | Arm | Train exact | Valid 1 demo | Valid 2 demos | Valid 4 demos | Valid 8 demos | Identifiable 8-demo | Shuffled 8-demo |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | Query-only | 49.54% | — | — | — | — | 15.00% baseline | — |
| 42 | Mixed evidence | 91.28% | 14.29% | 14.29% | 11.11% | 12.70% | 11.67% | 9.52% |
| 1337 | Query-only | 48.62% | — | — | — | — | 11.67% baseline | — |
| 1337 | Mixed evidence | 93.58% | 14.29% | 9.52% | 9.52% | 1.59% | 0.00% | 6.35% |

Aggregate mixed-model validation accuracy fell from 14.29% at one demonstration to 7.14% at eight. The aggregate identifiable-task accuracy at eight demonstrations was 5.83%, below the query-only baseline’s 11.67% on the same identifiable-task subsets. Shuffled eight-demonstration accuracy was 7.94%, slightly higher than valid eight-demonstration accuracy of 7.14%.

## Gate outcomes

| Gate | Required | Measured | Result |
|---|---:|---:|---|
| Evidence response | 8-demo raw accuracy at least 5 pp above 1-demo | −7.14 pp | Failed |
| Demonstration advantage | 8-demo identifiable accuracy at least 5 pp above query-only | −5.83 pp | Failed |
| Evidence dependence | valid 8-demo accuracy at least 5 pp above shuffled | −0.79 pp | Failed |

The test partition was not loaded or evaluated because the validation gates failed.

## Interpretation

The dense control fit its mixed training records, reaching more than 91% training exact accuracy in both seeds, but its validation behavior did not show evidence-dependent rule induction. The model appears to be learning distributional output patterns or task-specific associations rather than reliably using additional demonstrations to constrain the hidden rule.

This is a negative result for the current prompt, model, optimizer, training schedule and finite rule library. It does not establish that rule induction is impossible for Crucible, and it does not justify GPU scaling or architecture changes yet.

The shuffled-demonstration control is particularly informative: replacing valid examples with unrelated examples did not materially hurt performance. That means the pilot did not demonstrate that the model was causally using the supplied evidence.

## Engineering note

The first pilot launch was stopped before the mixed arm because the verbose eight-demonstration prompt exceeded the 320-token context window. That partial attempt is preserved separately and is not included in these results. The accepted v3 benchmark uses compact prompts with maximum serialized length 130 tokens and passed the context audit before retraining.

## Decision

Do not open the test split. Keep C2, E4 and GPU scaling blocked.

The next diagnostic should isolate prompt use from rule difficulty, for example with a much smaller rule library and an explicit candidate-consistency control, before increasing model size or adding recursive mechanisms.

## Artifacts

- [Pilot protocol](CRUCIBLE_RULE_INDUCTION_PILOT_PROTOCOL_20261024.md)
- [Pilot summary](../experiments/crucible_rule_induction_pilot_20261024_v3/SUMMARY.json)
- [Pilot hash manifest](../experiments/crucible_rule_induction_pilot_20261024_v3/HASHES.json)
- [Accepted benchmark audit](../experiments/crucible_rule_induction_benchmark_20261024_v3/AUDIT.json)
- [Accepted benchmark report](CRUCIBLE_RULE_INDUCTION_BENCHMARK_RESULTS_20261024.md)
