# Crucible Rule-Induction Benchmark Audit

Date: 2026-10-24  
Accepted version: `crucible_rule_induction_benchmark_20261024_v3`  
Status: generated and audited; no model training performed

## Benchmark contents

The accepted benchmark contains 314 behaviorally unique hidden rules from four finite families. Version v2 had the same rule and split structure, but its verbose eight-demonstration prompts were not compatible with the dense control's 320-token context window. Version v3 uses compact prompts and is the version authorized for model evaluation.

| Family | Candidate rules |
|---|---:|
| affine2 | 150 |
| threshold-select | 144 |
| max-shift | 10 |
| absolute-difference-shift | 10 |

Each task contains a hidden rule, eight ordered demonstrations and a distinct query. The same task can be evaluated with 1, 2, 4 or 8 demonstrations. Rule instances are split, rather than individual examples, so a hidden rule identifier cannot cross train, validation or test.

Accepted split counts:

| Split | Tasks |
|---|---:|
| Train | 218 |
| Validation | 63 |
| Test | 33 |

Every family contributes at least two validation and two test rules. The earlier v1 benchmark is preserved but superseded because its smallest families had only one test rule each.

## Identifiability audit

The audit compares each demonstration prefix against the complete candidate-rule library. `Query identifiable` means every rule consistent with the demonstrations produces the same answer to the query; this is the appropriate upper-bound condition for answer accuracy. Exact rule uniqueness is stricter and is not required when multiple rules agree on the query.

| Split | 1 demo identifiable | 2 demos identifiable | 4 demos identifiable | 8 demos identifiable |
|---|---:|---:|---:|---:|
| Train | 0.0% | 11.0% | 80.3% | 94.5% |
| Validation | 0.0% | 17.5% | 84.1% | 95.2% |
| Test | 0.0% | 18.2% | 78.8% | 90.9% |

One demonstration is intentionally underdetermined in the entire benchmark. A model should not be judged as though every one-demo task has a uniquely recoverable answer. Four and eight demonstrations provide the meaningful rule-induction conditions, with identifiability reported alongside accuracy.

At eight demonstrations, the v3 test set has 1.61 consistent candidates on average, and 90.9% of queries are identifiable. Exact rule uniqueness is lower, so the benchmark should score query answers rather than require recovery of a particular symbolic description. Maximum serialized context length is 130 tokens, below the dense control's 320-token limit.

## Integrity results

- Independent oracle errors: `0`.
- Duplicate demonstration or query-pair errors: `0`.
- Train/validation rule-instance overlap: `0`.
- Train/test rule-instance overlap: `0`.
- Validation/test rule-instance overlap: `0`.
- Artifact hash failures: `0` across seven generated files.
- Model training: not performed.
- GPU/cloud use: none.

## Scientific use

This benchmark can now test whether a model improves as demonstrations increase, while separating solvable from genuinely ambiguous cases. The first model experiment should report:

- exact query accuracy at 1/2/4/8 demonstrations;
- accuracy conditional on query identifiability;
- rule-family and split breakdowns;
- exact-rule identification separately from answer correctness;
- failures where additional demonstrations do not improve the answer.

The benchmark does not yet demonstrate rule induction by Crucible. It establishes a controlled, leakage-audited environment in which that question can be tested without confusing ambiguity with model failure.

## Preserved artifacts

- [Protocol](CRUCIBLE_RULE_INDUCTION_PROTOCOL_20261024.md)
- [Benchmark builder](../tools/build_crucible_rule_induction_benchmark_20261024.py)
- [Accepted v3 audit](../experiments/crucible_rule_induction_benchmark_20261024_v3/AUDIT.json)
- [Accepted v3 summary](../experiments/crucible_rule_induction_benchmark_20261024_v3/SUMMARY.json)
- [Accepted v3 rule library](../experiments/crucible_rule_induction_benchmark_20261024_v3/RULE_LIBRARY.json)
- [v3 hash manifest](../experiments/crucible_rule_induction_benchmark_20261024_v3/HASHES.json)
- [Superseded v2 audit](../experiments/crucible_rule_induction_benchmark_20261024_v2/AUDIT.json)
- [Superseded v1 audit](../experiments/crucible_rule_induction_benchmark_20261024/AUDIT.json)
