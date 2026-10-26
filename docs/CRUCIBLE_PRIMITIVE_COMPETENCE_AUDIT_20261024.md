# Crucible Primitive-Competence Audit

Date: 2026-10-24  
Status: completed  
Scope: CPU-only, read-only audit of the existing operation-position artifacts

## Executive finding

The task generator, oracle targets, split isolation, serialization and tokenizer alignment passed the audit. The current 3–4% held-out results therefore are not explained by an obvious target or leakage defect.

They are also not yet a clean test of primitive operation competence. The B-only arm uses a distinct prompt/interface, contains only 48 training records, and defines a feature as the complete tuple `(operation, supplied intermediate, operand)`. Every validation and held-out B-only tuple is therefore unseen in that full-feature space. The result measures transfer across new contexts under a sparse interface, not an isolated, well-powered test of whether the model learned operation B.

The appropriate next step is a controlled B-only positive-control gate with a matched interface, balanced coverage and a deliberately specified interpolation/generalization split. No architecture change, GPU run or recursive-reasoning evaluation is justified by this audit alone.

## Integrity checks

- Independent oracle errors: `0` across all six audited arms.
- Serialization/target-alignment errors: `0`.
- Train/validation/held-out overlaps: `0` for feature keys, inputs and record IDs in every arm.
- Composition target serialization: fixed three-byte target for all records; prefix length fixed at 198 tokens in the audited tokenizer path.
- B-only target serialization: fixed one-byte target; prefix length fixed at 160 tokens.
- Original benchmark modified: no.
- Training or new model evaluation performed by this audit: no.
- Audit artifact hashes: verified against `HASHES.json`.

The serialization audit validates target construction and tokenizer alignment. It does not independently rerun model-output parsing or retrain any checkpoint.

## Existing position-gate results

Free-running exact accuracy from the existing final checkpoints:

| Arm | Seed 42 validation | Seed 42 held-out | Seed 1337 validation | Seed 1337 held-out |
|---|---:|---:|---:|---:|
| A → B | 3.57% | 1.56% | 5.36% | 4.69% |
| B only | 12.50% | 0.00% | 4.17% | 8.33% |
| B → A | 8.93% | 6.25% | 8.93% | 0.00% |

The composition training sets were substantially fitted but not perfectly: A → B reached 86.6–97.3% training exact accuracy and B → A reached 94.6–95.5%. Both B-only seeds reached 100% training accuracy. This is consistent with memorization and weak transfer, but not by itself a proof of a capacity boundary.

For reference, the nominal uniform baselines are 10% for a one-digit B-only answer and 1% for an exact two-digit composition sequence. Majority-class baselines vary by split; on the held-out sets they range from 14.1–17.2% for composition final digits, 6.25–7.81% for composition sequences, and 20.8% for the B-only final digit. Small held-out sizes—64 composition examples and 24 B-only examples—make small percentage differences unstable.

## Distribution and interface findings

The composition protocol has 112/56/64 train/validation/held-out records. Seven operation pairs are trained with 16 examples each. The held-out pairs are `add+sub` and `mul+sub`, 32 examples each, so the held-out composition test includes unseen operation-pair combinations.

The B-only protocol has 48/24/24 records. Each of `add`, `mul` and `sub` appears 16 times in training and 8 times in validation and held-out. Thus operation identities are seen during training, but every complete B-only feature tuple is new in validation and held-out. The audit correctly reports a 100% full-feature-unseen rate for every validation and held-out split. That is clean split isolation, but it does not distinguish:

1. learning the operation rule;
2. interpolating to a new supplied-intermediate/operand combination; and
3. coping with the new prompt/interface.

The new position-gate prompt begins with `Task: position-gate modulo-ten arithmetic.` The earlier intermediate-supervision experiment used `Task: arithmetic is modulo ten. Apply both operations left-to-right.` The earlier experiment trained only `add+add` and `add+mul` with 128 training and 64 validation records. Consequently, its approximately 31–39% free-running results are not directly comparable to the new approximately 0–8% held-out results. The task family, prompt, operation-pair coverage and split construction changed.

The earlier frozen traces remain useful as historical evidence: they reported 76.56–87.50% first-step accuracy, 31.25–39.06% free-running end-to-end accuracy and approximately 40.8–44.6% final accuracy conditional on a correct intermediate. They do not resolve the new interface/distribution discrepancy.

## Scientific conclusion

The audit supports the following narrow conclusion:

> Under the new position-gate interface and split design, the tested checkpoints fit the training records but show weak held-out transfer. The audit found no oracle, serialization or leakage defect that explains this. However, the B-only arm is too sparse and interface-disparate to establish that the underlying primitive operation itself is unlearned.

It does not support conclusions that the Crucible architecture cannot compose, that operation position is the primary bottleneck, or that a larger model is required.

## Next gate: matched primitive competence

Before another composition or architecture experiment, freeze a new CPU protocol with:

- the same prompt and answer interface used by the composition task;
- a B-only control that supplies the intermediate state but asks only for the second operation;
- balanced coverage over operation, supplied intermediate and operand;
- a tiny-set overfit positive control;
- a split that tests interpolation over seen operation rules separately from extrapolation to held-out combinations or ranges;
- train, validation and untouched diagnostic test sets generated before training;
- per-operation accuracy, final-digit histograms, target-loss curves and exact confidence intervals;
- explicit comparison against the 10% uniform final-digit baseline and majority baseline.

The gate should first require reliable primitive transfer in the matched B-only condition. Only then should A → B versus B → A be used to attribute any remaining deficit to composition or operation position.

## Preserved artifacts

- Audit script: `tools/audit_crucible_primitive_competence_20261024.py`
- Audit JSON: `experiments/crucible_primitive_competence_audit_20261024/AUDIT.json`
- Run configuration: `experiments/crucible_primitive_competence_audit_20261024/RUN_CONFIG.json`
- Hash manifest: `experiments/crucible_primitive_competence_audit_20261024/HASHES.json`

