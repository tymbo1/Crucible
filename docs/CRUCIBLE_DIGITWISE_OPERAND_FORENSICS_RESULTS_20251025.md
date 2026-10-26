# Crucible Digitwise Operand-Interaction Forensics

Date: 2026-10-25  
Status: completed diagnostic  
Hardware: CPU only; no training; no GPU

## Scope

The analysis loaded all eight available Stage-1 checkpoints from the digitwise addition ladder: two interfaces, two training arms and two seeds. Each checkpoint was evaluated read-only on all 55 valid ordered digit pairs `(a,b)` where `a+b<10`. The original 40/15 train/validation membership was retained. The sealed test partition was not evaluated.

The complete machine-readable rows are in the per-model JSON files in [the forensic output](../experiments/crucible_digitwise_operand_forensics_20251025/SUMMARY.json), with the frozen scope recorded in [the protocol](CRUCIBLE_DIGITWISE_OPERAND_FORENSICS_PROTOCOL_20261025.md).

## Checkpoint results

| Interface | Arm | Seed | Train | Validation | All 55 pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| Canonical | Direct | 42 | 40/40 | 11/15 | 51/55 |
| Canonical | Curriculum | 42 | 40/40 | 11/15 | 51/55 |
| Canonical | Direct | 1337 | 40/40 | 9/15 | 49/55 |
| Canonical | Curriculum | 1337 | 40/40 | 9/15 | 49/55 |
| Factorized | Direct | 42 | 38/40 | 9/15 | 47/55 |
| Factorized | Curriculum | 42 | 38/40 | 9/15 | 47/55 |
| Factorized | Direct | 1337 | 40/40 | 11/15 | 51/55 |
| Factorized | Curriculum | 1337 | 40/40 | 11/15 | 51/55 |

The central pattern remains high training fit with incomplete transfer. The direct and curriculum checkpoints produced identical predictions within each interface/seed pair: curriculum ordering changed no measured prediction in this 500-update setting.

## Complete prediction matrices

These four matrices are the unique behaviours; direct and curriculum are identical for each interface/seed. `✓` is a correct prediction, a digit is an incorrect predicted digit, and `—` is an invalid pair because `a+b≥10`.

### Canonical, seed 42

```text
a\\b  0 1 2 3 4 5 6 7 8 9
 0    ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
 1    ✓ ✓ ✓ ✓ ✓ 7 ✓ ✓ ✓ —
 2    ✓ ✓ 7 ✓ ✓ ✓ ✓ ✓ — —
 3    ✓ ✓ ✓ 5 ✓ ✓ ✓ — — —
 4    ✓ ✓ ✓ ✓ ✓ ✓ — — — —
 5    ✓ 4 ✓ ✓ ✓ — — — — —
 6    ✓ ✓ ✓ ✓ — — — — — —
 7    ✓ ✓ ✓ — — — — — — —
 8    ✓ ✓ — — — — — — — —
 9    ✓ — — — — — — — — —
```

### Canonical, seed 1337

```text
a\\b  0 1 2 3 4 5 6 7 8 9
 0    ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
 1    ✓ ✓ ✓ ✓ ✓ 8 ✓ ✓ ✓ —
 2    ✓ ✓ 6 ✓ ✓ ✓ ✓ ✓ — —
 3    ✓ ✓ ✓ 7 ✓ ✓ ✓ — — —
 4    ✓ 2 ✓ ✓ ✓ ✓ — — — —
 5    ✓ 8 ✓ ✓ ✓ — — — — —
 6    ✓ ✓ ✓ ✓ — — — — — —
 7    ✓ ✓ ✓ — — — — — — —
 8    ✓ 8 — — — — — — — —
 9    ✓ — — — — — — — — —
```

### Factorized, seed 42

```text
a\\b  0 1 2 3 4 5 6 7 8 9
 0    ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
 1    ✓ ✓ ✓ 2 7 2 ✓ ✓ ✓ —
 2    ✓ ✓ 5 ✓ ✓ ✓ ✓ ✓ — —
 3    ✓ 2 ✓ 5 ✓ ✓ ✓ — — —
 4    ✓ 7 ✓ ✓ ✓ ✓ — — — —
 5    ✓ 2 ✓ ✓ ✓ — — — — —
 6    ✓ ✓ ✓ ✓ — — — — — —
 7    ✓ ✓ ✓ — — — — — — —
 8    ✓ ✓ — — — — — — — —
 9    ✓ — — — — — — — — —
```

### Factorized, seed 1337

```text
a\\b  0 1 2 3 4 5 6 7 8 9
 0    ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
 1    ✓ ✓ ✓ ✓ ✓ 8 ✓ ✓ ✓ —
 2    ✓ ✓ 5 ✓ ✓ ✓ ✓ ✓ — —
 3    ✓ ✓ ✓ 5 ✓ ✓ ✓ — — —
 4    ✓ ✓ ✓ ✓ ✓ ✓ — — — —
 5    ✓ 8 ✓ ✓ ✓ — — — — —
 6    ✓ ✓ ✓ ✓ — — — — — —
 7    ✓ ✓ ✓ — — — — — — —
 8    ✓ ✓ — — — — — — — —
 9    ✓ — — — — — — — — —
```

## Error structure

Four validation pairs failed in every one of the eight checkpoints:

| Pair | Target | Result |
| --- | ---: | --- |
| `(1,5)` | 6 | Incorrect in 8/8 checkpoints |
| `(2,2)` | 4 | Incorrect in 8/8 checkpoints |
| `(3,3)` | 6 | Incorrect in 8/8 checkpoints |
| `(5,1)` | 6 | Incorrect in 8/8 checkpoints |

The diagonal cases `(2,2)` and `(3,3)` account for all errors among the five equal-operand pairs except none of `(0,0)`, `(1,1)` and `(4,4)`. Equal-operand accuracy is therefore 3/5 (60%) in every model.

The universal failures are concentrated at sums 4 and 6, but this is not a simple large-sum effect. Every checkpoint scored 4/7 (57.14%) on the sum-6 pairs; the remaining errors differ by interface and seed. Factorized seed 42 also made two training-set errors, `(1,3)` and `(1,4)`, and their reversed pairs were also wrong.

## Commutativity and seed consistency

- Canonical seed 42: 24/25 unordered pairs correct in both directions; forward and reverse accuracy both 24/25.
- Canonical seed 1337: 22/25 both directions, with two forward-only pairs; forward 24/25 versus reverse 22/25. Directional asymmetry was 2/25 (8%).
- Factorized seed 42: 22/25 both directions and 3/25 neither; forward and reverse both 22/25.
- Factorized seed 1337: 24/25 both directions; forward and reverse both 24/25.

Thus there is no broad order bias across the checkpoints. One canonical seed shows a small directional asymmetry, while the most persistent failures are pair-specific and occur in both operand orders where both are present as hard validation cases.

Within each fixed interface, cross-seed prediction agreement was 49/55 (89.09%). Correctness agreement was 53/55 (96.36%) for canonical and 51/55 (92.73%) for factorized. The models often disagree on the wrong digit while agreeing that a hard pair is incorrect.

## Interpretation

This is not evidence that the model cannot combine operands at all: most pairs transfer, and the canonical checkpoints reach 60–73.3% validation accuracy as reported by the ladder. It does show that the remaining gap is structured rather than random.

The strongest immediate hypotheses are:

1. a diagonal/equal-operand interaction weakness;
2. a small set of pair-specific representations or decision boundaries that are not learned reliably;
3. seed/interface-dependent extrapolation for a few validation pairs, rather than a general left-to-right arithmetic bias.

The next intervention should therefore target the identified pair structure—especially equal operands and the sum-6 cases—with a preregistered control. It should not yet be described as a carry or multi-step composition problem.

## Artifact links

- [Forensic summary JSON](../experiments/crucible_digitwise_operand_forensics_20261025/SUMMARY.json)
- [Canonical direct seed 42](../experiments/crucible_digitwise_operand_forensics_20261025/canonical_direct_seed42.json)
- [Canonical direct seed 1337](../experiments/crucible_digitwise_operand_forensics_20261025/canonical_direct_seed1337.json)
- [Factorized direct seed 42](../experiments/crucible_digitwise_operand_forensics_20261025/factorized_direct_seed42.json)
- [Factorized direct seed 1337](../experiments/crucible_digitwise_operand_forensics_20261025/factorized_direct_seed1337.json)
