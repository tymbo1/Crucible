# Crucible Balanced Holdouts — Final Analysis

This is read-only post-processing of the completed 16-arm CPU run. No model was retrained and no sealed test corpus was accessed during finalization.

Fit gate: 15/16 arms passed the preregistered 95% training exact-accuracy threshold. The failed arm is `balanced_c_factorized_seed42`; its rows remain in the raw matrix but are excluded from fit-gated transfer claims.

## Main result

All validation observations: diagonal/equal-operand pairs 0/32 (0.00%); non-diagonal pairs 133/208 (63.94%). Fit-gated validation: diagonal 0/30 (0.00%); non-diagonal 127/195 (65.13%).

Every held-out diagonal observation was wrong (0/32), while trained diagonal observations were 47/48 correct. The fit-gated training diagonal result is 45/45. This supports a held-out diagonal transfer problem under these splits, not an inability to fit equal-operand examples.

## Four original failures under rotated holdouts

| Pair | Training correct | Validation correct | Interpretation |
|---|---:|---:|---|
| `(1,5)` | 8/8 | 3/8 | moves with split; partly recovers when held out |
| `(2,2)` | 8/8 | 0/8 | wrong whenever held out; correct when trained |
| `(3,3)` | 7/8 | 0/8 | wrong whenever held out; usually correct when trained |
| `(5,1)` | 7/8 | 4/8 | moves with split; partly recovers when held out |

Validation prediction distributions for those four pairs were: `(1,5)` → `5:1, 6:3, 7:4`; `(2,2)` → `2:4, 5:2, 6:2`; `(3,3)` → `5:4, 7:3, 8:1`; and `(5,1)` → `5:1, 6:4, 7:3`. Only the `6` predictions are correct for the two sum-six pairs; `(2,2)` and `(3,3)` had no correct validation output. The complete distributions, including training rows, are in the machine-readable analysis.

The original fixed-split four-pair pattern therefore does not survive as a fixed-pair phenomenon. The stable result is the diagonal/equal-operand held-out failure.

## Other requested diagnostics

- **Output format and positional errors:** no output-format errors occurred. EOS was correct on 236/240 validation rows and 640/640 training rows. Full-sequence accuracy is recorded separately and matches output-token accuracy here because the few EOS errors coincide with wrong output tokens.
- **Operand order:** per-arm commutative counterpart entries retain forward/reverse predictions and train/validation membership. Because the rotated partitions do not make every counterpart a matched validation pair, this study does not establish a broad order-sensitivity mechanism.
- **Sum magnitude, operand identity and difference:** rates with denominators and Wilson intervals are in [ANALYSIS_FINAL.json](ANALYSIS_FINAL.json). They are descriptive; sum, diagonal status and holdout membership are not independently balanced enough here to support a causal sum- or operand-specific claim beyond the diagonal result.
- **Split/evaluator integrity:** the final verification checks row counts, 40/15 membership, no within-arm pair overlap, labels, decoding relation, checkpoint identity, preflight status and the run's sealed-test/architecture flags.

## Conclusions

1. Errors move substantially with the train/validation partition; the original four failures are not intrinsic fixed-pair failures.
2. Equal-operand transfer is the clearest reproducible weakness: every held-out diagonal observation failed, while trained diagonal observations were mostly correct.
3. One arm failed the training-competence gate and is excluded from clean transfer claims.
4. No embedding collision, attention interference or architectural ceiling is established by these behavioral results.

## Single most informative next controlled experiment

Run a targeted diagonal-ablation study: compare equal-operand holdouts against matched non-equal holdouts with matched sums, operand frequencies, split sizes and fresh seeds, using the same evaluator. Inspect activations only if that behavioral contrast replicates. This directly tests whether the diagonal effect is a structural class-coverage issue rather than a consequence of the particular four partitions.

## Artifacts

- [Final analysis JSON](ANALYSIS_FINAL.json)
- [Predictions with fit status](PREDICTIONS_WITH_FIT.csv)
- [Per-arm JSON/confusion matrices](forensics/arm_outputs/)
- [Per-arm operand heatmaps and confusion SVGs](forensics/plots/)
- [Final artifact hashes](FINAL_HASHES.json)
- [Raw run report](REPORT.md)
- [Frozen partitions](PARTITIONS.json)
- [Preflight](PREFLIGHT.json)
