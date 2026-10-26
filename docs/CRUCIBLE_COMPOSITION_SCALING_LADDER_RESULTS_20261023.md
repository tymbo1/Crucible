# Crucible composition scaling ladder results

Date: 2026-10-23  
Status: Corrected CPU-only ladder complete; held-out and test splits remain sealed.

## Scope

This experiment held the dense model, optimizer, objective and 500-update
budget fixed while increasing the composition training set from 8 to 128
examples. Each arm used a fresh 21,968,640-parameter dense active-compute
control and was run at seeds 42 and 1337.

The training examples were balanced between `add+add` and `add+mul`. A single
64-example validation set per seed was balanced the same way and constructed
to be disjoint from the complete 128-example training pool. Therefore it was
also disjoint from every smaller nested training arm. There was no exact input
overlap in any accepted arm.

Only training and validation were evaluated. Held-out and test splits were not
loaded for model evaluation or checkpoint selection. No E4 model, GPU, cloud
resource or checkpoint download was used.

## Final exact accuracy

| Training examples | Seed 42 train | Seed 42 validation | Seed 1337 train | Seed 1337 validation |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 100.00% | 9.38% | 100.00% | 10.94% |
| 16 | 100.00% | 14.06% | 100.00% | 4.69% |
| 32 | 100.00% | 4.69% | 96.88% | 9.38% |
| 64 | 93.75% | 7.81% | 87.50% | 12.50% |
| 128 | 37.50% | 14.06% | 36.72% | 14.06% |

The validation task has ten possible answer digits, so the low validation
values are broadly near a single-digit exact-match baseline. No arm approached
the learnability gate of at least 90% training and 70% validation exact
accuracy.

## Interpretation

The result separates two effects:

- At 8–32 examples, the dense control can fit the observed examples but does
  not transfer the two-operation rule to disjoint examples. This is
  memorization without demonstrated compositional generalization.
- At 64 examples, training fit becomes incomplete within 500 updates.
- At 128 examples, training fit falls to approximately 37% at both seeds,
  showing an additional optimization, interference or coverage boundary.

The 128-example failure is therefore not the only problem. Even the smallest
arms fail to demonstrate reusable two-step computation on disjoint validation
examples. The experiment does not identify whether the limiting factor is
representation, optimization, prompt structure or the absence of an
intermediate-state learning signal, but it rules out simple data leakage as the
explanation for the validation gap.

Prediction-digit histograms, target probabilities, target ranks, losses and
gradient norms were recorded at steps 0, 50, 100, 200, 300 and 500 for every
arm. Per-arm artifact hashes were independently rechecked: 70 recorded hashes,
zero mismatches.

## Decision

Keep C2 and E4-versus-dense cognitive comparison blocked. Do not increase the
neural model or rent GPU capacity to address this result.

The next CPU gate should be a controlled representation/objective experiment:

1. retain the disjoint validation protocol;
2. test a simple staged composition task or an intermediate-state diagnostic;
3. compare final-answer supervision with explicitly supervised intermediate
   state only as a learnability diagnostic; and
4. return to the original composition benchmark only after a dense control
   demonstrates both fitting and transfer.

The intermediate-state condition must not be presented as evidence of final
compositional reasoning unless its evaluation remains independent and its
additional supervision is reported explicitly.

## Frozen artifacts

- Corrected accepted ladder: `experiments/crucible_composition_scaling_ladder_20261023_v2/`
- Preliminary ladder with validation overlap, retained for audit history:
  `experiments/crucible_composition_scaling_ladder_20261023/`
- Runner: `tools/run_crucible_composition_scaling_ladder_20261023.py`
- Prior positive-control report:
  `docs/CRUCIBLE_COMPOSITION_INTERFACE_DIAGNOSTIC_RESULTS_20261022.md`

The preliminary run is not used for the scientific validation conclusion; the
corrected v2 run is the accepted result.
