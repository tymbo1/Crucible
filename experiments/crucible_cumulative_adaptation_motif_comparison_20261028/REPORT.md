# Exploratory executable-operator comparison

This CPU-only run uses a fresh manifest and is not a registered scientific result.
Target answers were opened only after all arms finalized proposals; no GPU was allocated.

## Manifest

- Source procedures: 32
- Operator-validation tasks: 32 (16 per eligible root)
- Target tasks: 32 (16 per seed)
- Applicability was assigned by root signature before validation; no correctness filter was used.
- Source, validation and target structural/function hashes are disjoint; target answers are separate.

## Promotion and acquisition

- add_mod10 validation: 9/16
- sub_mod10 validation: 11/16
- Promoted operators: none; the 16/16 threshold was unchanged.
- Newly constructed adaptive candidate ASTs: none, because A3 remained unactivated.

| Arm | Proposal accepted | End-to-end acquisition after reset |
| --- | ---: | ---: |
| A0_FROZEN_INDUCTION | 4/32 | 1/32 |
| A2_SHAM_OPERATOR_LIBRARY | 4/32 | 1/32 |
| A3_VERIFIED_ADAPTIVE_LIBRARY | 4/32 | 1/32 |
| A3_NO_GUIDANCE_SAME_MEMORY | 4/32 | 1/32 |

The A3 and no-guidance ablation share identical operator memory, but the mechanism never activated, so their paired contrast is not evidence for or against adaptive guidance.
The measured A3-vs-A0, A3-vs-A2 and A3-vs-no-guidance task-level differences are all zero.
