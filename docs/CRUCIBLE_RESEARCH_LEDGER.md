# Crucible Research Ledger

This ledger is the repository-backed scientific memory for Crucible. It records
what a specific tested hypothesis established, failed, or left unresolved. A
status applies only to the documented hypothesis under its stated conditions;
`FAILED` does not mean that the broader architectural idea is impossible.

Status values:

- `ESTABLISHED` — supported by the accepted artifact and its preregistered controls.
- `FAILED` — the stated hypothesis failed under the documented conditions.
- `INCONCLUSIVE` — informative result, but a confound prevents the narrower causal claim.
- `UNTESTED` — proposed, incomplete, or not yet supported by an accepted result.

The machine-readable source is [CRUCIBLE_RESEARCH_LEDGER.json](CRUCIBLE_RESEARCH_LEDGER.json).
Use `tools/search_crucible_research_ledger.py` before proposing an experiment;
search by mechanism, title, status or unresolved question. Validate the ledger
with `tools/validate_crucible_research_ledger.py`, and run
`tools/audit_crucible_research_ledger_coverage.py` to identify experiment
reports that still need ledger entries.

## Current high-value entries

| ID | Status | Finding |
|---|---|---|
| `authority-two-candidate-s4` | ESTABLISHED | A causally valid final decision interface can select between two candidates under S4. |
| `authority-component-ablation` | INCONCLUSIVE | C1/C2/C3 passed; historical C4 failed, but objective and trajectory are not isolated. |
| `shared-scorer-compositional-transfer` | FAILED | The tested dynamic scorer did not transfer reliably from three to four candidates. |
| `explicit-id-cardinality-transfer` | FAILED | The oracle passed, but none of eight learned arms passed training competence; learned cardinality transfer is therefore unresolved. |
| `two-candidate-transfer-forensics-v1` | INCONCLUSIVE | Padding alone did not explain the gap; the original audit still confounded candidate-set regime with operand context. |
| `two-candidate-matched-context-probe-v1` | ESTABLISHED | Familiar candidate sets outperformed recombined sets on the same source-context strata for both cyclic checkpoints. |
| `two-candidate-identifier-forensics-v1` | INCONCLUSIVE | Errors concentrated in specific unordered candidate families and often preserved the same wrong identifier under reversal; mechanism remains unresolved. |
| `two-candidate-pair-coverage-v1` | INCONCLUSIVE | All arms fit training, but held-out pair transfer remained incomplete and strongly seed-by-panel dependent; repeated holdouts are still needed to separate coverage from pair-specific effects. |
| `two-candidate-pair-crossover-v1` | ESTABLISHED | The complete crossover supports training-partner/coverage-graph dependence: the same held-out panel can be 0/64 or 47/64 paired depending on the training panel, while no pair fails across all six evaluations. |
| `two-candidate-coverage-density-v1` | INCONCLUSIVE | At a fixed 144-row budget, nested D4/D8/D12 coverage did not produce a monotonic transfer improvement; one D8 arm failed training competence, and density was confounded with per-pair repetition, degree and connectivity. |
| `two-candidate-topology-control-v1` | INCONCLUSIVE | Matched-degree graph controls and the path oracle passed, but only 1/4 arms reached training competence; no diameter-5 arm passed, so the topology comparison is not adjudicated. |
| `topology-symbolic-hypothesis-audit-v1` | ESTABLISHED | The graph, modular-oracle and effective-resistance mathematics replicate; baseline neural errors do not yet validate those mechanisms inside Crucible. |
| `topology-constructive-relational-baseline-v1` | ESTABLISHED | A supplied shared latent-potential representation composes all tested unseen pairs exactly on both graphs; this is a symbolic constructive control, not neural discovery. |
| `two-candidate-topology-duration-extension-v1` | ESTABLISHED | All four topology arms passed the training gate after 2,000 updates, with exact 500-step reproduction; this rescues acquisition but does not establish topology causality. |
| `phase-vii-neural-shared-latent-transfer-v1` | FAILED | All eight shared-latent/cycle arms passed training competence, but none improved matched paired transfer over N0; a read-only replay found N1/N2 preserved predicted values under every candidate-order counterfactual (often the wrong value), so the tested neural inductive bias did not transfer on this benchmark. |
| `learning-lifecycle-v0-engineered-baseline` | ESTABLISHED | The bounded symbolic learner acquired and verified 24/24 procedures, retained and executed them correctly after reset on 288/288 queries, rejected insufficient/contradictory evidence, and revised a versioned procedure; contradiction invalidation was externally triggered. |
| `track-b-c1-engineered-controller-series-v1` | INCONCLUSIVE | C1f–C1r provide heterogeneous engineered evidence on reuse, applicability, active evidence and correction, but do not demonstrate a neural procedure lifecycle. |
| `digitwise-addition-ladder-forensics-v2-v3` | ESTABLISHED | Read-only checkpoint forensics reproduced the arithmetic metrics and separated digit/EOS/output-format effects; pair-level causes remain unresolved. |
| `track-b-procedure-interface-design-v1` | UNTESTED | The canonical AST interface, exact XOR semantics and exhaustive structural/functional novelty criteria are specified but untested. |
| `track-b-neural-proposer-protocol-v1` | UNTESTED | The preregistered learned-proposer versus deterministic AST-search comparison is defined but has not been executed. |
| `source-training-diagnostic-v1` | INCONCLUSIVE | Both seeds fit explicit execution training examples but failed the held-out execution gate; induction was correctly blocked. |
| `hybrid-procedural-controller-interface-audit-v1` | ESTABLISHED | A CPU-only synthetic audit passed 15/15 checks for a typed proposal boundary, independent AST verification, persistent reset execution, unsupported-request abstention and append-only revision; no neural learning claim. |
| `hybrid-reference-controller-substrate-path-v1` | ESTABLISHED | An engineered reference controller passed six end-to-end fixtures covering renamed/reordered bindings, unseen values, novel composition, reset execution, abstention and semantic mismatch detection; no neural learning claim. |
| `hybrid-neural-controller-proposal-v1` | UNTESTED | CPU preflight generated 512/128/256 manifests, passed leakage and novel-composition checks, and validated exact-memory/search baselines; controller training remains untested. |
| `track-a-cnss-partner-generalization-v1` | INCONCLUSIVE | The four-skill partnership pilot completed cleanly, but modular-addition skill D failed the 95% individual gate; partnership-locus results remain descriptive. |
| `track-a-cnss-d-skill-acquisition-diagnostic-v1` | INCONCLUSIVE | Oracle, encoding and partition checks passed, but D reached only 87.5% training and 68.75% test accuracy under the existing rank-8/2,000-update setup. |
| `track-a-cnss-d-skill-corrective-controls-v1` | INCONCLUSIVE | Rank 32 reached perfect training fit but only 85.94% mean held-out accuracy; no condition reached the 95% D gate, and condition-specific seed offsets limit strict paired capacity/horizon causality. |
| `track-a-cnss-d-generalization-forensics-v1` | INCONCLUSIVE | Rank-32 errors concentrate in higher output bits and local transition dimensions consistent with carry/high-bit difficulty, but uneven strata prevent a unique mechanistic conclusion. |

The ledger now contains 66 records and indexes 47 canonical experiment report
files (`REPORT_FINAL.md` and `REPORT.md`). The 23 older reports were extracted
from their accepted final reports and linked artifacts on 2026-10-26; no
outcomes were inferred from filenames or
conversation summaries. See
[CRUCIBLE_RESEARCH_CONSOLIDATION_DECISION_20261026.md](CRUCIBLE_RESEARCH_CONSOLIDATION_DECISION_20261026.md)
for the evidence consolidation, overlap groups and cardinality training audit.

## Complete indexed inventory

| ID | Status | Primary report |
|---|---|---|
| `authority-two-candidate-s4` | ESTABLISHED | `crucible_d2_readout_competence_ladder_20261026_v1` |
| `authority-component-ablation` | INCONCLUSIVE | `crucible_d2_component_ablation_20261026_v1` |
| `shared-scorer-compositional-transfer` | FAILED | `crucible_compositional_authority_transfer_20261026_v1` |
| `explicit-id-cardinality-transfer` | FAILED | `crucible_shared_candidate_scoring_cardinality_20261026_v1` |
| `authority-aware-role-binding-v1` | FAILED | `crucible_authority_aware_role_binding_20251025_v1` |
| `authority-aware-role-binding-v2` | FAILED | `crucible_authority_aware_role_binding_20251025_v2` |
| `authority-counterfactual-fixed-width-v1` | INCONCLUSIVE | `crucible_authority_counterfactual_20261026_v1` |
| `authority-counterfactual-raw-label-v2` | FAILED | `crucible_authority_counterfactual_20261026_v2` |
| `authority-representation-use` | ESTABLISHED | `crucible_authority_representation_use_20261026_v1` |
| `batch-composition-order` | INCONCLUSIVE | `crucible_batch_composition_order_20251025_v1` |
| `causally-valid-authority-decision` | ESTABLISHED | `crucible_causally_valid_authority_decision_20261026_v1` |
| `digitwise-balanced-holdouts` | ESTABLISHED | `crucible_digitwise_balanced_holdouts_20251025` |
| `digitwise-diagonal-ablation-v2` | INCONCLUSIVE | `crucible_digitwise_diagonal_ablation_20251025_v2` |
| `digitwise-role-swapped-control` | INCONCLUSIVE | `crucible_digitwise_role_swapped_control_20251025` |
| `equality-foundational-v2` | FAILED | `crucible_equality_foundational_20251025_v2` |
| `explicit-role-binding-gate` | FAILED | `crucible_explicit_role_binding_gate_20251025_v1` |
| `explicit-role-supervision-v3` | FAILED | `crucible_explicit_role_supervision_20251025_v3` |
| `operation-conditioning-forensics` | INCONCLUSIVE | `crucible_operation_conditioning_forensics_20251025_v1` |
| `operation-marker-interface-v3` | FAILED | `crucible_operation_marker_interface_20251025_v3` |
| `operation-position-matrix-v1` | ESTABLISHED | `crucible_operation_position_matrix_20251025_v1` |
| `operation-token-identification` | ESTABLISHED | `crucible_operation_token_identification_20251025_v1` |
| `paired-positional-exposure` | FAILED | `crucible_paired_positional_exposure_20251025_v1` |
| `position-generalization` | FAILED | `crucible_position_generalization_gate_20251025_v1` |
| `relevance-adjacency-v1` | ESTABLISHED | `crucible_relevance_adjacency_gate_20251025_v1` |
| `relevance-adjacency-v2` | ESTABLISHED | `crucible_relevance_adjacency_gate_20251025_v2` |
| `role-selection-competence-v5` | FAILED | `crucible_role_selection_competence_20251025_v5` |
| `variable-binding-v1` | FAILED | `crucible_variable_binding_20251025_v1` |
| `cardinality-competence-diagnostic-v1` | INCONCLUSIVE | `crucible_cardinality_competence_diagnostic_20261026_v1` |
| `two-candidate-competence-gate-v1` | INCONCLUSIVE | `crucible_two_candidate_competence_gate_20261027_v1` |
| `two-candidate-transfer-forensics-v1` | INCONCLUSIVE | `crucible_two_candidate_transfer_forensics_20261027_v1` |
| `two-candidate-matched-context-probe-v1` | ESTABLISHED | `crucible_two_candidate_matched_context_probe_20261027_v1` |
| `two-candidate-identifier-forensics-v1` | INCONCLUSIVE | `crucible_two_candidate_identifier_forensics_20261027_v1` |
| `two-candidate-pair-coverage-v1` | INCONCLUSIVE | `crucible_two_candidate_pair_coverage_20261027_v1` |
| `two-candidate-pair-crossover-v1` | ESTABLISHED | `crucible_two_candidate_pair_crossover_20261027_v1` |
| `two-candidate-coverage-density-v1` | INCONCLUSIVE | `crucible_two_candidate_coverage_density_20261027_v1` |
| `two-candidate-topology-control-v1` | INCONCLUSIVE | `crucible_two_candidate_topology_control_20261027_v1` |
| `topology-symbolic-hypothesis-audit-v1` | ESTABLISHED | `crucible_topology_symbolic_hypothesis_audit_20261027_v1` |
| `topology-constructive-relational-baseline-v1` | ESTABLISHED | `crucible_topology_constructive_baseline_20261027_v1` |
| `two-candidate-topology-duration-extension-v1` | ESTABLISHED | `crucible_two_candidate_topology_duration_extension_20261027_v1` |
| `phase-vii-neural-shared-latent-transfer-v1` | FAILED | `crucible_phase_vii_neural_shared_latent_relational_transfer_20261027_v1` |
| `learning-lifecycle-v0-engineered-baseline` | ESTABLISHED | `crucible_learning_lifecycle_v0_20251025` |
| `track-b-c1-engineered-controller-series-v1` | INCONCLUSIVE | `crucible_c1f_persistent_reuse_20261021` through `crucible_c1r_prediction_intervention_20261021` |
| `digitwise-addition-ladder-forensics-v2-v3` | ESTABLISHED | `crucible_digitwise_addition_ladder_20261025/forensics_20251025_v2` and `v3` |
| `track-b-procedure-interface-design-v1` | UNTESTED | design documents only |
| `track-b-neural-proposer-protocol-v1` | UNTESTED | design documents only |
| `source-training-diagnostic-v1` | INCONCLUSIVE | `experiments/crucible_source_training_diagnostic_20261031/SOURCE_TRAINING_DIAGNOSTIC_REPORT.md` |
| `hybrid-procedural-controller-interface-audit-v1` | ESTABLISHED | `experiments/crucible_hybrid_procedural_controller_novelty_audit_20261031/AUDIT_REPORT.md` |
| `hybrid-reference-controller-substrate-path-v1` | ESTABLISHED | `experiments/crucible_hybrid_reference_controller_test_20261031/REFERENCE_CONTROLLER_TEST_REPORT.md` |
| `hybrid-neural-controller-proposal-v1` | UNTESTED | `docs/CRUCIBLE_HYBRID_CONTROLLER_LEARNING_PROPOSAL_20261031.md` |
| `track-a-cnss-partner-generalization-v1` | INCONCLUSIVE | `experiments/crucible_track_a_cnss_partner_generalization_v1_20261101/CNSS_PARTNER_GENERALIZATION_RESULT.md` |
| `track-a-cnss-d-skill-acquisition-diagnostic-v1` | INCONCLUSIVE | `experiments/crucible_track_a_cnss_d_skill_acquisition_diagnostic_v1_20261101/D_SKILL_ACQUISITION_DIAGNOSTIC_REPORT.md` |
| `track-a-cnss-d-skill-corrective-controls-v1` | INCONCLUSIVE | `experiments/crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101/D_SKILL_CORRECTIVE_CONTROLS_RESULT.md` |
| `track-a-cnss-d-generalization-forensics-v1` | INCONCLUSIVE | `experiments/crucible_track_a_cnss_d_generalization_forensics_v1_20261101/D_GENERALIZATION_FORENSICS_RESULT.md` |
