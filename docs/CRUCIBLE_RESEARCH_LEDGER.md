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

The ledger now covers all 33 final experiment reports in the repository. The
23 older reports were extracted from their accepted final reports and linked
artifacts on 2026-10-26; no outcomes were inferred from filenames or
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
