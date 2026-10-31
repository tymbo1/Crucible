#!/usr/bin/env python3
"""Consolidate the CNSS Skill D acquisition campaign without training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_d_acquisition_campaign_review_v1_20261101"


def load(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.exists():
        raise RuntimeError(f"missing campaign artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    corrective_path = ROOT / "experiments/crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101/D_SKILL_CORRECTIVE_CONTROLS_RESULT.json"
    corrective = load("experiments/crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101/D_SKILL_CORRECTIVE_CONTROLS_RESULT.json")
    forensics = load("experiments/crucible_track_a_cnss_d_generalization_forensics_v1_20261101/D_GENERALIZATION_FORENSICS_RESULT.json")
    fault = load("experiments/crucible_track_a_cnss_d_fault_isolation_v1_corrected_20261101/D_FAULT_ISOLATION_RESULT.json")
    lambda_analysis = load("experiments/crucible_track_a_cnss_d_lambda_acquisition_analysis_v1_20261101/D_LAMBDA_ACQUISITION_ANALYSIS_RESULT.json")
    balanced = load("experiments/crucible_track_a_cnss_d_carry_balanced_acquisition_v1_20261101/CARRY_BALANCED_ACQUISITION_RESULT.json")
    transitions = load("experiments/crucible_track_a_cnss_d_carry_balanced_error_transition_audit_v1_20261101/D_CARRY_BALANCED_ERROR_TRANSITION_AUDIT.json")
    geometry = load("experiments/crucible_track_a_cnss_workspace_transformation_geometry_audit_v1_20261101/WORKSPACE_TRANSFORMATION_GEOMETRY_AUDIT.json")
    alternative = load("experiments/crucible_track_a_cnss_workspace_accessibility_causal_control_v1_20261101/WORKSPACE_ACCESSIBILITY_CAUSAL_CONTROL_RESULT.json")
    partner = load("experiments/crucible_track_a_cnss_partner_generalization_v1_20261101/CNSS_PARTNER_GENERALIZATION_RESULT.json")

    manifest_hash = corrective["manifest_sha256"]
    if manifest_hash != balanced["manifest_sha256"] or manifest_hash != alternative["manifest_sha256"] or manifest_hash != forensics["manifest_sha256"] or manifest_hash != fault["manifest_sha256"]:
        raise RuntimeError("campaign artifacts do not share the expected fixed manifest")
    condition = corrective["condition_summary"]
    review = {
        "schema_version": "crucible-cnss-d-acquisition-campaign-review-v1-20261101",
        "classification": "READ_ONLY_D_ACQUISITION_CAMPAIGN_REVIEW",
        "scope": "Consolidation of the registered Skill D acquisition, forensics, geometry, carry-balancing and workspace-accessibility evidence through 2026-11-01.",
        "training_performed_by_review": False,
        "partner_rerun_unlocked": False,
        "fixed_evaluation_audit": {
            "manifest_sha256": manifest_hash,
            "test_state_count": 64,
            "training_candidate_count": 128,
            "direct_scored_training_rows_on_same_test": {
                "rank_horizon_lambda_corrective_control": len(corrective["rows"]),
                "carry_balanced": len(balanced["rows"]),
                "alternative_workspace": len(alternative["rows"]),
                "total": len(corrective["rows"]) + len(balanced["rows"]) + len(alternative["rows"]),
            },
            "read_only_reuses": {
                "generalization_forensics_checkpoints": forensics["checkpoint_count"],
                "fault_isolation_checkpoints": fault["checkpoint_count"],
                "lambda_reanalysis_rows": len(lambda_analysis["rows"]),
                "carry_error_transition_comparisons": sum(transitions["aggregate_transition_counts"].values()),
            },
            "interpretation": "The fixed test was not used for gradient updates, but it was repeatedly used for checkpoint comparison, intervention selection and post-hoc diagnostics. It should now be retired from future model selection and preserved as a historical diagnostic only.",
        },
        "established_under_tested_conditions": [
            "The D oracle, partition boundaries, canonical workspace reconstruction and checkpoint integrity checks are operational.",
            "Rank 32 is materially more capable than rank 8 on the tested fixed task, but rank 32 does not reach the 95% held-out gate.",
            "All rank-32 corrective-control rows fit their training partitions at 100% while remaining below the individual held-out gate.",
            "Canonical input and target states decode exactly across the 256-state domain.",
            "The original workspace's D transformation has lower pairwise latent-distance correlation and a larger rank-32 affine residual than A, B or C in the read-only geometry proxy.",
            "The signed-bit alternative workspace is a valid exact-reconstruction control, but it performed worse under the matched registered training protocol.",
            "All authorized CPU runs in this campaign passed finite-tensor, checkpoint and fresh-process replay safeguards.",
        ],
        "hypotheses_weakened_or_rejected_under_documented_conditions": [
            {
                "hypothesis": "More rank or a longer horizon alone will reach reliable D acquisition.",
                "evidence": {"rank32_2000_mean_test": condition["rank_32_updates_2000"]["test_accuracy_mean"], "rank32_8000_mean_test": condition["rank_32_updates_8000"]["test_accuracy_mean"], "rank8_2000_mean_test": condition["rank_8_updates_2000"]["test_accuracy_mean"], "rank8_8000_mean_test": condition["rank_8_updates_8000"]["test_accuracy_mean"]},
                "status": "not_supported",
            },
            {
                "hypothesis": "Stronger canonical alignment reliably solves D acquisition.",
                "evidence": "The existing lambda analysis found lower displacement but non-monotonic held-out accuracy; no lambda arm reached 95%.",
                "status": "not_supported",
            },
            {
                "hypothesis": "Carry-count frequency balancing alone resolves D generalization.",
                "evidence": {"balanced_mean_test": balanced["paired_control"]["balanced_mean_test_accuracy"], "control_mean_test": balanced["paired_control"]["mean_test_accuracy"], "delta": balanced["paired_control"]["mean_delta_balanced_minus_control"]},
                "status": "not_supported",
            },
            {
                "hypothesis": "Explicitly exposed signed bit coordinates make D easier to acquire under the matched protocol.",
                "evidence": alternative["paired_comparison"],
                "status": "failed_for_this_construction",
            },
        ],
        "remaining_plausible_explanations": [
            "D's carry-dependent transformation is less compatible with the current adapter's inductive bias or optimization path than the tested A-C transformations.",
            "The original nonlinear workspace may encode useful relationships that the signed-bit workspace removes, even though both preserve all state values.",
            "The fixed training support and one addition constant may encourage shortcuts that do not transfer to new constants or carry configurations.",
            "Coordinate semantics, workspace complexity, optimizer interaction and carry propagation remain confounded in the signed-bit negative control.",
            "The 64-state test partition is too small and has sparse carry strata for reliable subcategory selection.",
        ],
        "not_established": [
            "A reusable carry algorithm inside the neural adapter.",
            "A general workspace design that improves arithmetic acquisition.",
            "Reliable neural acquisition at the 95% individual gate.",
            "Neural skill composition, positive forward transfer or cumulative learning.",
            "Optimality of the original workspace.",
        ],
        "future_evaluation_policy": {
            "current_test_status": "retire_from_model_selection; preserve as historical_diagnostic_v1",
            "required_before_next_major_intervention": [
                "Generate and seal a fresh manifest before training or architecture selection.",
                "Use disjoint train, development and final-test manifests with a new manifest hash.",
                "Hold out entire addition constants or operation instances, not only input states.",
                "Include carry-balanced coverage in train/development and report full carry-chain strata with sample counts.",
                "Keep the final test sealed and consult it once for the preregistered comparison.",
                "Separate explicit execution, generalization to unseen inputs, transfer to unseen constants and composition into distinct gates.",
                "Select checkpoints and hyperparameters only from training/development evidence.",
                "Keep the current 95% individual D gate and partner lock unchanged until the fresh domain passes.",
            ],
            "recommended_next_experiment": "A fresh preregistered arithmetic-acquisition domain that trains on several public constants or operation instances and evaluates on sealed unseen constants, with no further model selection on the current 64-state test.",
        },
        "source_artifacts": [
            "experiments/crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101/D_SKILL_CORRECTIVE_CONTROLS_RESULT.json",
            "experiments/crucible_track_a_cnss_d_generalization_forensics_v1_20261101/D_GENERALIZATION_FORENSICS_RESULT.json",
            "experiments/crucible_track_a_cnss_d_fault_isolation_v1_corrected_20261101/D_FAULT_ISOLATION_RESULT.json",
            "experiments/crucible_track_a_cnss_d_lambda_acquisition_analysis_v1_20261101/D_LAMBDA_ACQUISITION_ANALYSIS_RESULT.json",
            "experiments/crucible_track_a_cnss_d_carry_balanced_acquisition_v1_20261101/CARRY_BALANCED_ACQUISITION_RESULT.json",
            "experiments/crucible_track_a_cnss_workspace_transformation_geometry_audit_v1_20261101/WORKSPACE_TRANSFORMATION_GEOMETRY_AUDIT.json",
            "experiments/crucible_track_a_cnss_workspace_accessibility_causal_control_v1_20261101/WORKSPACE_ACCESSIBILITY_CAUSAL_CONTROL_RESULT.json",
            "experiments/crucible_track_a_cnss_partner_generalization_v1_20261101/CNSS_PARTNER_GENERALIZATION_RESULT.json",
        ],
    }
    review["source_artifact_sha256"] = {path: digest(ROOT / path) for path in review["source_artifacts"]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "D_ACQUISITION_CAMPAIGN_REVIEW.json", review)
    lines = [
        "# Crucible CNSS Skill D acquisition campaign review",
        "",
        "Classification: `READ_ONLY_D_ACQUISITION_CAMPAIGN_REVIEW`.",
        "",
        "This review consolidates the Skill D evidence without training or changing any checkpoint.",
        "",
        "## Bottom line",
        "",
        "D can fit the registered training support, but no tested condition reached the 95% individual held-out gate. The signed-bit workspace control was negative: 75.52% mean test accuracy versus 85.94% for the original workspace. The broader workspace-geometry hypothesis remains unresolved, but this particular alternative is rejected.",
        "",
        "## What is established",
        "",
        "- The oracle, partitions, exact workspace reconstruction and artifact safeguards are valid under the tested protocols.",
        "- Rank 32 improves acquisition relative to rank 8, but more rank/horizon alone did not reach reliable held-out competence.",
        "- Alignment reduced latent displacement without reliably improving D acquisition.",
        "- Carry-balanced exposure and the signed-bit workspace both failed to improve generalization.",
        "- D's canonical transformation is geometrically less affine/accessibly organized than A–C in the read-only proxy.",
        "",
        "## Evaluation-integrity decision",
        "",
        f"The same 64-state test partition was reused under manifest hash `{manifest_hash}` across at least {len(corrective['rows']) + len(balanced['rows']) + len(alternative['rows'])} direct scored model rows, in addition to read-only checkpoint audits. It was not used for gradient updates, but it has been used repeatedly for selection and diagnosis. It is now retired from future model selection and retained only as a historical diagnostic.",
        "",
        "## Required next protocol",
        "",
        "Before another major intervention, create a fresh sealed domain with disjoint train/development/final-test manifests, hold out entire addition constants or operation instances, cover carry structures explicitly, and select checkpoints only from development evidence. Keep execution, unseen-input generalization, unseen-constant transfer and composition as separate gates.",
        "",
        "No partner rerun or further optimization of the current D test is authorized by this review.",
    ]
    (OUTPUT / "D_ACQUISITION_CAMPAIGN_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": review["classification"],
        "records_reviewed": len(review["source_artifacts"]),
        "fixed_manifest_sha256": manifest_hash,
        "direct_scored_rows": review["fixed_evaluation_audit"]["direct_scored_training_rows_on_same_test"]["total"],
        "current_test_retired": True,
        "partner_rerun_unlocked": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
