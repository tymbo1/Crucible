#!/usr/bin/env python3
"""Read-only paired error-transition audit for the D carry-balanced control."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "crucible_track_a_cnss_d_skill_corrective_controls_v1_20261101" / "MANIFEST_PUBLIC.json"
BASELINE_ROOT = Path("/home/timbushnell/crucible_artifacts/crucible_cnss_d_skill_corrective_controls_v1_20261101")
BALANCED_ROOT = Path("/home/timbushnell/crucible_artifacts/crucible_cnss_d_carry_balanced_acquisition_v1_20261101")
OUTPUT = ROOT / "experiments" / "crucible_track_a_cnss_d_carry_balanced_error_transition_audit_v1_20261101"
SEALED_OUTPUT = BALANCED_ROOT / "D_CARRY_BALANCED_ERROR_TRANSITIONS_SEALED.json"
SEEDS = (42, 1337, 2025)
SKILL = "D"
CONSTANT = 0x37


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def features(value: int) -> tuple[int, int, bool]:
    carry = 0
    carry_outs = []
    for bit in range(8):
        total = ((value >> bit) & 1) + ((CONSTANT >> bit) & 1) + carry
        carry = int(total >= 2)
        carry_outs.append(carry)
    longest = 0
    current = 0
    for carry_out in carry_outs:
        if carry_out:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return sum(carry_outs), longest, bool(carry_outs[-1])


def accuracy(predictions: list[int], targets: list[int]) -> float:
    return sum(prediction == target for prediction, target in zip(predictions, targets)) / len(targets)


def mask_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return {f"0x{mask:02x}": count for mask, count in sorted(Counter(row[field] for row in rows).items())}


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        group: {
            "count": len(items),
            "regressions": sum(item["transition"] == "correct_to_wrong" for item in items),
            "improvements": sum(item["transition"] == "wrong_to_correct" for item in items),
            "control_wrong": sum(item["control_correct"] is False for item in items),
            "balanced_wrong": sum(item["balanced_correct"] is False for item in items),
        }
        for group, items in sorted(groups.items(), key=lambda item: item[0])
    }


def category_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    result = {}
    for group, items in sorted(groups.items(), key=lambda item: item[0]):
        control_correct = sum(item["control_correct"] for item in items)
        balanced_correct = sum(item["balanced_correct"] for item in items)
        result[group] = {
            "count": len(items),
            "control_correct": control_correct,
            "balanced_correct": balanced_correct,
            "control_accuracy": control_correct / len(items),
            "balanced_accuracy": balanced_correct / len(items),
            "delta_balanced_minus_control": (balanced_correct - control_correct) / len(items),
            "regressions": sum(item["transition"] == "correct_to_wrong" for item in items),
            "improvements": sum(item["transition"] == "wrong_to_correct" for item in items),
        }
    return result


def public_answer_free(value: Any) -> bool:
    if isinstance(value, dict):
        return all(key not in {"target", "targets", "prediction", "predictions"} and public_answer_free(item) for key, item in value.items())
    if isinstance(value, list):
        return all(public_answer_free(item) for item in value)
    return True


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    test_values = list(manifest["partitions"][SKILL]["test"])
    if len(test_values) != 64 or len(set(test_values)) != 64:
        raise RuntimeError("fixed D test partition changed")
    all_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    for seed in SEEDS:
        baseline_path = BASELINE_ROOT / "runs" / f"rank_32_updates_2000_lambda_0.1_seed_{seed}.json"
        balanced_path = BALANCED_ROOT / "runs" / f"seed_{seed}.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        balanced = json.loads(balanced_path.read_text(encoding="utf-8"))
        control_predictions = baseline["test"]["predictions"]
        balanced_predictions = balanced["test"]["predictions"]
        control_targets = baseline["test"]["targets"]
        balanced_targets = balanced["test"]["targets"]
        expected_targets = [((value + CONSTANT) & 0xFF) for value in test_values]
        if control_targets != expected_targets or balanced_targets != expected_targets:
            raise RuntimeError(f"oracle target mismatch seed={seed}")
        if len(control_predictions) != 64 or len(balanced_predictions) != 64:
            raise RuntimeError(f"prediction cardinality mismatch seed={seed}")
        rows: list[dict[str, Any]] = []
        for index, (value, target, control_prediction, balanced_prediction) in enumerate(zip(test_values, expected_targets, control_predictions, balanced_predictions)):
            control_correct = control_prediction == target
            balanced_correct = balanced_prediction == target
            if control_correct and balanced_correct:
                transition = "preserved_correct"
            elif control_correct and not balanced_correct:
                transition = "correct_to_wrong"
            elif not control_correct and balanced_correct:
                transition = "wrong_to_correct"
            else:
                transition = "preserved_wrong"
            carry_count, max_chain, wraparound = features(value)
            row = {
                "seed": seed,
                "test_index": index,
                "input_value": value,
                "target": target,
                "control_prediction": control_prediction,
                "balanced_prediction": balanced_prediction,
                "control_correct": control_correct,
                "balanced_correct": balanced_correct,
                "transition": transition,
                "control_error_mask": control_prediction ^ target,
                "balanced_error_mask": balanced_prediction ^ target,
                "carry_count": carry_count,
                "max_carry_chain": max_chain,
                "wraparound": wraparound,
            }
            rows.append(row)
            all_rows.append(row)
        regressions = [row for row in rows if row["transition"] == "correct_to_wrong"]
        improvements = [row for row in rows if row["transition"] == "wrong_to_correct"]
        control_bit_correct = [((prediction ^ target) >> bit) & 1 == 0 for prediction, target in zip(control_predictions, expected_targets) for bit in range(8)]
        balanced_bit_correct = [((prediction ^ target) >> bit) & 1 == 0 for prediction, target in zip(balanced_predictions, expected_targets) for bit in range(8)]
        seed_summaries.append({
            "seed": seed,
            "control_test_accuracy": accuracy(control_predictions, expected_targets),
            "balanced_test_accuracy": accuracy(balanced_predictions, expected_targets),
            "delta_balanced_minus_control": accuracy(balanced_predictions, expected_targets) - accuracy(control_predictions, expected_targets),
            "transition_counts": dict(sorted(Counter(row["transition"] for row in rows).items())),
            "regression_input_values": [row["input_value"] for row in regressions],
            "improvement_input_values": [row["input_value"] for row in improvements],
            "regression_error_masks": mask_counts(regressions, "balanced_error_mask"),
            "improvement_error_masks": mask_counts(improvements, "control_error_mask"),
            "by_carry_count": category_accuracy(rows, "carry_count"),
            "by_max_carry_chain": category_accuracy(rows, "max_carry_chain"),
            "by_wraparound": category_accuracy(rows, "wraparound"),
            "regression_by_carry_count": grouped_summary(regressions + improvements, "carry_count"),
            "regression_by_max_carry_chain": grouped_summary(regressions + improvements, "max_carry_chain"),
            "regression_by_wraparound": grouped_summary(regressions + improvements, "wraparound"),
            "control_bit_accuracy": sum(control_bit_correct) / len(control_bit_correct),
            "balanced_bit_accuracy": sum(balanced_bit_correct) / len(balanced_bit_correct),
            "control_run_sha256": digest_bytes(baseline_path.read_bytes()),
            "balanced_run_sha256": digest_bytes(balanced_path.read_bytes()),
        })
    regressions_by_seed = {summary["seed"]: set(summary["regression_input_values"]) for summary in seed_summaries}
    improvements_by_seed = {summary["seed"]: set(summary["improvement_input_values"]) for summary in seed_summaries}
    public = {
        "schema_version": "crucible-cnss-d-carry-balanced-error-transition-audit-v1-20261101",
        "classification": "READ_ONLY_PAIRED_ERROR_TRANSITION_AUDIT",
        "source_experiment": "track-a-cnss-d-carry-balanced-acquisition-v1",
        "manifest_sha256": digest_bytes(MANIFEST.read_bytes()),
        "test_count": len(test_values),
        "seeds": list(SEEDS),
        "operation": "x + 0x37 modulo 256",
        "seed_summaries": seed_summaries,
        "cross_seed_overlap": {
            "regression_inputs_all_seeds": sorted(set.intersection(*regressions_by_seed.values())) if regressions_by_seed else [],
            "regression_inputs_1337_and_2025": sorted(regressions_by_seed[1337] & regressions_by_seed[2025]),
            "improvement_inputs_all_seeds": sorted(set.intersection(*improvements_by_seed.values())) if improvements_by_seed else [],
        },
        "aggregate_transition_counts": dict(sorted(Counter(row["transition"] for row in all_rows).items())),
        "aggregate_regression_masks": mask_counts([row for row in all_rows if row["transition"] == "correct_to_wrong"], "balanced_error_mask"),
        "aggregate_improvement_masks": mask_counts([row for row in all_rows if row["transition"] == "wrong_to_correct"], "control_error_mask"),
        "aggregate_by_carry_count": category_accuracy(all_rows, "carry_count"),
        "aggregate_by_max_carry_chain": category_accuracy(all_rows, "max_carry_chain"),
        "aggregate_by_wraparound": category_accuracy(all_rows, "wraparound"),
        "aggregate_regression_by_carry_count": grouped_summary([row for row in all_rows if row["transition"] in {"correct_to_wrong", "wrong_to_correct"}], "carry_count"),
        "aggregate_regression_by_max_carry_chain": grouped_summary([row for row in all_rows if row["transition"] in {"correct_to_wrong", "wrong_to_correct"}], "max_carry_chain"),
        "aggregate_regression_by_wraparound": grouped_summary([row for row in all_rows if row["transition"] in {"correct_to_wrong", "wrong_to_correct"}], "wraparound"),
        "sealed_detail_sha256": None,
        "limitations": [
            "This is a read-only paired audit of the already completed fixed test partition.",
            "It identifies behavioural transitions and arithmetic subcategories but does not identify the internal cause of the transitions.",
            "The 64-state test partition and three seeds limit uncertainty estimates.",
            "No training, optimizer update, checkpoint mutation or partner rerun was performed.",
        ],
    }
    sealed = {
        "schema_version": "crucible-cnss-d-carry-balanced-error-transition-audit-sealed-v1-20261101",
        "public_summary_without_sealed_hash_sha256": digest_json(public),
        "manifest_sha256": digest_bytes(MANIFEST.read_bytes()),
        "test_values": test_values,
        "rows": all_rows,
    }
    write_json(SEALED_OUTPUT, sealed)
    public["sealed_detail_sha256"] = digest_bytes(SEALED_OUTPUT.read_bytes())
    if not public_answer_free(public):
        raise RuntimeError("public audit contains answer-bearing keys")
    write_json(OUTPUT / "D_CARRY_BALANCED_ERROR_TRANSITION_AUDIT.json", public)
    write_json(OUTPUT / "D_CARRY_BALANCED_ERROR_TRANSITION_AUDIT_SEALED_HASH.json", {"sealed_detail_sha256": public["sealed_detail_sha256"], "sealed_path": str(SEALED_OUTPUT)})
    lines = [
        "# D carry-balanced paired error-transition audit",
        "",
        "Classification: `READ_ONLY_PAIRED_ERROR_TRANSITION_AUDIT`.",
        "",
        "This audit compares the original rank-32/lambda-0.1 control with the carry-balanced arm on the identical 64-state D test partition. No training or checkpoint mutation occurred.",
        "",
        "| Seed | Preserved correct | Correct→wrong | Wrong→correct | Preserved wrong | Control test | Balanced test | Delta |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in seed_summaries:
        counts = summary["transition_counts"]
        lines.append(f"| {summary['seed']} | {counts.get('preserved_correct', 0)} | {counts.get('correct_to_wrong', 0)} | {counts.get('wrong_to_correct', 0)} | {counts.get('preserved_wrong', 0)} | {summary['control_test_accuracy']:.4f} | {summary['balanced_test_accuracy']:.4f} | {summary['delta_balanced_minus_control']:+.4f} |")
    lines.extend([
        "",
        f"Common regressions in seeds 1337 and 2025: `{public['cross_seed_overlap']['regression_inputs_1337_and_2025']}`.",
        f"Regressions common to all seeds: `{public['cross_seed_overlap']['regression_inputs_all_seeds']}`.",
        "",
        "The exact row-level transition records, including predictions and targets, are retained only in the sealed local artifact referenced by D_CARRY_BALANCED_ERROR_TRANSITION_AUDIT_SEALED_HASH.json.",
        "",
        "This audit is diagnostic evidence, not a causal explanation of why the balanced schedule changed the outputs.",
    ])
    (OUTPUT / "D_CARRY_BALANCED_ERROR_TRANSITION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "classification": public["classification"],
        "aggregate_transition_counts": public["aggregate_transition_counts"],
        "cross_seed_overlap": public["cross_seed_overlap"],
        "sealed_detail_sha256": public["sealed_detail_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
