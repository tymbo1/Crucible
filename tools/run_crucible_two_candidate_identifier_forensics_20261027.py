#!/usr/bin/env python3
"""Analyze committed two-candidate prediction matrices without inference."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "experiments" / "crucible_two_candidate_matched_context_probe_20261027_v1"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_IDENTIFIER_FORENSICS_PROTOCOL_20261027.md"
NOVELTY = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_IDENTIFIER_FORENSICS_NOVELTY_AUDIT_20261027.json"
OUTPUT = ROOT / "experiments" / "crucible_two_candidate_identifier_forensics_20261027_v1"
SEEDS = (42, 1337)
REGIMES = ("FAMILIAR", "RECOMBINED")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_rows(seed: int) -> List[Dict[str, Any]]:
    path = INPUT_ROOT / f"PREDICTIONS_SEED{seed}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 480:
        raise RuntimeError(f"seed {seed}: expected 480 rows, got {len(rows)}")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"seed {seed}: duplicate record IDs")
    for row in rows:
        candidates = [int(value) for value in row["candidates"].split(",")]
        authority = int(row["authority_id"])
        target = int(row["target_slot"])
        predicted = int(row["predicted_slot"])
        oracle = candidates.index(authority) if authority in candidates else 8
        if target != oracle:
            raise RuntimeError(f"seed {seed}: oracle mismatch for {row['record_id']}")
        if predicted not in (0, 1, 8):
            raise RuntimeError(f"seed {seed}: out-of-range prediction for {row['record_id']}")
        if (predicted == target) != (row["pointer_correct"] == "True"):
            raise RuntimeError(f"seed {seed}: correctness mismatch for {row['record_id']}")
        if row["layout"] != "TRAIN_COMPACT" or row["split"] != "matched_context_probe":
            raise RuntimeError(f"seed {seed}: unexpected source matrix condition")
    return rows


def predicted_identity(row: Mapping[str, Any]) -> str:
    predicted = int(row["predicted_slot"])
    if predicted == 8:
        return "ABSTAIN"
    return row["candidates"].split(",")[predicted]


def expected_identity(row: Mapping[str, Any]) -> str:
    target = int(row["target_slot"])
    if target == 8:
        return "ABSTAIN"
    return row["candidates"].split(",")[target]


def candidate_pair(row: Mapping[str, Any]) -> Tuple[int, int]:
    values = tuple(int(value) for value in row["candidates"].split(","))
    return tuple(sorted(values))


def accuracy_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    correct = sum(int(row["pointer_correct"] == "True") for row in rows)
    errors = [row for row in rows if row["pointer_correct"] != "True"]
    by_target: Dict[str, Dict[str, Any]] = {}
    for target in (0, 1, 8):
        subset = [row for row in rows if int(row["target_slot"]) == target]
        target_correct = sum(int(row["pointer_correct"] == "True") for row in subset)
        by_target[str(target)] = {"rows": len(subset), "correct": target_correct, "accuracy": target_correct / len(subset) if subset else None}
    confusion = Counter((int(row["target_slot"]), int(row["predicted_slot"])) for row in rows)
    return {
        "rows": len(rows),
        "correct": correct,
        "errors": len(errors),
        "accuracy": correct / len(rows) if rows else None,
        "by_target": by_target,
        "confusion": {f"target_{target}_pred_{predicted}": count for (target, predicted), count in sorted(confusion.items())},
        "abstention": by_target["8"],
        "wrong_expected_identities": dict(Counter(expected_identity(row) for row in errors)),
        "wrong_predicted_identities": dict(Counter(predicted_identity(row) for row in errors)),
        "wrong_identity_pairs": dict(Counter(f"{expected_identity(row)}->{predicted_identity(row)}" for row in errors)),
    }


def grouped(rows: Iterable[Mapping[str, Any]], key_fn: Any) -> Dict[Any, List[Mapping[str, Any]]]:
    groups: Dict[Any, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return dict(groups)


def ordered_pair_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for pair, subset in sorted(grouped(rows, lambda row: row["candidates"]).items()):
        by_source = {}
        for source in ("TRAIN_SOURCE", "VALIDATION_SOURCE"):
            by_source[source] = accuracy_block([row for row in subset if row["context_source"] == source])
        result[pair] = {"candidate_pair": pair, "overall": accuracy_block(subset), "by_context_source": by_source}
    return result


def unordered_pair_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for pair, subset in sorted(grouped(rows, candidate_pair).items()):
        key = ",".join(str(value) for value in pair)
        ordered = ordered_pair_summary(subset)
        result[key] = {"unordered_pair": key, "overall": accuracy_block(subset), "ordered_pairs": ordered}
    return result


def reversal_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    non_abstain = [row for row in rows if int(row["authority_id"]) != 9]
    groups = grouped(non_abstain, lambda row: (row["context_source"], int(row["context_a"]), int(row["context_b"]), candidate_pair(row), int(row["authority_id"])))
    aggregate = Counter()
    by_pair: Dict[str, Counter] = defaultdict(Counter)
    for key, subset in groups.items():
        if len(subset) != 2:
            raise RuntimeError(f"reversal group does not contain two ordered rows: {key}")
        left, right = subset
        aggregate["groups"] += 1
        pair_key = ",".join(str(value) for value in key[3])
        by_pair[pair_key]["groups"] += 1
        same_identity = predicted_identity(left) == predicted_identity(right)
        slot_flipped = int(left["predicted_slot"] != right["predicted_slot"])
        both_correct = left["pointer_correct"] == "True" and right["pointer_correct"] == "True"
        both_wrong_same_identity = left["pointer_correct"] != "True" and right["pointer_correct"] != "True" and same_identity
        for name, value in (("predicted_identity_same", same_identity), ("predicted_slot_flipped", slot_flipped), ("both_correct", both_correct), ("both_wrong_same_identity", both_wrong_same_identity)):
            aggregate[name] += int(value)
            by_pair[pair_key][name] += int(value)
    return {
        "overall": dict(aggregate),
        "by_unordered_pair": {pair: dict(values) for pair, values in sorted(by_pair.items())},
    }


def error_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for row in rows:
        if row["pointer_correct"] == "True":
            continue
        result.append({
            "seed": int(row["seed"]),
            "candidate_regime": row["candidate_regime"],
            "context_source": row["context_source"],
            "context_a": int(row["context_a"]),
            "context_b": int(row["context_b"]),
            "ordered_candidates": row["candidates"],
            "unordered_candidates": ",".join(str(value) for value in candidate_pair(row)),
            "authority_id": int(row["authority_id"]),
            "target_slot": int(row["target_slot"]),
            "predicted_slot": int(row["predicted_slot"]),
            "expected_identity": expected_identity(row),
            "predicted_identity": predicted_identity(row),
            "error_kind": "wrong_candidate_identity" if int(row["target_slot"]) in (0, 1) and int(row["predicted_slot"]) in (0, 1) else "unexpected_output",
            "record_id": row["record_id"],
            "counterfactual_pair_id": row["counterfactual_pair_id"],
        })
    return result


def main() -> int:
    if OUTPUT.exists() and (OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"refusing to overwrite completed output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    analysis: Dict[str, Any] = {}
    all_errors: List[Dict[str, Any]] = []
    for seed in SEEDS:
        rows = load_rows(seed)
        all_rows[str(seed)] = rows
        analysis[str(seed)] = {
            "overall": accuracy_block(rows),
            "by_regime": {regime: accuracy_block([row for row in rows if row["candidate_regime"] == regime]) for regime in REGIMES},
            "ordered_pairs": {regime: ordered_pair_summary([row for row in rows if row["candidate_regime"] == regime]) for regime in REGIMES},
            "unordered_pairs": {regime: unordered_pair_summary([row for row in rows if row["candidate_regime"] == regime]) for regime in REGIMES},
            "reversal": {regime: reversal_summary([row for row in rows if row["candidate_regime"] == regime]) for regime in REGIMES},
        }
        all_errors.extend(error_rows(rows))

    data_audit = {
        "experiment": "crucible_two_candidate_identifier_forensics_20261027_v1",
        "input_root": rel(INPUT_ROOT),
        "input_prediction_hashes": {f"PREDICTIONS_SEED{seed}.csv": sha256(INPUT_ROOT / f"PREDICTIONS_SEED{seed}.csv") for seed in SEEDS},
        "rows_per_seed": {str(seed): len(all_rows[str(seed)]) for seed in SEEDS},
        "new_inference": False,
        "training_performed": False,
        "sealed_test_accessed": False,
        "checkpoints_modified": False,
    }
    write_json(OUTPUT / "DATA_AUDIT.json", data_audit)
    write_json(OUTPUT / "NOVELTY_AUDIT.json", json.loads(NOVELTY.read_text(encoding="utf-8")))

    write_json(OUTPUT / "PAIR_ANALYSIS.json", analysis)
    fields = [
        "seed", "candidate_regime", "context_source", "context_a", "context_b",
        "ordered_candidates", "unordered_candidates", "authority_id", "target_slot",
        "predicted_slot", "expected_identity", "predicted_identity", "error_kind",
        "record_id", "counterfactual_pair_id",
    ]
    with (OUTPUT / "ERROR_ROWS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_errors)

    report_lines = [
        "# Crucible Two-Candidate Identifier Forensics v1",
        "",
        "Post-hoc, read-only analysis of the committed matched-context prediction matrices. No inference, retraining, checkpoint modification or sealed-test access occurred.",
        "",
        "## Ordered recombined-pair accuracy",
        "",
        "| Seed | Ordered pair | Correct / rows | Accuracy | Error identities (expected → predicted) |",
        "|---:|---|---:|---:|---|",
    ]
    for seed in SEEDS:
        for pair, block in analysis[str(seed)]["ordered_pairs"]["RECOMBINED"].items():
            errors = block["overall"]["wrong_identity_pairs"]
            error_text = ", ".join(f"{key}: {value}" for key, value in sorted(errors.items())) or "none"
            overall = block["overall"]
            report_lines.append(f"| {seed} | `{pair}` | {overall['correct']}/{overall['rows']} | {overall['accuracy']:.3f} | {error_text} |")
    report_lines += [
        "",
        "## Unordered recombined-pair results",
        "",
        "| Seed | Unordered pair | Correct / rows | Accuracy | Errors |",
        "|---:|---|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for pair, block in analysis[str(seed)]["unordered_pairs"]["RECOMBINED"].items():
            overall = block["overall"]
            report_lines.append(f"| {seed} | `{pair}` | {overall['correct']}/{overall['rows']} | {overall['accuracy']:.3f} | {overall['errors']} |")
    report_lines += [
        "",
        "## Target-slot and abstention behavior",
        "",
        "| Seed | Regime | Slot 0 | Slot 1 | Abstention | Wrong predicted identities |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for seed in SEEDS:
        for regime in REGIMES:
            block = analysis[str(seed)]["by_regime"][regime]
            targets = block["by_target"]
            wrong_ids = ", ".join(f"{key}: {value}" for key, value in sorted(block["wrong_predicted_identities"].items())) or "none"
            report_lines.append(f"| {seed} | {regime} | {targets['0']['correct']}/{targets['0']['rows']} | {targets['1']['correct']}/{targets['1']['rows']} | {targets['8']['correct']}/{targets['8']['rows']} | {wrong_ids} |")
    report_lines += [
        "",
        "## Reversal consistency",
        "",
        "For non-abstention rows, a reversal group holds the source context, unordered pair and authority fixed while swapping candidate order. A correct matching rule should preserve the predicted identifier and flip the selected slot.",
        "",
        "| Seed | Regime | Reversal groups | Same predicted identity | Slot flipped | Both correct | Both wrong, same identity |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for regime in REGIMES:
            rev = analysis[str(seed)]["reversal"][regime]["overall"]
            report_lines.append(f"| {seed} | {regime} | {rev.get('groups', 0)} | {rev.get('predicted_identity_same', 0)} | {rev.get('predicted_slot_flipped', 0)} | {rev.get('both_correct', 0)} | {rev.get('both_wrong_same_identity', 0)} |")
    report_lines += [
        "",
        "## Interpretation boundary",
        "",
        "Persistent pair-level errors support a pair-specific behavioral weakness under this distribution. They do not prove that the model literally memorized pairs, identify an embedding mechanism, or establish a general architectural limitation. The matrices contain repeated predictions from two checkpoints and are not independent observations.",
        "",
        "## Artifacts",
        "",
        f"- [Analysis protocol](../../{rel(PROTOCOL)})",
        f"- [Novelty audit](../../{rel(NOVELTY)})",
        "- [Machine-readable pair analysis](PAIR_ANALYSIS.json)",
        "- [Machine-readable error rows](ERROR_ROWS.csv)",
    ]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    interpretation = """# Scientific Interpretation

This is a post-hoc analysis of the committed matched-context prediction
matrices. It did not run the model again. No checkpoints, labels or source
matrices were modified.

## Directly supported patterns

- Familiar candidate sets were perfect for both seeds across both source-context
  strata.
- Recombined errors are concentrated by unordered candidate family. The `(2,5)`
  and `(3,6)` families fail persistently in both seeds; `(0,3)` is perfect in
  both; `(1,4)` is seed-dependent.
- Every recombined abstention row is correct. Non-abstention errors select the
  other candidate rather than producing a new output class.
- Target-slot performance is not a stable one-sided shortcut: seed 42 has a
  modest slot-0/slot-1 difference, while seed 1337 is nearly balanced.
- Candidate-order reversal usually preserves the predicted identifier and
  flips the predicted slot. This weakens a simple fixed-slot explanation. In
  the persistent error families, the model often preserves the same *wrong*
  candidate identity across reversal.

## What remains unresolved

These data support a pair-specific behavioral weakness, but cannot distinguish
exact pair association from an imperfect identifier-matching rule. Only four
unseen unordered recombined pairs are available, and the analysis contains two
checkpoints rather than independent model populations. No neural mechanism or
architectural conclusion is justified.

## Single next controlled experiment

Run a preregistered balanced pair-coverage retraining experiment, keeping the
22M dense `SHARED_SCORER`, cyclic schedule, compact layout, optimizer budget and
competence gate fixed. Rotate held-out unordered candidate pairs across several
balanced splits while ensuring every identifier, target slot, candidate order
and source context is represented in training. Evaluate both held-out pair
orders with counterfactual authority groups.

If errors move with the held-out pairs, exact pair coverage is implicated. If
the same identifier families fail across rotated holdouts, an identifier- or
representation-specific effect is more plausible. This experiment should be
run only after a repository novelty audit; it was not run here.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")

    summary = {
        "experiment": "crucible_two_candidate_identifier_forensics_20261027_v1",
        "status": "completed",
        "protocol": rel(PROTOCOL),
        "novelty_audit": rel(NOVELTY),
        "data_audit": rel(OUTPUT / "DATA_AUDIT.json"),
        "pair_analysis": rel(OUTPUT / "PAIR_ANALYSIS.json"),
        "error_rows": rel(OUTPUT / "ERROR_ROWS.csv"),
        "input_root": rel(INPUT_ROOT),
        "seeds": list(SEEDS),
        "new_inference": False,
        "training_performed": False,
        "sealed_test_accessed": False,
        "checkpoints_modified": False,
        "error_row_count": len(all_errors),
    }
    write_json(OUTPUT / "SUMMARY.json", summary)

    hash_paths = [
        PROTOCOL,
        NOVELTY,
        Path(__file__).resolve(),
        INPUT_ROOT / "PREDICTIONS_SEED42.csv",
        INPUT_ROOT / "PREDICTIONS_SEED1337.csv",
        OUTPUT / "DATA_AUDIT.json",
        OUTPUT / "NOVELTY_AUDIT.json",
        OUTPUT / "PAIR_ANALYSIS.json",
        OUTPUT / "ERROR_ROWS.csv",
        OUTPUT / "REPORT_FINAL.md",
        OUTPUT / "SCIENTIFIC_INTERPRETATION.md",
        OUTPUT / "SUMMARY.json",
    ]
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    write_json(OUTPUT / "FINALIZATION.json", {
        "experiment": summary["experiment"],
        "status": "completed",
        "verification_pass": True,
        "new_inference": False,
        "training_performed": False,
        "sealed_test_accessed": False,
        "checkpoints_modified": False,
        "input_rows": 960,
        "error_rows": len(all_errors),
        "hashes": rel(OUTPUT / "HASHES.json"),
    })
    print(json.dumps({"experiment": summary["experiment"], "status": "completed", "new_inference": False, "error_rows": len(all_errors)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
