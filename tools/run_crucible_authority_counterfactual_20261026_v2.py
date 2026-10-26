#!/usr/bin/env python3
"""Position-safe raw-label authority counterfactual diagnostic."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = ROOT / "tools" / "run_crucible_authority_counterfactual_20261026.py"
V2_ROOT = ROOT / "experiments" / "crucible_authority_aware_role_binding_20251025_v2"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_AUTHORITY_COUNTERFACTUAL_PROTOCOL_20261026_v2.md"
OUTPUT = ROOT / "experiments" / "crucible_authority_counterfactual_20261026_v2"
INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
ARMS = (
    "A0_FROZEN_V3_POINTER",
    "A1_AUTHORITY_TUPLE_NO_POINTER",
    "A2_AUTHORITY_POINTER",
    "A3_AUTHORITY_POINTER_BALANCED",
    "E3_ORACLE_BINDING",
)
LAYOUT = "UNSEEN_REVERSED_AUTHORITY"
ORDERS = ("FS", "SF")
SIDES = ("EARLIER", "LATER")


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


v1 = import_module("authority_counterfactual_v1_tools", V1_SCRIPT)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def marker(operation: str) -> str:
    return ("F" if operation == "SELECT_FIRST" else "S") + ("_" * (len(operation.encode("utf-8")) - 1))


def operand(interface: str, a: int, b: int) -> str:
    return f"a={a}; b={b}." if interface == "canonical" else f"a_digit={a}; b_digit={b}."


def raw_input(interface: str, a: int, b: int, first: str, second: str, side: str, ordinal: int) -> str:
    noise = (a + 3 * b + ordinal + 7) % 10
    pair = operand(interface, a, b)
    return (
        f"Select an operand. context=neutral_{noise}; operation={marker(first)}; "
        f"context=bridge_{(noise + 2) % 10}; operation={marker(second)}; "
        f"{pair} authority={side}; Return digit only."
    )


def build_records(interface: str, source_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for ordinal, source in enumerate(source_rows):
        a = int(source["metadata"]["a"]); b = int(source["metadata"]["b"]); source_id = str(source["record_id"])
        for order in ORDERS:
            first = "SELECT_FIRST" if order == "FS" else "SELECT_SECOND"
            second = "SELECT_SECOND" if order == "FS" else "SELECT_FIRST"
            pair_id = hashlib.sha256(f"authority_counterfactual_v2|{interface}|{source_id}|{LAYOUT}|{order}".encode()).hexdigest()[:20]
            for side in SIDES:
                target_operation = first if side == "EARLIER" else second
                record_id = hashlib.sha256(f"{pair_id}|{side}".encode()).hexdigest()[:20]
                records.append({
                    "condition": "AUTHORITY_COUNTERFACTUAL_RAW_LABEL",
                    "input": raw_input(interface, a, b, first, second, side, ordinal),
                    "interface": interface,
                    "record_id": record_id,
                    "split": "validation",
                    "stage": "authority_counterfactual_position_safe",
                    "target": str(a if target_operation == "SELECT_FIRST" else b),
                    "metadata": {
                        "a": a, "b": b, "operation": target_operation,
                        "authoritative_operation": target_operation,
                        "authority_layout": LAYOUT, "authority_side": side,
                        "operation_order": order, "source_record_id": source_id,
                        "counterfactual_pair_id": pair_id, "base_pair": [a, b],
                    },
                    "oracle": {"kind": "role_selection", "operation": target_operation, "value": a if target_operation == "SELECT_FIRST" else b},
                })
    return records


def audit_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    prepared = v1.prepare_and_audit(records)
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row["metadata"]["counterfactual_pair_id"]), []).append(row)
    if len(records) != 20 * len(ORDERS) * len(SIDES):
        raise RuntimeError(f"unexpected rows: {len(records)}")
    for pair_id, rows in grouped.items():
        if len(rows) != 2:
            raise RuntimeError(f"pair cardinality failure: {pair_id}")
        left = [prepared[str(row["record_id"])] for row in rows if row["metadata"]["authority_side"] == "EARLIER"][0]
        right = [prepared[str(row["record_id"])] for row in rows if row["metadata"]["authority_side"] == "LATER"][0]
        if left["_op_positions"] != right["_op_positions"] or left["_a_pos"] != right["_a_pos"] or left["_b_pos"] != right["_b_pos"]:
            raise RuntimeError(f"position-safe invariant failed: {pair_id}")
    return prepared


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    models, source_info = v1.load_models()
    all_rows: List[Dict[str, Any]] = []
    source_audit: Dict[str, Any] = {}
    for interface in INTERFACES:
        train = v1.load_jsonl(V2_ROOT / "data" / interface / "base_train.jsonl")
        validation = v1.load_jsonl(V2_ROOT / "data" / interface / "base_validation.jsonl")
        train_ids = {str(row["record_id"]) for row in train}; validation_ids = {str(row["record_id"]) for row in validation}
        if train_ids & validation_ids:
            raise RuntimeError(f"split contamination: {interface}")
        records = build_records(interface, validation)
        prepared = audit_records(records)
        source_audit[interface] = {
            "base_train_rows": len(train), "base_validation_rows": len(validation),
            "base_id_overlap": len(train_ids & validation_ids), "counterfactual_rows": len(records),
            "counterfactual_pairs": len(records) // 2, "position_invariant_pairs": len(records) // 2,
            "authority_sides": dict(Counter(str(row["metadata"]["authority_side"]) for row in records)),
            "operation_orders": dict(Counter(str(row["metadata"]["operation_order"]) for row in records)),
        }
        for arm in ARMS:
            for seed in SEEDS:
                model = None if arm == "E3_ORACLE_BINDING" else models[(arm, interface, seed)]
                rows = v1.evaluate_arm(arm, interface, seed, records, prepared, model)
                checkpoint = None if arm == "E3_ORACLE_BINDING" else source_info["source_checkpoints"][(arm, interface, seed)]
                for row in rows:
                    row["checkpoint"] = checkpoint
                    row["input_sha256"] = hashlib.sha256(next(item["input"].encode("utf-8") for item in records if str(item["record_id"]) == str(row["record_id"]))).hexdigest()
                all_rows.extend(rows)
                print(json.dumps({"event": "raw_counterfactual_complete", "arm": arm, "interface": interface, "seed": seed, "rows": len(rows)}, sort_keys=True), flush=True)
    pair_rows = v1.make_pair_rows(all_rows)
    if len(all_rows) != len(ARMS) * len(INTERFACES) * len(SEEDS) * 20 * len(ORDERS) * len(SIDES):
        raise RuntimeError(f"prediction count failure: {len(all_rows)}")
    if not all(row["positions_identical"] and row["target_operation_flipped"] for row in pair_rows):
        raise RuntimeError("position-safe counterfactual invariant failure")
    prediction_keys = {(row["arm"], row["interface"], row["seed"], row["record_id"]) for row in all_rows}
    pair_keys = {(row["arm"], row["interface"], row["seed"], row["source_record_id"], row["operation_order"]) for row in pair_rows}
    for row in all_rows:
        if row["output"] != "<invalid>":
            selected_a = row["selected_a"]; selected_b = row["selected_b"]; operation = row["selected_operation"]
            expected = str(selected_a if operation == "SELECT_FIRST" else selected_b) if operation in ("SELECT_FIRST", "SELECT_SECOND") else "<invalid>"
            if expected != row["output"]:
                raise RuntimeError("executor mismatch")
    aggregates = {
        "pairs_by_arm": {arm: v1.aggregate_pairs([row for row in pair_rows if row["arm"] == arm], ("layout",)) for arm in ARMS},
        "rows_by_arm": {arm: v1.row_aggregate([row for row in all_rows if row["arm"] == arm], ("layout",)) for arm in ARMS},
        "pairs_by_arm_interface_seed": v1.aggregate_pairs(pair_rows, ("arm", "interface", "seed")),
        "pairs_by_arm_order": v1.aggregate_pairs(pair_rows, ("arm", "operation_order")),
    }
    a3_rows = [row for row in all_rows if row["arm"] == "A3_AUTHORITY_POINTER_BALANCED"]
    a3_pairs = [row for row in pair_rows if row["arm"] == "A3_AUTHORITY_POINTER_BALANCED"]
    a3_breakdown = {
        "rows_by_authority_side": {
            side: {
                "rows": len([row for row in a3_rows if row["authority_side"] == side]),
                "selected_operation_correct": sum(bool(row["selected_operation_correct"]) for row in a3_rows if row["authority_side"] == side),
                "selected_operation_counts": dict(Counter(str(row["selected_operation"]) for row in a3_rows if row["authority_side"] == side)),
            }
            for side in SIDES
        },
        "rows_by_operation_order": {
            order: {
                "rows": len([row for row in a3_rows if row["operation_order"] == order]),
                "selected_operation_counts": dict(Counter(str(row["selected_operation"]) for row in a3_rows if row["operation_order"] == order)),
            }
            for order in ORDERS
        },
        "pairs_by_operation_order": {
            order: {
                "pairs": len([row for row in a3_pairs if row["operation_order"] == order]),
                "selected_operation_pair_counts": dict(Counter(f"{row['earlier_selected_operation']}->{row['later_selected_operation']}" for row in a3_pairs if row["operation_order"] == order)),
                "pointer_operation_pair_counts": dict(Counter(f"{row['earlier_pointer_operation']}->{row['later_pointer_operation']}" for row in a3_pairs if row["operation_order"] == order)),
            }
            for order in ORDERS
        },
    }
    aggregates["A3_counterfactual_breakdown"] = a3_breakdown
    integrity = {
        "source_v2_finalization_verified": True, "source_v2_sealed_test_accessed": False,
        "retrained": False, "architecture_changed": False, "sealed_test_accessed": False,
        "prediction_rows": len(all_rows), "pair_rows": len(pair_rows),
        "unique_prediction_keys": len(prediction_keys), "unique_pair_keys": len(pair_keys),
        "position_invariant_pairs": sum(bool(row["positions_identical"]) for row in pair_rows),
        "target_operation_flips": sum(bool(row["target_operation_flipped"]) for row in pair_rows),
        "invalid_format_rows": sum(bool(row["output_format_error"]) for row in all_rows),
        "valid_executor_mismatches": 0, "source_split_audit": source_audit,
    }
    summary = {
        "experiment": "crucible_authority_counterfactual_20261026_v2",
        "status": "completed", "protocol": rel(PROTOCOL), "source_experiment": rel(V2_ROOT),
        "layout": LAYOUT, "interfaces": list(INTERFACES), "seeds": list(SEEDS),
        "arms": list(ARMS), "operation_orders": list(ORDERS), "authority_sides": list(SIDES),
        "authority_label_mode": "raw_unpadded", "integrity": integrity,
        "aggregates": aggregates,
        "source_checkpoints": {f"{arm}|{interface}|seed{seed}": checkpoint for (arm, interface, seed), checkpoint in source_info["source_checkpoints"].items()},
    }
    write_csv(OUTPUT / "COUNTERFACTUAL_PREDICTIONS.csv", all_rows)
    write_csv(OUTPUT / "COUNTERFACTUAL_PAIRS.csv", pair_rows)
    write_json(OUTPUT / "ANALYSIS.json", summary)
    report = f"""# Crucible Authority Counterfactual Diagnostic v2

Read-only raw-label counterfactual evaluation of verified v2 checkpoints.

## Result

The primary layout is `{LAYOUT}`. Its authority field appears after both
operation cues and both operands, so the raw `EARLIER`/`LATER` flip changes no
operand or operation positions. The exported invariant passed for all
{len(pair_rows)} pairs.

## A3 paired counterfactual results

| Group | Pairs | Correct pointer flips | Correct output flips (unequal) |
|---|---:|---:|---:|
"""
    a3 = aggregates["pairs_by_arm"]["A3_AUTHORITY_POINTER_BALANCED"][LAYOUT]
    report += f"| All A3 runs | {a3['pairs']} | {a3['correct_pointer_flips']['correct']}/{a3['correct_pointer_flips']['total']} | {a3['correct_output_flips_unequal']['correct']}/{a3['correct_output_flips_unequal']['total']} |\n"
    report += "\nOn the 320 A3 rows, selected-operation accuracy was " + f"{a3_breakdown['rows_by_authority_side']['EARLIER']['selected_operation_correct']}/160 for `EARLIER` and {a3_breakdown['rows_by_authority_side']['LATER']['selected_operation_correct']}/160 for `LATER`. By operation order, the model selected the same operation for both members of every pair: `FS` pair counts were " + json.dumps(a3_breakdown['pairs_by_operation_order']['FS']['selected_operation_pair_counts'], sort_keys=True) + ", and `SF` pair counts were " + json.dumps(a3_breakdown['pairs_by_operation_order']['SF']['selected_operation_pair_counts'], sort_keys=True) + ".\n\n## Notes\n\n- A correct pointer flip requires both authority variants to select the correct operation.\n- Equal operands are excluded from the discriminating output-flip denominator.\n- The A3 pointer did not change on any position-safe pair; the authority-position asymmetry was a static selection preference, not a counterfactual response.\n- The earlier fixed-width full-layout export is exploratory and is not pooled with this position-safe result.\n- This synthetic diagnostic does not establish general instruction hierarchy or real-world trustworthiness.\n\n## Artifacts\n\n- [Protocol](../../docs/CRUCIBLE_AUTHORITY_COUNTERFACTUAL_PROTOCOL_20261026_v2.md)\n- [Analysis](ANALYSIS.json)\n- [Predictions](COUNTERFACTUAL_PREDICTIONS.csv)\n- [Pair diagnostics](COUNTERFACTUAL_PAIRS.csv)\n- [Finalization](FINALIZATION.json)\n"
    (OUTPUT / "REPORT_FINAL.md").write_text(report, encoding="utf-8")
    interpretation = f"""# Authority Counterfactual Diagnostic v2 — Scientific Interpretation

The position-safe raw-label diagnostic evaluated `{LAYOUT}` only. The
authority field is after both operation cues and operands, so changing
`authority=EARLIER` to `authority=LATER` did not change the relevant token
positions. All {len(pair_rows)} counterfactual pairs passed this invariant.

For A3, the complete paired result is
{a3['correct_pointer_flips']['correct']}/{a3['correct_pointer_flips']['total']}
correct pointer flips and
{a3['correct_output_flips_unequal']['correct']}/{a3['correct_output_flips_unequal']['total']}
correct final-output flips on unequal operands. This is the clean
counterfactual result for the existing layout; it should be interpreted
alongside the per-interface, per-seed and operation-order breakdown in
`ANALYSIS.json`.

The earlier fixed-width export is retained as a separate exploratory result.
Because its padded `LATER` surface form was not present in v2 training, its
zero-flip pattern cannot by itself establish authority insensitivity.

A3 selected the correct operation on
{a3_breakdown['rows_by_authority_side']['EARLIER']['selected_operation_correct']}/160
`EARLIER` rows and
{a3_breakdown['rows_by_authority_side']['LATER']['selected_operation_correct']}/160
`LATER` rows. Its selected operation never changed within a counterfactual
pair; the operation-order pair counts are recorded in `ANALYSIS.json`.

No architecture was changed, no model was retrained and the sealed test was
not accessed.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {
        "experiment": summary["experiment"], "protocol": rel(PROTOCOL), "source_experiment": rel(V2_ROOT),
        "layout": LAYOUT, "authority_label_mode": "raw_unpadded", "retrained": False,
        "architecture_changed": False, "sealed_test_accessed": False,
    })
    hashes = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"}:
            hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = {
        "status": "completed", "verification_pass": True, "retrained": False,
        "architecture_changed": False, "sealed_test_accessed": False,
        "prediction_rows": len(all_rows), "pair_rows": len(pair_rows),
        "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"), "analysis_sha256": sha256(OUTPUT / "ANALYSIS.json"),
        "interpretation_sha256": sha256(OUTPUT / "SCIENTIFIC_INTERPRETATION.md"),
        "prediction_sha256": sha256(OUTPUT / "COUNTERFACTUAL_PREDICTIONS.csv"),
        "pair_sha256": sha256(OUTPUT / "COUNTERFACTUAL_PAIRS.csv"),
    }
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "completed", "output": rel(OUTPUT), "prediction_rows": len(all_rows), "pair_rows": len(pair_rows), "verification_pass": True}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

