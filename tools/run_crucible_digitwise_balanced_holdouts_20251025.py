#!/usr/bin/env python3
"""Run the frozen repeated balanced pair-holdout study.

The runner writes and audits new 40/15 partitions before invoking the existing
direct-training recipe. It never opens the original sealed test corpus.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
LADDER_ROOT = ROOT / "experiments" / "crucible_digitwise_addition_ladder_20261025"
OUTPUT = ROOT / "experiments" / "crucible_digitwise_balanced_holdouts_20251025"
LADDER_TOOL = ROOT / "tools" / "run_crucible_digitwise_addition_ladder_20261025.py"
STAGE_A_TOOL = ROOT / "tools" / "run_crucible_stage_a_instrumentation_20261022.py"
INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
STAGE = "single_digit_no_carry"
STEPS = 500
PAIRS = [(a, b) for a in range(10) for b in range(10) if a + b < 10]
PAIR_SET = set(PAIRS)
PROBLEMATIC = {(1, 5), (2, 2), (3, 3), (5, 1)}

# These lists are frozen before training and copied verbatim into PARTITIONS.json.
VALIDATION_PAIRS: Dict[str, List[Tuple[int, int]]] = {
    "balanced_a": [(0, 7), (0, 9), (1, 0), (1, 4), (1, 5), (2, 1), (2, 2), (3, 3), (3, 6), (4, 0), (5, 1), (5, 3), (6, 2), (7, 0), (8, 1)],
    "balanced_b": [(0, 0), (0, 2), (0, 7), (1, 2), (1, 4), (1, 6), (2, 1), (2, 3), (3, 1), (3, 6), (4, 4), (4, 5), (5, 3), (6, 0), (6, 1)],
    "balanced_c": [(0, 6), (0, 8), (1, 1), (1, 3), (1, 5), (2, 2), (2, 6), (3, 2), (3, 4), (4, 3), (5, 0), (5, 4), (6, 2), (7, 0), (8, 1)],
    "balanced_d": [(0, 0), (0, 5), (0, 9), (1, 2), (1, 4), (2, 0), (2, 6), (3, 3), (3, 4), (4, 3), (4, 5), (5, 1), (6, 1), (6, 2), (7, 1)],
}


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


ladder = import_module("balanced_holdout_ladder", LADDER_TOOL)
stage_a = import_module("balanced_holdout_stage_a", STAGE_A_TOOL)
torch = stage_a.torch


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(value), sort_keys=True) for value in values) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair(record: Mapping[str, Any]) -> Tuple[int, int]:
    return int(record["metadata"]["a"]), int(record["metadata"]["b"])


def records_for_partition(partition: str, interface: str) -> Dict[str, List[Dict[str, Any]]]:
    validation = set(VALIDATION_PAIRS[partition])
    train = sorted(PAIR_SET - validation)
    result: Dict[str, List[Dict[str, Any]]] = {}
    for split, selected in (("train", train), ("validation", sorted(validation))):
        values: List[Dict[str, Any]] = []
        for index, (a, b) in enumerate(selected):
            record = ladder.make_record(interface, STAGE, split, index, a, b)
            record["partition"] = partition
            record["record_id"] = hashlib.sha256(f"balanced_holdout|{partition}|{interface}|{split}|{index}|{a}|{b}".encode()).hexdigest()[:20]
            values.append(record)
        result[split] = values
    return result


def record_errors(record: Mapping[str, Any], interface: str, partition: str, split: str) -> List[str]:
    errors: List[str] = []
    a, b = pair(record)
    if record.get("interface") != interface or record.get("partition") != partition or record.get("split") != split:
        errors.append("metadata")
    if a + b >= 10 or str(record["target"]) != str(a + b):
        errors.append("label")
    if int(record["oracle"]["value"]) != a + b or int(record["metadata"]["answer"]) != a + b:
        errors.append("oracle")
    if int(record["metadata"]["output_length"]) != 1 or len(str(record["target"])) != 1:
        errors.append("output_length")
    return errors


def partition_audit(partition: str, records: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    train_pairs = {pair(record) for record in records["train"]}
    validation_pairs = {pair(record) for record in records["validation"]}
    errors: List[str] = []
    if len(train_pairs) != 40 or len(validation_pairs) != 15:
        errors.append("counts")
    if train_pairs & validation_pairs:
        errors.append("overlap")
    if train_pairs | validation_pairs != PAIR_SET:
        errors.append("universe")
    if sum(a == b for a, b in validation_pairs) != 2:
        errors.append("diagonal_validation_count")
    if not PROBLEMATIC & validation_pairs or PROBLEMATIC - validation_pairs == set():
        # The first condition is intentionally permissive per partition; the
        # across-partition rotation is checked below.
        pass
    for split in ("train", "validation"):
        for record in records[split]:
            errors.extend(f"{split}:{record['record_id']}:{error}" for error in record_errors(record, record["interface"], partition, split))
    return {
        "partition": partition,
        "train_count": len(train_pairs),
        "validation_count": len(validation_pairs),
        "train_pairs": sorted(train_pairs),
        "validation_pairs": sorted(validation_pairs),
        "validation_diagonal_pairs": sorted((a, b) for a, b in validation_pairs if a == b),
        "validation_by_first": dict(sorted(Counter(a for a, _ in validation_pairs).items())),
        "validation_by_second": dict(sorted(Counter(b for _, b in validation_pairs).items())),
        "validation_by_sum": dict(sorted(Counter(a + b for a, b in validation_pairs).items())),
        "validation_by_absolute_difference": dict(sorted(Counter(abs(a - b) for a, b in validation_pairs).items())),
        "errors": errors,
        "pass": not errors,
    }


def build_preflight() -> Tuple[Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]], Dict[str, Any]]:
    records: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    audits: Dict[str, Any] = {}
    for partition in VALIDATION_PAIRS:
        records[partition] = {}
        audits[partition] = None
        for interface in INTERFACES:
            records[partition][interface] = records_for_partition(partition, interface)
        # Audit the canonical records; the factorized records share all pairs
        # and labels and receive their own interface check below.
        audit = partition_audit(partition, records[partition]["canonical"])
        interface_errors: Dict[str, List[str]] = {}
        for interface in INTERFACES:
            interface_errors[interface] = [
                f"{split}:{record['record_id']}:{error}"
                for split in ("train", "validation")
                for record in records[partition][interface][split]
                for error in record_errors(record, interface, partition, split)
            ]
        audit["interface_errors"] = interface_errors
        audit["pass"] = audit["pass"] and not any(interface_errors.values())
        audits[partition] = audit
    val_sets = {name: set(values) for name, values in VALIDATION_PAIRS.items()}
    rotation = {
        f"({a},{b})": {name: (a, b) in val_sets[name] for name in val_sets}
        for a, b in sorted(PROBLEMATIC)
    }
    global_errors: List[str] = []
    if not all(audit["pass"] for audit in audits.values()):
        global_errors.append("partition_audit")
    for name, validation in val_sets.items():
        if len(validation) != 15:
            global_errors.append(f"{name}:validation_count")
    if any(all(states.values()) or not any(states.values()) for states in rotation.values()):
        global_errors.append("problematic_pair_not_rotated")
    preflight = {
        "experiment": "crucible_digitwise_balanced_holdouts_20251025",
        "status": "frozen_before_training",
        "partitions": audits,
        "problematic_pair_rotation": rotation,
        "interfaces": list(INTERFACES),
        "seeds": list(SEEDS),
        "steps": STEPS,
        "architecture_source": rel(STAGE_A_TOOL),
        "trainer_source": rel(LADDER_TOOL),
        "sealed_test_accessed": False,
        "training_performed": False,
        "errors": global_errors,
        "pass": not global_errors,
    }
    return records, preflight


def write_partition_artifacts(records: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]]) -> None:
    for partition, by_interface in records.items():
        for interface, splits in by_interface.items():
            base = OUTPUT / "data" / partition / interface
            for split in ("train", "validation"):
                write_jsonl(base / f"{split}.jsonl", splits[split])


def decode_output(token_id: int) -> Tuple[str, bool]:
    value = token_id - int(stage_a.RAW_BYTE_START)
    if 48 <= value <= 57:
        return chr(value), False
    return f"<token:{token_id}>", True


def predict_rows(model: Any, records: Sequence[Mapping[str, Any]], arm: Mapping[str, Any]) -> List[Dict[str, Any]]:
    prepared = ladder.prepare(records)
    rows: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prepared), int(stage_a.BATCH_SIZE)):
            indices = list(range(start, min(start + int(stage_a.BATCH_SIZE), len(prepared))))
            batch = stage_a.batchify(prepared, indices)
            logits = stage_a.forward_model(model, batch)
            for local, index in enumerate(indices):
                item = prepared[index]
                output_position = int(item["_prefix_len"]) - 1
                eos_position = output_position + 1
                predicted_id = int(logits[local, output_position].argmax().item())
                predicted_output, format_error = decode_output(predicted_id)
                predicted_eos_id = int(logits[local, eos_position].argmax().item())
                target_id = int(batch["y"][local, output_position].item())
                target_eos_id = int(batch["y"][local, eos_position].item())
                token_correct = predicted_id == target_id
                eos_correct = predicted_eos_id == target_eos_id
                if int(batch["target_mask"][local].sum().item()) != 2:
                    raise RuntimeError(f"unexpected target mask for {item['record_id']}")
                rows.append({
                    **{key: arm[key] for key in ("arm_id", "partition", "interface", "seed", "checkpoint", "checkpoint_sha256")},
                    "record_id": item["record_id"],
                    "split": item["split"],
                    "a": int(item["metadata"]["a"]),
                    "b": int(item["metadata"]["b"]),
                    "expected_sum": int(item["metadata"]["a"]) + int(item["metadata"]["b"]),
                    "target_output": str(item["target"]),
                    "predicted_output": predicted_output,
                    "target_token_id": target_id,
                    "predicted_token_id": predicted_id,
                    "target_eos_token_id": target_eos_id,
                    "predicted_eos_token_id": predicted_eos_id,
                    "output_format_error": format_error,
                    "output_token_correct": token_correct,
                    "eos_correct": eos_correct,
                    "full_sequence_correct": token_correct and eos_correct,
                    "absolute_difference": abs(int(item["metadata"]["a"]) - int(item["metadata"]["b"])),
                    "signed_difference": int(item["metadata"]["a"]) - int(item["metadata"]["b"]),
                })
    return rows


def stat(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(bool(row["output_token_correct"]) for row in rows)
    full = sum(bool(row["full_sequence_correct"]) for row in rows)
    eos = sum(bool(row["eos_correct"]) for row in rows)
    return {
        "examples": n,
        "output_correct": correct,
        "output_accuracy": correct / n if n else None,
        "full_sequence_correct": full,
        "full_sequence_accuracy": full / n if n else None,
        "eos_correct": eos,
        "eos_errors": n - eos,
        "format_errors": sum(bool(row["output_format_error"]) for row in rows),
    }


def grouped(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    values: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        values[str(row[field])].append(row)
    return {key: stat(group) for key, group in sorted(values.items(), key=lambda item: int(item[0]))}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def arm_id(partition: str, interface: str, seed: int) -> str:
    return f"{partition}_{interface}_seed{seed}"


def build_report(summary: Mapping[str, Any], all_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Crucible Repeated Balanced Operand-Pair Holdouts",
        "",
        "Date: 2026-10-25  ",
        "Status: completed exploratory split study  ",
        "Hardware: CPU only; no sealed test access",
        "",
        "## Frozen controls",
        "",
        "This study kept the existing dense model, byte tokenizer, evaluator, optimizer and 500-update direct-training recipe unchanged. It trained two seeds across four deterministic 40/15 partitions and both existing interfaces. No architecture, task definition, test corpus or original ladder artifact was changed.",
        "",
        "[Protocol](../../docs/CRUCIBLE_DIGITWISE_BALANCED_HOLDOUT_PROTOCOL_20251025.md) and [preflight manifest](PREFLIGHT.json) were written before training.",
        "",
        "## Training and validation results",
        "",
        "| Partition | Interface | Seed | Train exact | Validation exact | Fit gate |",
        "|---|---|---:|---:|---:|---|",
    ]
    for arm in summary["arms"]:
        lines.append(f"| {arm['partition']} | {arm['interface']} | {arm['seed']} | {arm['train']['full_sequence_accuracy']:.2%} ({arm['train']['full_sequence_correct']}/{arm['train']['examples']}) | {arm['validation']['full_sequence_accuracy']:.2%} ({arm['validation']['full_sequence_correct']}/{arm['validation']['examples']}) | {'Pass' if arm['fit_pass'] else 'Fail'} |")
    lines += ["", "## Pair movement across partitions", "", "The four prior failures were deliberately placed in validation in some partitions and training in others.", "", "| Pair | Validation partitions | Training partitions |", "|---|---|---|"]
    for pair_key, states in summary["problematic_pair_rotation"].items():
        val = [name for name, is_val in states.items() if is_val]
        train = [name for name, is_val in states.items() if not is_val]
        lines.append(f"| `{pair_key}` | {', '.join(val)} | {', '.join(train)} |")
    lines += ["", "## Held-out pair results", "", "The authoritative per-pair outputs are in [PREDICTIONS.csv](PREDICTIONS.csv) and [PAIR_SUMMARY.json](PAIR_SUMMARY.json). Training rows are not treated as independent held-out evidence.", "", "| Pair | Validation observations | Output correct | Accuracy |", "|---|---:|---:|---:|"]
    pair_groups: Dict[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if row["split"] == "validation":
            pair_groups[(int(row["a"]), int(row["b"]))].append(row)
    for key, group in sorted(pair_groups.items()):
        correct = sum(bool(row["output_token_correct"]) for row in group)
        lines.append(f"| `{key}` | {len(group)} | {correct} | {correct / len(group):.2%} |")
    lines += ["", "## Diagonal versus non-diagonal control", "", "Each partition holds out exactly two diagonal pairs and thirteen non-diagonal pairs. The machine-readable results report this comparison per partition, interface and seed. This is a matched structural control, not evidence of embedding collision.", "", "| Interface | Diagonal validation accuracy | Non-diagonal validation accuracy |", "|---|---:|---:|"]
    for interface in INTERFACES:
        selected = [row for row in all_rows if row["interface"] == interface and row["split"] == "validation"]
        diag = [row for row in selected if int(row["a"]) == int(row["b"])]
        non = [row for row in selected if int(row["a"]) != int(row["b"])]
        lines.append(f"| {interface} | {sum(bool(row['output_token_correct']) for row in diag)}/{len(diag)} ({sum(bool(row['output_token_correct']) for row in diag)/len(diag):.2%}) | {sum(bool(row['output_token_correct']) for row in non)}/{len(non)} ({sum(bool(row['output_token_correct']) for row in non)/len(non):.2%}) |")
    lines += ["", "## Interpretation", "", "If the same pair remains difficult when its training context changes, that supports a pair-specific representation or optimization hypothesis. If errors follow the held-out set, the original fixed split was the stronger explanation. These results are exploratory and do not establish an internal mechanism.", "", "## Artifacts", "", "- [Machine-readable summary](SUMMARY.json)", "- [Partition manifest](PARTITIONS.json)", "- [Preflight audit](PREFLIGHT.json)", "- [Per-arm results](arm_outputs/)", "- [All predictions](PREDICTIONS.csv)", "- [Per-pair summary](PAIR_SUMMARY.json)", "- [Reproducibility hashes](HASHES.json)"]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records, preflight = build_preflight()
    if not preflight["pass"]:
        write_json(OUTPUT / "PREFLIGHT.json", preflight)
        raise SystemExit(f"preflight failed: {json.dumps(preflight, sort_keys=True)}")
    write_partition_artifacts(records)
    partition_manifest = {
        "experiment": "crucible_digitwise_balanced_holdouts_20251025",
        "partitions": {
            partition: {
                "validation_pairs": [list(pair) for pair in sorted(VALIDATION_PAIRS[partition])],
                "train_pairs": [list(pair) for pair in sorted(PAIR_SET - set(VALIDATION_PAIRS[partition]))],
            }
            for partition in VALIDATION_PAIRS
        },
        "fixed_before_training": True,
    }
    write_json(OUTPUT / "PARTITIONS.json", partition_manifest)
    preflight["status"] = "passed_before_training"
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    print(json.dumps({"status": "preflight_passed", "output": rel(OUTPUT), "arms_expected": len(VALIDATION_PAIRS) * len(INTERFACES) * len(SEEDS)}, sort_keys=True), flush=True)

    arms: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for partition in VALIDATION_PAIRS:
        for interface in INTERFACES:
            split_records = records[partition][interface]
            for seed in SEEDS:
                aid = arm_id(partition, interface, seed)
                arm_output = OUTPUT / "arm_outputs" / partition / interface / f"seed{seed}" / STAGE
                print(json.dumps({"event": "start", "arm_id": aid}, sort_keys=True), flush=True)
                result = ladder.train_model(seed, interface, STAGE, split_records, arm_output, STEPS)
                result.update({"arm_id": aid, "partition": partition, "arm": "direct", "interface": interface, "seed": seed})
                checkpoint = arm_output / "checkpoint.pt"
                result["checkpoint"] = rel(checkpoint)
                result["checkpoint_sha256"] = sha256(checkpoint)
                result["fit_pass"] = bool(result["train_exact"] >= ladder.TRAIN_THRESHOLD)
                write_json(arm_output / "RESULTS.json", result)
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = stage_a.make_dense(seed)
                model.load_state_dict(state["model"], strict=True)
                prediction_arm = {"arm_id": aid, "partition": partition, "interface": interface, "seed": seed, "checkpoint": rel(checkpoint), "checkpoint_sha256": result["checkpoint_sha256"]}
                rows = predict_rows(model, split_records["train"] + split_records["validation"], prediction_arm)
                split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation")}
                arm_summary = {
                    **{key: result[key] for key in ("arm_id", "partition", "arm", "interface", "seed", "steps", "train_exact", "validation_exact", "fit_pass", "elapsed_seconds", "checkpoint", "checkpoint_sha256")},
                    "train": stat(split_rows["train"]),
                    "validation": stat(split_rows["validation"]),
                    "by_split": {split: {"by_first": grouped(split_rows[split], "a"), "by_second": grouped(split_rows[split], "b"), "by_sum": grouped(split_rows[split], "expected_sum"), "by_abs_difference": grouped(split_rows[split], "absolute_difference")} for split in ("train", "validation")},
                }
                write_json(arm_output / "FORENSIC.json", {"summary": arm_summary, "rows": rows})
                arms.append(arm_summary)
                all_rows.extend(rows)
                print(json.dumps({"event": "complete", "arm_id": aid, "train": result["train_exact"], "validation": result["validation_exact"]}, sort_keys=True), flush=True)
                del model, state
                gc.collect()

    write_csv(OUTPUT / "PREDICTIONS.csv", all_rows)
    pair_summary: Dict[str, Any] = {}
    for a, b in PAIRS:
        group = [row for row in all_rows if row["split"] == "validation" and int(row["a"]) == a and int(row["b"]) == b]
        pair_summary[f"({a},{b})"] = {"validation_observations": len(group), "correct": sum(bool(row["output_token_correct"]) for row in group), "accuracy": sum(bool(row["output_token_correct"]) for row in group) / len(group) if group else None, "partitions": sorted({row["partition"] for row in group})}
    write_json(OUTPUT / "PAIR_SUMMARY.json", pair_summary)
    rotation = preflight["problematic_pair_rotation"]
    summary = {
        "experiment": "crucible_digitwise_balanced_holdouts_20251025",
        "status": "completed",
        "training_performed": True,
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "split_regenerated_original": False,
        "partition_count": len(VALIDATION_PAIRS),
        "arm_count": len(arms),
        "prediction_rows": len(all_rows),
        "problematic_pair_rotation": rotation,
        "arms": arms,
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
    write_json(OUTPUT / "PARTITION_AUDIT.json", {"preflight": preflight, "pair_summary": pair_summary})
    (OUTPUT / "REPORT.md").write_text(build_report(summary, all_rows), encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "HASHES.json" and path.suffix != ".pt":
            hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    print(json.dumps({"status": "completed", "arms": len(arms), "prediction_rows": len(all_rows), "output": rel(OUTPUT)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
