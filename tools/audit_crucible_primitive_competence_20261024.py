#!/usr/bin/env python3
"""CPU/read-only audit of the primitive-competence experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
POSITION_ROOT = ROOT / "experiments" / "crucible_operation_position_gate_20261024"
OLD_ROOT = ROOT / "experiments" / "crucible_intermediate_supervision_20261023"
OLD_TRACE_ROOT = ROOT / "experiments" / "crucible_free_running_composition_diagnostic_20261023"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_primitive_competence_audit_20261024"
SEEDS = (42, 1337)
CONDITIONS = ("A_to_B", "B_only", "B_to_A")
RAW_BYTE_START = 3
EOS_ID = 2


def import_ladder() -> Any:
    path = ROOT / "tools" / "run_crucible_cognitive_learnability_ladder_20261022.py"
    spec = importlib.util.spec_from_file_location("primitive_audit_ladder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ladder = import_ladder()
stage_a = ladder.stage_a


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def apply_mod(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return (value + operand) % 10
    if operation == "sub":
        return (value - operand) % 10
    if operation == "mul":
        return (value * operand) % 10
    raise ValueError(operation)


def condition_of(record: Mapping[str, Any]) -> str:
    return str(record.get("condition", record.get("metadata", {}).get("condition", "unknown")))


def feature_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    metadata = record["metadata"]
    condition = condition_of(record)
    if condition == "B_only":
        return ("B_only", metadata["operation"], int(metadata["supplied_intermediate"]), int(metadata["operand"]))
    return (
        condition,
        tuple(metadata["composition"]),
        int(metadata["start"]),
        int(metadata["arg1"]),
        int(metadata["arg2"]),
    )


def operation_key(record: Mapping[str, Any]) -> Tuple[str, ...]:
    metadata = record["metadata"]
    return tuple(str(value) for value in metadata["composition"])


def final_value(record: Mapping[str, Any]) -> int:
    return int(record["metadata"]["final"])


def intermediate_value(record: Mapping[str, Any]) -> int | None:
    metadata = record["metadata"]
    return int(metadata["intermediate"]) if "intermediate" in metadata else None


def independent_target_ok(record: Mapping[str, Any]) -> bool:
    metadata = record["metadata"]
    condition = condition_of(record)
    if condition == "B_only":
        expected = apply_mod(int(metadata["supplied_intermediate"]), str(metadata["operation"]), int(metadata["operand"]))
        return str(record["target"]) == str(expected) and int(metadata["final"]) == expected
    value = int(metadata["start"])
    value = apply_mod(value, str(metadata["composition"][0]), int(metadata["arg1"]))
    intermediate = value
    value = apply_mod(value, str(metadata["composition"][1]), int(metadata["arg2"]))
    return str(record["target"]) == f"{intermediate},{value}" and int(metadata["intermediate"]) == intermediate and int(metadata["final"]) == value


def tokenization_audit(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prepared = ladder.prepare_records(records)
    errors: List[str] = []
    prefix_lengths: List[int] = []
    target_lengths: List[int] = []
    for record in prepared:
        prefix_len = int(record["_prefix_len"])
        full = list(record["_full"])
        target_bytes = str(record["target"]).encode("utf-8")
        expected_target = [RAW_BYTE_START + byte for byte in target_bytes] + [EOS_ID]
        actual_target = full[prefix_len:]
        prefix_lengths.append(prefix_len)
        target_lengths.append(len(target_bytes))
        if actual_target != expected_target:
            errors.append(str(record["record_id"]))
    return {
        "records": len(records),
        "serialization_errors": errors,
        "serialization_pass": not errors,
        "prefix_length_min": min(prefix_lengths) if prefix_lengths else None,
        "prefix_length_max": max(prefix_lengths) if prefix_lengths else None,
        "target_byte_lengths": dict(sorted(Counter(target_lengths).items())),
    }


def distribution(records: Sequence[Mapping[str, Any]], train_records: Sequence[Mapping[str, Any]] | None = None) -> Dict[str, Any]:
    feature_set = {feature_key(record) for record in records}
    train_features = {feature_key(record) for record in train_records} if train_records is not None else set()
    final_counts = Counter(final_value(record) for record in records)
    target_counts = Counter(str(record["target"]) for record in records)
    operation_counts = Counter(operation_key(record) for record in records)
    metadata = [record["metadata"] for record in records]
    condition = condition_of(records[0]) if records else "unknown"
    result: Dict[str, Any] = {
        "records": len(records),
        "unique_inputs": len({str(record["input"]) for record in records}),
        "unique_record_ids": len({str(record["record_id"]) for record in records}),
        "condition": condition,
        "operation_counts": {"+".join(key): value for key, value in sorted(operation_counts.items())},
        "final_counts": {str(key): value for key, value in sorted(final_counts.items())},
        "target_counts": dict(sorted(target_counts.items())),
        "feature_records": len(feature_set),
    }
    if condition == "B_only":
        result["operation_counts_flat"] = dict(sorted(Counter(str(item["operation"]) for item in metadata).items()))
        result["intermediate_counts"] = dict(sorted(Counter(int(item["supplied_intermediate"]) for item in metadata).items()))
        result["operand_counts"] = dict(sorted(Counter(int(item["operand"]) for item in metadata).items()))
    else:
        result["start_counts"] = dict(sorted(Counter(int(item["start"]) for item in metadata).items()))
        result["arg1_counts"] = dict(sorted(Counter(int(item["arg1"]) for item in metadata).items()))
        result["arg2_counts"] = dict(sorted(Counter(int(item["arg2"]) for item in metadata).items()))
        result["intermediate_counts"] = dict(sorted(Counter(int(item["intermediate"]) for item in metadata).items()))
    if train_records is not None:
        result["feature_unseen_vs_train"] = sum(feature_key(record) not in train_features for record in records)
        result["feature_unseen_vs_train_rate"] = result["feature_unseen_vs_train"] / len(records) if records else None
    return result


def overlap_counts(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    sets = {
        split: {
            "input": {str(record["input"]) for record in records},
            "feature": {feature_key(record) for record in records},
            "record_id": {str(record["record_id"]) for record in records},
        }
        for split, records in splits.items()
    }
    result: Dict[str, Any] = {}
    for left, right in (("train", "validation"), ("train", "heldout"), ("validation", "heldout")):
        result[f"{left}_{right}"] = {
            key: len(sets[left][key] & sets[right][key]) for key in ("input", "feature", "record_id")
        }
    return result


def baselines(records: Sequence[Mapping[str, Any]], condition: str) -> Dict[str, Any]:
    n = len(records)
    final_counts = Counter(final_value(record) for record in records)
    target_counts = Counter(str(record["target"]) for record in records)
    return {
        "examples": n,
        "uniform_random_final_digit": 0.10,
        "uniform_random_end_to_end": 0.10 if condition == "B_only" else 0.01,
        "majority_final_digit": max(final_counts.values()) / n if n else None,
        "majority_target_sequence": max(target_counts.values()) / n if n else None,
        "majority_final_digit_value": max(final_counts, key=final_counts.get) if final_counts else None,
        "majority_target_sequence_value": max(target_counts, key=target_counts.get) if target_counts else None,
    }


def legacy_distribution(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize the earlier manifest, whose metadata lacks numeric fields."""
    pair_counts = Counter("+".join(str(value) for value in record["metadata"]["composition"]) for record in records)
    final_counts = Counter(int(record["metadata"]["final"]) for record in records)
    target_counts = Counter(str(record["target"]) for record in records)
    return {
        "records": len(records),
        "unique_inputs": len({str(record["input"]) for record in records}),
        "operation_counts": dict(sorted(pair_counts.items())),
        "final_counts": {str(key): value for key, value in sorted(final_counts.items())},
        "target_counts": dict(sorted(target_counts.items())),
        "uniform_random_final_digit": 0.10,
        "uniform_random_end_to_end": 0.01,
        "majority_final_digit": max(final_counts.values()) / len(records) if records else None,
        "majority_target_sequence": max(target_counts.values()) / len(records) if records else None,
    }


def summarize_model_metrics() -> Dict[str, Any]:
    new_summary = json.loads((POSITION_ROOT / "SUMMARY.json").read_text(encoding="utf-8"))
    new: Dict[str, Any] = {}
    for result in new_summary["results"]:
        final_row = result["rows"][-1]
        new[f"seed{result['seed']}_{result['condition']}"] = {
            "train": final_row["metrics"]["train"]["overall"],
            "validation": final_row["metrics"]["validation"]["overall"],
            "heldout": final_row["metrics"]["heldout"]["overall"],
            "validation_sequence": final_row["metrics"]["validation"]["sequence"],
            "heldout_sequence": final_row["metrics"]["heldout"]["sequence"],
        }
    old: Dict[str, Any] = {}
    for seed in SEEDS:
        payload = json.loads((OLD_TRACE_ROOT / f"seed{seed}.json").read_text(encoding="utf-8"))
        final = next(row for row in payload["rows"] if int(row["step"]) == 2_000)
        old[f"seed{seed}"] = {
            "teacher_forced_and_free": final["metrics"],
        }
    return {"new_position_gate_final_checkpoint": new, "old_free_running_final_checkpoint": old}


def old_new_comparison() -> Dict[str, Any]:
    old_seed42 = {
        split: load_jsonl(OLD_ROOT / "seed42_intermediate_and_final" / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    new_seed42 = {
        condition: {
            split: load_jsonl(POSITION_ROOT / f"seed42_{condition}" / f"{split}.jsonl")
            for split in ("train", "validation", "heldout")
        }
        for condition in CONDITIONS
    }
    return {
        "old_intermediate_supervision_seed42": {
            split: legacy_distribution(records) for split, records in old_seed42.items()
        },
        "new_position_gate_seed42": {
            condition: {split: distribution(records) for split, records in splits.items()}
            for condition, splits in new_seed42.items()
        },
        "interface_difference": {
            "old_prompt_prefix": "Task: arithmetic is modulo ten. Apply both operations left-to-right.",
            "new_prompt_prefix": "Task: position-gate modulo-ten arithmetic.",
            "old_pairs": ["add+add", "add+mul"],
            "new_composition_train_pairs": ["add+add", "add+mul", "mul+add", "mul+mul", "sub+add", "sub+mul", "sub+sub"],
            "new_composition_heldout_pairs": ["add+sub", "mul+sub"],
        },
    }


def audit() -> Dict[str, Any]:
    arms: Dict[str, Any] = {}
    independent_errors: Dict[str, List[str]] = {}
    serialization: Dict[str, Any] = {}
    overlaps: Dict[str, Any] = {}
    chance: Dict[str, Any] = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            key = f"seed{seed}_{condition}"
            splits = {
                split: load_jsonl(POSITION_ROOT / key / f"{split}.jsonl")
                for split in ("train", "validation", "heldout")
            }
            errors = [str(record["record_id"]) for records in splits.values() for record in records if not independent_target_ok(record)]
            independent_errors[key] = errors
            serialization[key] = {split: tokenization_audit(records) for split, records in splits.items()}
            overlaps[key] = overlap_counts(splits)
            train = splits["train"]
            arms[key] = {split: distribution(records, train if split != "train" else None) for split, records in splits.items()}
            chance[key] = {split: baselines(records, condition) for split, records in splits.items()}
    return {
        "experiment": "crucible_primitive_competence_audit_20261024",
        "status": "completed",
        "training_performed": False,
        "model_evaluation_performed": False,
        "original_benchmark_modified": False,
        "independent_oracle_errors": independent_errors,
        "tokenization": serialization,
        "split_overlaps": overlaps,
        "distributions_and_baselines": {"arms": arms, "chance": chance},
        "old_new_comparison": old_new_comparison(),
        "model_metrics": summarize_model_metrics(),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result = audit()
    write_json(output / "RUN_CONFIG.json", {
        "experiment": "crucible_primitive_competence_audit_20261024",
        "status": "read_only_cpu_audit",
        "source_position_gate": str(POSITION_ROOT.relative_to(ROOT)),
        "source_old_intermediate_run": str(OLD_ROOT.relative_to(ROOT)),
        "training_performed": False,
        "model_evaluation_performed": False,
        "original_benchmark_modified": False,
    })
    write_json(output / "AUDIT.json", result)
    hashes: Dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "HASHES.json":
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    write_json(output / "HASHES.json", hashes)
    print(json.dumps({"output": str(output), "status": "completed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
