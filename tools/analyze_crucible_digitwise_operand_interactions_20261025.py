#!/usr/bin/env python3
"""Checkpoint-only operand interaction forensic analysis."""

from __future__ import annotations

import gc
import importlib.util
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
LADDER_PATH = ROOT / "tools" / "run_crucible_digitwise_addition_ladder_20261025.py"
CHECKPOINT_ROOT = ROOT / "experiments" / "crucible_digitwise_addition_ladder_20261025"
OUTPUT_DEFAULT = ROOT / "experiments" / "crucible_digitwise_operand_forensics_20261025"
INTERFACES = ("canonical", "factorized")
ARMS = ("direct", "curriculum")
SEEDS = (42, 1337)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ladder = import_module("digitwise_ladder_for_forensics", LADDER_PATH)
stage_a = ladder.stage_a


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def all_pairs() -> List[Tuple[int, int]]:
    return [(a, b) for a in range(10) for b in range(10) if a + b < 10]


def split_membership(interface: str) -> Dict[Tuple[int, int], str]:
    root = CHECKPOINT_ROOT / "data" / interface / "single_digit_no_carry"
    membership: Dict[Tuple[int, int], str] = {}
    for split in ("train", "validation"):
        for item in load_jsonl(root / f"{split}.jsonl"):
            pair = (int(item["metadata"]["a"]), int(item["metadata"]["b"]))
            membership[pair] = split
    return membership


def predict(model: Any, records: Sequence[Mapping[str, Any]]) -> List[str]:
    prepared = ladder.prepare(records)
    torch = stage_a.torch
    predictions: List[str] = []
    model.eval()
    with torch.no_grad():
        batch = stage_a.batchify(prepared, list(range(len(prepared))))
        logits = stage_a.forward_model(model, batch)
        start = stage_a.RAW_BYTE_START + ord("0")
        for row, item in enumerate(prepared):
            position = int(item["_prefix_len"]) - 1
            token = int(logits[row, position].argmax().item())
            predictions.append(str(token - start) if start <= token <= start + 9 else f"token:{token}")
    return predictions


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = [bool(row["correct"]) for row in rows]
    return {"examples": n, "correct": sum(correct), "accuracy": sum(correct) / n if n else None}


def grouped(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: summarize_rows(values) for key, values in sorted(groups.items(), key=lambda item: item[0])}


def analyze_model(interface: str, arm: str, seed: int, checkpoint: Path) -> Dict[str, Any]:
    torch = stage_a.torch
    model = stage_a.make_dense(seed)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    membership = split_membership(interface)
    pairs = all_pairs()
    records = [
        ladder.make_record(interface, "single_digit_no_carry", "exploratory_all_pairs", index, a, b)
        for index, (a, b) in enumerate(pairs)
    ]
    predictions = predict(model, records)
    rows: List[Dict[str, Any]] = []
    for pair, record, prediction in zip(pairs, records, predictions):
        a, b = pair
        target = int(record["target"])
        numeric_prediction = int(prediction) if prediction.isdigit() else None
        rows.append({
            "a": a,
            "b": b,
            "target": target,
            "prediction": prediction,
            "correct": numeric_prediction == target,
            "sum": a + b,
            "membership": membership.get(pair, "exploratory"),
            "modular_error": (numeric_prediction - target) % 10 if numeric_prediction is not None else None,
        })
    unordered: List[Dict[str, Any]] = []
    row_map = {(int(row["a"]), int(row["b"])): row for row in rows}
    for a, b in combinations(range(10), 2):
        if a + b >= 10:
            continue
        forward = row_map[(a, b)]
        reverse = row_map[(b, a)]
        unordered.append({
            "a": a,
            "b": b,
            "forward_correct": forward["correct"],
            "reverse_correct": reverse["correct"],
            "category": "both" if forward["correct"] and reverse["correct"] else "forward_only" if forward["correct"] else "reverse_only" if reverse["correct"] else "neither",
        })
    categories = Counter(item["category"] for item in unordered)
    correct_directional = [item for pair, item in row_map.items() if pair[0] != pair[1] and pair[0] + pair[1] < 10]
    forward_rows = [row_map[(a, b)] for a, b in pairs if a < b and a + b < 10]
    reverse_rows = [row_map[(a, b)] for a, b in pairs if a > b and a + b < 10]
    result = {
        "interface": interface,
        "arm": arm,
        "seed": seed,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_step": int(state.get("step", -1)),
        "overall": summarize_rows(rows),
        "by_membership": grouped(rows, "membership"),
        "by_sum": grouped(rows, "sum"),
        "by_first_operand": grouped(rows, "a"),
        "by_second_operand": grouped(rows, "b"),
        "order": {
            "a_lt_b": summarize_rows(forward_rows),
            "a_gt_b": summarize_rows(reverse_rows),
            "a_eq_b": summarize_rows([row for row in rows if row["a"] == row["b"]]),
        },
        "commutativity": {
            "unordered_pairs": len(unordered),
            "categories": dict(sorted(categories.items())),
            "directional_asymmetry_rate": sum(item["category"] in {"forward_only", "reverse_only"} for item in unordered) / len(unordered) if unordered else None,
            "forward_accuracy": summarize_rows(forward_rows),
            "reverse_accuracy": summarize_rows(reverse_rows),
        },
        "prediction_histogram": dict(sorted(Counter(row["prediction"] for row in rows).items())),
        "modular_error_histogram": dict(sorted(Counter(str(row["modular_error"]) for row in rows if row["modular_error"] is not None).items())),
        "rows": rows,
    }
    del model, state
    gc.collect()
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    models: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []
    for interface in INTERFACES:
        for arm in ARMS:
            for seed in SEEDS:
                checkpoint = CHECKPOINT_ROOT / arm / interface / f"seed{seed}" / "single_digit_no_carry" / "checkpoint.pt"
                if checkpoint.is_file():
                    models.append(analyze_model(interface, arm, seed, checkpoint))
                else:
                    unavailable.append({"interface": interface, "arm": arm, "seed": seed, "reason": "repair-only result has no checkpoint"})
    consistency: List[Dict[str, Any]] = []
    for left_index, left in enumerate(models):
        left_rows = {(int(row["a"]), int(row["b"])): row for row in left["rows"]}
        for right in models[left_index + 1 :]:
            if left["interface"] != right["interface"]:
                continue
            right_rows = {(int(row["a"]), int(row["b"])): row for row in right["rows"]}
            pairs = sorted(set(left_rows) & set(right_rows))
            consistency.append({
                "left": {"interface": left["interface"], "arm": left["arm"], "seed": left["seed"]},
                "right": {"interface": right["interface"], "arm": right["arm"], "seed": right["seed"]},
                "pairs": len(pairs),
                "prediction_agreement": sum(left_rows[pair]["prediction"] == right_rows[pair]["prediction"] for pair in pairs) / len(pairs),
                "correctness_agreement": sum(left_rows[pair]["correct"] == right_rows[pair]["correct"] for pair in pairs) / len(pairs),
            })
    write_json(output / "SUMMARY.json", {"experiment": "crucible_digitwise_operand_forensics_20261025", "status": "completed", "training_performed": False, "test_evaluated": False, "models_analyzed": len(models), "unavailable": unavailable, "consistency": consistency, "models": [{key: value for key, value in model.items() if key != "rows"} for model in models]})
    for model in models:
        write_json(output / f"{model['interface']}_{model['arm']}_seed{model['seed']}.json", model)
    write_json(output / "PROTOCOL.json", {"scope": "checkpoint_only", "pairs": len(all_pairs()), "training_performed": False, "test_evaluated": False, "unavailable": unavailable})
    print(json.dumps({"status": "completed", "models_analyzed": len(models), "unavailable": len(unavailable)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
