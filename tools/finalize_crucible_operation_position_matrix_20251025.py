#!/usr/bin/env python3
"""Complete checkpoint-only metrics for the finished A0--A3 matrix."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_operation_position_matrix_20251025_v1"
RUNNER = ROOT / "tools" / "run_crucible_operation_position_matrix_20251025.py"


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


runner = import_module("operation_position_finished_runner", RUNNER)
torch = runner.torch
forensics = runner.forensics


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value == "True"
    return bool(value)


def pair_stats(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    selected = [row for row in rows if row["split"] == split and int(row["a"]) != int(row["b"])]
    by_pair: Dict[Tuple[int, int], Dict[str, Mapping[str, Any]]] = {}
    for row in selected:
        by_pair.setdefault((int(row["a"]), int(row["b"])), {})[str(row["operation"])] = row
    complete = [value for value in by_pair.values() if set(value) == {"SELECT_FIRST", "SELECT_SECOND"}]
    both = sum(bool_value(value["SELECT_FIRST"]["output_token_correct"]) and bool_value(value["SELECT_SECOND"]["output_token_correct"]) for value in complete)
    switched = sum(value["SELECT_FIRST"]["predicted_output"] != value["SELECT_SECOND"]["predicted_output"] for value in complete)
    correct_switched = sum(
        bool_value(value["SELECT_FIRST"]["output_token_correct"])
        and bool_value(value["SELECT_SECOND"]["output_token_correct"])
        and value["SELECT_FIRST"]["predicted_output"] != value["SELECT_SECOND"]["predicted_output"]
        for value in complete
    )
    return {
        "pair_groups": len(complete),
        "both_correct": both,
        "both_correct_rate": both / len(complete) if complete else None,
        "predictions_switched": switched,
        "correct_and_switched": correct_switched,
        "correct_and_switched_rate": correct_switched / len(complete) if complete else None,
        "scope": "unequal operand pairs",
        "split": split,
    }


def operation_sensitivity(rows: Sequence[Mapping[str, Any]], logits: torch.Tensor, record_ids: Sequence[str], split: str) -> Dict[str, Any]:
    row_by_id = {str(row["record_id"]): row for row in rows if row["split"] == split and int(row["a"]) != int(row["b"])}
    index_by_id = {str(record_id): index for index, record_id in enumerate(record_ids)}
    by_pair: Dict[Tuple[int, int], Dict[str, Mapping[str, Any]]] = {}
    for row in row_by_id.values():
        by_pair.setdefault((int(row["a"]), int(row["b"])), {})[str(row["operation"])] = row
    pair_rows: List[Dict[str, Any]] = []
    for pair, values in sorted(by_pair.items()):
        if set(values) != {"SELECT_FIRST", "SELECT_SECOND"}:
            continue
        a, b = pair
        first = logits[index_by_id[str(values["SELECT_FIRST"]["record_id"])]].float()
        second = logits[index_by_id[str(values["SELECT_SECOND"]["record_id"])]].float()
        first_a = int(runner.forensics.token_id_for_digit(a))
        first_b = int(runner.forensics.token_id_for_digit(b))
        first_pred = int(first.argmax().item())
        second_pred = int(second.argmax().item())
        pair_rows.append({
            "pair": [a, b],
            "first_prediction": runner.forensics.decode_token(first_pred)[0],
            "second_prediction": runner.forensics.decode_token(second_pred)[0],
            "first_output_token_correct": first_pred == first_a,
            "second_output_token_correct": second_pred == first_b,
            "operation_changes_argmax": first_pred != second_pred,
            "first_toward_correct_margin": float((first[first_a] - first[first_b]).item()),
            "second_toward_correct_margin": float((second[first_b] - second[first_a]).item()),
            "first_toward_correct": bool(first[first_a] > first[first_b]),
            "second_toward_correct": bool(second[first_b] > second[first_a]),
            "logit_l2": float(torch.linalg.vector_norm(first - second).item()),
            "logit_cosine": runner.forensics.cosine(first, second),
            "mean_absolute_logit_delta": float((first - second).abs().mean().item()),
        })
    return {
        "split": split,
        "pair_count": len(pair_rows),
        "pairs": pair_rows,
        "argmax_changes": sum(bool(row["operation_changes_argmax"]) for row in pair_rows),
        "first_toward_correct": sum(bool(row["first_toward_correct"]) for row in pair_rows),
        "second_toward_correct": sum(bool(row["second_toward_correct"]) for row in pair_rows),
        "both_toward_correct": sum(bool(row["first_toward_correct"]) and bool(row["second_toward_correct"]) for row in pair_rows),
        "mean_logit_l2": math.fsum(row["logit_l2"] for row in pair_rows) / len(pair_rows) if pair_rows else None,
        "mean_logit_cosine": math.fsum(row["logit_cosine"] for row in pair_rows) / len(pair_rows) if pair_rows else None,
        "mean_absolute_logit_delta": math.fsum(row["mean_absolute_logit_delta"] for row in pair_rows) / len(pair_rows) if pair_rows else None,
        "interpretation": "Operation-only input swap with operands held fixed; sensitivity is evidence of conditional influence, not by itself a causal proof.",
    }


def build_report(summary: Mapping[str, Any], sensitivity: Mapping[str, Any]) -> str:
    lines = [
        "# Crucible Operation-Position Matrix — Final Report",
        "",
        "CPU-only A0–A3 comparison. A0/A1 are verified existing artifacts; A2/A3 are newly trained. No sealed-test access.",
        "",
        "## Results",
        "",
        "| Condition | Interface | Seed | Train | Validation | Unequal validation | Train paired | Held-out paired | Correct-and-switched | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in summary["arms"]:
        metrics = result["metrics"]
        train_pair = metrics["train_paired_unequal"]
        valid_pair = metrics["validation_paired_unequal"]
        lines.append(
            f"| {result['condition']} | {result['interface']} | {result['seed']} | "
            f"{metrics['train']['accuracy']:.2%} | {metrics['validation']['accuracy']:.2%} | "
            f"{metrics['validation_unequal']['accuracy']:.2%} | "
            f"{train_pair['both_correct']}/{train_pair['pair_groups']} | "
            f"{valid_pair['both_correct']}/{valid_pair['pair_groups']} | "
            f"{valid_pair['correct_and_switched']}/{valid_pair['pair_groups']} | "
            f"{'PASS' if result['gate_pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Train paired correctness is reported separately from held-out paired correctness. The paired metric uses unequal operand pairs so FIRST and SECOND produce distinguishable outputs.",
        "",
        "## Operation-only output sensitivity",
        "",
        "These diagnostics hold operands fixed and swap only the operation field. `argmax changes` counts pairs where the predicted output token changes; `both toward correct` counts pairs where each operation's target logit exceeds the other operand's logit.",
        "",
        "| Condition | Interface | Seed | Split | Pairs | Argmax changes | First toward | Second toward | Both toward | Mean logit L2 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(sensitivity):
        condition, interface, seed = key.rsplit("_", 2)
        for split in ("train", "validation"):
            value = sensitivity[key][split]
            lines.append(
                f"| {condition} | {interface} | {seed.replace('seed', '')} | {split} | {value['pair_count']} | "
                f"{value['argmax_changes']} | {value['first_toward_correct']} | {value['second_toward_correct']} | "
                f"{value['both_toward_correct']} | {value['mean_logit_l2']:.3f} |"
            )
    lines += [
        "",
        "## Answer-position probes",
        "",
        "The operation-identity probe is fitted on training states and evaluated on disjoint validation states. Probe decodability is not treated as proof of causal use.",
        "",
        "| Condition | Interface | Seed | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Final norm |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in summary["answer_position_probes"].items():
        condition, interface, seed = key.rsplit("_", 2)
        row = [condition, interface, seed.replace("seed", "")]
        for state in ("layer_0", "layer_1", "layer_2", "layer_3", "final_norm"):
            row.append(f"{values[state]['ridge']['validation_accuracy']:.2%}")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "A2/A3 move the operation field to the answer boundary, changing distance, absolute positions and prompt token distributions together. An improvement therefore establishes position/interface sensitivity in this tested setup, not a unique attention-routing mechanism.",
        "",
        "## Verification",
        "",
        f"Source-control prediction reproduction passed for all {len(summary['source_reproduction'])} A0/A1 arms. Verification errors: {summary.get('verification_errors', []) or 'none'}.",
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_OPERATION_POSITION_MATRIX_PROTOCOL_20251025.md)",
        "- [Summary](SUMMARY.json)",
        "- [Predictions](PREDICTIONS.csv)",
        "- [Answer-position probes](ANSWER_POSITION_PROBES.json)",
        "- [Operation sensitivity](OUTPUT_SENSITIVITY.json)",
        "- [Finalization](FINALIZATION.json)",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    summary = json.loads((OUTPUT / "SUMMARY.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((OUTPUT / "PREDICTIONS.csv").open(newline="", encoding="utf-8")))
    paired: Dict[str, Dict[str, Any]] = {}
    sensitivity: Dict[str, Dict[str, Any]] = {}
    for arm in summary["arms"]:
        key = arm["arm_id"]
        arm_rows = [row for row in rows if row["arm_id"] == key]
        metrics = arm["metrics"]
        metrics["train_paired_unequal"] = pair_stats(arm_rows, "train")
        metrics["validation_paired_unequal"] = pair_stats(arm_rows, "validation")
        paired[key] = {"train": metrics["train_paired_unequal"], "validation": metrics["validation_paired_unequal"]}
        hidden = torch.load(OUTPUT / "hidden_states" / f"{key}.pt", map_location="cpu", weights_only=False)
        sensitivity[key] = {
            "train": operation_sensitivity(arm_rows, hidden["output_logits"], hidden["record_ids"], "train"),
            "validation": operation_sensitivity(arm_rows, hidden["output_logits"], hidden["record_ids"], "validation"),
        }
    summary["paired_conditioning"] = paired
    summary["output_sensitivity_by_split"] = sensitivity
    write_json(OUTPUT / "PAIRED_CONDITIONAL.json", paired)
    write_json(OUTPUT / "OUTPUT_SENSITIVITY.json", sensitivity)
    (OUTPUT / "REPORT_FINAL.md").write_text(build_report(summary, sensitivity), encoding="utf-8")
    write_json(OUTPUT / "SUMMARY.json", summary)
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = json.loads((OUTPUT / "FINALIZATION.json").read_text(encoding="utf-8"))
    finalization["postprocessed_without_retraining"] = True
    finalization["paired_and_sensitivity_complete"] = True
    finalization["report_sha256"] = sha256(OUTPUT / "REPORT_FINAL.md")
    finalization["summary_sha256"] = sha256(OUTPUT / "SUMMARY.json")
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "postprocessed", "output": str(OUTPUT.relative_to(ROOT)), "verification_pass": finalization.get("verification_pass")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
