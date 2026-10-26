#!/usr/bin/env python3
"""Run the frozen CPU matched diagonal-ablation study.

This runner reuses the already-audited dense training/evaluation implementation
but writes a new experiment tree. It never opens the sealed test corpus or the
previous experiment's checkpoints.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_digitwise_diagonal_ablation_20251025_v2"
BALANCED_TOOL = ROOT / "tools" / "run_crucible_digitwise_balanced_holdouts_20251025.py"
INTERFACES = ("canonical", "factorized")
SEEDS = (2025, 2026)
STAGE = "single_digit_no_carry"
STEPS = 500


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


base = import_module("diagonal_ablation_balanced_runner", BALANCED_TOOL)
ladder = base.ladder
stage_a = base.stage_a
torch = base.torch
PAIR_SET = set(base.PAIRS)

DIAGONAL_SPECIAL = {(2, 2), (3, 3)}
UNEQUAL_SPECIAL = {(2, 4), (3, 1)}
MIXED_SPECIAL = {(2, 2), (2, 4)}
COMMON_BACKGROUND = tuple(sorted(set(base.VALIDATION_PAIRS["balanced_a"]) - DIAGONAL_SPECIAL))
VALIDATION_PAIRS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "diagonal": tuple(sorted(set(COMMON_BACKGROUND) | DIAGONAL_SPECIAL)),
    "unequal_matched": tuple(sorted(set(COMMON_BACKGROUND) | UNEQUAL_SPECIAL)),
    "mixed": tuple(sorted(set(COMMON_BACKGROUND) | MIXED_SPECIAL)),
}
SPECIAL = {
    "diagonal": DIAGONAL_SPECIAL,
    "unequal_matched": UNEQUAL_SPECIAL,
    "mixed": MIXED_SPECIAL,
}
ANCHOR_DIR = OUTPUT / "generated_reference_e4" / "crucible_manifest"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_fresh_anchors() -> Dict[str, str]:
    """Create deterministic E4 anchors locally without touching prior artifacts."""
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    ladder.stage_a.E4_ANCHOR_DIR = ANCHOR_DIR
    base.stage_a.E4_ANCHOR_DIR = ANCHOR_DIR
    hashes: Dict[str, str] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = ladder.stage_a.BraidPrototypeLM(
            ladder.stage_a.VOCAB_SIZE,
            ladder.stage_a.N_LAYERS,
            ladder.stage_a.D_MODEL,
            ladder.stage_a.N_HEADS,
            ladder.stage_a.LATENT_KV,
            ladder.stage_a.LATENT_Q,
            ladder.stage_a.PE_DIM,
            ladder.stage_a.E4_EXPERTS,
            ladder.stage_a.E4_D_FF,
            top_k=ladder.stage_a.E4_TOP_K,
        )
        path = ANCHOR_DIR / f"initial_seed_anchor_{seed}.pt"
        torch.save(model.state_dict(), path)
        hashes[str(seed)] = sha256(path)
        del model
    return hashes


def condition_records(condition: str, interface: str) -> Dict[str, List[Dict[str, Any]]]:
    validation = set(VALIDATION_PAIRS[condition])
    result: Dict[str, List[Dict[str, Any]]] = {}
    for split, selected in (("train", sorted(PAIR_SET - validation)), ("validation", sorted(validation))):
        values: List[Dict[str, Any]] = []
        for index, (a, b) in enumerate(selected):
            record = ladder.make_record(interface, STAGE, split, index, a, b)
            record["condition"] = condition
            record["partition"] = condition
            record["record_id"] = hashlib.sha256(
                f"diagonal_ablation|{condition}|{interface}|{split}|{index}|{a}|{b}".encode()
            ).hexdigest()[:20]
            values.append(record)
        result[split] = values
    return result


def pair(record: Mapping[str, Any]) -> Tuple[int, int]:
    return int(record["metadata"]["a"]), int(record["metadata"]["b"])


def audit_condition(condition: str, records: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    train = {pair(record) for record in records["train"]}
    validation = {pair(record) for record in records["validation"]}
    errors: List[str] = []
    if len(train) != 40 or len(validation) != 15:
        errors.append("counts")
    if train & validation:
        errors.append("overlap")
    if train | validation != PAIR_SET:
        errors.append("universe")
    if any(a + b >= 10 for a, b in validation):
        errors.append("invalid_pair")
    for split in ("train", "validation"):
        for record in records[split]:
            a, b = pair(record)
            if str(record["target"]) != str(a + b) or int(record["oracle"]["value"]) != a + b:
                errors.append(f"label:{split}:{record['record_id']}")
            if record.get("condition") != condition or record.get("partition") != condition:
                errors.append(f"metadata:{split}:{record['record_id']}")
    return {
        "condition": condition,
        "train_count": len(train),
        "validation_count": len(validation),
        "validation_pairs": sorted(validation),
        "validation_special_pairs": sorted(SPECIAL[condition]),
        "validation_diagonal_pairs": sorted((a, b) for a, b in validation if a == b),
        "validation_by_first": dict(sorted(Counter(a for a, _ in validation).items())),
        "validation_by_second": dict(sorted(Counter(b for _, b in validation).items())),
        "validation_by_sum": dict(sorted(Counter(a + b for a, b in validation).items())),
        "validation_by_absolute_difference": dict(sorted(Counter(abs(a - b) for a, b in validation).items())),
        "errors": errors,
        "pass": not errors,
    }


def preflight(records: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]]) -> Dict[str, Any]:
    audits = {condition: audit_condition(condition, records[condition]["canonical"]) for condition in VALIDATION_PAIRS}
    interface_errors: Dict[str, List[str]] = {}
    for condition in VALIDATION_PAIRS:
        for interface in INTERFACES:
            errors: List[str] = []
            for split in ("train", "validation"):
                for record in records[condition][interface][split]:
                    if record.get("interface") != interface:
                        errors.append(f"interface:{condition}:{interface}:{record['record_id']}")
                    if record.get("condition") != condition:
                        errors.append(f"condition:{condition}:{interface}:{record['record_id']}")
            interface_errors[f"{condition}:{interface}"] = errors

    validation_sets = {condition: set(values) for condition, values in VALIDATION_PAIRS.items()}
    balance_errors: List[str] = []
    if any(set(COMMON_BACKGROUND) - validation_sets[condition] for condition in validation_sets):
        balance_errors.append("common_background")
    if sorted(a + b for a, b in SPECIAL["diagonal"]) != sorted(a + b for a, b in SPECIAL["unequal_matched"]):
        balance_errors.append("special_sum_multiset")
    diagonal_counts = {condition: sum(a == b for a, b in values) for condition, values in validation_sets.items()}
    if diagonal_counts != {"diagonal": 2, "unequal_matched": 0, "mixed": 1}:
        balance_errors.append("diagonal_counts")
    first_diagonal = Counter(a for a, _ in sorted(DIAGONAL_SPECIAL))
    first_unequal = Counter(a for a, _ in sorted(UNEQUAL_SPECIAL))
    if first_diagonal != first_unequal:
        balance_errors.append("first_operand_match")

    errors = [f"{condition}:{error}" for condition, audit in audits.items() for error in audit["errors"]]
    errors.extend(f"{key}:{error}" for key, values in interface_errors.items() for error in values)
    errors.extend(balance_errors)
    return {
        "experiment": "crucible_digitwise_diagonal_ablation_20251025",
        "status": "frozen_before_training",
        "conditions": {condition: [list(pair) for pair in pairs] for condition, pairs in VALIDATION_PAIRS.items()},
        "common_background": [list(pair) for pair in COMMON_BACKGROUND],
        "special_pairs": {condition: [list(pair) for pair in pairs] for condition, pairs in SPECIAL.items()},
        "condition_audits": audits,
        "interface_errors": interface_errors,
        "balance_diagnostics": {
            "special_sum_values": {condition: sorted(a + b for a, b in pairs) for condition, pairs in SPECIAL.items()},
            "special_first_operand_counts": {condition: dict(sorted(Counter(a for a, _ in pairs).items())) for condition, pairs in SPECIAL.items()},
            "special_second_operand_counts": {condition: dict(sorted(Counter(b for _, b in pairs).items())) for condition, pairs in SPECIAL.items()},
            "diagonal_counts": diagonal_counts,
            "unavoidable_second_operand_difference": first_diagonal != first_unequal or Counter(b for _, b in DIAGONAL_SPECIAL) != Counter(b for _, b in UNEQUAL_SPECIAL),
        },
        "interfaces": list(INTERFACES),
        "seeds": list(SEEDS),
        "steps": STEPS,
        "architecture_changed": False,
        "sealed_test_accessed": False,
        "errors": errors,
        "pass": not errors,
    }


def arm_id(condition: str, interface: str, seed: int) -> str:
    return f"{condition}_{interface}_seed{seed}"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary: Mapping[str, Any], all_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Crucible Matched Diagonal Ablation",
        "",
        "Date: 2026-10-25  ",
        "Status: completed exploratory CPU study  ",
        "Hardware: CPU only; no sealed test access",
        "",
        f"The study used the frozen 22M dense model, existing evaluator and 500-update direct-training recipe. It used fresh seeds 2025 and 2026 across {summary['condition_count']} pre-frozen conditions and both interfaces.",
        "",
        "## Arm results",
        "",
        "| Condition | Interface | Seed | Train exact | Validation exact | Fit gate |",
        "|---|---|---:|---:|---:|---|",
    ]
    for arm in summary["arms"]:
        lines.append(f"| {arm['condition']} | {arm['interface']} | {arm['seed']} | {arm['train']['full_sequence_accuracy']:.2%} ({arm['train']['full_sequence_correct']}/{arm['train']['examples']}) | {arm['validation']['full_sequence_accuracy']:.2%} ({arm['validation']['full_sequence_correct']}/{arm['validation']['examples']}) | {'Pass' if arm['fit_pass'] else 'Fail'} |")
    lines += [
        "",
        "## Condition comparison",
        "",
        "The final analysis reports fit-gated and all-arm results separately. Training rows are audit evidence, not independent held-out observations.",
        "",
        "| Condition | Diagonal validation | Unequal validation |",
        "|---|---:|---:|",
    ]
    for condition in VALIDATION_PAIRS:
        selected = [row for row in all_rows if row["condition"] == condition and row["split"] == "validation"]
        diag = [row for row in selected if int(row["a"]) == int(row["b"])]
        unequal = [row for row in selected if int(row["a"]) != int(row["b"])]
        d_correct = sum(bool(row["output_token_correct"]) for row in diag)
        u_correct = sum(bool(row["output_token_correct"]) for row in unequal)
        lines.append(f"| {condition} | {d_correct}/{len(diag)} | {u_correct}/{len(unequal)} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "A diagonal-versus-unequal difference supports an equality-specific transfer hypothesis only if it persists across fit-competent seeds and interfaces. The special conditions match sums and first-operand frequencies, but the protocol records an unavoidable second-operand-frequency difference. No internal mechanism is inferred.",
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_DIGITWISE_DIAGONAL_ABLATION_PROTOCOL_20251025.md)",
        "- [Summary](SUMMARY.json)",
        "- [Preflight](PREFLIGHT.json)",
        "- [Predictions](PREDICTIONS.csv)",
        "- [Per-pair results](PAIR_SUMMARY.json)",
        "- [Hashes](HASHES.json)",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    records: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    for condition in VALIDATION_PAIRS:
        records[condition] = {interface: condition_records(condition, interface) for interface in INTERFACES}
    frozen_preflight = preflight(records)
    write_json(OUTPUT / "PREFLIGHT.json", frozen_preflight)
    if not frozen_preflight["pass"]:
        raise SystemExit(f"preflight failed: {json.dumps(frozen_preflight, sort_keys=True)}")
    anchor_hashes = prepare_fresh_anchors()
    write_json(OUTPUT / "INITIAL_ANCHORS.json", {"directory": rel(ANCHOR_DIR), "seeds": list(SEEDS), "sha256": anchor_hashes, "generated_without_training": True})
    for condition, by_interface in records.items():
        for interface, splits in by_interface.items():
            base_path = OUTPUT / "data" / condition / interface
            for split, values in splits.items():
                base_path.mkdir(parents=True, exist_ok=True)
                (base_path / f"{split}.jsonl").write_text("\n".join(json.dumps(value, sort_keys=True) for value in values) + "\n", encoding="utf-8")
    write_json(OUTPUT / "CONDITIONS.json", frozen_preflight["conditions"])
    print(json.dumps({"status": "preflight_passed", "arms_expected": len(VALIDATION_PAIRS) * len(INTERFACES) * len(SEEDS)}, sort_keys=True), flush=True)

    arms: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for condition in VALIDATION_PAIRS:
        for interface in INTERFACES:
            split_records = records[condition][interface]
            for seed in SEEDS:
                aid = arm_id(condition, interface, seed)
                arm_output = OUTPUT / "arm_outputs" / condition / interface / f"seed{seed}" / STAGE
                print(json.dumps({"event": "start", "arm_id": aid}, sort_keys=True), flush=True)
                result = ladder.train_model(seed, interface, STAGE, split_records, arm_output, STEPS)
                result.update({"arm_id": aid, "condition": condition, "partition": condition, "arm": "direct", "interface": interface, "seed": seed})
                checkpoint = arm_output / "checkpoint.pt"
                result["checkpoint"] = rel(checkpoint)
                result["checkpoint_sha256"] = sha256(checkpoint)
                result["fit_pass"] = bool(result["train_exact"] >= ladder.TRAIN_THRESHOLD)
                write_json(arm_output / "RESULTS.json", result)
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = stage_a.make_dense(seed)
                model.load_state_dict(state["model"], strict=True)
                prediction_arm = {
                    "arm_id": aid,
                    "partition": condition,
                    "interface": interface,
                    "seed": seed,
                    "checkpoint": rel(checkpoint),
                    "checkpoint_sha256": result["checkpoint_sha256"],
                }
                rows = base.predict_rows(model, split_records["train"] + split_records["validation"], prediction_arm)
                for row in rows:
                    row["condition"] = condition
                split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation")}
                arm_summary = {
                    **{key: result[key] for key in ("arm_id", "condition", "partition", "arm", "interface", "seed", "steps", "train_exact", "validation_exact", "fit_pass", "elapsed_seconds", "checkpoint", "checkpoint_sha256")},
                    "train": base.stat(split_rows["train"]),
                    "validation": base.stat(split_rows["validation"]),
                }
                write_json(arm_output / "FORENSIC.json", {"summary": arm_summary, "rows": rows})
                arms.append(arm_summary)
                all_rows.extend(rows)
                print(json.dumps({"event": "complete", "arm_id": aid, "train": result["train_exact"], "validation": result["validation_exact"]}, sort_keys=True), flush=True)
                del model, state
                gc.collect()

    write_csv(OUTPUT / "PREDICTIONS.csv", all_rows)
    pair_summary: Dict[str, Any] = {}
    for condition in VALIDATION_PAIRS:
        for a, b in sorted(PAIR_SET):
            group = [row for row in all_rows if row["condition"] == condition and row["split"] == "validation" and int(row["a"]) == a and int(row["b"]) == b]
            pair_summary[f"{condition}|({a},{b})"] = {
                "condition": condition,
                "a": a,
                "b": b,
                "validation_observations": len(group),
                "correct": sum(bool(row["output_token_correct"]) for row in group),
                "accuracy": sum(bool(row["output_token_correct"]) for row in group) / len(group) if group else None,
                "predictions": dict(sorted(Counter(str(row["predicted_output"]) for row in group).items())),
            }
    write_json(OUTPUT / "PAIR_SUMMARY.json", pair_summary)
    summary = {
        "experiment": "crucible_digitwise_diagonal_ablation_20251025",
        "status": "completed",
        "training_performed": True,
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "initial_anchor_manifest": rel(OUTPUT / "INITIAL_ANCHORS.json"),
        "condition_count": len(VALIDATION_PAIRS),
        "arm_count": len(arms),
        "prediction_rows": len(all_rows),
        "arms": arms,
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
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
