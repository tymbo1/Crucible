#!/usr/bin/env python3
"""Add fit-gated analysis to the completed balanced holdout run.

This is read-only post-processing of the completed run. It does not load a
model, train, alter checkpoints or access any test data.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_digitwise_balanced_holdouts_20251025"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows() -> Tuple[List[Dict[str, Any]], Dict[str, bool], Dict[str, Dict[str, Any]]]:
    summary = json.loads((OUTPUT / "SUMMARY.json").read_text())
    fit_by_arm = {arm["arm_id"]: bool(arm["fit_pass"]) for arm in summary["arms"]}
    arm_by_id = {arm["arm_id"]: arm for arm in summary["arms"]}
    with (OUTPUT / "PREDICTIONS.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["fit_pass"] = fit_by_arm[row["arm_id"]]
        row["a"] = int(row["a"])
        row["b"] = int(row["b"])
        row["expected_sum"] = int(row["expected_sum"])
        for key in ("output_token_correct", "full_sequence_correct", "eos_correct", "output_format_error"):
            row[key] = row[key] == "True"
    return rows, fit_by_arm, arm_by_id


def stat(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    output_correct = sum(bool(row["output_token_correct"]) for row in rows)
    eos_correct = sum(bool(row["eos_correct"]) for row in rows)
    full_correct = sum(bool(row["full_sequence_correct"]) for row in rows)
    format_errors = sum(bool(row["output_format_error"]) for row in rows)
    return {
        "observations": n,
        "correct": output_correct,
        "accuracy": output_correct / n if n else None,
        "output_token_correct": output_correct,
        "output_token_accuracy": output_correct / n if n else None,
        "eos_correct": eos_correct,
        "eos_accuracy": eos_correct / n if n else None,
        "full_sequence_correct": full_correct,
        "full_sequence_accuracy": full_correct / n if n else None,
        "output_format_errors": format_errors,
        "output_format_error_rate": format_errors / n if n else None,
    }


def prediction_counts(rows: Sequence[Mapping[str, Any]], incorrect_only: bool = False) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        if incorrect_only and row["output_token_correct"]:
            continue
        counts[str(row["predicted_output"])] += 1
    return dict(sorted(counts.items()))


def wilson(correct: int, total: int, z: float = 1.96) -> Dict[str, Any]:
    if total == 0:
        return {"lower": None, "upper": None}
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return {"lower": max(0.0, centre - radius), "upper": min(1.0, centre + radius)}


def grouped(rows: Sequence[Mapping[str, Any]], key_fn) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    result = {}
    for key, group in sorted(groups.items(), key=lambda item: int(item[0])):
        value = stat(group)
        value["wilson_95"] = wilson(value["correct"], value["observations"])
        result[key] = value
    return result


def confusion(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labels = [str(i) for i in range(10)] + ["<non-digit>"]
    counts = {target: {predicted: 0 for predicted in labels} for target in labels}
    for row in rows:
        target = str(row["target_output"])
        predicted = row["predicted_output"] if not row["output_format_error"] else "<non-digit>"
        counts[target][predicted] += 1
    return {"labels": labels, "counts": counts}


def svg_heatmap(rows: Sequence[Mapping[str, Any]], path: Path, title: str) -> None:
    row_map = {(int(row["a"]), int(row["b"])): row for row in rows}
    cell = 46
    left = 72
    top = 72
    width = left + cell * 10 + 10
    height = top + cell * 10 + 36
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:12px}.small{font-size:9px}</style>',
        f'<text x="8" y="18" font-size="14">{html.escape(title)}</text>',
        '<text x="34" y="48">a\\b</text>',
    ]
    for b in range(10):
        parts.append(f'<text x="{left + b * cell + 18}" y="52">{b}</text>')
    for a in range(10):
        parts.append(f'<text x="48" y="{top + a * cell + 28}">{a}</text>')
        for b in range(10):
            x = left + b * cell
            y = top + a * cell
            row = row_map.get((a, b))
            if row is None:
                fill, label, split = "#eeeeee", "—", ""
            else:
                fill = "#cce8cc" if row["output_token_correct"] else "#f2c4c4"
                label = "✓" if row["output_token_correct"] else str(row["predicted_output"])
                split = "T" if row["split"] == "train" else "V"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" fill="{fill}" stroke="#777"/>')
            parts.append(f'<text x="{x + 18}" y="{y + 24}">{html.escape(label)}</text>')
            if split:
                parts.append(f'<text class="small" x="{x + 4}" y="{y + 42}">{split}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def svg_confusion(matrix: Mapping[str, Any], path: Path, title: str) -> None:
    labels = list(matrix["labels"])
    counts = matrix["counts"]
    cell = 42
    left = 90
    top = 78
    width = left + cell * len(labels) + 8
    height = top + cell * len(labels) + 28
    max_count = max(max(row.values()) for row in counts.values()) if counts else 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:11px}</style>',
        f'<text x="8" y="18" font-size="14">{html.escape(title)}</text>',
        '<text x="12" y="52">target\\pred</text>',
    ]
    for j, label in enumerate(labels):
        short = "ND" if label == "<non-digit>" else label
        parts.append(f'<text x="{left + j * cell + 12}" y="52">{html.escape(short)}</text>')
    for i, target in enumerate(labels):
        short_target = "ND" if target == "<non-digit>" else target
        parts.append(f'<text x="48" y="{top + i * cell + 25}">{html.escape(short_target)}</text>')
        for j, predicted in enumerate(labels):
            count = int(counts[target][predicted])
            intensity = int(245 - (count / max_count) * 150) if max_count else 245
            fill = f"rgb(255,{intensity},{intensity})" if count else "#ffffff"
            x = left + j * cell
            y = top + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" fill="{fill}" stroke="#777"/>')
            parts.append(f'<text x="{x + 16}" y="{y + 24}">{count}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def commutative(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compare (a,b) with (b,a), retaining each row's split membership."""
    row_map = {(int(row["a"]), int(row["b"])): row for row in rows}
    entries: List[Dict[str, Any]] = []
    for a in range(10):
        for b in range(a + 1, 10):
            if (a, b) not in row_map or (b, a) not in row_map:
                continue
            forward = row_map[(a, b)]
            reverse = row_map[(b, a)]
            f_correct = bool(forward["output_token_correct"])
            r_correct = bool(reverse["output_token_correct"])
            if f_correct and r_correct:
                category = "both_correct"
            elif f_correct:
                category = "forward_only"
            elif r_correct:
                category = "reverse_only"
            else:
                category = "neither"
            entries.append({
                "a": a,
                "b": b,
                "forward_split": forward["split"],
                "reverse_split": reverse["split"],
                "forward_prediction": forward["predicted_output"],
                "reverse_prediction": reverse["predicted_output"],
                "forward_correct": f_correct,
                "reverse_correct": r_correct,
                "category": category,
            })
    counts = defaultdict(int)
    membership = defaultdict(int)
    for entry in entries:
        counts[entry["category"]] += 1
        membership[f"{entry['forward_split']}/{entry['reverse_split']}"] += 1
    return {
        "distinct_unordered_pairs": len(entries),
        "category_counts": dict(sorted(counts.items())),
        "directional_asymmetry_pairs": counts["forward_only"] + counts["reverse_only"],
        "membership_pattern_counts": dict(sorted(membership.items())),
        "entries": entries,
    }


def main() -> int:
    rows, fit_by_arm, arm_by_id = load_rows()
    if not rows:
        raise RuntimeError("PREDICTIONS.csv is empty")

    # Preserve the complete row-level matrix, adding only the preregistered
    # training-competence status from SUMMARY.json.
    fields = list(rows[0])
    with (OUTPUT / "PREDICTIONS_WITH_FIT.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    all_validation = [row for row in rows if row["split"] == "validation"]
    fit_validation = [row for row in all_validation if row["fit_pass"]]
    all_training = [row for row in rows if row["split"] == "train"]
    fit_training = [row for row in all_training if row["fit_pass"]]

    pair_rows: Dict[str, Any] = {}
    for a in range(10):
        for b in range(10 - a):
            key = f"({a},{b})"
            pair_rows[key] = {
                "all_training": stat([row for row in all_training if row["a"] == a and row["b"] == b]),
                "fit_training": stat([row for row in fit_training if row["a"] == a and row["b"] == b]),
                "all_validation": stat([row for row in all_validation if row["a"] == a and row["b"] == b]),
                "fit_validation": stat([row for row in fit_validation if row["a"] == a and row["b"] == b]),
                "all_validation_predictions": prediction_counts([row for row in all_validation if row["a"] == a and row["b"] == b]),
                "all_validation_wrong_predictions": prediction_counts([row for row in all_validation if row["a"] == a and row["b"] == b], incorrect_only=True),
                "validation_partitions": sorted({row["partition"] for row in all_validation if row["a"] == a and row["b"] == b}),
                "training_partitions": sorted({row["partition"] for row in all_training if row["a"] == a and row["b"] == b}),
            }

    diagonal_all_validation = [row for row in all_validation if row["a"] == row["b"]]
    non_diagonal_all_validation = [row for row in all_validation if row["a"] != row["b"]]
    diagonal_fit_validation = [row for row in fit_validation if row["a"] == row["b"]]
    non_diagonal_fit_validation = [row for row in fit_validation if row["a"] != row["b"]]
    diagonal_all_training = [row for row in all_training if row["a"] == row["b"]]
    diagonal_fit_training = [row for row in fit_training if row["a"] == row["b"]]

    def with_interval(selected: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        result = stat(selected)
        result["wilson_95"] = wilson(result["output_token_correct"], result["observations"])
        return result

    # Verify the immutable row-level invariants without opening checkpoints.
    verification_errors: List[str] = []
    by_arm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm_id"]].append(row)
        if row["a"] + row["b"] >= 10:
            verification_errors.append(f"invalid_pair:{row['arm_id']}:{row['a']},{row['b']}")
        if row["target_output"] != str(row["expected_sum"]):
            verification_errors.append(f"label_mismatch:{row['record_id']}")
        if row["full_sequence_correct"] != (row["output_token_correct"] and row["eos_correct"]):
            verification_errors.append(f"evaluator_mismatch:{row['record_id']}")
        if not row["output_format_error"] and not (len(str(row["predicted_output"])) == 1 and str(row["predicted_output"]).isdigit()):
            verification_errors.append(f"decode_mismatch:{row['record_id']}")
    for arm_id, arm_rows in sorted(by_arm.items()):
        if len(arm_rows) != 55:
            verification_errors.append(f"arm_row_count:{arm_id}:{len(arm_rows)}")
        split_counts = {split: sum(row["split"] == split for row in arm_rows) for split in ("train", "validation")}
        if split_counts != {"train": 40, "validation": 15}:
            verification_errors.append(f"split_counts:{arm_id}:{split_counts}")
        train_pairs = {(row["a"], row["b"]) for row in arm_rows if row["split"] == "train"}
        validation_pairs = {(row["a"], row["b"]) for row in arm_rows if row["split"] == "validation"}
        if train_pairs & validation_pairs:
            verification_errors.append(f"split_overlap:{arm_id}")
        if len({row["checkpoint_sha256"] for row in arm_rows}) != 1:
            verification_errors.append(f"checkpoint_identity:{arm_id}")

    summary = json.loads((OUTPUT / "SUMMARY.json").read_text(encoding="utf-8"))
    preflight = json.loads((OUTPUT / "PREFLIGHT.json").read_text(encoding="utf-8"))
    partition_audit = json.loads((OUTPUT / "PARTITION_AUDIT.json").read_text(encoding="utf-8"))
    if summary.get("sealed_test_accessed") is not False:
        verification_errors.append("summary_sealed_test_accessed")
    if summary.get("architecture_changed") is not False:
        verification_errors.append("summary_architecture_changed")
    if preflight.get("pass") is not True:
        verification_errors.append("preflight_failed")
    audited_partitions = partition_audit.get("preflight", {}).get("partitions", {})
    if not audited_partitions or any(not value.get("pass", False) for value in audited_partitions.values()):
        verification_errors.append("partition_audit_failed")

    reported_metric_errors: List[str] = []
    for arm_id, arm_rows in sorted(by_arm.items()):
        arm_meta = arm_by_id[arm_id]
        result_path = OUTPUT / "arm_outputs" / arm_meta["partition"] / arm_meta["interface"] / f"seed{arm_meta['seed']}" / "single_digit_no_carry" / "RESULTS.json"
        if not result_path.exists():
            reported_metric_errors.append(f"missing_results:{arm_id}")
            continue
        recorded = json.loads(result_path.read_text(encoding="utf-8"))
        for split in ("train", "validation"):
            observed = stat([row for row in arm_rows if row["split"] == split])["full_sequence_accuracy"]
            expected = recorded[split]["aggregate"]["teacher_forced_exact_accuracy"]
            if not math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12):
                reported_metric_errors.append(f"reported_metric:{arm_id}:{split}:{observed}:{expected}")
    verification_errors.extend(reported_metric_errors)

    arm_forensics: Dict[str, Any] = {}
    for arm_id, arm_rows in sorted(by_arm.items()):
        split_rows = {
            "all": arm_rows,
            "train": [row for row in arm_rows if row["split"] == "train"],
            "validation": [row for row in arm_rows if row["split"] == "validation"],
        }
        arm_forensics[arm_id] = {
            "arm_id": arm_id,
            "fit_pass": fit_by_arm[arm_id],
            "run_summary": arm_by_id[arm_id],
            "all": stat(split_rows["all"]),
            "train": stat(split_rows["train"]),
            "validation": stat(split_rows["validation"]),
            "confusion": {name: confusion(selected) for name, selected in split_rows.items()},
            "commutativity": {
                "all": commutative(split_rows["all"]),
                "validation_only": commutative(split_rows["validation"]),
            },
        }
        base = OUTPUT / "forensics" / "arm_outputs"
        write_json(base / f"{arm_id}.json", arm_forensics[arm_id])
        for label, selected in split_rows.items():
            svg_heatmap(selected, OUTPUT / "forensics" / "plots" / f"heatmap_{arm_id}_{label}.svg", f"{arm_id} — {label} operand pairs")
            svg_confusion(arm_forensics[arm_id]["confusion"][label], OUTPUT / "forensics" / "plots" / f"confusion_{arm_id}_{label}.svg", f"{arm_id} — {label} target/predicted output")

    def arm_group_stats(selected_arm_ids: Sequence[str]) -> Dict[str, Any]:
        values = [arm_forensics[arm_id]["validation"]["output_token_accuracy"] for arm_id in selected_arm_ids]
        return {
            "arms": list(selected_arm_ids),
            "count": len(values),
            "mean_validation_accuracy": statistics.mean(values) if values else None,
            "min_validation_accuracy": min(values) if values else None,
            "max_validation_accuracy": max(values) if values else None,
            "population_stdev_validation_accuracy": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
        }

    by_interface_seed: Dict[str, Any] = {}
    for interface in ("canonical", "factorized"):
        for seed in (42, 1337):
            ids = [arm_id for arm_id in sorted(arm_forensics) if arm_by_id[arm_id]["interface"] == interface and int(arm_by_id[arm_id]["seed"]) == seed]
            fit_ids = [arm_id for arm_id in ids if fit_by_arm[arm_id]]
            by_interface_seed[f"{interface}_seed{seed}"] = {
                "all": arm_group_stats(ids),
                "fit_gated": arm_group_stats(fit_ids),
            }

    diagnostic = {
        "experiment": "crucible_digitwise_balanced_holdouts_20251025",
        "postprocessing_training_performed": False,
        "run_training_performed": bool(summary.get("training_performed")),
        "sealed_test_accessed": False,
        "fit_pass_arms": sum(fit_by_arm.values()),
        "total_arms": len(fit_by_arm),
        "failed_fit_arms": [arm_id for arm_id, passed in sorted(fit_by_arm.items()) if not passed],
        "all_validation": with_interval(all_validation),
        "fit_validation": with_interval(fit_validation),
        "all_training": with_interval(all_training),
        "fit_training": with_interval(fit_training),
        "diagonal": {
            "all_validation": with_interval(diagonal_all_validation),
            "fit_validation": with_interval(diagonal_fit_validation),
            "all_training": with_interval(diagonal_all_training),
            "fit_training": with_interval(diagonal_fit_training),
        },
        "non_diagonal": {
            "all_validation": with_interval(non_diagonal_all_validation),
            "fit_validation": with_interval(non_diagonal_fit_validation),
        },
        "by_sum": {
            "all_validation": grouped(all_validation, lambda row: row["expected_sum"]),
            "fit_validation": grouped(fit_validation, lambda row: row["expected_sum"]),
            "all_training": grouped(all_training, lambda row: row["expected_sum"]),
            "fit_training": grouped(fit_training, lambda row: row["expected_sum"]),
        },
        "by_first_operand": {
            "all_validation": grouped(all_validation, lambda row: row["a"]),
            "fit_validation": grouped(fit_validation, lambda row: row["a"]),
            "all_training": grouped(all_training, lambda row: row["a"]),
            "fit_training": grouped(fit_training, lambda row: row["a"]),
        },
        "by_second_operand": {
            "all_validation": grouped(all_validation, lambda row: row["b"]),
            "fit_validation": grouped(fit_validation, lambda row: row["b"]),
            "all_training": grouped(all_training, lambda row: row["b"]),
            "fit_training": grouped(fit_training, lambda row: row["b"]),
        },
        "by_absolute_difference": {
            "all_validation": grouped(all_validation, lambda row: abs(row["a"] - row["b"])),
            "fit_validation": grouped(fit_validation, lambda row: abs(row["a"] - row["b"])),
            "all_training": grouped(all_training, lambda row: abs(row["a"] - row["b"])),
            "fit_training": grouped(fit_training, lambda row: abs(row["a"] - row["b"])),
        },
        "pair_summary": pair_rows,
        "by_interface_seed": by_interface_seed,
        "arm_forensics": arm_forensics,
        "verification": {
            "pass": not verification_errors,
            "errors": verification_errors,
            "rows": len(rows),
            "arms": len(by_arm),
            "reported_metrics_reproduced": not reported_metric_errors,
        },
    }
    write_json(OUTPUT / "ANALYSIS_FINAL.json", diagnostic)

    prior = {"(1,5)", "(2,2)", "(3,3)", "(5,1)"}
    diagonal_all = diagnostic["diagonal"]["all_validation"]
    diagonal_fit = diagnostic["diagonal"]["fit_validation"]
    non_diagonal_all = diagnostic["non_diagonal"]["all_validation"]
    non_diagonal_fit = diagnostic["non_diagonal"]["fit_validation"]
    report = [
        "# Crucible Balanced Holdouts — Final Analysis",
        "",
        "This is read-only post-processing of the completed 16-arm CPU run. No model was retrained and no sealed test corpus was accessed during finalization.",
        "",
        f"Fit gate: {diagnostic['fit_pass_arms']}/{diagnostic['total_arms']} arms passed the preregistered 95% training exact-accuracy threshold. The failed arm is `{diagnostic['failed_fit_arms'][0]}`; its rows remain in the raw matrix but are excluded from fit-gated transfer claims.",
        "",
        "## Main result",
        "",
        f"All validation observations: diagonal/equal-operand pairs {diagonal_all['correct']}/{diagonal_all['observations']} ({diagonal_all['accuracy']:.2%}); non-diagonal pairs {non_diagonal_all['correct']}/{non_diagonal_all['observations']} ({non_diagonal_all['accuracy']:.2%}). Fit-gated validation: diagonal {diagonal_fit['correct']}/{diagonal_fit['observations']} ({diagonal_fit['accuracy']:.2%}); non-diagonal {non_diagonal_fit['correct']}/{non_diagonal_fit['observations']} ({non_diagonal_fit['accuracy']:.2%}).",
        "",
        f"Every held-out diagonal observation was wrong ({diagonal_all['correct']}/{diagonal_all['observations']}), while trained diagonal observations were {diagnostic['diagonal']['all_training']['correct']}/{diagnostic['diagonal']['all_training']['observations']} correct. The fit-gated training diagonal result is {diagnostic['diagonal']['fit_training']['correct']}/{diagnostic['diagonal']['fit_training']['observations']}. This supports a held-out diagonal transfer problem under these splits, not an inability to fit equal-operand examples.",
        "",
        "## Four original failures under rotated holdouts",
        "",
        "| Pair | Training correct | Validation correct | Interpretation |",
        "|---|---:|---:|---|",
    ]
    interpretations = {
        "(1,5)": "moves with split; partly recovers when held out",
        "(2,2)": "wrong whenever held out; correct when trained",
        "(3,3)": "wrong whenever held out; usually correct when trained",
        "(5,1)": "moves with split; partly recovers when held out",
    }
    for key in sorted(prior):
        item = pair_rows[key]
        report.append(f"| `{key}` | {item['all_training']['correct']}/{item['all_training']['observations']} | {item['all_validation']['correct']}/{item['all_validation']['observations']} | {interpretations[key]} |")
    report += [
        "",
        "Validation prediction distributions for those four pairs were: `(1,5)` → `5:1, 6:3, 7:4`; `(2,2)` → `2:4, 5:2, 6:2`; `(3,3)` → `5:4, 7:3, 8:1`; and `(5,1)` → `5:1, 6:4, 7:3`. Only the `6` predictions are correct for the two sum-six pairs; `(2,2)` and `(3,3)` had no correct validation output. The complete distributions, including training rows, are in the machine-readable analysis.",
        "",
        "The original fixed-split four-pair pattern therefore does not survive as a fixed-pair phenomenon. The stable result is the diagonal/equal-operand held-out failure.",
        "",
        "## Other requested diagnostics",
        "",
        f"- **Output format and positional errors:** no output-format errors occurred. EOS was correct on {diagnostic['all_validation']['eos_correct']}/{diagnostic['all_validation']['observations']} validation rows and {diagnostic['all_training']['eos_correct']}/{diagnostic['all_training']['observations']} training rows. Full-sequence accuracy is recorded separately and matches output-token accuracy here because the few EOS errors coincide with wrong output tokens.",
        "- **Operand order:** per-arm commutative counterpart entries retain forward/reverse predictions and train/validation membership. Because the rotated partitions do not make every counterpart a matched validation pair, this study does not establish a broad order-sensitivity mechanism.",
        "- **Sum magnitude, operand identity and difference:** rates with denominators and Wilson intervals are in [ANALYSIS_FINAL.json](ANALYSIS_FINAL.json). They are descriptive; sum, diagonal status and holdout membership are not independently balanced enough here to support a causal sum- or operand-specific claim beyond the diagonal result.",
        "- **Split/evaluator integrity:** the final verification checks row counts, 40/15 membership, no within-arm pair overlap, labels, decoding relation, checkpoint identity, preflight status and the run's sealed-test/architecture flags.",
        "",
        "## Conclusions",
        "",
        "1. Errors move substantially with the train/validation partition; the original four failures are not intrinsic fixed-pair failures.",
        "2. Equal-operand transfer is the clearest reproducible weakness: every held-out diagonal observation failed, while trained diagonal observations were mostly correct.",
        "3. One arm failed the training-competence gate and is excluded from clean transfer claims.",
        "4. No embedding collision, attention interference or architectural ceiling is established by these behavioral results.",
        "",
        "## Single most informative next controlled experiment",
        "",
        "Run a targeted diagonal-ablation study: compare equal-operand holdouts against matched non-equal holdouts with matched sums, operand frequencies, split sizes and fresh seeds, using the same evaluator. Inspect activations only if that behavioral contrast replicates. This directly tests whether the diagonal effect is a structural class-coverage issue rather than a consequence of the particular four partitions.",
        "",
        "## Artifacts",
        "",
        "- [Final analysis JSON](ANALYSIS_FINAL.json)",
        "- [Predictions with fit status](PREDICTIONS_WITH_FIT.csv)",
        "- [Per-arm JSON/confusion matrices](forensics/arm_outputs/)",
        "- [Per-arm operand heatmaps and confusion SVGs](forensics/plots/)",
        "- [Final artifact hashes](FINAL_HASHES.json)",
        "- [Raw run report](REPORT.md)",
        "- [Frozen partitions](PARTITIONS.json)",
        "- [Preflight](PREFLIGHT.json)",
    ]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUTPUT / "FINALIZATION.json", {
        "status": "completed",
        "script": "tools/finalize_crucible_balanced_holdouts_20251025.py",
        "analysis_sha256": sha256(OUTPUT / "ANALYSIS_FINAL.json"),
        "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"),
        "prediction_matrix_sha256": sha256(OUTPUT / "PREDICTIONS_WITH_FIT.csv"),
        "plots": 96,
        "verification_pass": diagnostic["verification"]["pass"],
    })
    final_hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "FINAL_HASHES.json":
            final_hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "FINAL_HASHES.json", final_hashes)
    print(json.dumps({"status": "finalized", "fit_pass_arms": diagnostic["fit_pass_arms"], "failed_fit_arms": diagnostic["failed_fit_arms"], "verification_pass": diagnostic["verification"]["pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
