#!/usr/bin/env python3
"""Read-only forensic finalization for the matched diagonal-ablation run."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_digitwise_diagonal_ablation_20251025_v2"
CONDITIONS = ("diagonal", "unequal_matched", "mixed")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load() -> tuple[List[Dict[str, Any]], Dict[str, bool], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    summary = json.loads((OUTPUT / "SUMMARY.json").read_text(encoding="utf-8"))
    fit_by_arm = {arm["arm_id"]: bool(arm["fit_pass"]) for arm in summary["arms"]}
    arm_by_id = {arm["arm_id"]: arm for arm in summary["arms"]}
    with (OUTPUT / "PREDICTIONS.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["fit_pass"] = fit_by_arm[row["arm_id"]]
        for key in ("a", "b", "expected_sum"):
            row[key] = int(row[key])
        for key in ("output_format_error", "output_token_correct", "eos_correct", "full_sequence_correct"):
            row[key] = row[key] == "True"
    return rows, fit_by_arm, arm_by_id, summary


def stat(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    output_correct = sum(bool(row["output_token_correct"]) for row in rows)
    eos_correct = sum(bool(row["eos_correct"]) for row in rows)
    full_correct = sum(bool(row["full_sequence_correct"]) for row in rows)
    format_errors = sum(bool(row["output_format_error"]) for row in rows)
    return {
        "observations": n,
        "output_token_correct": output_correct,
        "output_token_accuracy": output_correct / n if n else None,
        "eos_correct": eos_correct,
        "eos_accuracy": eos_correct / n if n else None,
        "full_sequence_correct": full_correct,
        "full_sequence_accuracy": full_correct / n if n else None,
        "output_format_errors": format_errors,
        "output_format_error_rate": format_errors / n if n else None,
    }


def wilson(correct: int, total: int, z: float = 1.96) -> Dict[str, Any]:
    if not total:
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
    output: Dict[str, Any] = {}
    for key, values in sorted(groups.items(), key=lambda item: int(item[0])):
        item = stat(values)
        item["wilson_95"] = wilson(item["output_token_correct"], item["observations"])
        output[key] = item
    return output


def confusion(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labels = [str(i) for i in range(10)] + ["<non-digit>"]
    counts = {target: {predicted: 0 for predicted in labels} for target in labels}
    for row in rows:
        target = str(row["target_output"])
        predicted = "<non-digit>" if row["output_format_error"] else str(row["predicted_output"])
        counts[target][predicted] += 1
    return {"labels": labels, "counts": counts}


def commutative(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairs = {(row["a"], row["b"]): row for row in rows}
    entries: List[Dict[str, Any]] = []
    for a in range(10):
        for b in range(a + 1, 10):
            if (a, b) not in pairs or (b, a) not in pairs:
                continue
            forward, reverse = pairs[(a, b)], pairs[(b, a)]
            f, r = bool(forward["output_token_correct"]), bool(reverse["output_token_correct"])
            category = "both_correct" if f and r else "forward_only" if f else "reverse_only" if r else "neither"
            entries.append({
                "a": a,
                "b": b,
                "forward_split": forward["split"],
                "reverse_split": reverse["split"],
                "forward_prediction": forward["predicted_output"],
                "reverse_prediction": reverse["predicted_output"],
                "forward_correct": f,
                "reverse_correct": r,
                "category": category,
            })
    return {"entries": entries, "category_counts": dict(Counter(entry["category"] for entry in entries))}


def svg_heatmap(rows: Sequence[Mapping[str, Any]], path: Path, title: str) -> None:
    row_map = {(row["a"], row["b"]): row for row in rows}
    cell, left, top = 46, 72, 72
    width, height = left + cell * 10 + 10, top + cell * 10 + 36
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<style>text{font-family:monospace;font-size:12px}.small{font-size:9px}</style>', f'<text x="8" y="18" font-size="14">{html.escape(title)}</text>', '<text x="34" y="48">a\\b</text>']
    for b in range(10):
        parts.append(f'<text x="{left + b * cell + 18}" y="52">{b}</text>')
    for a in range(10):
        parts.append(f'<text x="48" y="{top + a * cell + 28}">{a}</text>')
        for b in range(10):
            x, y = left + b * cell, top + a * cell
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
    labels, counts = matrix["labels"], matrix["counts"]
    cell, left, top = 42, 90, 78
    width, height = left + cell * len(labels) + 8, top + cell * len(labels) + 28
    maximum = max(max(row.values()) for row in counts.values()) if counts else 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<style>text{font-family:monospace;font-size:11px}</style>', f'<text x="8" y="18" font-size="14">{html.escape(title)}</text>', '<text x="12" y="52">target\\pred</text>']
    for j, label in enumerate(labels):
        parts.append(f'<text x="{left + j * cell + 12}" y="52">{html.escape("ND" if label == "<non-digit>" else label)}</text>')
    for i, target in enumerate(labels):
        parts.append(f'<text x="48" y="{top + i * cell + 25}">{html.escape("ND" if target == "<non-digit>" else target)}</text>')
        for j, predicted in enumerate(labels):
            count = int(counts[target][predicted])
            intensity = int(245 - (count / maximum) * 150) if maximum else 245
            x, y = left + j * cell, top + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" fill="{f"rgb(255,{intensity},{intensity})" if count else "#ffffff"}" stroke="#777"/>')
            parts.append(f'<text x="{x + 16}" y="{y + 24}">{count}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def main() -> int:
    rows, fit_by_arm, arm_by_id, summary = load()
    if not rows:
        raise RuntimeError("empty prediction matrix")
    conditions = tuple(sorted({arm["condition"] for arm in summary["arms"]}))
    with (OUTPUT / "PREDICTIONS_WITH_FIT.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_arm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm_id"]].append(row)
    all_validation = [row for row in rows if row["split"] == "validation"]
    fit_validation = [row for row in all_validation if row["fit_pass"]]
    all_training = [row for row in rows if row["split"] == "train"]
    fit_training = [row for row in all_training if row["fit_pass"]]
    errors: List[str] = []
    for arm_id, arm_rows in sorted(by_arm.items()):
        if len(arm_rows) != 55:
            errors.append(f"row_count:{arm_id}")
        train = {(row["a"], row["b"]) for row in arm_rows if row["split"] == "train"}
        validation = {(row["a"], row["b"]) for row in arm_rows if row["split"] == "validation"}
        if len(train) != 40 or len(validation) != 15 or train & validation:
            errors.append(f"split:{arm_id}")
        if any(row["target_output"] != str(row["expected_sum"]) for row in arm_rows):
            errors.append(f"label:{arm_id}")
        if any(row["full_sequence_correct"] != (row["output_token_correct"] and row["eos_correct"]) for row in arm_rows):
            errors.append(f"evaluator:{arm_id}")
        if len({row["checkpoint_sha256"] for row in arm_rows}) != 1:
            errors.append(f"checkpoint:{arm_id}")
        for split in ("train", "validation"):
            result_path = OUTPUT / "arm_outputs" / arm_by_id[arm_id]["condition"] / arm_by_id[arm_id]["interface"] / f"seed{arm_by_id[arm_id]['seed']}" / "single_digit_no_carry" / "RESULTS.json"
            if not result_path.exists():
                result_path = OUTPUT / "arm_outputs" / arm_by_id[arm_id]["condition"] / arm_by_id[arm_id]["interface"] / f"seed{arm_by_id[arm_id]['seed']}" / "addition" / "RESULTS.json"
            recorded = json.loads(result_path.read_text(encoding="utf-8"))
            observed = stat([row for row in arm_rows if row["split"] == split])["full_sequence_accuracy"]
            expected = recorded[split]["aggregate"]["teacher_forced_exact_accuracy"]
            if not math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12):
                errors.append(f"reported_metric:{arm_id}:{split}")

    preflight = json.loads((OUTPUT / "PREFLIGHT.json").read_text(encoding="utf-8"))
    if preflight.get("pass") is not True:
        errors.append("preflight")

    arm_forensics: Dict[str, Any] = {}
    for arm_id, arm_rows in sorted(by_arm.items()):
        split_rows = {name: [row for row in arm_rows if name == "all" or row["split"] == name] for name in ("all", "train", "validation")}
        arm_forensics[arm_id] = {
            "arm_id": arm_id,
            "condition": arm_by_id[arm_id]["condition"],
            "interface": arm_by_id[arm_id]["interface"],
            "seed": arm_by_id[arm_id]["seed"],
            "fit_pass": fit_by_arm[arm_id],
            "summary": arm_by_id[arm_id],
            "all": stat(split_rows["all"]),
            "train": stat(split_rows["train"]),
            "validation": stat(split_rows["validation"]),
            "confusion": {name: confusion(selected) for name, selected in split_rows.items()},
            "commutativity": commutative(split_rows["all"]),
        }
        write_json(OUTPUT / "forensics" / "arm_outputs" / f"{arm_id}.json", arm_forensics[arm_id])
        for name, selected in split_rows.items():
            svg_heatmap(selected, OUTPUT / "forensics" / "plots" / f"heatmap_{arm_id}_{name}.svg", f"{arm_id} — {name} operand pairs")
            svg_confusion(arm_forensics[arm_id]["confusion"][name], OUTPUT / "forensics" / "plots" / f"confusion_{arm_id}_{name}.svg", f"{arm_id} — {name} target/predicted output")

    condition_special = {
        condition: {tuple(pair) for pair in preflight["special_pairs"][condition]}
        for condition in conditions
    }
    pair_summary: Dict[str, Any] = {}
    condition_summary: Dict[str, Any] = {}
    for condition in conditions:
        validation = [row for row in all_validation if row["condition"] == condition]
        fit_val = [row for row in fit_validation if row["condition"] == condition]
        special = [row for row in validation if (row["a"], row["b"]) in condition_special[condition]]
        special_fit = [row for row in fit_val if (row["a"], row["b"]) in condition_special[condition]]
        diagonal = [row for row in validation if row["a"] == row["b"]]
        unequal = [row for row in validation if row["a"] != row["b"]]
        condition_summary[condition] = {
            "all_validation": stat(validation),
            "fit_validation": stat(fit_val),
            "special_validation": stat(special),
            "special_fit_validation": stat(special_fit),
            "diagonal_validation": stat(diagonal),
            "unequal_validation": stat(unequal),
            "diagonal_fit_validation": stat([row for row in fit_val if row["a"] == row["b"]]),
            "unequal_fit_validation": stat([row for row in fit_val if row["a"] != row["b"]]),
            "by_sum": grouped(validation, lambda row: row["expected_sum"]),
            "by_first_operand": grouped(validation, lambda row: row["a"]),
            "by_second_operand": grouped(validation, lambda row: row["b"]),
            "by_absolute_difference": grouped(validation, lambda row: abs(row["a"] - row["b"])),
            "arm_ids": sorted(arm_id for arm_id, arm in arm_by_id.items() if arm["condition"] == condition),
        }
        for a in range(10):
            for b in range(10 - a):
                selected = [row for row in validation if row["a"] == a and row["b"] == b]
                pair_summary[f"{condition}|({a},{b})"] = {
                    "condition": condition,
                    "a": a,
                    "b": b,
                    "validation": stat(selected),
                    "predictions": dict(sorted(Counter(str(row["predicted_output"]) for row in selected).items())),
                    "validation_partitions": sorted({row["partition"] for row in selected}),
                }

    interface_seed: Dict[str, Any] = {}
    for condition in conditions:
        for interface in ("canonical", "factorized"):
            for seed in (2025, 2026):
                arm_ids = [arm_id for arm_id, arm in arm_by_id.items() if arm["condition"] == condition and arm["interface"] == interface and int(arm["seed"]) == seed]
                values = [arm_forensics[arm_id]["validation"]["full_sequence_accuracy"] for arm_id in arm_ids]
                interface_seed[f"{condition}_{interface}_seed{seed}"] = {"arm_ids": arm_ids, "validation_accuracies": values}

    analysis = {
        "experiment": summary.get("experiment", "unknown"),
        "training_performed": False,
        "run_training_performed": bool(summary.get("training_performed")),
        "sealed_test_accessed": False,
        "fit_pass_arms": sum(fit_by_arm.values()),
        "total_arms": len(fit_by_arm),
        "failed_fit_arms": [arm_id for arm_id, passed in sorted(fit_by_arm.items()) if not passed],
        "all_validation": stat(all_validation),
        "fit_validation": stat(fit_validation),
        "all_training": stat(all_training),
        "fit_training": stat(fit_training),
        "condition_summary": condition_summary,
        "pair_summary": pair_summary,
        "interface_seed": interface_seed,
        "arm_forensics": arm_forensics,
        "verification": {"pass": not errors, "errors": errors, "rows": len(rows), "arms": len(by_arm), "reported_metrics_reproduced": not any(error.startswith("reported_metric:") for error in errors)},
    }
    write_json(OUTPUT / "ANALYSIS_FINAL.json", analysis)

    if "unequal_role_swapped" in conditions:
        matching_note = "The diagonal and role-swapped unequal special conditions use the same special sum values and the same second-operand frequencies. The first-operand frequency difference is the explicit complementary limitation."
        mixed_note = "There is no mixed condition in this role-swapped control."
    else:
        matching_note = "The diagonal and unequal special conditions use the same special sum values and the same first-operand frequencies. The protocol records the unavoidable second-operand-frequency difference; therefore this is a controlled but not perfect matching."
        mixed_note = "The mixed condition tests whether a one-diagonal/one-unequal special holdout behaves between the two extremes."
    report = [
        "# Crucible Matched Diagonal Ablation — Final Analysis",
        "",
        "This is read-only post-processing. No model was retrained during finalization and no sealed test data was accessed.",
        "",
        f"Fit gate: {analysis['fit_pass_arms']}/{analysis['total_arms']} arms passed. Verification: {'PASS' if analysis['verification']['pass'] else 'FAIL'}.",
        "",
        "There are four independently initialized arms per condition (two interfaces × two seeds). Individual prediction rows within an arm are correlated, so pooled row counts are descriptive rather than independent-trial estimates.",
        "",
        "## Condition results",
        "",
        "| Condition | Special holdout accuracy | All validation accuracy | Diagonal validation | Unequal validation |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        item = condition_summary[condition]
        report.append(f"| {condition} | {item['special_validation']['output_token_correct']}/{item['special_validation']['observations']} ({item['special_validation']['output_token_accuracy']:.2%}) | {item['all_validation']['output_token_correct']}/{item['all_validation']['observations']} ({item['all_validation']['output_token_accuracy']:.2%}) | {item['diagonal_validation']['output_token_correct']}/{item['diagonal_validation']['observations']} | {item['unequal_validation']['output_token_correct']}/{item['unequal_validation']['observations']} |")
    report += [
        "",
        matching_note,
        "",
        "## Conclusions",
        "",
        "- The diagonal condition's special held-out pairs are compared directly with the matched unequal special pairs, rather than with the entire validation set alone.",
        f"- {mixed_note}",
        "- Output-token, full-sequence, EOS, operand, sum, difference, pair and commutativity diagnostics are in `ANALYSIS_FINAL.json`.",
        "- No result by itself establishes embedding collision, attention interference or an architectural ceiling.",
        "",
        "## Artifacts",
        "",
        "- [Final analysis](ANALYSIS_FINAL.json)",
        "- [Final prediction matrix](PREDICTIONS_WITH_FIT.csv)",
        "- [Per-arm JSON/confusion data](forensics/arm_outputs/)",
        "- [Per-arm heatmaps and confusion SVGs](forensics/plots/)",
        "- [Raw report](REPORT.md)",
        "- [Preflight](PREFLIGHT.json)",
        "- [Initial anchors](INITIAL_ANCHORS.json)",
    ]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"FINAL_HASHES.json"}:
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "FINAL_HASHES.json", hashes)
    write_json(OUTPUT / "FINALIZATION.json", {"status": "completed", "verification_pass": analysis["verification"]["pass"], "analysis_sha256": sha256(OUTPUT / "ANALYSIS_FINAL.json"), "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"), "prediction_matrix_sha256": sha256(OUTPUT / "PREDICTIONS_WITH_FIT.csv"), "plot_count": len(list((OUTPUT / "forensics" / "plots").glob("*.svg")))})
    print(json.dumps({"status": "finalized", "fit_pass_arms": analysis["fit_pass_arms"], "verification_pass": analysis["verification"]["pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
