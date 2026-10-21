#!/usr/bin/env python3
"""Audit simple non-reasoning shortcuts in cognitive task manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Mapping, Sequence

from validate_cognitive_manifests import load_records


SPLITS = ("train", "validation", "test")


def class_label(record: Mapping[str, Any]) -> str:
    family = record["family"]
    target = record["target"]
    if family == "error_correction":
        return target.split("|", 1)[0]
    return target


def accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    if not targets:
        return 0.0
    return sum(prediction == target for prediction, target in zip(predictions, targets)) / len(targets)


def majority_predictor(
    train: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
) -> float:
    majority = Counter(class_label(record) for record in train).most_common(1)[0][0]
    return accuracy([majority] * len(evaluation), [class_label(record) for record in evaluation])


def length_bucket_predictor(
    train: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
) -> float:
    by_bucket: Dict[int, Counter[str]] = defaultdict(Counter)
    overall = Counter(class_label(record) for record in train)
    for record in train:
        by_bucket[len(record["input"]) // 8][class_label(record)] += 1
    predictions = []
    for record in evaluation:
        bucket = by_bucket.get(len(record["input"]) // 8, overall)
        predictions.append(bucket.most_common(1)[0][0])
    return accuracy(predictions, [class_label(record) for record in evaluation])


def audit(records_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    families = sorted({record["family"] for rows in records_by_split.values() for record in rows})
    report: Dict[str, Any] = {"families": {}}
    for family in families:
        train = [record for record in records_by_split["train"] if record["family"] == family]
        validation = [record for record in records_by_split["validation"] if record["family"] == family]
        test = [record for record in records_by_split["test"] if record["family"] == family]
        lengths = [len(record["input"]) for record in train + validation + test]
        target_counts = {
            split: dict(Counter(class_label(record) for record in records_by_split[split] if record["family"] == family))
            for split in SPLITS
        }
        report["families"][family] = {
            "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
            "target_distribution": target_counts,
            "prompt_length": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": mean(lengths),
                "population_stddev": pstdev(lengths),
            },
            "majority_baseline_accuracy": {
                "validation": majority_predictor(train, validation),
                "test": majority_predictor(train, test),
            },
            "length_bucket_baseline_accuracy": {
                "validation": length_bucket_predictor(train, validation),
                "test": length_bucket_predictor(train, test),
            },
        }
    report["interpretation"] = (
        "These are cheap baselines, not evidence of task solvability. A high "
        "non-reasoning baseline requires task redesign before model training."
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(load_records(args.manifests))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
