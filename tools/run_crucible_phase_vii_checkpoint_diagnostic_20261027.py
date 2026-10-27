#!/usr/bin/env python3
"""Read-only forensic comparison of the Phase VII and N0 checkpoints.

This diagnostic never calls an optimizer or changes a checkpoint. It replays
the committed final checkpoints, verifies their predictions against the
committed CSVs, compares error strata and candidate-order behavior, and
summarizes the final hidden representations. The centroid probe is a
deterministic descriptive probe fit on training representations only; it is
not a trained Crucible model or a new scientific arm.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
N0_ROOT = ROOT / "experiments" / "crucible_two_candidate_topology_duration_extension_20261027_v1"
PHASE_ROOT = ROOT / "experiments" / "crucible_phase_vii_neural_shared_latent_relational_transfer_20261027_v1"
N0_RUNNER = ROOT / "tools" / "run_crucible_two_candidate_topology_control_20261027.py"
PHASE_RUNNER = ROOT / "tools" / "run_crucible_phase_vii_neural_shared_latent_20261027.py"
OUTPUT = PHASE_ROOT
CONDITIONS = ("P12_PRISM", "R12_DIAMETER5")
SEEDS = (42, 1337)
VARIANTS = ("N0_SHARED_SCORER", "N1_SHARED_LATENT", "N2_SHARED_LATENT_CYCLE")
MODULUS = 5
BATCH_SIZE = 12


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


topology = import_module("phase_vii_diagnostic_topology", N0_RUNNER)
phase = import_module("phase_vii_diagnostic_shared_latent", PHASE_RUNNER)
torch = phase.torch
F = torch.nn.functional


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_records(condition: str, split: str) -> List[Dict[str, Any]]:
    path = topology.OUTPUT / "data" / f"{condition}_{split}.jsonl"
    return topology.load_jsonl(path)


def load_model(variant: str, condition: str, seed: int, graph: Any) -> Tuple[Any, Path]:
    if variant == "N0_SHARED_SCORER":
        checkpoint = N0_ROOT / f"{condition}_SEED{seed}" / "checkpoint.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = topology.source.CandidateReadout(seed, "SHARED_SCORER")
    else:
        checkpoint = PHASE_ROOT / f"{variant}_{condition}_SEED{seed}" / "checkpoint.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = phase.SharedLatentRelationalModel(graph, variant, seed)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, checkpoint


def batch_for(variant: str, prepared: Sequence[Mapping[str, Any]], indices: Sequence[int], graph: Any) -> Dict[str, Any]:
    if variant == "N0_SHARED_SCORER":
        return topology.batchify(prepared, indices)
    return phase.batchify(prepared, indices, graph)


def base_row(record: Mapping[str, Any], predicted_slot: int, predicted_value: int) -> Dict[str, Any]:
    metadata = record["metadata"]
    return {
        "record_id": str(record["record_id"]),
        "condition": str(record.get("condition", metadata["condition"])),
        "split": str(record["split"]),
        "counterfactual_pair_id": str(metadata["counterfactual_pair_id"]),
        "source_node": int(metadata["source_node"]),
        "target_node": int(metadata["target_node"]),
        "unordered_pair": ",".join(map(str, metadata["unordered_pair"])),
        "query_direction": str(metadata["query_direction"]),
        "graph_distance": int(metadata["graph_distance"]),
        "candidates": ",".join(map(str, metadata["candidates"])),
        "target_value": int(metadata["target_value"]),
        "predicted_value": None if predicted_value is None else int(predicted_value),
        "target_slot": int(metadata["target_slot"]),
        "predicted_slot": int(predicted_slot),
        "value_correct": predicted_value is not None and int(predicted_value) == int(metadata["target_value"]),
        "slot_correct": int(predicted_slot) == int(record["metadata"]["target_slot"]),
    }


@torch.inference_mode()
def replay(
    model: Any,
    variant: str,
    condition: str,
    split: str,
    records: Sequence[Mapping[str, Any]],
    prepared: Sequence[Mapping[str, Any]],
    graph: Any,
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    rows: List[Dict[str, Any]] = []
    features: List[torch.Tensor] = []
    for start in range(0, len(prepared), BATCH_SIZE):
        indices = list(range(start, min(start + BATCH_SIZE, len(prepared))))
        batch = batch_for(variant, prepared, indices, graph)
        if variant == "N0_SHARED_SCORER":
            hidden = model.hidden_sequence(batch["x"])
            row_index = torch.arange(hidden.size(0), dtype=torch.long)
            feature = hidden[row_index, batch["decision_pos"]]
            logits = model.decision_logits(batch)
        else:
            hidden = model.encode(batch)
            row_index = torch.arange(hidden.size(0), dtype=torch.long)
            source_hidden = hidden[row_index, batch["source"]]
            target_hidden = hidden[row_index, batch["target"]]
            feature = target_hidden - source_hidden
            query_logits = model.relation_logits(source_hidden, target_hidden)
            logits = query_logits.gather(1, batch["candidates"])
        predicted_slots = logits.argmax(dim=-1).tolist()
        features.append(feature.detach().cpu())
        for local, index in enumerate(indices):
            record = records[index]
            candidates = [int(value) for value in record["metadata"]["candidates"]]
            slot = int(predicted_slots[local])
            predicted_value = candidates[slot] if 0 <= slot < len(candidates) else None
            rows.append(base_row(record, slot, predicted_value))
    return rows, torch.cat(features, dim=0)


def candidate_order(candidates: str) -> str:
    return "forward" if candidates == "0,1,2,3,4" else "reverse" if candidates == "4,3,2,1,0" else "other"


def bool_value(value: Any) -> bool:
    return value is True or value == "True" or value == 1


def compact_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["counterfactual_pair_id"])].append(row)
    complete = [values for values in groups.values() if len(values) == 2]
    order_groups = []
    for values in complete:
        order_groups.append({
            "value_preserved": values[0]["predicted_value"] == values[1]["predicted_value"],
            "slot_changed": int(values[0]["predicted_slot"]) != int(values[1]["predicted_slot"]),
            "both_correct": all(bool_value(value["value_correct"]) for value in values),
        })
    by_distance: Dict[str, Dict[str, Any]] = {}
    for distance in sorted({int(row["graph_distance"]) for row in rows}):
        subset = [row for row in rows if int(row["graph_distance"]) == distance]
        correct = sum(bool_value(row["value_correct"]) for row in subset)
        by_distance[str(distance)] = {"correct": correct, "rows": len(subset), "accuracy": correct / len(subset)}
    by_target: Dict[str, Dict[str, Any]] = {}
    for target in range(MODULUS):
        subset = [row for row in rows if int(row["target_value"]) == target]
        if subset:
            correct = sum(bool_value(row["value_correct"]) for row in subset)
            by_target[str(target)] = {"correct": correct, "rows": len(subset), "accuracy": correct / len(subset)}
    by_direction: Dict[str, Dict[str, Any]] = {}
    for direction in sorted({str(row["query_direction"]) for row in rows}):
        subset = [row for row in rows if str(row["query_direction"]) == direction]
        correct = sum(bool_value(row["value_correct"]) for row in subset)
        by_direction[direction] = {"correct": correct, "rows": len(subset), "accuracy": correct / len(subset)}
    by_pair: Dict[str, Dict[str, Any]] = {}
    for pair in sorted({str(row["unordered_pair"]) for row in rows}):
        subset = [row for row in rows if str(row["unordered_pair"]) == pair]
        correct = sum(bool_value(row["value_correct"]) for row in subset)
        by_pair[pair] = {"correct": correct, "rows": len(subset), "accuracy": correct / len(subset)}
    by_order: Dict[str, Dict[str, Any]] = {}
    for order in ("forward", "reverse"):
        subset = [row for row in rows if candidate_order(str(row["candidates"])) == order]
        correct = sum(bool_value(row["value_correct"]) for row in subset)
        by_order[order] = {"correct": correct, "rows": len(subset), "accuracy": correct / len(subset) if subset else None}
    return {
        "rows": len(rows),
        "value_correct": sum(bool_value(row["value_correct"]) for row in rows),
        "value_accuracy": sum(bool_value(row["value_correct"]) for row in rows) / len(rows),
        "paired_groups": len(complete),
        "paired_correct": sum(int(item["both_correct"]) for item in order_groups),
        "paired_accuracy": sum(int(item["both_correct"]) for item in order_groups) / len(order_groups),
        "candidate_order_groups": len(order_groups),
        "candidate_order_value_preserved": sum(int(item["value_preserved"]) for item in order_groups),
        "candidate_order_slot_changed": sum(int(item["slot_changed"]) for item in order_groups),
        "by_distance": by_distance,
        "by_target_value": by_target,
        "by_query_direction": by_direction,
        "by_unordered_pair": by_pair,
        "by_candidate_order": by_order,
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = float(torch.linalg.vector_norm(left).item())
    right_norm = float(torch.linalg.vector_norm(right).item())
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(torch.dot(left, right).item() / (left_norm * right_norm))


def representation_summary(
    train_rows: Sequence[Mapping[str, Any]],
    train_features: torch.Tensor,
    validation_rows: Sequence[Mapping[str, Any]],
    validation_features: torch.Tensor,
) -> Dict[str, Any]:
    def stats(features: torch.Tensor) -> Dict[str, float]:
        norms = torch.linalg.vector_norm(features, dim=1)
        return {
            "feature_dim": int(features.size(1)),
            "norm_mean": float(norms.mean().item()),
            "norm_std": float(norms.std(unbiased=False).item()),
            "component_mean_abs": float(features.mean(dim=0).abs().mean().item()),
            "component_std_mean": float(features.std(dim=0, unbiased=False).mean().item()),
        }

    def grouped_distance(rows: Sequence[Mapping[str, Any]], features: torch.Tensor, key_fn: Any) -> Dict[str, Any]:
        groups: Dict[Any, List[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[key_fn(row)].append(index)
        distances: List[float] = []
        similarities: List[float] = []
        complete = 0
        for indices in groups.values():
            if len(indices) != 2:
                continue
            complete += 1
            left, right = features[indices[0]], features[indices[1]]
            distances.append(float(torch.linalg.vector_norm(left - right).item()))
            similarities.append(cosine(left, right))
        return {
            "complete_groups": complete,
            "mean_l2": sum(distances) / len(distances) if distances else None,
            "mean_cosine": sum(similarities) / len(similarities) if similarities else None,
            "max_l2": max(distances) if distances else None,
        }

    train_labels = torch.tensor([int(row["target_value"]) for row in train_rows], dtype=torch.long)
    validation_labels = torch.tensor([int(row["target_value"]) for row in validation_rows], dtype=torch.long)
    centroids = []
    for label in range(MODULUS):
        selected = train_features[train_labels == label]
        if selected.size(0) == 0:
            centroids.append(torch.zeros(train_features.size(1)))
        else:
            centroids.append(selected.mean(dim=0))
    centroid_tensor = F.normalize(torch.stack(centroids), dim=1)

    def probe(features: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        scores = F.normalize(features, dim=1) @ centroid_tensor.T
        predictions = scores.argmax(dim=1)
        confusion = [[0 for _ in range(MODULUS)] for _ in range(MODULUS)]
        for actual, predicted in zip(labels.tolist(), predictions.tolist()):
            confusion[int(actual)][int(predicted)] += 1
        return {"correct": int((predictions == labels).sum().item()), "rows": int(labels.numel()), "accuracy": float((predictions == labels).float().mean().item()), "confusion": confusion}

    return {
        "feature": {"train": stats(train_features), "validation": stats(validation_features)},
        "same_query_candidate_order": grouped_distance(validation_rows, validation_features, lambda row: row["counterfactual_pair_id"]),
        "centroid_probe_target_value": {"train": probe(train_features, train_labels), "validation": probe(validation_features, validation_labels)},
    }


def transition_summary(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_id = {str(row["record_id"]): row for row in left}
    paired = {str(row["record_id"]): row for row in right}
    counts = Counter()
    for record_id, lrow in by_id.items():
        rrow = paired[record_id]
        lcorrect = bool_value(lrow["value_correct"])
        rcorrect = bool_value(rrow["value_correct"])
        counts["both_correct" if lcorrect and rcorrect else "left_correct_right_wrong" if lcorrect else "left_wrong_right_correct" if rcorrect else "both_wrong"] += 1
    return {key: int(counts[key]) for key in ("both_correct", "left_correct_right_wrong", "left_wrong_right_correct", "both_wrong")}


def load_existing_predictions(variant: str, condition: str, seed: int, split: str) -> List[Dict[str, str]]:
    if variant == "N0_SHARED_SCORER":
        path = N0_ROOT / f"{condition}_SEED{seed}" / f"{split.upper()}_PREDICTIONS.csv"
    else:
        path = PHASE_ROOT / f"{variant}_{condition}_SEED{seed}" / f"{split.upper()}_PREDICTIONS.csv"
    return read_csv(path)


def main() -> int:
    failures: List[str] = []
    all_prediction_rows: List[Dict[str, Any]] = []
    model_summaries: Dict[str, Any] = {}
    model_rows: Dict[str, List[Dict[str, Any]]] = {}
    model_features: Dict[str, Dict[str, torch.Tensor]] = {}
    hash_checks: Dict[str, bool] = {}
    graph_manifest = phase.graph_manifest()

    for condition in CONDITIONS:
        graph = phase.graph_from(graph_manifest[condition])
        raw_train = load_records(condition, "train")
        raw_validation = load_records(condition, "validation")
        for seed in SEEDS:
            for variant in VARIANTS:
                key = f"{variant}_{condition}_SEED{seed}"
                model, checkpoint = load_model(variant, condition, seed, graph)
                hash_checks[key] = sha256(checkpoint) == (load_json(PHASE_ROOT / "HASHES.json").get(rel(checkpoint)) if variant != "N0_SHARED_SCORER" else load_json(N0_ROOT / "HASHES.json").get(rel(checkpoint)))
                if not hash_checks[key]:
                    failures.append(f"{key}: checkpoint hash mismatch")

                if variant == "N0_SHARED_SCORER":
                    prepared_train = topology.prepare_records(raw_train)
                    prepared_validation = topology.prepare_records(raw_validation)
                else:
                    prepared_train = phase.prepare_records(condition, "train", raw_train, graph)
                    prepared_validation = phase.prepare_records(condition, "validation", raw_validation, graph)
                train_rows, train_features = replay(model, variant, condition, "train", raw_train, prepared_train, graph)
                validation_rows, validation_features = replay(model, variant, condition, "validation", raw_validation, prepared_validation, graph)
                model_rows[key] = validation_rows
                model_features[key] = {"train": train_features, "validation": validation_features}
                recorded_train = load_existing_predictions(variant, condition, seed, "train")
                recorded_validation = load_existing_predictions(variant, condition, seed, "validation")
                for split, recomputed, recorded in (("train", train_rows, recorded_train), ("validation", validation_rows, recorded_validation)):
                    expected = {str(row["record_id"]): row for row in recorded}
                    for row in recomputed:
                        prior = expected.get(str(row["record_id"]))
                        if prior is None:
                            failures.append(f"{key}/{split}: missing recorded row {row['record_id']}")
                            continue
                        for field in ("predicted_slot", "predicted_value", "target_slot", "target_value"):
                            left = row[field]
                            right = None if prior[field] in ("", "None", None) else int(prior[field])
                            if left != right:
                                failures.append(f"{key}/{split}/{row['record_id']}: replay mismatch in {field}")
                for row in validation_rows:
                    all_prediction_rows.append({"model": variant, "condition": condition, "seed": seed, **row})

                model_summaries[key] = {
                    "variant": variant,
                    "condition": condition,
                    "seed": seed,
                    "checkpoint": rel(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "replay_train_metrics": compact_metrics(train_rows),
                    "replay_validation_metrics": compact_metrics(validation_rows),
                    "representation": representation_summary(train_rows, train_features, validation_rows, validation_features),
                    "recorded_replay_match": True,
                }
                del model

    transitions: Dict[str, Any] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            n0 = model_rows[f"N0_SHARED_SCORER_{condition}_SEED{seed}"]
            n1 = model_rows[f"N1_SHARED_LATENT_{condition}_SEED{seed}"]
            n2 = model_rows[f"N2_SHARED_LATENT_CYCLE_{condition}_SEED{seed}"]
            transitions[f"N0_to_N1_{condition}_SEED{seed}"] = transition_summary(n0, n1)
            transitions[f"N0_to_N2_{condition}_SEED{seed}"] = transition_summary(n0, n2)
            transitions[f"N1_to_N2_{condition}_SEED{seed}"] = transition_summary(n1, n2)

    trajectories: Dict[str, List[Dict[str, Any]]] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            for variant in VARIANTS:
                key = f"{variant}_{condition}_SEED{seed}"
                if variant == "N0_SHARED_SCORER":
                    trace_path = N0_ROOT / f"{condition}_SEED{seed}" / "TRAINING_TRACE.jsonl"
                else:
                    trace_path = PHASE_ROOT / key / "TRAINING_TRACE.jsonl"
                trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                trajectories[key] = [
                    {
                        "step": int(row["step"]),
                        "loss": float(row["loss"]),
                        "cycle_loss": float(row.get("cycle_loss", 0.0)),
                        "train_value_accuracy": float(row["training"]["value_accuracy"]),
                        "validation_value_accuracy": float(row["validation"]["value_accuracy"]),
                        "train_paired_accuracy": float(row["training"]["paired_accuracy"]),
                        "validation_paired_accuracy": float(row["validation"]["paired_accuracy"]),
                    }
                    for row in trace
                    if "training" in row and "validation" in row
                ]

    # Add a compact paired transition table to the diagnostic JSON.
    diagnostic = {
        "schema_version": "crucible_phase_vii_checkpoint_diagnostic_v1",
        "experiment": "crucible_phase_vii_neural_shared_latent_relational_transfer_20261027_v1",
        "status": "completed",
        "read_only": True,
        "training_performed": False,
        "sealed_test_accessed": False,
        "checkpoint_hashes_verified": all(hash_checks.values()),
        "replay_predictions_match_committed_csvs": not any("replay mismatch" in failure or "missing recorded row" in failure for failure in failures),
        "models": model_summaries,
        "transitions": transitions,
        "trajectories": trajectories,
        "hash_checks": hash_checks,
        "failures": failures,
        "interpretation_boundary": "This is a read-only behavioral and representation diagnostic. It can localize error patterns and show whether final hidden states contain simple reusable target-value structure; it cannot identify a unique causal mechanism or prove how the networks reason.",
    }
    diagnostic_json = OUTPUT / "CHECKPOINT_DIAGNOSTIC.json"
    prediction_csv = OUTPUT / "CHECKPOINT_DIAGNOSTIC_PREDICTIONS.csv"
    diagnostic_md = OUTPUT / "CHECKPOINT_DIAGNOSTIC.md"
    write_json(diagnostic_json, diagnostic)
    write_csv(prediction_csv, all_prediction_rows)

    report = [
        "# Phase VII read-only checkpoint diagnostic",
        "",
        "Verification pass: **%s**." % ("yes" if not failures else "no"),
        "",
        "This diagnostic replayed the committed N0, N1 and N2 final checkpoints without optimizer calls, verified their predictions against the committed train/validation CSVs, and compared candidate-order, query-direction, error-stratum and representation summaries.",
        "",
        "## Paired validation by model",
        "",
        "| Model | Condition | Seed | Paired | Value accuracy | Candidate-order value preserved | Centroid probe validation |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(model_summaries):
        item = model_summaries[key]
        metrics = item["replay_validation_metrics"]
        probe = item["representation"]["centroid_probe_target_value"]["validation"]
        report.append(
            f"| {item['variant']} | {item['condition']} | {item['seed']} | {metrics['paired_correct']}/{metrics['paired_groups']} | {metrics['value_accuracy']:.3f} | {metrics['candidate_order_value_preserved']}/{metrics['candidate_order_groups']} | {probe['correct']}/{probe['rows']} ({probe['accuracy']:.3f}) |"
        )
    report.extend([
        "",
        "## Interpretation",
        "",
        "The detailed machine-readable comparisons are in `CHECKPOINT_DIAGNOSTIC.json`; per-row replay outputs are in `CHECKPOINT_DIAGNOSTIC_PREDICTIONS.csv`.",
        "",
        "The diagnostic does not train a probe, alter checkpoints, or establish a unique internal mechanism. A centroid probe is included only as a fixed descriptive test of whether final representations cluster by target value when centroids are computed from training representations.",
    ])
    diagnostic_md.write_text("\n".join(report) + "\n", encoding="utf-8")

    diagnostic_hashes = {
        rel(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        rel(diagnostic_json): sha256(diagnostic_json),
        rel(prediction_csv): sha256(prediction_csv),
        rel(diagnostic_md): sha256(diagnostic_md),
    }
    write_json(OUTPUT / "CHECKPOINT_DIAGNOSTIC_HASHES.json", diagnostic_hashes)

    if failures:
        print(json.dumps({"status": "failed", "read_only": True, "failures": failures}, indent=2))
        return 1
    print(json.dumps({"status": "completed", "read_only": True, "models": len(model_summaries), "replay_predictions_match": True, "checkpoint_hashes_verified": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
