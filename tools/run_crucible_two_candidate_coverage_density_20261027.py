#!/usr/bin/env python3
"""Run the preregistered two-candidate coverage-density experiment.

The D4 endpoint is reused from the immutable M0->M2 crossover cell. D8 and
D12 are newly trained with the same 144-row budget and fixed M2 validation
records. This script intentionally treats graph descriptors as descriptive;
the task is direct identifier matching, not transitive graph reasoning.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
PAIR_RUNNER = ROOT / "tools" / "run_crucible_two_candidate_pair_coverage_20261027.py"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_COVERAGE_DENSITY_PROTOCOL_20261027.md"
NOVELTY = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_COVERAGE_DENSITY_NOVELTY_AUDIT_20261027.json"
OUTPUT = ROOT / "experiments" / "crucible_two_candidate_coverage_density_20261027_v1"
CROSSOVER_OUTPUT = ROOT / "experiments" / "crucible_two_candidate_pair_crossover_20261027_v1"
FIXED_VALIDATION = CROSSOVER_OUTPUT / "data" / "T_M0_V_M2_validation.jsonl"
FIXED_TRAIN_REFERENCE = CROSSOVER_OUTPUT / "data" / "T_M0_V_M2_train.jsonl"

SEEDS = (42, 1337)
BATCH_SIZE = 12
STEPS = 500
SHUFFLE_SEED = 20261027
TRAIN_THRESHOLD = 0.95
CHECKPOINT_STEPS = (1, 50, 100, 200, 300, 400, 500)
TRAIN_CONTEXTS = ((1, 6), (1, 8), (2, 7), (2, 9), (3, 8), (4, 9))
VALIDATION_CONTEXTS = ((1, 6), (2, 7), (3, 8), (4, 9))
PANELS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "M0": ((0, 1), (2, 3), (4, 5), (6, 7)),
    "M1": ((0, 2), (1, 3), (4, 6), (5, 7)),
    "M2": ((0, 3), (1, 2), (4, 7), (5, 6)),
    "M3": ((0, 4), (1, 5), (2, 6), (3, 7)),
}
DENSITY: Dict[str, Dict[str, Any]] = {
    "D4": {"panels": ("M0",), "contexts_per_edge": 6},
    "D8": {"panels": ("M0", "M1"), "contexts_per_edge": 3},
    "D12": {"panels": ("M0", "M1", "M3"), "contexts_per_edge": 2},
}
VALIDATION_PANEL = "M2"


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


pair = import_module("pair_coverage_for_density", PAIR_RUNNER)
base = pair.base
source = pair.source
torch = pair.torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(row), sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_hashes(hashes: Mapping[str, str]) -> List[Dict[str, str]]:
    failures: List[Dict[str, str]] = []
    for name, expected in hashes.items():
        path = ROOT / name
        if not path.exists():
            failures.append({"path": name, "error": "missing"})
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append({"path": name, "expected": expected, "actual": actual})
    return failures


def edge_label(edge: Sequence[int]) -> str:
    return f"{min(edge)},{max(edge)}"


def panel_signature(panel: Sequence[Sequence[int]]) -> List[str]:
    return [edge_label(pair_value) for pair_value in panel]


def density_edges(condition: str) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for panel_name in DENSITY[condition]["panels"]:
        edges.extend(tuple(edge) for edge in PANELS[panel_name])
    if len(set(edges)) != len(edges):
        raise RuntimeError(f"{condition}: duplicate training edge")
    return edges


def graph_summary(edges: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    adjacency = {node: set() for node in range(8)}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: List[List[int]] = []
    seen = set()
    for start in range(8):
        if start in seen:
            continue
        component: List[int] = []
        queue = [start]
        seen.add(start)
        while queue:
            node = queue.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))

    distances: Dict[str, int | None] = {}
    finite_distances: List[int] = []
    for source_node in range(8):
        distance = {source_node: 0}
        queue = deque([source_node])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        for target_node in range(source_node + 1, 8):
            if target_node in distance:
                finite_distances.append(distance[target_node])

    for edge in PANELS[VALIDATION_PANEL]:
        left, right = edge
        distance = {left: 0}
        queue = deque([left])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        distances[edge_label(edge)] = distance.get(right)

    return {
        "edges": [edge_label(edge) for edge in edges],
        "edge_count": len(edges),
        "degree_sequence": [len(adjacency[node]) for node in range(8)],
        "components": components,
        "component_count": len(components),
        "max_finite_pair_distance": max(finite_distances) if finite_distances else None,
        "held_out_M2_shortest_paths": distances,
        "interpretation": "descriptive only; direct identifier matching does not require transitive graph composition",
    }


def balanced_context_indices(edge_count: int, contexts_per_edge: int) -> List[Tuple[int, ...]]:
    total = edge_count * contexts_per_edge
    if total != 24:
        raise RuntimeError(f"expected 24 edge-context assignments, got {total}")
    slots = [index % len(TRAIN_CONTEXTS) for index in range(total)]
    return [tuple(slots[start:start + contexts_per_edge]) for start in range(0, total, contexts_per_edge)]


def make_density_records(condition: str) -> List[Dict[str, Any]]:
    edges = density_edges(condition)
    context_indices = balanced_context_indices(len(edges), DENSITY[condition]["contexts_per_edge"])
    rows: List[Dict[str, Any]] = []
    ordinal = 0
    for edge_index, unordered in enumerate(edges):
        for context_index in context_indices[edge_index]:
            a, b = TRAIN_CONTEXTS[context_index]
            for candidates in (tuple(unordered), (unordered[1], unordered[0])):
                pair_key = f"coverage_density|{condition}|train|{edge_index}|{context_index}|{a}|{b}|{candidates}"
                counterfactual_pair_id = hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:20]
                for authority in (candidates[0], candidates[1], 9):
                    target = candidates.index(authority) if authority in candidates else 8
                    record_key = f"{pair_key}|{authority}|{ordinal}"
                    record_id = hashlib.sha256(record_key.encode("utf-8")).hexdigest()[:20]
                    rows.append({
                        "schema_version": "crucible_two_candidate_coverage_density_v1",
                        "record_id": record_id,
                        "interface": "canonical",
                        "split": "train",
                        "layout": "TRAIN_COMPACT",
                        "input": source.input_text("canonical", "TRAIN_COMPACT", a, b, candidates, authority, ordinal),
                        "target": str(target),
                        "oracle": {"kind": "identifier_match", "target_slot": target, "authority_id": authority},
                        "metadata": {
                            "a": a,
                            "b": b,
                            "candidates": list(candidates),
                            "authority_id": authority,
                            "target_slot": target,
                            "counterfactual_pair_id": counterfactual_pair_id if authority != 9 else None,
                            "partition": f"{condition}_V_M2",
                            "panel": condition,
                            "unordered_pair": list(unordered),
                            "context_index": context_index,
                            "context_pool": "TRAIN_SOURCE_POOL",
                            "edge_index": edge_index,
                            "training_edge": list(unordered),
                            "coverage_condition": condition,
                        },
                    })
                    ordinal += 1
    return rows


def oracle_target(row: Mapping[str, Any]) -> int:
    candidates = [int(value) for value in row["metadata"]["candidates"]]
    authority = int(row["metadata"]["authority_id"])
    return candidates.index(authority) if authority in candidates else 8


def validate_records(name: str, rows: Sequence[Mapping[str, Any]], expected_rows: int, expected_pairs: Iterable[Tuple[int, int]], expected_contexts: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    if len(rows) != expected_rows:
        raise RuntimeError(f"{name}: expected {expected_rows} rows, got {len(rows)}")
    if len({str(row["record_id"]) for row in rows}) != len(rows):
        raise RuntimeError(f"{name}: duplicate record IDs")
    if len({str(row["input"]) for row in rows}) != len(rows):
        raise RuntimeError(f"{name}: duplicate inputs")
    if any(str(row["layout"]) != "TRAIN_COMPACT" for row in rows):
        raise RuntimeError(f"{name}: layout changed")
    if any(int(row["metadata"]["target_slot"]) != oracle_target(row) for row in rows):
        raise RuntimeError(f"{name}: oracle mismatch")
    counts = {str(key): value for key, value in sorted(Counter(int(row["metadata"]["target_slot"]) for row in rows).items())}
    expected_counts = {"0": expected_rows // 3, "1": expected_rows // 3, "8": expected_rows // 3}
    if counts != expected_counts:
        raise RuntimeError(f"{name}: class balance mismatch {counts}")
    actual_contexts = sorted({(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in rows})
    if actual_contexts != sorted(expected_contexts):
        raise RuntimeError(f"{name}: context mismatch {actual_contexts}")
    actual_pairs = {tuple(sorted(int(value) for value in row["metadata"]["candidates"])) for row in rows}
    expected_pair_set = {tuple(sorted(pair_value)) for pair_value in expected_pairs}
    if actual_pairs != expected_pair_set:
        raise RuntimeError(f"{name}: pair mismatch {actual_pairs} != {expected_pair_set}")
    return {
        "rows": len(rows),
        "target_slot_counts": counts,
        "unique_record_ids": len({str(row["record_id"]) for row in rows}),
        "unique_inputs": len({str(row["input"]) for row in rows}),
        "counterfactual_groups": len({str(row["metadata"]["counterfactual_pair_id"]) for row in rows if row["metadata"]["counterfactual_pair_id"]}),
        "contexts": [f"{a},{b}" for a, b in sorted(actual_contexts)],
        "unordered_pairs": [edge_label(pair_value) for pair_value in sorted(expected_pair_set)],
    }


def selection_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = [row for row in rows if int(row["target_slot"]) != 8]
    correct = sum(int(bool(row["pointer_correct"])) for row in selected)
    return {"rows": len(selected), "correct": correct, "accuracy": correct / len(selected) if selected else None}


def load_model(checkpoint: Path, seed: int) -> Any:
    base.SEED = seed
    model = base.model_for("dense")
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    return model


def train_density_cell(cell: str, condition: str, train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    base.SEED = seed
    model = base.model_for("dense")
    prepared_train = base.runner.prepare_records(train_records)
    prepared_validation = base.runner.prepare_records(validation_records)
    optimizer = base.runner.stage_a.optimizer_for(model)
    batches = base.schedule(len(train_records), "cyclic")
    output = OUTPUT / cell
    output.mkdir(parents=True, exist_ok=True)
    trace: List[Dict[str, Any]] = []
    for step, indices in enumerate(batches, start=1):
        model.train()
        batch = base.dense_batch(train_records, prepared_train, indices)
        optimizer.zero_grad(set_to_none=True)
        logits = model.decision_logits(batch)
        loss = base.F.cross_entropy(logits, batch["target"])
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), base.runner.stage_a.CLIP_NORM).item())
        optimizer.step()
        entry: Dict[str, Any] = {
            "cell": cell,
            "condition": condition,
            "seed": seed,
            "step": step,
            "examples": len(indices),
            "loss": float(loss.detach().item()),
            "grad_norm_preclip": grad_norm,
        }
        if step in CHECKPOINT_STEPS:
            train_rows = base.evaluate(model, "dense", train_records, prepared_train)
            validation_rows = base.evaluate(model, "dense", validation_records, prepared_validation)
            entry["training"] = base.row_metrics(train_rows)
            entry["training_selection"] = selection_metrics(train_rows)
            entry["validation"] = base.row_metrics(validation_rows)
            entry["validation_selection"] = selection_metrics(validation_rows)
        trace.append(entry)

    final_train_rows = base.evaluate(model, "dense", train_records, prepared_train)
    final_validation_rows = base.evaluate(model, "dense", validation_records, prepared_validation)
    checkpoint = output / "checkpoint.pt"
    torch.save({"cell": cell, "condition": condition, "representation": "dense", "schedule": "cyclic", "seed": seed, "model": model.state_dict()}, checkpoint)
    write_jsonl(output / "TRAINING_TRACE.jsonl", trace)
    write_csv(output / "PREDICTIONS.csv", final_train_rows)
    train_enriched = pair.enrich_predictions(final_train_rows, train_records)
    validation_enriched = pair.enrich_predictions(final_validation_rows, validation_records)
    write_csv(output / "TRAIN_PREDICTIONS_ENRICHED.csv", train_enriched)
    write_csv(output / "VALIDATION_PREDICTIONS.csv", validation_enriched)
    train_metrics = base.row_metrics(final_train_rows)
    validation_metrics = base.row_metrics(final_validation_rows)
    result = {
        "cell": cell,
        "source": "new",
        "condition": condition,
        "seed": seed,
        "representation": "dense",
        "schedule": "cyclic",
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "source_rows": len(train_records),
        "train_metrics": train_metrics,
        "train_selection_metrics": selection_metrics(final_train_rows),
        "train_slot_metrics": pair.slot_metrics(final_train_rows),
        "validation_metrics": validation_metrics,
        "validation_selection_metrics": selection_metrics(final_validation_rows),
        "validation_slot_metrics": pair.slot_metrics(final_validation_rows),
        "training_gate_threshold": TRAIN_THRESHOLD,
        "training_gate_pass": bool(train_metrics["pointer_accuracy"] >= TRAIN_THRESHOLD),
        "checkpoint": rel(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_breakdown": pair.metric_breakdown(train_enriched),
        "validation_breakdown": pair.metric_breakdown(validation_enriched),
        "validation_trace_steps": list(CHECKPOINT_STEPS),
    }
    write_json(output / "RESULTS.json", result)
    del optimizer, model
    return result


def prior_d4_result(seed: int, prior_error_breakdown: Mapping[str, Any]) -> Dict[str, Any]:
    cell = f"D4_SEED{seed}"
    source_cell = f"T_M0_V_M2_SEED{seed}"
    payload = json.loads((CROSSOVER_OUTPUT / source_cell / "RESULTS.json").read_text(encoding="utf-8"))
    if payload["train_panel"] != "M0" or payload["validation_panel"] != "M2":
        raise RuntimeError(f"unexpected D4 source cell: {source_cell}")
    return {
        "cell": cell,
        "source": "prior_crossover",
        "source_cell": source_cell,
        "condition": "D4",
        "seed": seed,
        "representation": payload["representation"],
        "schedule": payload["schedule"],
        "steps": payload["steps"],
        "batch_size": payload["batch_size"],
        "source_rows": payload["source_rows"],
        "train_metrics": payload["train_metrics"],
        "train_selection_metrics": selection_metrics_from_payload_breakdown(prior_error_breakdown[source_cell]["train"]),
        "train_slot_metrics": payload.get("train_slot_metrics"),
        "validation_metrics": payload["validation_metrics"],
        "validation_selection_metrics": selection_metrics_from_payload_breakdown(prior_error_breakdown[source_cell]["validation"]),
        "validation_slot_metrics": payload.get("validation_slot_metrics"),
        "training_gate_threshold": payload["training_gate_threshold"],
        "training_gate_pass": payload["training_gate_pass"],
        "checkpoint": payload["checkpoint"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "parameter_count": payload["parameter_count"],
        "train_breakdown": prior_error_breakdown[source_cell]["train"],
        "validation_breakdown": prior_error_breakdown[source_cell]["validation"],
        "validation_trace_steps": None,
        "graph": graph_summary(density_edges("D4")),
    }


def selection_metrics_from_payload_breakdown(breakdown: Mapping[str, Any]) -> Dict[str, Any]:
    overall = breakdown["overall"]
    rows = overall["rows"] - overall["abstention"]["rows"]
    correct = overall["correct"] - overall["abstention"]["correct"]
    return {"rows": rows, "correct": correct, "accuracy": correct / rows if rows else None}


def aggregate_by_condition(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for condition in DENSITY:
        subset = [result for result in results if result["condition"] == condition]
        train_rows = sum(result["train_metrics"]["rows"] for result in subset)
        train_correct = sum(result["train_metrics"]["pointer_correct"] for result in subset)
        validation_rows = sum(result["validation_metrics"]["rows"] for result in subset)
        validation_correct = sum(result["validation_metrics"]["pointer_correct"] for result in subset)
        selection_rows = sum(result["validation_selection_metrics"]["rows"] for result in subset)
        selection_correct = sum(result["validation_selection_metrics"]["correct"] for result in subset)
        paired_groups = sum(result["validation_metrics"]["paired_groups"] for result in subset)
        paired_correct = sum(result["validation_metrics"]["paired_correct"] for result in subset)
        abstain_rows = sum(result["validation_metrics"]["abstain_rows"] for result in subset)
        abstain_correct = sum(result["validation_metrics"]["abstain_correct"] for result in subset)
        aggregate[condition] = {
            "arms": len(subset),
            "training_gate_passes": sum(int(result["training_gate_pass"]) for result in subset),
            "training": {"correct": train_correct, "rows": train_rows, "accuracy": train_correct / train_rows if train_rows else None},
            "validation": {"correct": validation_correct, "rows": validation_rows, "accuracy": validation_correct / validation_rows if validation_rows else None},
            "candidate_selection": {"correct": selection_correct, "rows": selection_rows, "accuracy": selection_correct / selection_rows if selection_rows else None},
            "paired": {"correct": paired_correct, "groups": paired_groups, "accuracy": paired_correct / paired_groups if paired_groups else None},
            "abstention": {"correct": abstain_correct, "rows": abstain_rows, "accuracy": abstain_correct / abstain_rows if abstain_rows else None},
            "seed_rows": [{"seed": result["seed"], "validation": result["validation_metrics"], "paired": result["validation_metrics"]["paired_correct"] / result["validation_metrics"]["paired_groups"]} for result in subset],
        }
    return aggregate


def main() -> int:
    execute = "--execute" in sys.argv[1:]
    if OUTPUT.exists() and (OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"refusing to overwrite completed output: {OUTPUT}")
    if not CROSSOVER_OUTPUT.exists() or not (CROSSOVER_OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"missing completed crossover output: {CROSSOVER_OUTPUT}")
    if not FIXED_VALIDATION.exists() or not FIXED_TRAIN_REFERENCE.exists():
        raise SystemExit("missing immutable M0->M2 crossover data")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    prior_hashes = json.loads((CROSSOVER_OUTPUT / "HASHES.json").read_text(encoding="utf-8"))
    prior_hash_failures = verify_hashes(prior_hashes)
    if prior_hash_failures:
        raise SystemExit(f"prior crossover hash verification failed: {prior_hash_failures[:3]}")

    fixed_validation = load_jsonl(FIXED_VALIDATION)
    fixed_train_reference = load_jsonl(FIXED_TRAIN_REFERENCE)
    expected_validation = pair.make_records("T_M0_V_M2", "validation", "M2", VALIDATION_CONTEXTS)
    expected_train = pair.make_records("T_M0_V_M2", "train", "M0", TRAIN_CONTEXTS)
    if fixed_validation != expected_validation or fixed_train_reference != expected_train:
        raise RuntimeError("immutable crossover M0->M2 records do not reproduce exactly")
    validation_audit = pair.audit_rows("fixed_M2_validation", fixed_validation, 96, "M2", VALIDATION_CONTEXTS)
    reference_train_audit = pair.audit_rows("reference_M0_train", fixed_train_reference, 144, "M0", TRAIN_CONTEXTS)
    if pair.cross_split_audit(fixed_train_reference, fixed_validation) != {"record_ids": 0, "exact_inputs": 0, "unordered_pairs": 0}:
        raise RuntimeError("reference M0/M2 contamination")

    source_train = load_jsonl(pair.TRAIN_SOURCE)
    source_contexts = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in source_train if len(row["metadata"]["candidates"]) == 2}
    if not set(TRAIN_CONTEXTS).issubset(source_contexts) or not set(VALIDATION_CONTEXTS).issubset(source_contexts):
        raise RuntimeError("configured contexts absent from canonical source pool")
    if not set(VALIDATION_CONTEXTS).issubset(set(TRAIN_CONTEXTS)):
        raise RuntimeError("validation contexts are not represented in the training context pool")

    conditions: Dict[str, Dict[str, Any]] = {}
    for condition in ("D8", "D12"):
        records = make_density_records(condition)
        audit = validate_records(f"{condition}_train", records, 144, density_edges(condition), TRAIN_CONTEXTS)
        context_counts = Counter(int(row["metadata"]["context_index"]) for row in records[::6])
        if sorted(context_counts.values()) != [4, 4, 4, 4, 4, 4]:
            raise RuntimeError(f"{condition}: unbalanced context assignment {context_counts}")
        overlap = pair.cross_split_audit(records, fixed_validation)
        if overlap != {"record_ids": 0, "exact_inputs": 0, "unordered_pairs": 0}:
            raise RuntimeError(f"{condition}: train/validation contamination {overlap}")
        conditions[condition] = {
            "records": records,
            "audit": audit,
            "context_counts": {str(key): value for key, value in sorted(context_counts.items())},
            "graph": graph_summary(density_edges(condition)),
            "cross_split_overlap": overlap,
        }

    preflight = {
        "experiment": "crucible_two_candidate_coverage_density_20261027_v1",
        "status": "preflight_passed",
        "execute_requested": execute,
        "protocol": rel(PROTOCOL),
        "novelty_audit": rel(NOVELTY),
        "prior_crossover": rel(CROSSOVER_OUTPUT),
        "prior_hash_files": len(prior_hashes),
        "prior_hash_verification_failures": prior_hash_failures,
        "fixed_validation": rel(FIXED_VALIDATION),
        "fixed_validation_sha256": sha256(FIXED_VALIDATION),
        "fixed_train_reference": rel(FIXED_TRAIN_REFERENCE),
        "fixed_train_reference_sha256": sha256(FIXED_TRAIN_REFERENCE),
        "fixed_validation_audit": validation_audit,
        "reference_train_audit": reference_train_audit,
        "conditions": {condition: {"audit": data["audit"], "context_counts": data["context_counts"], "graph": data["graph"], "cross_split_overlap": data["cross_split_overlap"]} for condition, data in conditions.items()},
        "reused_condition": "D4",
        "new_conditions": ["D8", "D12"],
        "architecture": "dense SHARED_SCORER",
        "seeds": list(SEEDS),
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "schedule": "cyclic",
        "training_gate_threshold": TRAIN_THRESHOLD,
        "training_performed": False,
        "sealed_test_accessed": False,
        "prior_checkpoints_modified": False,
    }
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    write_json(OUTPUT / "NOVELTY_AUDIT.json", json.loads(NOVELTY.read_text(encoding="utf-8")))
    write_jsonl(OUTPUT / "data" / "M2_validation.jsonl", fixed_validation)
    for condition, data in conditions.items():
        write_jsonl(OUTPUT / "data" / f"{condition}_train.jsonl", data["records"])
    write_json(OUTPUT / "data" / "GRAPH_METADATA.json", {condition: data["graph"] for condition, data in conditions.items()} | {"D4": graph_summary(density_edges("D4"))})
    if not execute:
        print(json.dumps({"experiment": preflight["experiment"], "status": "preflight_passed", "new_arms": 4, "training_performed": False, "execute_with": "--execute"}, sort_keys=True))
        return 0

    base.OUTPUT = OUTPUT
    base.STEPS = STEPS
    base.BATCH_SIZE = BATCH_SIZE
    base.SHUFFLE_SEED = SHUFFLE_SEED
    base.TRAIN_THRESHOLD = TRAIN_THRESHOLD
    prior_breakdown = json.loads((CROSSOVER_OUTPUT / "ERROR_BREAKDOWN.json").read_text(encoding="utf-8"))
    results: List[Dict[str, Any]] = [prior_d4_result(seed, prior_breakdown) for seed in SEEDS]
    for condition in ("D8", "D12"):
        for seed in SEEDS:
            cell = f"{condition}_SEED{seed}"
            result = train_density_cell(cell, condition, conditions[condition]["records"], fixed_validation, seed)
            result["graph"] = conditions[condition]["graph"]
            write_json(OUTPUT / cell / "RESULTS.json", result)
            results.append(result)

    for result in results:
        if "graph" not in result:
            result["graph"] = graph_summary(density_edges(result["condition"]))

    summary = {
        "experiment": "crucible_two_candidate_coverage_density_20261027_v1",
        "status": "completed",
        "protocol": rel(PROTOCOL),
        "novelty_audit": rel(NOVELTY),
        "preflight": rel(OUTPUT / "PREFLIGHT.json"),
        "data_directory": rel(OUTPUT / "data"),
        "prior_crossover": rel(CROSSOVER_OUTPUT),
        "fixed_validation": rel(FIXED_VALIDATION),
        "fixed_validation_sha256": sha256(FIXED_VALIDATION),
        "reused_condition": "D4",
        "new_conditions": ["D8", "D12"],
        "reused_arm_count": 2,
        "new_arm_count": 4,
        "total_arm_count": len(results),
        "conditions": list(DENSITY),
        "seeds": list(SEEDS),
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "schedule": "cyclic",
        "training_gate_threshold": TRAIN_THRESHOLD,
        "training_performed": True,
        "sealed_test_accessed": False,
        "prior_checkpoints_modified": False,
        "training_gate_passes_new": sum(int(result["training_gate_pass"]) for result in results if result["source"] == "new"),
        "training_gate_passes_combined": sum(int(result["training_gate_pass"]) for result in results),
        "results": results,
        "by_condition": aggregate_by_condition(results),
        "graph_descriptors": {condition: graph_summary(density_edges(condition)) for condition in DENSITY},
    }
    write_json(OUTPUT / "SUMMARY.json", summary)

    report_lines = [
        "# Crucible Two-Candidate Coverage-Density Experiment v1",
        "",
        "Fixed M2 held-out panel; D4 is reused from the immutable crossover and D8/D12 are newly trained with the same 144-row budget.",
        "",
        "| Condition | Seed | Source | Train | Gate | Validation | Candidate selection | Paired | Abstention |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for result in results:
        report_lines.append(
            f"| {result['condition']} | {result['seed']} | {result['source']} | {result['train_metrics']['pointer_correct']}/{result['train_metrics']['rows']} ({result['train_metrics']['pointer_accuracy']:.3f}) | {'PASS' if result['training_gate_pass'] else 'FAIL'} | {result['validation_metrics']['pointer_correct']}/{result['validation_metrics']['rows']} ({result['validation_metrics']['pointer_accuracy']:.3f}) | {result['validation_selection_metrics']['correct']}/{result['validation_selection_metrics']['rows']} ({result['validation_selection_metrics']['accuracy']:.3f}) | {result['validation_metrics']['paired_correct']}/{result['validation_metrics']['paired_groups']} ({result['validation_metrics']['paired_correct'] / result['validation_metrics']['paired_groups']:.3f}) | {result['validation_metrics']['abstain_correct']}/{result['validation_metrics']['abstain_rows']} |"
        )
    report_lines += ["", "## Pooled condition results", "", "| Condition | Train gates | Validation | Candidate selection | Paired | Abstention |", "|---|---:|---:|---:|---:|---:|"]
    for condition, values in summary["by_condition"].items():
        report_lines.append(
            f"| {condition} | {values['training_gate_passes']}/{values['arms']} | {values['validation']['correct']}/{values['validation']['rows']} ({values['validation']['accuracy']:.3f}) | {values['candidate_selection']['correct']}/{values['candidate_selection']['rows']} ({values['candidate_selection']['accuracy']:.3f}) | {values['paired']['correct']}/{values['paired']['groups']} ({values['paired']['accuracy']:.3f}) | {values['abstention']['correct']}/{values['abstention']['rows']} ({values['abstention']['accuracy']:.3f}) |"
        )
    report_lines += ["", "## Structural descriptors", "", "| Condition | Edges | Components | Degree sequence | Held-out M2 distances |", "|---|---:|---:|---|---|"]
    for condition, graph in summary["graph_descriptors"].items():
        report_lines.append(f"| {condition} | {graph['edge_count']} | {graph['component_count']} | `{graph['degree_sequence']}` | `{graph['held_out_M2_shortest_paths']}` |")
    report_lines += [
        "",
        "## Interpretation boundary",
        "",
        "A result is interpreted as transfer only after the 95% training gate. The fixed M2 endpoint tests pair coverage under this direct identifier-matching task. D4/D8/D12 are nested and change degree and connectivity together with coverage; their graph descriptors are descriptive and cannot establish topology or path-length causality. No variable-cardinality or sealed-test claim is made.",
        "",
        "## Artifacts",
        "",
        f"- [Protocol](../../{rel(PROTOCOL)})",
        f"- [Novelty audit](../../{rel(NOVELTY)})",
        "- [Preflight](PREFLIGHT.json)",
        "- [Summary](SUMMARY.json)",
        "- Fixed validation: `data/M2_validation.jsonl`",
        "- New condition data, checkpoints, traces and prediction matrices are under `data/`, `D8_SEED*` and `D12_SEED*`.",
        f"- Immutable D4 source: [`{rel(CROSSOVER_OUTPUT)}`](../../{rel(CROSSOVER_OUTPUT)})",
    ]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    interpretation = """# Scientific Interpretation

This experiment holds the M2 candidate panel, context pool, serialization,
dense SHARED_SCORER, cyclic schedule, optimizer path, 500-update budget and
row count fixed while increasing nested training-pair coverage from D4 to D8
to D12. D4 is an immutable M0->M2 crossover reference; D8 and D12 are new
training arms.

The primary behavioral question is whether broader pair coverage improves
transfer after training competence is established. The graph descriptors are
included to make structural changes auditable, but this task directly asks for
identifier matching and does not require transitive paths. Since the density
conditions also change degree, connectivity and finite distances, this run
cannot identify graph topology or path-length causality.

See SUMMARY.json, ERROR_BREAKDOWN.json and each cell's validation trace for
complete denominators and seed variation. No architecture, tokenizer,
optimizer, schedule or sealed-test boundary was changed.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")

    error_breakdown = {
        result["cell"]: {
            "source": result["source"],
            "condition": result["condition"],
            "seed": result["seed"],
            "training_gate_pass": result["training_gate_pass"],
            "train": result["train_breakdown"],
            "validation": result["validation_breakdown"],
        }
        for result in results
    }
    write_json(OUTPUT / "ERROR_BREAKDOWN.json", error_breakdown)

    hash_paths = [
        PROTOCOL,
        NOVELTY,
        Path(__file__).resolve(),
        pair.TRAIN_SOURCE,
        pair.VALIDATION_SOURCE,
        FIXED_VALIDATION,
        FIXED_TRAIN_REFERENCE,
        CROSSOVER_OUTPUT / "SUMMARY.json",
        CROSSOVER_OUTPUT / "HASHES.json",
        CROSSOVER_OUTPUT / "FINALIZATION.json",
        OUTPUT / "PREFLIGHT.json",
        OUTPUT / "NOVELTY_AUDIT.json",
        OUTPUT / "SUMMARY.json",
        OUTPUT / "REPORT_FINAL.md",
        OUTPUT / "SCIENTIFIC_INTERPRETATION.md",
        OUTPUT / "ERROR_BREAKDOWN.json",
        OUTPUT / "data" / "M2_validation.jsonl",
        OUTPUT / "data" / "D8_train.jsonl",
        OUTPUT / "data" / "D12_train.jsonl",
        OUTPUT / "data" / "GRAPH_METADATA.json",
    ]
    for condition in ("D8", "D12"):
        for seed in SEEDS:
            cell_dir = OUTPUT / f"{condition}_SEED{seed}"
            hash_paths.extend([
                cell_dir / "RESULTS.json",
                cell_dir / "PREDICTIONS.csv",
                cell_dir / "TRAIN_PREDICTIONS_ENRICHED.csv",
                cell_dir / "VALIDATION_PREDICTIONS.csv",
                cell_dir / "TRAINING_TRACE.jsonl",
                cell_dir / "checkpoint.pt",
            ])
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    write_json(OUTPUT / "FINALIZATION.json", {
        "experiment": summary["experiment"],
        "status": "completed",
        "verification_pass": True,
        "training_performed": True,
        "sealed_test_accessed": False,
        "prior_checkpoints_modified": False,
        "prior_hash_verification_failures": prior_hash_failures,
        "reused_arm_count": 2,
        "new_arm_count": 4,
        "total_arm_count": len(results),
        "training_gate_passes_new": summary["training_gate_passes_new"],
        "training_gate_passes_combined": summary["training_gate_passes_combined"],
        "hashes": rel(OUTPUT / "HASHES.json"),
    })
    print(json.dumps({"experiment": summary["experiment"], "status": "completed", "new_arms": 4, "total_arms": len(results), "training_gate_passes_new": summary["training_gate_passes_new"], "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
