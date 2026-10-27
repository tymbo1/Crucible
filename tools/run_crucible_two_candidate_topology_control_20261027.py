#!/usr/bin/env python3
"""Run the preregistered matched-degree relational topology control."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import itertools
import json
import random
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUNNER = ROOT / "tools" / "run_crucible_shared_candidate_scoring_cardinality_20261026.py"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_TOPOLOGY_CONTROL_PROTOCOL_20261027.md"
NOVELTY = ROOT / "docs" / "CRUCIBLE_TWO_CANDIDATE_TOPOLOGY_CONTROL_NOVELTY_AUDIT_20261027.json"
OUTPUT = ROOT / "experiments" / "crucible_two_candidate_topology_control_20261027_v1"

SEEDS = (42, 1337)
DATA_SEED = 20261027
STEPS = 500
BATCH_SIZE = 12
TRAIN_THRESHOLD = 0.95
CHECKPOINT_STEPS = (1, 50, 100, 200, 300, 400, 500)
MODULUS = 5
NODES = tuple(range(12))
ANSWER_VALUES = tuple(range(MODULUS))
HELD_OUT_PAIRS = ((0, 3), (0, 4), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 3))

GRAPH_FAMILIES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "P12_PRISM": (
        (0, 1), (0, 5), (0, 6), (1, 2), (1, 7), (2, 3), (2, 8), (3, 4), (3, 9),
        (4, 5), (4, 10), (5, 11), (6, 7), (6, 11), (7, 8), (8, 9), (9, 10), (10, 11),
    ),
    "R12_DIAMETER5": (
        (0, 1), (0, 2), (0, 6), (1, 2), (1, 9), (2, 9), (3, 4), (3, 5), (3, 10),
        (4, 8), (4, 11), (5, 7), (5, 11), (6, 8), (6, 9), (7, 8), (7, 10), (10, 11),
    ),
}


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


source = import_module("topology_source_runner", SOURCE_RUNNER)
torch = source.torch
nn = torch.nn
stage_a = source.stage_a


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


def graph_distances(edges: Sequence[Tuple[int, int]]) -> Dict[int, Dict[int, int]]:
    graph = {node: set() for node in NODES}
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)
    result: Dict[int, Dict[int, int]] = {}
    for start in NODES:
        distances = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(graph[node]):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        result[start] = distances
    return result


def graph_manifest(name: str, edges: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    distances = graph_distances(edges)
    finite = [distances[left][right] for left, right in itertools.combinations(NODES, 2)]
    return {
        "name": name,
        "nodes": list(NODES),
        "edges": [f"{min(left, right)},{max(left, right)}" for left, right in edges],
        "edge_count": len(edges),
        "degree_sequence": [sum(node in edge for edge in edges) for node in NODES],
        "component_count": 1 if all(node in distances[0] for node in NODES) else None,
        "diameter": max(finite),
        "distance_distribution": {str(distance): finite.count(distance) for distance in sorted(set(finite))},
        "held_out_distances": {f"{left},{right}": distances[left][right] for left, right in HELD_OUT_PAIRS},
    }


def canonical_pair(pair_value: Sequence[int]) -> Tuple[int, int]:
    return tuple(sorted((int(pair_value[0]), int(pair_value[1]))))


def input_text(edges: Sequence[Tuple[int, int]], deltas: Mapping[Tuple[int, int], int], source_node: int, target_node: int, candidates: Sequence[int], ordinal: int) -> str:
    facts = [f"e{left},{right},{deltas[(left, right)]};" for left, right in edges]
    random.Random(DATA_SEED + ordinal * 7919).shuffle(facts)
    candidate_text = " ".join(f"id={value};" for value in candidates)
    return "F:" + "".join(facts) + f" sid={source_node};qid={target_node}; {candidate_text} decision=Q;"


def target_value(potentials: Sequence[int], source_node: int, target_node: int) -> int:
    return (int(potentials[target_node]) - int(potentials[source_node])) % MODULUS


def deltas_for(edges: Sequence[Tuple[int, int]], potentials: Sequence[int]) -> Dict[Tuple[int, int], int]:
    return {(left, right): (potentials[right] - potentials[left]) % MODULUS for left, right in edges}


def query_pairs() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    held = {canonical_pair(pair_value) for pair_value in HELD_OUT_PAIRS}
    all_pairs = list(itertools.combinations(NODES, 2))
    train_unordered = [pair_value for pair_value in all_pairs if pair_value not in held][:36]
    train = [ordered for pair_value in train_unordered for ordered in (pair_value, (pair_value[1], pair_value[0]))]
    validation = [ordered for pair_value in HELD_OUT_PAIRS for ordered in (pair_value, (pair_value[1], pair_value[0]))]
    return train, validation


def make_records(condition: str, split: str, edges: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    train_queries, validation_queries = query_pairs()
    queries = train_queries if split == "train" else validation_queries
    semantic_count = 72 if split == "train" else 48
    rows: List[Dict[str, Any]] = []
    ordinal = 0
    for semantic_index in range(semantic_count):
        source_node, target_node = queries[semantic_index % len(queries)]
        rng = random.Random(DATA_SEED + semantic_index * 104729 + (0 if split == "train" else 1_000_003))
        potentials = [rng.randrange(MODULUS) for _ in NODES]
        desired = semantic_index % MODULUS
        potentials[target_node] = (potentials[source_node] + desired) % MODULUS
        deltas = deltas_for(edges, potentials)
        value = target_value(potentials, source_node, target_node)
        if value != desired:
            raise RuntimeError("potential construction failed")
        semantic_key = f"topology_offset|{condition}|{split}|{semantic_index}|{source_node}|{target_node}|{potentials}"
        group_id = hashlib.sha256(semantic_key.encode("utf-8")).hexdigest()[:20]
        permutations = (tuple(ANSWER_VALUES), tuple(reversed(ANSWER_VALUES)))
        for candidate_order, candidates in enumerate(permutations):
            target_slot = candidates.index(value)
            record_id = hashlib.sha256(f"{semantic_key}|{candidate_order}".encode("utf-8")).hexdigest()[:20]
            rows.append({
                "schema_version": "crucible_two_candidate_topology_control_v1",
                "record_id": record_id,
                "interface": "topology_offset",
                "split": split,
                "layout": "GRAPH_FACTS_QUERY",
                "input": input_text(edges, deltas, source_node, target_node, candidates, ordinal),
                "target": str(target_slot),
                "oracle": {"kind": "modular_path_offset", "target_value": value, "target_slot": target_slot, "modulus": MODULUS},
                "metadata": {
                    "condition": condition,
                    "source_node": source_node,
                    "target_node": target_node,
                    "unordered_pair": list(canonical_pair((source_node, target_node))),
                    "query_direction": f"{source_node}->{target_node}",
                    "candidates": list(candidates),
                    "target_value": value,
                    "target_slot": target_slot,
                    "counterfactual_pair_id": group_id,
                    "potentials": potentials,
                    "edge_deltas": {f"{left},{right}": delta for (left, right), delta in deltas.items()},
                    "graph_distance": graph_distances(edges)[source_node][target_node],
                },
            })
            ordinal += 1
    return rows


def prepare_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    candidate_pattern = re.compile(rb"(?<![A-Za-z_])id=([0-9])")
    query_pattern = re.compile(rb"(?<![A-Za-z_])qid=([0-9]+)")
    decision_pattern = re.compile(rb"decision=(Q)")
    prepared: List[Dict[str, Any]] = []
    for original in records:
        item = dict(original)
        raw = source.auth.base.base.parser_input(item).encode("utf-8")
        candidates = list(candidate_pattern.finditer(raw))
        query = query_pattern.search(raw)
        decision = decision_pattern.search(raw)
        expected = [int(value) for value in item["metadata"]["candidates"]]
        found = [int(match.group(1)) for match in candidates]
        if found != expected or query is None or decision is None or len(found) != 5:
            raise RuntimeError(f"topology span failure: {item['record_id']} found={found} expected={expected}")
        full = [stage_a.BOS_ID] + stage_a.serialize_bytes(raw)
        if len(full) > stage_a.CONTEXT_LENGTH:
            raise RuntimeError(f"topology context overflow: {item['record_id']}")
        item["_parser_full"] = full
        item["_parser_length"] = len(full)
        item["_candidate_positions"] = [1 + match.start(1) for match in candidates]
        item["_decision_position"] = 1 + decision.start(1)
        prepared.append(item)
    return prepared


def batchify(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    selected = [records[index] for index in indices]
    max_len = max(int(row["_parser_length"]) for row in selected)
    x = torch.full((len(selected), max_len), stage_a.PAD_ID, dtype=torch.long)
    positions = torch.zeros((len(selected), source.MAX_CANDIDATES), dtype=torch.long)
    mask = torch.zeros((len(selected), source.MAX_CANDIDATES), dtype=torch.bool)
    targets: List[int] = []
    decisions: List[int] = []
    lengths: List[int] = []
    for row_index, item in enumerate(selected):
        full = torch.tensor(item["_parser_full"], dtype=torch.long)
        x[row_index, :full.numel()] = full
        lengths.append(int(full.numel()))
        candidate_positions = [int(value) for value in item["_candidate_positions"]]
        positions[row_index, :len(candidate_positions)] = torch.tensor(candidate_positions, dtype=torch.long)
        mask[row_index, :len(candidate_positions)] = True
        targets.append(int(item["metadata"]["target_slot"]))
        decisions.append(int(item["_decision_position"]))
    return {"x": x, "lengths": torch.tensor(lengths, dtype=torch.long), "candidate_positions": positions, "candidate_mask": mask, "target": torch.tensor(targets, dtype=torch.long), "decision_pos": torch.tensor(decisions, dtype=torch.long)}


def evaluate(model: Any, records: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(prepared), BATCH_SIZE):
        indices = list(range(start, min(start + BATCH_SIZE, len(prepared))))
        batch = batchify(prepared, indices)
        with torch.inference_mode():
            predicted = model.decision_logits(batch).argmax(dim=-1).tolist()
        for local, index in enumerate(indices):
            item = records[index]
            target_value = int(item["metadata"]["target_value"])
            predicted_slot = int(predicted[local])
            candidates = [int(value) for value in item["metadata"]["candidates"]]
            predicted_value = candidates[predicted_slot] if predicted_slot < len(candidates) else None
            rows.append({
                "record_id": str(item["record_id"]),
                "condition": item["metadata"]["condition"],
                "split": item["split"],
                "counterfactual_pair_id": item["metadata"]["counterfactual_pair_id"],
                "source_node": item["metadata"]["source_node"],
                "target_node": item["metadata"]["target_node"],
                "unordered_pair": ",".join(str(value) for value in item["metadata"]["unordered_pair"]),
                "query_direction": item["metadata"]["query_direction"],
                "graph_distance": item["metadata"]["graph_distance"],
                "candidates": ",".join(str(value) for value in candidates),
                "target_value": target_value,
                "predicted_value": predicted_value,
                "target_slot": int(item["metadata"]["target_slot"]),
                "predicted_slot": predicted_slot,
                "value_correct": predicted_value == target_value,
                "slot_correct": predicted_slot == int(item["metadata"]["target_slot"]),
            })
    return rows


def metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    correct_value = sum(int(bool(row["value_correct"])) for row in rows)
    correct_slot = sum(int(bool(row["slot_correct"])) for row in rows)
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["counterfactual_pair_id"])].append(row)
    complete = [values for values in groups.values() if len(values) == 2]
    paired = sum(int(all(bool(row["value_correct"]) for row in values)) for values in complete)
    flips = sum(int(values[0]["predicted_value"] != values[1]["predicted_value"]) for values in complete)
    by_distance: Dict[str, Dict[str, Any]] = {}
    for distance in sorted({int(row["graph_distance"]) for row in rows}):
        subset = [row for row in rows if int(row["graph_distance"]) == distance]
        n = len(subset)
        c = sum(int(bool(row["value_correct"])) for row in subset)
        by_distance[str(distance)] = {"correct": c, "rows": n, "accuracy": c / n if n else None}
    return {
        "rows": len(rows),
        "value_correct": correct_value,
        "value_accuracy": correct_value / len(rows) if rows else None,
        "slot_correct": correct_slot,
        "slot_accuracy": correct_slot / len(rows) if rows else None,
        "paired_groups": len(complete),
        "paired_correct": paired,
        "paired_accuracy": paired / len(complete) if complete else None,
        "candidate_order_flips": flips,
        "by_distance": by_distance,
    }


def train_cell(cell: str, condition: str, train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = source.CandidateReadout(seed, "SHARED_SCORER")
    optimizer = source.stage_a.optimizer_for(model)
    prepared_train = prepare_records(train_records)
    prepared_validation = prepare_records(validation_records)
    batches = source.schedule(len(prepared_train))
    output = OUTPUT / cell
    output.mkdir(parents=True, exist_ok=True)
    trace: List[Dict[str, Any]] = []
    for step, indices in enumerate(batches, start=1):
        model.train()
        batch = batchify(prepared_train, indices)
        optimizer.zero_grad(set_to_none=True)
        loss = source.F.cross_entropy(model.decision_logits(batch), batch["target"])
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), source.stage_a.CLIP_NORM).item())
        optimizer.step()
        entry: Dict[str, Any] = {"cell": cell, "condition": condition, "seed": seed, "step": step, "loss": float(loss.detach().item()), "grad_norm_preclip": grad_norm}
        if step in CHECKPOINT_STEPS:
            train_rows = evaluate(model, train_records, prepared_train)
            validation_rows = evaluate(model, validation_records, prepared_validation)
            entry["training"] = metrics(train_rows)
            entry["validation"] = metrics(validation_rows)
        trace.append(entry)
    train_rows = evaluate(model, train_records, prepared_train)
    validation_rows = evaluate(model, validation_records, prepared_validation)
    checkpoint = output / "checkpoint.pt"
    torch.save({"cell": cell, "condition": condition, "seed": seed, "variant": "SHARED_SCORER", "model": model.state_dict()}, checkpoint)
    write_jsonl(output / "TRAINING_TRACE.jsonl", trace)
    write_csv(output / "TRAIN_PREDICTIONS.csv", train_rows)
    write_csv(output / "VALIDATION_PREDICTIONS.csv", validation_rows)
    result = {
        "cell": cell, "condition": condition, "seed": seed, "source": "new", "variant": "SHARED_SCORER", "steps": STEPS, "batch_size": BATCH_SIZE,
        "train_metrics": metrics(train_rows), "validation_metrics": metrics(validation_rows), "training_gate_threshold": TRAIN_THRESHOLD,
        "training_gate_pass": metrics(train_rows)["value_accuracy"] >= TRAIN_THRESHOLD, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "trace_steps": list(CHECKPOINT_STEPS),
    }
    write_json(output / "RESULTS.json", result)
    del optimizer, model
    return result


def validate_records(condition: str, split: str, records: Sequence[Mapping[str, Any]], edges: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    expected_rows = 144 if split == "train" else 96
    if len(records) != expected_rows or len({str(row["record_id"]) for row in records}) != expected_rows or len({str(row["input"]) for row in records}) != expected_rows:
        raise RuntimeError(f"{condition}/{split}: row identity/count failure")
    if any(int(row["metadata"]["target_value"]) not in ANSWER_VALUES for row in records):
        raise RuntimeError(f"{condition}/{split}: target value failure")
    if any(int(row["metadata"]["target_value"]) != target_value(row["metadata"]["potentials"], row["metadata"]["source_node"], row["metadata"]["target_node"]) for row in records):
        raise RuntimeError(f"{condition}/{split}: potential oracle mismatch")
    expected_pairs = {canonical_pair(pair_value) for pair_value in HELD_OUT_PAIRS} if split == "validation" else set(query_pairs()[0][::2])
    actual_pairs = {canonical_pair(row["metadata"]["unordered_pair"]) for row in records}
    if split == "validation" and actual_pairs != expected_pairs:
        raise RuntimeError(f"{condition}/{split}: fixed held-out pair mismatch")
    if split == "train" and actual_pairs & {canonical_pair(pair_value) for pair_value in HELD_OUT_PAIRS}:
        raise RuntimeError(f"{condition}/{split}: held-out pair contamination")
    counterfactual_sizes = Counter(str(row["metadata"]["counterfactual_pair_id"]) for row in records)
    if set(counterfactual_sizes.values()) != {2}:
        raise RuntimeError(f"{condition}/{split}: candidate-order group failure")
    return {"rows": len(records), "unique_inputs": len({str(row["input"]) for row in records}), "target_value_counts": {str(key): value for key, value in sorted(Counter(int(row["metadata"]["target_value"]) for row in records).items())}, "semantic_groups": len(counterfactual_sizes), "query_pairs": sorted(edge_label for edge_label in {f"{min(a,b)},{max(a,b)}" for a,b in actual_pairs})}


def edge_label(pair_value: Sequence[int]) -> str:
    return f"{min(pair_value)},{max(pair_value)}"


def main() -> int:
    execute = "--execute" in sys.argv[1:]
    if OUTPUT.exists() and (OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"refusing to overwrite completed output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    graph_info = {name: graph_manifest(name, edges) for name, edges in GRAPH_FAMILIES.items()}
    if any(info["edge_count"] != 18 or info["degree_sequence"] != [3] * 12 or info["component_count"] != 1 for info in graph_info.values()):
        raise RuntimeError(f"graph feasibility failure: {graph_info}")
    held = {canonical_pair(pair_value) for pair_value in HELD_OUT_PAIRS}
    for name, edges in GRAPH_FAMILIES.items():
        if any(pair_value in {canonical_pair(edge) for edge in edges} for pair_value in held):
            raise RuntimeError(f"{name}: held-out pair is a direct edge")

    records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    audits: Dict[str, Any] = {}
    for condition, edges in GRAPH_FAMILIES.items():
        train = make_records(condition, "train", edges)
        validation = make_records(condition, "validation", edges)
        train_audit = validate_records(condition, "train", train, edges)
        validation_audit = validate_records(condition, "validation", validation, edges)
        train_pairs = {canonical_pair(row["metadata"]["unordered_pair"]) for row in train}
        validation_pairs = {canonical_pair(row["metadata"]["unordered_pair"]) for row in validation}
        if train_pairs & validation_pairs or len(train) != 144 or len(validation) != 96:
            raise RuntimeError(f"{condition}: split contamination or row count failure")
        for row in train + validation:
            raw = row["input"]
            if f"target={row['metadata']['target_value']}" in raw or "potentials=" in raw:
                raise RuntimeError(f"{condition}: answer leakage in input")
        prepare_records(train)
        prepare_records(validation)
        records[condition] = {"train": train, "validation": validation}
        audits[condition] = {"train": train_audit, "validation": validation_audit, "split_pair_overlap": 0, "graph": graph_info[condition]}

    preflight = {
        "experiment": "crucible_two_candidate_topology_control_20261027_v1", "status": "preflight_passed", "execute_requested": execute,
        "protocol": rel(PROTOCOL), "novelty_audit": rel(NOVELTY), "data_seed": DATA_SEED, "modulus": MODULUS, "held_out_pairs": [edge_label(pair_value) for pair_value in HELD_OUT_PAIRS],
        "graphs": graph_info, "audits": audits, "seeds": list(SEEDS), "steps": STEPS, "batch_size": BATCH_SIZE, "schedule": "cyclic", "training_gate_threshold": TRAIN_THRESHOLD,
        "training_performed": False, "sealed_test_accessed": False, "prior_checkpoints_modified": False,
    }
    write_json(OUTPUT / "PREFLIGHT.json", preflight)
    write_json(OUTPUT / "NOVELTY_AUDIT.json", json.loads(NOVELTY.read_text(encoding="utf-8")))
    write_json(OUTPUT / "GRAPH_MANIFEST.json", graph_info)
    for condition, data in records.items():
        write_jsonl(OUTPUT / "data" / f"{condition}_train.jsonl", data["train"])
        write_jsonl(OUTPUT / "data" / f"{condition}_validation.jsonl", data["validation"])
    if not execute:
        print(json.dumps({"experiment": preflight["experiment"], "status": "preflight_passed", "new_arms": 4, "training_performed": False, "execute_with": "--execute"}, sort_keys=True))
        return 0

    results: List[Dict[str, Any]] = []
    for condition in GRAPH_FAMILIES:
        for seed in SEEDS:
            results.append(train_cell(f"{condition}_SEED{seed}", condition, records[condition]["train"], records[condition]["validation"], seed))
    summary = {
        "experiment": "crucible_two_candidate_topology_control_20261027_v1", "status": "completed", "protocol": rel(PROTOCOL), "novelty_audit": rel(NOVELTY), "preflight": rel(OUTPUT / "PREFLIGHT.json"), "graphs": graph_info, "results": results,
        "seeds": list(SEEDS), "steps": STEPS, "batch_size": BATCH_SIZE, "schedule": "cyclic", "training_gate_threshold": TRAIN_THRESHOLD, "training_performed": True, "sealed_test_accessed": False, "prior_checkpoints_modified": False,
        "training_gate_passes": sum(int(result["training_gate_pass"]) for result in results),
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
    report = ["# Crucible Relational Topology Control v1", "", "Matched degree-3, 18-edge graph families on the explicit modular edge-offset composition task.", "", "| Condition | Seed | Train | Gate | Validation value | Paired |", "|---|---:|---:|---|---:|---:|"]
    for result in results:
        train = result["train_metrics"]; validation = result["validation_metrics"]
        report.append(f"| {result['condition']} | {result['seed']} | {train['value_correct']}/{train['rows']} ({train['value_accuracy']:.3f}) | {'PASS' if result['training_gate_pass'] else 'FAIL'} | {validation['value_correct']}/{validation['rows']} ({validation['value_accuracy']:.3f}) | {validation['paired_correct']}/{validation['paired_groups']} |")
    report += ["", "## Interpretation boundary", "", "Validation is interpreted only after the 95% training gate. This task has an explicitly path-compositional oracle, but a positive result would establish behavior on this synthetic offset task, not a general neural mechanism or broad graph reasoning.", "", "## Artifacts", "", f"- [Protocol](../../{rel(PROTOCOL)})", f"- [Novelty audit](../../{rel(NOVELTY)})", "- [Preflight](PREFLIGHT.json)", "- [Graph manifest](GRAPH_MANIFEST.json)", "- [Summary](SUMMARY.json)"]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text("""# Scientific Interpretation\n\nThe graph families have matched vertex degree and edge count but different shortest-path structure. The task target is a modular offset recoverable by summing path-consistent edge offsets, so graph paths are logically relevant. Interpret only gated arms; retain pair-, direction- and distance-level denominators.\n\nNo result from this task should be pooled with the direct identifier-matching pair-coverage lineage or described as a general reasoning result.\n""", encoding="utf-8")
    error_breakdown = {result["cell"]: result for result in results}
    write_json(OUTPUT / "ERROR_BREAKDOWN.json", error_breakdown)
    hash_paths = [PROTOCOL, NOVELTY, Path(__file__).resolve(), OUTPUT / "PREFLIGHT.json", OUTPUT / "NOVELTY_AUDIT.json", OUTPUT / "GRAPH_MANIFEST.json", OUTPUT / "SUMMARY.json", OUTPUT / "REPORT_FINAL.md", OUTPUT / "SCIENTIFIC_INTERPRETATION.md", OUTPUT / "ERROR_BREAKDOWN.json"]
    for condition in GRAPH_FAMILIES:
        hash_paths.extend([OUTPUT / "data" / f"{condition}_train.jsonl", OUTPUT / "data" / f"{condition}_validation.jsonl"])
        for seed in SEEDS:
            cell = OUTPUT / f"{condition}_SEED{seed}"
            hash_paths.extend([cell / "RESULTS.json", cell / "TRAINING_TRACE.jsonl", cell / "TRAIN_PREDICTIONS.csv", cell / "VALIDATION_PREDICTIONS.csv", cell / "checkpoint.pt"])
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    write_json(OUTPUT / "FINALIZATION.json", {"experiment": summary["experiment"], "status": "completed", "verification_pass": True, "training_performed": True, "sealed_test_accessed": False, "prior_checkpoints_modified": False, "training_gate_passes": summary["training_gate_passes"], "hashes": rel(OUTPUT / "HASHES.json")})
    print(json.dumps({"experiment": summary["experiment"], "status": "completed", "arms": len(results), "training_gate_passes": summary["training_gate_passes"], "sealed_test_accessed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
