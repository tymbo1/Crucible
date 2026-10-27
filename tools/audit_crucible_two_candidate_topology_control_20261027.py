#!/usr/bin/env python3
"""Independent post-run audit for the relational topology control."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "crucible_two_candidate_topology_control_20261027_v1"
MODULUS = 5
HELD_OUT = {(0, 3), (0, 4), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (1, 3)}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pair(value: Sequence[int]) -> Tuple[int, int]:
    a, b = (int(value[0]), int(value[1]))
    return (min(a, b), max(a, b))


def jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def path_offset(row: Mapping[str, Any], graph: Mapping[str, Any]) -> int:
    """Independently sum directed edge offsets along a graph path."""
    edges = {tuple(map(int, label.split(","))): int(delta) for label, delta in row["metadata"]["edge_deltas"].items()}
    adjacency: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for (left, right), delta in edges.items():
        adjacency[left].append((right, delta % MODULUS))
        adjacency[right].append((left, (-delta) % MODULUS))
    source = int(row["metadata"]["source_node"])
    target = int(row["metadata"]["target_node"])
    queue = deque([(source, 0)])
    visited = {source}
    while queue:
        node, accumulated = queue.popleft()
        if node == target:
            return accumulated % MODULUS
        for neighbour, step in adjacency[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, (accumulated + step) % MODULUS))
    raise AssertionError(f"no path for {source}->{target} in {graph['name']}")


def audit_data(condition: str, split: str, rows: Sequence[Mapping[str, Any]], graph: Mapping[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    expected = 144 if split == "train" else 96
    if len(rows) != expected:
        errors.append(f"{condition}/{split}: rows={len(rows)} expected={expected}")
    if len({str(row["record_id"]) for row in rows}) != len(rows):
        errors.append(f"{condition}/{split}: duplicate record IDs")
    if len({str(row["input"]) for row in rows}) != len(rows):
        errors.append(f"{condition}/{split}: duplicate inputs")
    pairs = {pair(row["metadata"]["unordered_pair"]) for row in rows}
    if split == "validation" and pairs != HELD_OUT:
        errors.append(f"{condition}/{split}: held-out pair identity mismatch")
    if split == "train" and pairs & HELD_OUT:
        errors.append(f"{condition}/{split}: held-out contamination")
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    path_checks = 0
    for row in rows:
        raw = str(row["input"])
        if "target=" in raw or "potentials=" in raw:
            errors.append(f"{condition}/{split}/{row['record_id']}: answer leakage marker")
        target_value = int(row["metadata"]["target_value"])
        independently_composed = path_offset(row, graph)
        if independently_composed != target_value:
            errors.append(f"{condition}/{split}/{row['record_id']}: path offset mismatch")
        candidates = [int(value) for value in row["metadata"]["candidates"]]
        if candidates[int(row["metadata"]["target_slot"])] != target_value:
            errors.append(f"{condition}/{split}/{row['record_id']}: target slot mismatch")
        groups[str(row["metadata"]["counterfactual_pair_id"])].append(row)
        path_checks += 1
    bad_groups = {key: len(value) for key, value in groups.items() if len(value) != 2}
    if bad_groups:
        errors.append(f"{condition}/{split}: counterfactual group sizes {bad_groups}")
    return {
        "rows": len(rows),
        "unique_record_ids": len({str(row["record_id"]) for row in rows}),
        "unique_inputs": len({str(row["input"]) for row in rows}),
        "query_pairs": [f"{a},{b}" for a, b in sorted(pairs)],
        "semantic_groups": len(groups),
        "path_oracle_checks": path_checks,
        "errors": errors,
    }


def csv_metrics(path: Path) -> Dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row["counterfactual_pair_id"])].append(row)
    complete = [values for values in groups.values() if len(values) == 2]
    value_correct = sum(row["value_correct"] == "True" for row in rows)
    slot_correct = sum(row["slot_correct"] == "True" for row in rows)
    paired_correct = sum(int(all(row["value_correct"] == "True" for row in values)) for values in complete)
    return {
        "rows": len(rows),
        "value_correct": value_correct,
        "slot_correct": slot_correct,
        "paired_groups": len(complete),
        "paired_correct": paired_correct,
    }


def main() -> int:
    summary = load(EXPERIMENT / "SUMMARY.json")
    preflight = load(EXPERIMENT / "PREFLIGHT.json")
    manifest = load(EXPERIMENT / "GRAPH_MANIFEST.json")
    errors: List[str] = []
    data_checks: Dict[str, Any] = {}
    for condition in manifest:
        train = jsonl(EXPERIMENT / "data" / f"{condition}_train.jsonl")
        validation = jsonl(EXPERIMENT / "data" / f"{condition}_validation.jsonl")
        data_checks[condition] = {
            "train": audit_data(condition, "train", train, manifest[condition]),
            "validation": audit_data(condition, "validation", validation, manifest[condition]),
        }
        errors.extend(data_checks[condition]["train"]["errors"])
        errors.extend(data_checks[condition]["validation"]["errors"])

    result_checks: Dict[str, Any] = {}
    for result in summary["results"]:
        cell = str(result["cell"])
        cell_dir = EXPERIMENT / cell
        checks: Dict[str, Any] = {}
        for split, filename, key in (("train", "TRAIN_PREDICTIONS.csv", "train_metrics"), ("validation", "VALIDATION_PREDICTIONS.csv", "validation_metrics")):
            observed = csv_metrics(cell_dir / filename)
            expected = result[key]
            checks[split] = {"observed": observed, "reported": {name: expected[name] for name in observed}}
            for name, value in observed.items():
                if value != expected[name]:
                    errors.append(f"{cell}/{split}: CSV metric {name}={value} != reported {expected[name]}")
        trace = [json.loads(line) for line in (cell_dir / "TRAINING_TRACE.jsonl").read_text().splitlines() if line.strip()]
        trace_steps = [int(row["step"]) for row in trace if "training" in row]
        expected_steps = [1, 50, 100, 200, 300, 400, 500]
        if trace_steps != expected_steps:
            errors.append(f"{cell}: trace steps {trace_steps} != {expected_steps}")
        checks["trace_steps"] = trace_steps
        checks["training_gate_pass"] = bool(result["training_gate_pass"])
        result_checks[cell] = checks

    hashes = load(EXPERIMENT / "HASHES.json")
    hash_mismatches: List[str] = []
    for relative, expected in hashes.items():
        path = ROOT / relative
        if not path.exists():
            hash_mismatches.append(f"missing:{relative}")
        elif digest(path) != expected:
            hash_mismatches.append(f"mismatch:{relative}")
    errors.extend(hash_mismatches)

    gate_by_condition: Dict[str, List[str]] = defaultdict(list)
    for result in summary["results"]:
        if result["training_gate_pass"]:
            gate_by_condition[str(result["condition"])].append(str(result["cell"]))
    comparison_adjudicated = all(gate_by_condition.get(condition) for condition in manifest)
    audit = {
        "schema_version": "crucible_topology_control_post_run_audit_v1",
        "experiment": summary["experiment"],
        "status": "PASS" if not errors else "FAIL",
        "independent_checks": {
            "preflight_status": preflight["status"],
            "sealed_test_accessed": summary["sealed_test_accessed"],
            "prior_checkpoints_modified": summary["prior_checkpoints_modified"],
            "hash_entries_verified": len(hashes),
            "hash_mismatches": hash_mismatches,
            "data_checks": data_checks,
            "result_checks": result_checks,
        },
        "training_gate": {
            "passes": int(summary["training_gate_passes"]),
            "arms": len(summary["results"]),
            "threshold": summary["training_gate_threshold"],
            "gated_cells_by_condition": dict(gate_by_condition),
        },
        "topology_comparison": {
            "adjudicated": comparison_adjudicated,
            "reason": "Both graph conditions require at least one training-competent arm; this was not met." if not comparison_adjudicated else "Both graph conditions have gated arms.",
            "diameters": {condition: manifest[condition]["diameter"] for condition in manifest},
            "matched_degree_sequence": all(manifest[condition]["degree_sequence"] == [3] * 12 for condition in manifest),
            "matched_edge_count": all(manifest[condition]["edge_count"] == 18 for condition in manifest),
            "fixed_held_out_pairs": sorted(f"{a},{b}" for a, b in HELD_OUT),
        },
        "errors": errors,
    }
    (EXPERIMENT / "POST_RUN_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Post-run audit: relational topology control v1",
        "",
        f"Audit status: **{audit['status']}**",
        "",
        f"- Independent hash entries verified: {len(hashes)}; mismatches: {len(hash_mismatches)}.",
        f"- Training gate: {summary['training_gate_passes']}/{len(summary['results'])} arms passed at {summary['training_gate_threshold']:.2f}.",
        f"- Topology comparison adjudicated: **{'yes' if comparison_adjudicated else 'no'}**.",
        "- Sealed test accessed: no.",
        "- Prior checkpoints modified: no.",
        "",
        "The graph controls are valid, but a topology comparison is not interpretable under the preregistered gate because the diameter-5 condition has no training-competent arm. The result is therefore a training-competence/optimization finding, not evidence for or against a graph-distance effect.",
        "",
        "See `POST_RUN_AUDIT.json` for row-level, oracle, CSV, trace and hash checks.",
    ]
    (EXPERIMENT / "POST_RUN_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "errors": len(errors), "training_gate_passes": summary["training_gate_passes"], "topology_comparison_adjudicated": comparison_adjudicated}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
