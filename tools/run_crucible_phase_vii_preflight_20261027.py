#!/usr/bin/env python3
"""Run the non-training preflight for Crucible Phase VII.

The preflight validates frozen data, N0 reference checkpoints, graph cycle
bases and the structured-input leakage boundary. It deliberately does not
instantiate or train a Phase VII neural model.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "crucible_two_candidate_topology_control_20261027_v1"
N0 = ROOT / "experiments" / "crucible_two_candidate_topology_duration_extension_20261027_v1"
OUTPUT = ROOT / "experiments" / "crucible_phase_vii_neural_shared_latent_relational_transfer_20261027_v1"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_PHASE_VII_NEURAL_SHARED_LATENT_PROTOCOL_20261027.md"
LEAKAGE_SPEC = ROOT / "docs" / "CRUCIBLE_PHASE_VII_NEURAL_SHARED_LATENT_LEAKAGE_AUDIT_20261027.json"
NOVELTY = ROOT / "docs" / "CRUCIBLE_PHASE_VII_NEURAL_SHARED_LATENT_NOVELTY_AUDIT_20261027.json"
CONDITIONS = ("P12_PRISM", "R12_DIAMETER5")
SEEDS = (42, 1337)
MODULUS = 5
EXPECTED_PARAM_COUNT = 22106906
FORBIDDEN_INPUT_TERMS = ("potentials", "target_value", "target_slot", "oracle", "held_out", "validation")
EDGE_PATTERN = re.compile(r"e(\d+),(\d+),(\d+);")
QUERY_PATTERN = re.compile(r"sid=(\d+);qid=(\d+);")
CANDIDATE_PATTERN = re.compile(r"(?<![A-Za-z_])id=(\d+);")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def canonical_edge(left: int, right: int) -> Tuple[int, int]:
    return (min(left, right), max(left, right))


def load_records(condition: str, split: str) -> List[Dict[str, Any]]:
    path = SOURCE / "data" / f"{condition}_{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def graph_from_manifest(info: Mapping[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(int(node) for node in info["nodes"])
    graph.add_edges_from(canonical_edge(*map(int, edge.split(","))) for edge in info["edges"])
    return graph


def normalize_cycle(cycle: Sequence[int]) -> List[int]:
    values = list(map(int, cycle))
    rotations = [values[index:] + values[:index] for index in range(len(values))]
    reversed_values = list(reversed(values))
    rotations.extend(reversed_values[index:] + reversed_values[:index] for index in range(len(values)))
    return list(min(tuple(item) for item in rotations))


def cycle_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for condition in CONDITIONS:
        graph = graph_from_manifest(manifest[condition])
        cycles = sorted(normalize_cycle(cycle) for cycle in nx.cycle_basis(graph))
        output[condition] = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "cycle_rank": graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph),
            "cycle_basis": cycles,
            "cycle_basis_sha256": hashlib.sha256(json.dumps(cycles, separators=(",", ":")).encode("utf-8")).hexdigest(),
        }
    return output


def parse_structured_view(text: str) -> Dict[str, Any]:
    edge_matches = EDGE_PATTERN.findall(text)
    edges = {canonical_edge(int(left), int(right)): int(delta) % MODULUS for left, right, delta in edge_matches}
    query_match = QUERY_PATTERN.search(text)
    candidates = [int(value) for value in CANDIDATE_PATTERN.findall(text)]
    return {
        "edges": edges,
        "source": int(query_match.group(1)) if query_match else None,
        "target": int(query_match.group(2)) if query_match else None,
        "candidates": candidates,
        "forbidden_terms": [term for term in FORBIDDEN_INPUT_TERMS if term in text],
    }


def observed_cycle_residuals(view: Mapping[str, Any], cycles: Sequence[Sequence[int]]) -> List[int]:
    residuals: List[int] = []
    edges = view["edges"]
    for cycle in cycles:
        total = 0
        for left, right in zip(cycle, list(cycle[1:]) + [cycle[0]]):
            edge = canonical_edge(left, right)
            offset = int(edges[edge])
            total += offset if left < right else -offset
        residuals.append(total % MODULUS)
    return residuals


def source_hash_audit() -> Dict[str, Any]:
    hashes = load(N0 / "HASHES.json")
    mismatches = []
    for path, expected in hashes.items():
        candidate = ROOT / path
        if not candidate.exists() or digest(candidate) != expected:
            mismatches.append(path)
    return {"entries": len(hashes), "mismatches": mismatches, "pass": not mismatches}


def n0_reference_audit() -> Dict[str, Any]:
    summary = load(N0 / "SUMMARY.json")
    cells: List[Dict[str, Any]] = []
    failures: List[str] = []
    for result in summary["results"]:
        cell = N0 / result["cell"]
        checkpoint = ROOT / result["checkpoint"]
        exists = checkpoint.exists()
        checkpoint_hash_ok = exists and digest(checkpoint) == result["checkpoint_sha256"]
        item = {
            "cell": result["cell"],
            "condition": result["condition"],
            "seed": result["seed"],
            "checkpoint_exists": exists,
            "checkpoint_hash_ok": checkpoint_hash_ok,
            "parameter_count": result["parameter_count"],
            "training_gate_pass": result["training_gate_pass"],
            "training_accuracy": result["train_metrics"]["value_accuracy"],
            "validation_paired_accuracy": result["validation_metrics"]["paired_accuracy"],
        }
        cells.append(item)
        if not exists or not checkpoint_hash_ok:
            failures.append(f"{result['cell']}: checkpoint hash")
        if int(result["parameter_count"]) != EXPECTED_PARAM_COUNT:
            failures.append(f"{result['cell']}: parameter count")
        if not result["training_gate_pass"]:
            failures.append(f"{result['cell']}: training gate")
    return {"summary_status": summary.get("status"), "cells": cells, "failures": failures, "pass": not failures}


def data_and_leakage_audit(manifest: Mapping[str, Any], cycles: Mapping[str, Any]) -> Dict[str, Any]:
    failures: List[str] = []
    condition_reports: Dict[str, Any] = {}
    for condition in CONDITIONS:
        graph = graph_from_manifest(manifest[condition])
        expected_edges = set(graph.edges())
        train = load_records(condition, "train")
        validation = load_records(condition, "validation")
        train_ids = {row["record_id"] for row in train}
        validation_ids = {row["record_id"] for row in validation}
        train_pairs = {tuple(sorted(row["metadata"]["unordered_pair"])) for row in train}
        validation_pairs = {tuple(sorted(row["metadata"]["unordered_pair"])) for row in validation}
        row_reports: List[Dict[str, Any]] = []
        for split, rows in (("train", train), ("validation", validation)):
            for row in rows:
                view = parse_structured_view(row["input"])
                parsed_edges = set(view["edges"])
                if parsed_edges != expected_edges:
                    failures.append(f"{condition}/{split}/{row['record_id']}: edge parse mismatch")
                if view["source"] != int(row["metadata"]["source_node"]) or view["target"] != int(row["metadata"]["target_node"]):
                    failures.append(f"{condition}/{split}/{row['record_id']}: query parse mismatch")
                if view["candidates"] != [int(value) for value in row["metadata"]["candidates"]]:
                    failures.append(f"{condition}/{split}/{row['record_id']}: candidate parse mismatch")
                if view["forbidden_terms"]:
                    failures.append(f"{condition}/{split}/{row['record_id']}: forbidden input terms {view['forbidden_terms']}")
                residuals = observed_cycle_residuals(view, cycles[condition]["cycle_basis"])
                if any(residuals):
                    failures.append(f"{condition}/{split}/{row['record_id']}: cycle residual {residuals}")
                row_reports.append({"split": split, "record_id": row["record_id"], "cycle_residuals": residuals, "parsed_edge_count": len(parsed_edges)})
        condition_reports[condition] = {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_ids": len(train_ids),
            "validation_ids": len(validation_ids),
            "split_id_overlap": len(train_ids & validation_ids),
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "pair_overlap": len(train_pairs & validation_pairs),
            "validation_pairs_expected": sorted([list(pair) for pair in validation_pairs]),
            "row_checks": len(row_reports),
        }
        if train_ids & validation_ids:
            failures.append(f"{condition}: train/validation record overlap")
        if train_pairs & validation_pairs:
            failures.append(f"{condition}: train/validation pair overlap")
        if len(train) != 144 or len(validation) != 96:
            failures.append(f"{condition}: unexpected row count")
    return {"conditions": condition_reports, "failures": failures, "pass": not failures}


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = load(SOURCE / "GRAPH_MANIFEST.json")
    cycles = cycle_manifest(manifest)
    data_audit = data_and_leakage_audit(manifest, cycles)
    reference_audit = n0_reference_audit()
    hashes = source_hash_audit()
    leakage_spec = load(LEAKAGE_SPEC)
    novelty = load(NOVELTY)
    results = {
        "schema_version": "crucible_phase_vii_preflight_v1",
        "experiment": "crucible_phase_vii_neural_shared_latent_relational_transfer_20261027_v1",
        "status": "preflight_passed_implementation_pending" if data_audit["pass"] and reference_audit["pass"] and hashes["pass"] else "preflight_failed",
        "training_launched": False,
        "sealed_test_accessed": False,
        "neural_model_implemented": False,
        "protocol": rel(PROTOCOL),
        "leakage_spec": rel(LEAKAGE_SPEC),
        "novelty_audit": rel(NOVELTY),
        "source_hash_audit": hashes,
        "n0_reference_audit": reference_audit,
        "cycle_basis": cycles,
        "data_and_leakage_audit": data_audit,
        "structured_input_contract": {
            "allowed": ["observed edge endpoints", "observed modular edge offsets", "query source and target", "candidate values and order"],
            "forbidden": leakage_spec["forbidden_inputs"],
            "source": "serialized input only",
        },
        "expected_cost": {
            "new_training_arms": 4,
            "steps_per_arm": 2000,
            "total_new_updates": 8000,
            "estimate": "approximately 3.5-5 CPU hours and 0.5-1 GB artifacts; measure with the required 50-step throughput preflight",
        },
    }
    (OUTPUT / "CYCLE_BASIS.json").write_text(json.dumps(cycles, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "DATA_LEAKAGE_AUDIT.json").write_text(json.dumps(data_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "N0_REFERENCE_AUDIT.json").write_text(json.dumps(reference_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "PREFLIGHT.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Crucible Phase VII preflight",
        "",
        f"Status: **{results['status']}**.",
        "",
        "This preflight performs no model training. It verifies the frozen N0 reference, immutable splits, graph cycle bases, serialized structured-input parsing and forbidden-field boundary.",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Source hashes | {'PASS' if hashes['pass'] else 'FAIL'} |",
        f"| N0 references | {'PASS' if reference_audit['pass'] else 'FAIL'} |",
        f"| Data and leakage audit | {'PASS' if data_audit['pass'] else 'FAIL'} |",
        "| Neural model implementation | PENDING; no Phase VII training launched |",
        "",
        "The data/reference preflight is complete. Implementation of N1/N2 remains a separately reviewable step; no structured model, cycle objective or training checkpoint is included in this artifact.",
    ]
    (OUTPUT / "REPORT_PREFLIGHT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hash_paths = [PROTOCOL, LEAKAGE_SPEC, NOVELTY, Path(__file__).resolve(), SOURCE / "GRAPH_MANIFEST.json", SOURCE / "SUMMARY.json", OUTPUT / "CYCLE_BASIS.json", OUTPUT / "DATA_LEAKAGE_AUDIT.json", OUTPUT / "N0_REFERENCE_AUDIT.json", OUTPUT / "PREFLIGHT.json", OUTPUT / "REPORT_PREFLIGHT.md"]
    (OUTPUT / "HASHES.json").write_text(json.dumps({rel(path): digest(path) for path in hash_paths}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUTPUT / "FINALIZATION.json").write_text(json.dumps({"experiment": results["experiment"], "status": results["status"], "training_launched": False, "sealed_test_accessed": False, "hashes": rel(OUTPUT / "HASHES.json")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"experiment": results["experiment"], "status": results["status"], "training_launched": False, "sealed_test_accessed": False}, sort_keys=True))
    return 0 if results["status"] != "preflight_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
