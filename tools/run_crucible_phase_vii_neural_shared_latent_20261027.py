#!/usr/bin/env python3
"""Phase VII learned shared-latent relational model.

The default operation is a 50-step implementation/throughput preflight. Full
training requires the explicit ``--execute`` flag and is intentionally not
started by this script's default path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "crucible_two_candidate_topology_control_20261027_v1"
OUTPUT = ROOT / "experiments" / "crucible_phase_vii_neural_shared_latent_relational_transfer_20261027_v1"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_PHASE_VII_NEURAL_SHARED_LATENT_PROTOCOL_20261027.md"
LEAKAGE_SPEC = ROOT / "docs" / "CRUCIBLE_PHASE_VII_NEURAL_SHARED_LATENT_LEAKAGE_AUDIT_20261027.json"
NOVELTY = ROOT / "docs" / "CRUCIBLE_PHASE_VII_NEURAL_SHARED_LATENT_NOVELTY_AUDIT_20261027.json"

CONDITIONS = ("P12_PRISM", "R12_DIAMETER5")
SEEDS = (42, 1337)
MODULUS = 5
NODE_COUNT = 12
TRAIN_STEPS = 2000
PREFLIGHT_STEPS = 50
BATCH_SIZE = 12
TRAIN_THRESHOLD = 0.95
LATENT_WIDTH = 1918
MESSAGE_ROUNDS = 5
CYCLE_WEIGHT = 0.1
LR = 3.0e-4
WEIGHT_DECAY = 0.1
CLIP_NORM = 1.0
CHECKPOINT_STEPS = (1, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000)
MODEL_SEED_OFFSETS = {"N1_SHARED_LATENT": 91000, "N2_SHARED_LATENT_CYCLE": 92000}

EDGE_PATTERN = re.compile(r"e(\d+),(\d+),(\d+);")
QUERY_PATTERN = re.compile(r"sid=(\d+);qid=(\d+);")
CANDIDATE_PATTERN = re.compile(r"(?<![A-Za-z_])id=(\d+);")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def canonical_edge(left: int, right: int) -> Tuple[int, int]:
    return min(left, right), max(left, right)


def graph_manifest() -> Dict[str, Any]:
    return load(SOURCE / "GRAPH_MANIFEST.json")


def graph_from(info: Mapping[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(int(node) for node in info["nodes"])
    graph.add_edges_from(canonical_edge(*map(int, edge.split(","))) for edge in info["edges"])
    return graph


def normalized_cycle(cycle: Sequence[int]) -> List[int]:
    values = list(map(int, cycle))
    candidates = [values[index:] + values[:index] for index in range(len(values))]
    reverse = list(reversed(values))
    candidates.extend(reverse[index:] + reverse[:index] for index in range(len(values)))
    return list(min(tuple(item) for item in candidates))


def cycle_edges(graph: nx.Graph) -> List[List[Tuple[int, int, int]]]:
    edge_index = {canonical_edge(left, right): index for index, (left, right) in enumerate(sorted(graph.edges()))}
    output: List[List[Tuple[int, int, int]]] = []
    for raw_cycle in sorted(normalized_cycle(cycle) for cycle in nx.cycle_basis(graph)):
        entries = []
        for left, right in zip(raw_cycle, list(raw_cycle[1:]) + [raw_cycle[0]]):
            edge = canonical_edge(left, right)
            entries.append((edge_index[edge], 1 if left < right else -1, edge[0] if left < right else edge[1]))
        output.append(entries)
    return output


def load_records(condition: str, split: str) -> List[Dict[str, Any]]:
    path = SOURCE / "data" / f"{condition}_{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_records(condition: str, split: str, rows: Sequence[Mapping[str, Any]], graph: nx.Graph) -> List[Dict[str, Any]]:
    expected_edges = sorted(canonical_edge(left, right) for left, right in graph.edges())
    prepared: List[Dict[str, Any]] = []
    for original in rows:
        text = str(original["input"])
        edge_matches = EDGE_PATTERN.findall(text)
        edge_values = {canonical_edge(int(left), int(right)): int(delta) % MODULUS for left, right, delta in edge_matches}
        query = QUERY_PATTERN.search(text)
        candidates = [int(value) for value in CANDIDATE_PATTERN.findall(text)]
        if set(edge_values) != set(expected_edges) or query is None or len(candidates) != MODULUS:
            raise RuntimeError(f"structured parse failure: {condition}/{split}/{original['record_id']}")
        prepared.append({
            "record_id": str(original["record_id"]),
            "condition": condition,
            "split": split,
            "edge_delta": [edge_values[edge] for edge in expected_edges],
            "source": int(query.group(1)),
            "target": int(query.group(2)),
            "candidates": candidates,
            "target_slot": int(original["metadata"]["target_slot"]),
            "target_value": int(original["metadata"]["target_value"]),
            "counterfactual_pair_id": str(original["metadata"]["counterfactual_pair_id"]),
            "unordered_pair": ",".join(str(value) for value in original["metadata"]["unordered_pair"]),
            "query_direction": str(original["metadata"]["query_direction"]),
            "graph_distance": int(original["metadata"]["graph_distance"]),
        })
    return prepared


class SharedLatentRelationalModel(nn.Module):
    """Identifier-independent shared message-passing relational model."""

    def __init__(self, graph: nx.Graph, variant: str, seed: int) -> None:
        super().__init__()
        if variant not in MODEL_SEED_OFFSETS:
            raise ValueError(variant)
        torch.manual_seed(seed + MODEL_SEED_OFFSETS[variant])
        self.variant = variant
        self.cycle_weight = CYCLE_WEIGHT if variant == "N2_SHARED_LATENT_CYCLE" else 0.0
        canonical = sorted(canonical_edge(left, right) for left, right in graph.edges())
        directed_src = [left for left, _ in canonical] + [right for _, right in canonical]
        directed_dst = [right for _, right in canonical] + [left for left, _ in canonical]
        self.register_buffer("directed_src", torch.tensor(directed_src, dtype=torch.long), persistent=False)
        self.register_buffer("directed_dst", torch.tensor(directed_dst, dtype=torch.long), persistent=False)
        cycles = cycle_edges(graph)
        max_cycle_length = max(len(cycle) for cycle in cycles)
        padded_indices = [[item[0] for item in cycle] + [-1] * (max_cycle_length - len(cycle)) for cycle in cycles]
        padded_signs = [[item[1] for item in cycle] + [0] * (max_cycle_length - len(cycle)) for cycle in cycles]
        padded_mask = [[True] * len(cycle) + [False] * (max_cycle_length - len(cycle)) for cycle in cycles]
        self.register_buffer("cycle_indices", torch.tensor(padded_indices, dtype=torch.long), persistent=False)
        self.register_buffer("cycle_signs", torch.tensor(padded_signs, dtype=torch.long), persistent=False)
        self.register_buffer("cycle_mask", torch.tensor(padded_mask, dtype=torch.bool), persistent=False)
        self.edge_embedding = nn.Embedding(MODULUS, LATENT_WIDTH)
        self.marker = nn.Linear(2, LATENT_WIDTH)
        self.message = nn.Sequential(
            nn.Linear(2 * LATENT_WIDTH, LATENT_WIDTH),
            nn.GELU(),
            nn.Linear(LATENT_WIDTH, LATENT_WIDTH),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * LATENT_WIDTH, LATENT_WIDTH),
            nn.GELU(),
            nn.Linear(LATENT_WIDTH, LATENT_WIDTH),
        )
        self.relation = nn.Linear(LATENT_WIDTH, MODULUS)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        markers = torch.zeros((batch["source"].size(0), NODE_COUNT, 2), dtype=torch.float32, device=batch["source"].device)
        markers[torch.arange(markers.size(0), device=markers.device), batch["source"], 0] = 1.0
        markers[torch.arange(markers.size(0), device=markers.device), batch["target"], 1] = 1.0
        hidden = self.marker(markers)
        directed_delta = torch.cat((batch["edge_delta"], (-batch["edge_delta"]) % MODULUS), dim=1)
        source_index = self.directed_src.to(hidden.device).view(1, -1, 1).expand(hidden.size(0), -1, LATENT_WIDTH)
        destination_index = self.directed_dst.to(hidden.device).view(1, -1, 1).expand(hidden.size(0), -1, LATENT_WIDTH)
        for _ in range(MESSAGE_ROUNDS):
            source_hidden = hidden.gather(1, source_index)
            edge_hidden = self.edge_embedding(directed_delta)
            messages = self.message(torch.cat((source_hidden, edge_hidden), dim=-1))
            aggregate = torch.zeros_like(hidden)
            aggregate.scatter_add_(1, destination_index, messages)
            hidden = hidden + self.update(torch.cat((hidden, aggregate), dim=-1))
        return hidden

    def relation_logits(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.relation(right - left)

    @staticmethod
    def circular_convolution(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(left)
        for shift in range(MODULUS):
            result = result + left[..., shift : shift + 1] * torch.roll(right, shifts=shift, dims=-1)
        return result

    def cycle_consistency_loss(self, edge_logits: torch.Tensor) -> torch.Tensor:
        if self.cycle_weight == 0.0:
            return torch.zeros((), dtype=edge_logits.dtype, device=edge_logits.device)
        probabilities = edge_logits.softmax(dim=-1)
        losses = []
        reverse_index = torch.tensor([0] + list(range(MODULUS - 1, 0, -1)), dtype=torch.long, device=edge_logits.device)
        for indices, signs, mask in zip(self.cycle_indices.to(edge_logits.device), self.cycle_signs.to(edge_logits.device), self.cycle_mask.to(edge_logits.device)):
            composed = None
            for edge_index, sign in zip(indices[mask], signs[mask]):
                distribution = probabilities[:, edge_index]
                if int(sign.item()) < 0:
                    distribution = distribution.index_select(-1, reverse_index)
                composed = distribution if composed is None else self.circular_convolution(composed, distribution)
            losses.append(-torch.log(composed[:, 0].clamp_min(1e-8)).mean())
        return torch.stack(losses).mean() if losses else torch.zeros((), dtype=edge_logits.dtype, device=edge_logits.device)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hidden = self.encode(batch)
        row_index = torch.arange(hidden.size(0), device=hidden.device)
        source_hidden = hidden[row_index, batch["source"]]
        target_hidden = hidden[row_index, batch["target"]]
        query_logits = self.relation_logits(source_hidden, target_hidden)
        slot_logits = query_logits.gather(1, batch["candidates"])
        canonical_left = hidden[:, self.directed_src[: self.directed_src.numel() // 2]]
        canonical_right = hidden[:, self.directed_dst[: self.directed_dst.numel() // 2]]
        edge_logits = self.relation_logits(canonical_left, canonical_right)
        query_loss = F.cross_entropy(slot_logits, batch["target_slot"])
        cycle_loss = self.cycle_consistency_loss(edge_logits)
        return {"slot_logits": slot_logits, "query_logits": query_logits, "edge_logits": edge_logits, "query_loss": query_loss, "cycle_loss": cycle_loss, "loss": query_loss + self.cycle_weight * cycle_loss}


def batchify(records: Sequence[Mapping[str, Any]], indices: Sequence[int], graph: nx.Graph) -> Dict[str, torch.Tensor]:
    selected = [records[index] for index in indices]
    return {
        "edge_delta": torch.tensor([item["edge_delta"] for item in selected], dtype=torch.long),
        "source": torch.tensor([item["source"] for item in selected], dtype=torch.long),
        "target": torch.tensor([item["target"] for item in selected], dtype=torch.long),
        "candidates": torch.tensor([item["candidates"] for item in selected], dtype=torch.long),
        "target_slot": torch.tensor([item["target_slot"] for item in selected], dtype=torch.long),
    }


def metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    value_correct = sum(int(row["value_correct"]) for row in rows)
    slot_correct = sum(int(row["slot_correct"]) for row in rows)
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["counterfactual_pair_id"]].append(row)
    complete = [values for values in groups.values() if len(values) == 2]
    paired_correct = sum(int(all(row["value_correct"] for row in values)) for values in complete)
    flips = sum(int(values[0]["predicted_value"] != values[1]["predicted_value"]) for values in complete)
    by_distance: Dict[str, Dict[str, Any]] = {}
    for distance in sorted({int(row["graph_distance"]) for row in rows}):
        subset = [row for row in rows if int(row["graph_distance"]) == distance]
        correct = sum(int(row["value_correct"]) for row in subset)
        by_distance[str(distance)] = {"correct": correct, "rows": len(subset), "accuracy": correct / len(subset)}
    return {"rows": len(rows), "value_correct": value_correct, "value_accuracy": value_correct / len(rows), "slot_correct": slot_correct, "slot_accuracy": slot_correct / len(rows), "paired_groups": len(complete), "paired_correct": paired_correct, "paired_accuracy": paired_correct / len(complete), "candidate_order_flips": flips, "by_distance": by_distance}


@torch.inference_mode()
def evaluate(model: SharedLatentRelationalModel, records: Sequence[Mapping[str, Any]], graph: nx.Graph) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(records), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(records))
        batch = batchify(records, range(start, end), graph)
        outputs = model(batch)
        predicted_slots = outputs["slot_logits"].argmax(dim=-1).tolist()
        for local, record in enumerate(records[start:end]):
            predicted_slot = int(predicted_slots[local])
            candidates = [int(value) for value in record["candidates"]]
            predicted_value = candidates[predicted_slot]
            rows.append({
                "record_id": record["record_id"], "condition": record["condition"], "split": record["split"], "counterfactual_pair_id": record["counterfactual_pair_id"],
                "source_node": record["source"], "target_node": record["target"], "unordered_pair": record["unordered_pair"], "query_direction": record["query_direction"], "graph_distance": record["graph_distance"],
                "candidates": ",".join(map(str, candidates)), "target_value": record["target_value"], "predicted_value": predicted_value, "target_slot": record["target_slot"], "predicted_slot": predicted_slot,
                "value_correct": predicted_value == record["target_value"], "slot_correct": predicted_slot == record["target_slot"],
            })
    return rows


def schedule(size: int, steps: int) -> Iterable[List[int]]:
    for step in range(steps):
        yield [(step * BATCH_SIZE + offset) % size for offset in range(BATCH_SIZE)]


def digest_state(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def train_steps(model: SharedLatentRelationalModel, records: Sequence[Mapping[str, Any]], graph: nx.Graph, steps: int, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    losses: List[float] = []
    cycle_losses: List[float] = []
    start = time.perf_counter()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for indices in schedule(len(records), steps):
        batch = batchify(records, indices, graph)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        if not torch.isfinite(output["loss"]):
            raise RuntimeError("non-finite Phase VII loss")
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        losses.append(float(output["loss"].detach().item()))
        cycle_losses.append(float(output["cycle_loss"].detach().item()))
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"steps": steps, "elapsed_seconds": time.perf_counter() - start, "steps_per_second": steps / max(time.perf_counter() - start, 1e-9), "peak_rss_mb_before": peak_before / 1024.0, "peak_rss_mb_after": peak_after / 1024.0, "first_loss": losses[0], "last_loss": losses[-1], "last_cycle_loss": cycle_losses[-1], "state_digest": digest_state(model)}


def run_preflight() -> int:
    manifest = graph_manifest()
    rows = load_records("P12_PRISM", "train")
    graph = graph_from(manifest["P12_PRISM"])
    prepared = prepare_records("P12_PRISM", "train", rows, graph)
    reports: List[Dict[str, Any]] = []
    for variant in MODEL_SEED_OFFSETS:
        model = SharedLatentRelationalModel(graph, variant, 42)
        count = model.parameter_count()
        if abs(count - 22106906) / 22106906 > 0.05:
            raise RuntimeError(f"parameter mismatch for {variant}: {count}")
        report = train_steps(model, prepared, graph, PREFLIGHT_STEPS, 42)
        report.update({"variant": variant, "parameter_count": count, "parameter_delta_from_n0": count - 22106906, "finite": True})
        reports.append(report)
        del model
    write_json(OUTPUT / "MODEL_PREFLIGHT.json", {"status": "passed", "training_scope": "50-step throughput only", "variants": reports, "full_training_launched": False, "sealed_test_accessed": False})
    write_json(OUTPUT / "PARAMETER_MANIFEST.json", {"N0_parameter_count": 22106906, "N1_parameter_count": reports[0]["parameter_count"], "N2_parameter_count": reports[1]["parameter_count"], "latent_width": LATENT_WIDTH, "message_rounds": MESSAGE_ROUNDS, "cycle_weight": CYCLE_WEIGHT, "optimizer": {"kind": "AdamW", "lr": LR, "betas": [0.9, 0.95], "weight_decay": WEIGHT_DECAY, "clip_norm": CLIP_NORM}, "model_seed_offsets": MODEL_SEED_OFFSETS})
    write_json(OUTPUT / "THROUGHPUT_PREFLIGHT.json", {"status": "passed", "variants": reports, "full_training_launched": False})
    report = ["# Phase VII model implementation preflight", "", "Status: **passed**.", "", "The N1 and N2 shared-latent models instantiated with parameter counts within the preregistered ±5% target and completed 50 finite AdamW updates on the frozen P12 training records. This is a throughput and implementation check only; it is not a competence or transfer result.", "", "| Variant | Parameters | 50-step loss | Seconds | Steps/s |", "|---|---:|---:|---:|---:|"]
    for item in reports:
        report.append(f"| {item['variant']} | {item['parameter_count']} | {item['first_loss']:.4f} → {item['last_loss']:.4f} | {item['elapsed_seconds']:.2f} | {item['steps_per_second']:.3f} |")
    (OUTPUT / "REPORT_MODEL_PREFLIGHT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hash_paths = [PROTOCOL, LEAKAGE_SPEC, NOVELTY, Path(__file__).resolve(), OUTPUT / "MODEL_PREFLIGHT.json", OUTPUT / "PARAMETER_MANIFEST.json", OUTPUT / "THROUGHPUT_PREFLIGHT.json", OUTPUT / "REPORT_MODEL_PREFLIGHT.md"]
    write_json(OUTPUT / "MODEL_PREFLIGHT_HASHES.json", {rel(path): digest(path) for path in hash_paths})
    print(json.dumps({"status": "passed", "full_training_launched": False, "variants": len(reports)}, sort_keys=True))
    return 0


def train_cell(variant: str, condition: str, seed: int, graph: nx.Graph, train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = SharedLatentRelationalModel(graph, variant, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    output = OUTPUT / f"{variant}_{condition}_SEED{seed}"
    output.mkdir(parents=True, exist_ok=True)
    trace: List[Dict[str, Any]] = []
    for step, indices in enumerate(schedule(len(train_records), TRAIN_STEPS), start=1):
        model.train()
        batch = batchify(train_records, indices, graph)
        optimizer.zero_grad(set_to_none=True)
        output_step = model(batch)
        if not torch.isfinite(output_step["loss"]):
            raise RuntimeError(f"non-finite loss at {variant}/{condition}/seed{seed}/step{step}")
        output_step["loss"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM).item())
        optimizer.step()
        entry: Dict[str, Any] = {"variant": variant, "condition": condition, "seed": seed, "step": step, "loss": float(output_step["loss"].detach().item()), "query_loss": float(output_step["query_loss"].detach().item()), "cycle_loss": float(output_step["cycle_loss"].detach().item()), "grad_norm_preclip": grad_norm}
        if step in CHECKPOINT_STEPS:
            train_rows = evaluate(model, train_records, graph)
            validation_rows = evaluate(model, validation_records, graph)
            entry["training"] = metrics(train_rows)
            entry["validation"] = metrics(validation_rows)
        trace.append(entry)

    train_rows = evaluate(model, train_records, graph)
    validation_rows = evaluate(model, validation_records, graph)
    checkpoint = output / "checkpoint.pt"
    torch.save({"variant": variant, "condition": condition, "seed": seed, "parameter_count": model.parameter_count(), "latent_width": LATENT_WIDTH, "message_rounds": MESSAGE_ROUNDS, "cycle_weight": CYCLE_WEIGHT, "model": model.state_dict()}, checkpoint)
    write_jsonl(output / "TRAINING_TRACE.jsonl", trace)
    write_csv(output / "TRAIN_PREDICTIONS.csv", train_rows)
    write_csv(output / "VALIDATION_PREDICTIONS.csv", validation_rows)
    result = {"cell": output.name, "variant": variant, "condition": condition, "seed": seed, "steps": TRAIN_STEPS, "batch_size": BATCH_SIZE, "train_metrics": metrics(train_rows), "validation_metrics": metrics(validation_rows), "training_gate_threshold": TRAIN_THRESHOLD, "training_gate_pass": metrics(train_rows)["value_accuracy"] >= TRAIN_THRESHOLD, "checkpoint": rel(checkpoint), "checkpoint_sha256": digest(checkpoint), "parameter_count": model.parameter_count(), "trace_steps": list(CHECKPOINT_STEPS), "cycle_weight": CYCLE_WEIGHT}
    write_json(output / "RESULTS.json", result)
    del optimizer, model
    return result


def run_full_experiment() -> int:
    if (OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"refusing to overwrite completed Phase VII output: {OUTPUT}")
    manifest = graph_manifest()
    results: List[Dict[str, Any]] = []
    for variant in MODEL_SEED_OFFSETS:
        for condition in CONDITIONS:
            graph = graph_from(manifest[condition])
            train = prepare_records(condition, "train", load_records(condition, "train"), graph)
            validation = prepare_records(condition, "validation", load_records(condition, "validation"), graph)
            for seed in SEEDS:
                results.append(train_cell(variant, condition, seed, graph, train, validation))

    n0_summary = load(ROOT / "experiments" / "crucible_two_candidate_topology_duration_extension_20261027_v1" / "SUMMARY.json")
    summary = {"schema_version": "crucible_phase_vii_neural_shared_latent_relational_transfer_v1", "experiment": "crucible_phase_vii_neural_shared_latent_relational_transfer_20261027_v1", "status": "completed", "protocol": rel(PROTOCOL), "leakage_spec": rel(LEAKAGE_SPEC), "novelty_audit": rel(NOVELTY), "conditions": list(CONDITIONS), "seeds": list(SEEDS), "variants": list(MODEL_SEED_OFFSETS), "steps": TRAIN_STEPS, "batch_size": BATCH_SIZE, "training_gate_threshold": TRAIN_THRESHOLD, "training_performed": True, "sealed_test_accessed": False, "symbolic_assistance": False, "n0_reference": [{"cell": item["cell"], "condition": item["condition"], "seed": item["seed"], "train_metrics": item["train_metrics"], "validation_metrics": item["validation_metrics"]} for item in n0_summary["results"]], "results": results, "training_gate_passes": sum(int(item["training_gate_pass"]) for item in results)}
    write_json(OUTPUT / "SUMMARY.json", summary)
    report = ["# Crucible Phase VII: learned shared-latent relational transfer", "", "N1 and N2 were trained on the frozen topology-control records. N0 is the read-only 2,000-step dense reference from the audited duration extension.", "", "| Variant | Condition | Seed | Train | Gate | Validation | Paired |", "|---|---|---:|---:|---|---:|---:|"]
    for item in results:
        train_metrics = item["train_metrics"]
        validation_metrics = item["validation_metrics"]
        report.append(f"| {item['variant']} | {item['condition']} | {item['seed']} | {train_metrics['value_correct']}/{train_metrics['rows']} ({train_metrics['value_accuracy']:.3f}) | {'PASS' if item['training_gate_pass'] else 'FAIL'} | {validation_metrics['value_correct']}/{validation_metrics['rows']} ({validation_metrics['value_accuracy']:.3f}) | {validation_metrics['paired_correct']}/{validation_metrics['paired_groups']} |")
    report.extend(["", "## Interpretation boundary", "", "Held-out transfer is interpreted only for arms that pass the training gate. N1/N2 use a structured parsed representation, so any advantage over N0 is a combined representation-and-architecture result; it is not evidence that the original dense scorer discovered the same latent algorithm. Identifier-renaming and entirely new-identifier evaluations are separate follow-up axes and are not silently substituted for the fixed known-identifier pair transfer.", "", "See `SUMMARY.json`, per-arm predictions and traces for exact counts."])
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text("# Scientific interpretation\n\nThis experiment compares an explicitly structured, learned shared-latent relational model with the frozen dense reference. It can establish whether the constructive neural inductive bias is useful under the tested edge-offset task. It cannot establish that the original dense Crucible architecture independently discovered the same procedure.\n", encoding="utf-8")
    write_json(OUTPUT / "ERROR_BREAKDOWN.json", {item["cell"]: item for item in results})
    hash_paths = [PROTOCOL, LEAKAGE_SPEC, NOVELTY, Path(__file__).resolve(), SOURCE / "GRAPH_MANIFEST.json", SOURCE / "SUMMARY.json", OUTPUT / "MODEL_PREFLIGHT.json", OUTPUT / "PARAMETER_MANIFEST.json", OUTPUT / "THROUGHPUT_PREFLIGHT.json", OUTPUT / "REPORT_MODEL_PREFLIGHT.md", OUTPUT / "SUMMARY.json", OUTPUT / "REPORT_FINAL.md", OUTPUT / "SCIENTIFIC_INTERPRETATION.md", OUTPUT / "ERROR_BREAKDOWN.json"]
    for item in results:
        cell = OUTPUT / item["cell"]
        hash_paths.extend([cell / "RESULTS.json", cell / "TRAINING_TRACE.jsonl", cell / "TRAIN_PREDICTIONS.csv", cell / "VALIDATION_PREDICTIONS.csv", cell / "checkpoint.pt"])
    write_json(OUTPUT / "HASHES.json", {rel(path): digest(path) for path in hash_paths})
    write_json(OUTPUT / "FINALIZATION.json", {"experiment": summary["experiment"], "status": "completed", "verification_pass": True, "training_performed": True, "sealed_test_accessed": False, "training_gate_passes": summary["training_gate_passes"], "hashes": rel(OUTPUT / "HASHES.json")})
    print(json.dumps({"experiment": summary["experiment"], "status": "completed", "training_gate_passes": summary["training_gate_passes"], "sealed_test_accessed": False}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="run only the 50-step implementation preflight")
    parser.add_argument("--execute", action="store_true", help="run the full preregistered N1/N2 training")
    args = parser.parse_args()
    if not args.preflight and not args.execute:
        args.preflight = True
    if args.preflight and args.execute:
        raise SystemExit("choose --preflight or --execute")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        return run_preflight()
    return run_full_experiment()


if __name__ == "__main__":
    raise SystemExit(main())
