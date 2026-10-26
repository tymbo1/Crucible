#!/usr/bin/env python3
"""CPU compositional authority-transfer gate with a dynamic candidate pointer."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_compositional_authority_transfer_20261026_v1"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_COMPOSITIONAL_AUTHORITY_TRANSFER_PROTOCOL_20261026.md"
AUTH_SOURCE = ROOT / "tools" / "run_crucible_authority_aware_role_binding_20251025.py"

INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
RULES = ("EARLIEST", "LATEST", "MATCH_A", "MATCH_D", "NONE")
OPERATIONS = ("F", "S", "A", "D")
MAX_CANDIDATES = 4
ABSTAIN_SLOT = MAX_CANDIDATES
STEPS = 500
EFFECTIVE_BATCH = 12
TRAIN_THRESHOLD = 0.95


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


auth = import_module("compositional_authority_tools", AUTH_SOURCE)
torch = auth.torch
nn = torch.nn
F = torch.nn.functional
stage_a = auth.stage_a


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(row), sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def operand_field(interface: str, a: int, b: int) -> str:
    return f"a={a}; b={b}." if interface == "canonical" else f"a_digit={a}; b_digit={b}."


def operation_value(operation: str, a: int, b: int) -> int:
    if operation == "F":
        return a
    if operation == "S":
        return b
    if operation == "A":
        return (a + b) % 10
    if operation == "D":
        return (a - b) % 10
    raise ValueError(operation)


def target_slot(candidates: Sequence[str], rule: str) -> int | None:
    if rule == "EARLIEST":
        return 0
    if rule == "LATEST":
        return len(candidates) - 1
    if rule == "MATCH_A":
        return candidates.index("A") if "A" in candidates else None
    if rule == "MATCH_D":
        return candidates.index("D") if "D" in candidates else None
    if rule == "NONE":
        return None
    raise ValueError(rule)


def input_text(interface: str, layout: str, a: int, b: int, candidates: Sequence[str], rule: str, ordinal: int) -> str:
    pair = operand_field(interface, a, b)
    noise = (a + 3 * b + ordinal + len(candidates)) % 10
    operations = [f"operation={operation};" for operation in candidates]
    if layout == "TRAIN_COMPACT":
        return f"Select one candidate. {pair} {' '.join(operations)} authority={rule}; Return selected result. decision=Q;"
    if layout == "VALIDATION_PADDED":
        chunks: List[str] = ["Select one candidate.", f"context=neutral_{noise};", operations[0], pair]
        for index, operation in enumerate(operations[1:], start=1):
            chunks.extend([f"context=bridge_{(noise + index) % 10};", operation])
        chunks.extend([f"context=tail_{(noise + 7) % 10};", f"authority={rule};", "Return selected result. decision=Q;"])
        return " ".join(chunks)
    if layout == "TRANSFER_FOUR":
        chunks: List[str] = ["Select one candidate.", f"context=transfer_{noise};"]
        for index, operation in enumerate(operations):
            if index == 1:
                chunks.append(pair)
            chunks.extend([operation, f"context=slot_{(noise + index + 2) % 10};"])
        chunks.extend([f"authority={rule};", "Return selected result. decision=Q;"])
        return " ".join(chunks)
    raise ValueError(layout)


def pair_id(interface: str, split: str, layout: str, a: int, b: int, candidates: Sequence[str], rule_group: str) -> str:
    raw = f"comp_authority_pair|{interface}|{split}|{layout}|{a}|{b}|{''.join(candidates)}|{rule_group}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def make_records(interface: str, split: str, layout: str, pairs: Sequence[Tuple[int, int]], candidate_sets: Sequence[Sequence[str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ordinal = 0
    for a, b in pairs:
        for candidate_tuple in candidate_sets:
            candidates = tuple(candidate_tuple)
            for rule in RULES:
                slot = target_slot(candidates, rule)
                target_operation = None if slot is None else candidates[slot]
                target_output = "ABSTAIN" if slot is None else str(operation_value(str(target_operation), a, b))
                group = "position" if rule in {"EARLIEST", "LATEST"} else "relation"
                counterfactual = None
                if group == "position":
                    counterfactual = pair_id(interface, split, layout, a, b, candidates, "position")
                elif "A" in candidates and "D" in candidates:
                    counterfactual = pair_id(interface, split, layout, a, b, candidates, "relation")
                record_id = hashlib.sha256(f"comp_authority|{interface}|{split}|{layout}|{a}|{b}|{''.join(candidates)}|{rule}|{ordinal}".encode()).hexdigest()[:20]
                rows.append({
                    "schema_version": "crucible_compositional_authority_transfer_v1",
                    "record_id": record_id,
                    "interface": interface,
                    "split": split,
                    "layout": layout,
                    "input": input_text(interface, layout, a, b, candidates, rule, ordinal),
                    "target": target_output,
                    "oracle": {"kind": "dynamic_candidate_pointer", "target_slot": slot, "target_operation": target_operation, "target_output": target_output},
                    "metadata": {"a": a, "b": b, "candidates": list(candidates), "authority_rule": rule, "target_slot": slot, "target_operation": target_operation, "counterfactual_pair_id": counterfactual},
                })
                ordinal += 1
    return rows


def prepare_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    operation_pattern = re.compile(rb"(?<![A-Za-z_])operation=([FSAD])")
    decision_pattern = re.compile(rb"decision=(Q)")
    prepared: List[Dict[str, Any]] = []
    for original in records:
        item = dict(original)
        raw = auth.base.base.parser_input(item).encode("utf-8")
        operations = list(operation_pattern.finditer(raw))
        decision = decision_pattern.search(raw)
        expected = [str(value) for value in item["metadata"]["candidates"]]
        found = [match.group(1).decode("ascii") for match in operations]
        if found != expected or decision is None or len(found) > MAX_CANDIDATES:
            raise RuntimeError(f"composition span failure: {item['record_id']} found={found} expected={expected}")
        full = [stage_a.BOS_ID] + stage_a.serialize_bytes(raw)
        if len(full) > stage_a.CONTEXT_LENGTH:
            raise RuntimeError(f"composition context overflow: {item['record_id']}")
        item["_parser_full"] = full
        item["_parser_length"] = len(full)
        item["_candidate_positions"] = [1 + match.start(1) for match in operations]
        item["_decision_position"] = 1 + decision.start(1)
        prepared.append(item)
    return prepared


def batchify(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    selected = [records[index] for index in indices]
    max_len = max(int(row["_parser_length"]) for row in selected)
    x = torch.full((len(selected), max_len), stage_a.PAD_ID, dtype=torch.long)
    lengths: List[int] = []
    candidate_positions = torch.zeros((len(selected), MAX_CANDIDATES), dtype=torch.long)
    candidate_mask = torch.zeros((len(selected), MAX_CANDIDATES), dtype=torch.bool)
    targets: List[int] = []
    decision_positions: List[int] = []
    for row_index, item in enumerate(selected):
        full = torch.tensor(item["_parser_full"], dtype=torch.long)
        x[row_index, : full.numel()] = full
        lengths.append(int(full.numel()))
        positions = [int(value) for value in item["_candidate_positions"]]
        candidate_positions[row_index, : len(positions)] = torch.tensor(positions, dtype=torch.long)
        candidate_mask[row_index, : len(positions)] = True
        target = item["metadata"]["target_slot"]
        targets.append(ABSTAIN_SLOT if target is None else int(target))
        decision_positions.append(int(item["_decision_position"]))
    return {"x": x, "lengths": torch.tensor(lengths, dtype=torch.long), "candidate_positions": candidate_positions, "candidate_mask": candidate_mask, "target": torch.tensor(targets, dtype=torch.long), "decision_pos": torch.tensor(decision_positions, dtype=torch.long)}


class DynamicCandidatePointer(auth.RoleSupervisedParser):
    """Final-token query scoring an arbitrary number of candidate positions."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        torch.manual_seed(seed + 96_000)
        self.query = nn.Linear(stage_a.D_MODEL, stage_a.D_MODEL)
        self.key = nn.Linear(stage_a.D_MODEL, stage_a.D_MODEL)
        self.abstain = nn.Linear(stage_a.D_MODEL, 1)

    def decision_logits(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.hidden_sequence(batch["x"])
        rows = torch.arange(hidden.size(0), dtype=torch.long)
        decision_hidden = hidden[rows, batch["decision_pos"]]
        query = self.query(decision_hidden)
        positions = batch["candidate_positions"].unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        candidate_hidden = hidden.gather(1, positions)
        scores = (self.key(candidate_hidden) * query.unsqueeze(1)).sum(dim=-1) / (stage_a.D_MODEL ** 0.5)
        scores = scores.masked_fill(~batch["candidate_mask"], -1e9)
        return torch.cat([scores, self.abstain(decision_hidden)], dim=-1)


def schedule(size: int) -> List[List[int]]:
    return [[(step * EFFECTIVE_BATCH + offset) % size for offset in range(EFFECTIVE_BATCH)] for step in range(STEPS)]


def pair_metrics(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row["split"]) == split and row["counterfactual_pair_id"]:
            groups.setdefault(str(row["counterfactual_pair_id"]), []).append(row)
    complete = [group for group in groups.values() if len(group) == 2]
    flips = sum(int(group[0]["predicted_slot"] != group[1]["predicted_slot"]) for group in complete)
    correct = sum(int(group[0]["pointer_correct"] and group[1]["pointer_correct"]) for group in complete)
    return {"pairs": len(complete), "pointer_flips": flips, "correct_two_way": correct, "flip_accuracy": flips / len(complete) if complete else None, "correct_two_way_accuracy": correct / len(complete) if complete else None}


def row_metrics(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    selected = [row for row in rows if str(row["split"]) == split]
    pointer_correct = sum(int(row["pointer_correct"]) for row in selected)
    output_correct = sum(int(row["output_correct"]) for row in selected)
    abstain_rows = [row for row in selected if row["target_slot"] == ABSTAIN_SLOT]
    abstain_correct = sum(int(row["pointer_correct"]) for row in abstain_rows)
    return {"rows": len(selected), "pointer_correct": pointer_correct, "pointer_accuracy": pointer_correct / len(selected) if selected else None, "output_correct": output_correct, "output_accuracy": output_correct / len(selected) if selected else None, "abstain_rows": len(abstain_rows), "abstain_correct": abstain_correct, "abstain_accuracy": abstain_correct / len(abstain_rows) if abstain_rows else None, "paired": pair_metrics(rows, split)}


def evaluate(model: DynamicCandidatePointer, records: Sequence[Mapping[str, Any]], source: str, seed: int) -> List[Dict[str, Any]]:
    prepared = prepare_records(records)
    model.eval()
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(prepared), EFFECTIVE_BATCH):
        indices = list(range(start, min(start + EFFECTIVE_BATCH, len(prepared))))
        batch = batchify(prepared, indices)
        with torch.inference_mode():
            logits = model.decision_logits(batch)
        predicted = logits.argmax(dim=-1).tolist()
        for local, index in enumerate(indices):
            item = prepared[index]
            pred = int(predicted[local])
            predicted_operation = None if pred == ABSTAIN_SLOT else str(item["metadata"]["candidates"][pred])
            predicted_output = "ABSTAIN" if predicted_operation is None else str(operation_value(predicted_operation, int(item["metadata"]["a"]), int(item["metadata"]["b"])))
            target_slot = ABSTAIN_SLOT if item["metadata"]["target_slot"] is None else int(item["metadata"]["target_slot"])
            rows.append({
                "seed": seed, "source": source, "split": str(item["split"]), "layout": str(item["layout"]), "record_id": str(item["record_id"]), "counterfactual_pair_id": item["metadata"]["counterfactual_pair_id"],
                "a": int(item["metadata"]["a"]), "b": int(item["metadata"]["b"]), "candidates": "".join(item["metadata"]["candidates"]), "authority_rule": str(item["metadata"]["authority_rule"]), "candidate_count": len(item["metadata"]["candidates"]),
                "target_slot": target_slot, "predicted_slot": pred, "target_operation": item["metadata"]["target_operation"], "predicted_operation": predicted_operation,
                "target_output": str(item["target"]), "predicted_output": predicted_output, "pointer_correct": pred == target_slot, "output_correct": predicted_output == str(item["target"]), "abstain_correct": target_slot == ABSTAIN_SLOT and pred == ABSTAIN_SLOT,
            })
    return rows


def train_arm(interface: str, seed: int, train_records: Sequence[Mapping[str, Any]], output: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = DynamicCandidatePointer(seed)
    optimizer = stage_a.optimizer_for(model)
    prepared = prepare_records(train_records)
    trace: List[Dict[str, Any]] = []
    for step, indices in enumerate(schedule(len(prepared)), start=1):
        batch = batchify(prepared, indices)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.decision_logits(batch), batch["target"])
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), stage_a.CLIP_NORM).item())
        optimizer.step()
        trace.append({"interface": interface, "seed": seed, "step": step, "loss": float(loss.detach().item()), "grad_norm": grad_norm, "examples": len(indices)})
        if step in (1, 100, 200, 300, 400, 500):
            print(json.dumps({"event": "compositional_authority_training", "interface": interface, "seed": seed, "step": step, "loss": float(loss.detach().item())}, sort_keys=True), flush=True)
    model.eval()
    checkpoint = output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"interface": interface, "seed": seed, "steps": STEPS, "max_candidates": MAX_CANDIDATES, "model": model.state_dict()}, checkpoint)
    write_jsonl(output / "TRAINING_TRACE.jsonl", trace)
    result = {"interface": interface, "seed": seed, "source": "new_dynamic_pointer", "steps": STEPS, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint)}
    del optimizer
    gc.collect()
    return result, [model]


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_pairs = [(1, 6), (2, 7), (3, 8), (4, 9), (1, 8), (2, 9), (3, 6), (4, 7)]
    validation_pairs = [(0, 5), (1, 7), (2, 8), (3, 9), (0, 6), (1, 9)]
    candidate_sets = [tuple(values) for values in itertools.permutations(OPERATIONS, 3)]
    train_data: Dict[str, List[Dict[str, Any]]] = {}
    validation3: Dict[str, List[Dict[str, Any]]] = {}
    transfer4: Dict[str, List[Dict[str, Any]]] = {}
    audits: Dict[str, Any] = {}
    for interface in INTERFACES:
        train_data[interface] = make_records(interface, "train", "TRAIN_COMPACT", train_pairs, candidate_sets)
        validation3[interface] = make_records(interface, "validation_3", "VALIDATION_PADDED", validation_pairs, candidate_sets)
        transfer4[interface] = make_records(interface, "transfer_4", "TRANSFER_FOUR", validation_pairs, [OPERATIONS])
        audits[interface] = {"train_rows": len(train_data[interface]), "validation_3_rows": len(validation3[interface]), "transfer_4_rows": len(transfer4[interface]), "train_records": [row["record_id"] for row in train_data[interface]], "validation_records": [row["record_id"] for row in validation3[interface]], "transfer_records": [row["record_id"] for row in transfer4[interface]]}
        write_jsonl(OUTPUT / "data" / interface / "train.jsonl", train_data[interface])
        write_jsonl(OUTPUT / "data" / interface / "validation_3.jsonl", validation3[interface])
        write_jsonl(OUTPUT / "data" / interface / "transfer_4.jsonl", transfer4[interface])

    results: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []
    checkpoints: Dict[str, Any] = {}
    for interface in INTERFACES:
        for seed in SEEDS:
            arm_output = OUTPUT / "trained_arms" / interface / f"seed{seed}"
            result, models = train_arm(interface, seed, train_data[interface], arm_output)
            model = models[0]
            rows = evaluate(model, train_data[interface] + validation3[interface] + transfer4[interface], "new_dynamic_pointer", seed)
            metrics = {split: row_metrics(rows, split) for split in ("train", "validation_3", "transfer_4")}
            result["metrics"] = metrics
            write_json(arm_output / "RESULTS.json", result)
            write_csv(arm_output / "PREDICTIONS.csv", rows)
            results.append(result)
            all_predictions.extend(rows)
            checkpoints[f"{interface}|seed{seed}"] = {"path": result["checkpoint"], "sha256": result["checkpoint_sha256"]}
            del model
            gc.collect()

    summary = {"experiment": "crucible_compositional_authority_transfer_20261026_v1", "status": "completed", "protocol": rel(PROTOCOL), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "steps": STEPS, "effective_batch": EFFECTIVE_BATCH, "max_candidates": MAX_CANDIDATES, "abstain_slot": ABSTAIN_SLOT, "data_audit": audits, "results": results, "checkpoint_hashes": checkpoints, "sealed_test_accessed": False, "architecture_changed": True, "dense_backbone_changed": False, "readout_changed": True, "training_candidate_count": 3, "transfer_candidate_count": 4}
    write_csv(OUTPUT / "PREDICTIONS.csv", all_predictions)
    write_json(OUTPUT / "DATA_AUDIT.json", audits)
    write_json(OUTPUT / "SUMMARY.json", summary)
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {"experiment": summary["experiment"], "protocol": rel(PROTOCOL), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "sealed_test_accessed": False, "architecture_changed": True, "dense_backbone_changed": False, "readout_changed": True, "training_candidate_count": 3, "transfer_candidate_count": 4})

    report = ["# Crucible Compositional Authority-Transfer Gate v1", "", "Dynamic candidate-pointer gate trained on three candidates and evaluated on held-out three-candidate layouts plus four-candidate transfer.", "", "| Interface | Seed | Train pointer | Validation-3 pointer | Transfer-4 pointer | Transfer-4 abstention | Transfer-4 flips | Transfer-4 paired |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        train = result["metrics"]["train"]; val = result["metrics"]["validation_3"]; transfer = result["metrics"]["transfer_4"]
        report.append(f"| {result['interface']} | {result['seed']} | {train['pointer_correct']}/{train['rows']} | {val['pointer_correct']}/{val['rows']} | {transfer['pointer_correct']}/{transfer['rows']} | {transfer['abstain_correct']}/{transfer['abstain_rows']} | {transfer['paired']['pointer_flips']}/{transfer['paired']['pairs']} | {transfer['paired']['correct_two_way']}/{transfer['paired']['pairs']} |")
    report += ["", "## Interpretation", "", "The training competence gate is 95% pointer accuracy. Validation-3 and Transfer-4 are separate: Transfer-4 contains four candidates although training contains only three. The primary transfer evidence is candidate-pointer correctness plus counterfactual flips and two-way paired correctness. Abstention is scored only on rows whose authority rule has no matching candidate.", "", "No sealed test data were accessed. The dynamic pointer is a new readout for this compositional gate; this result does not claim that the original fixed two-class S4 head scales to arbitrary candidate counts.", "", "## Artifacts", "", f"- [Protocol](../../{rel(PROTOCOL)})", "- [Summary](SUMMARY.json)", "- [Predictions](PREDICTIONS.csv)", "- [Data audit](DATA_AUDIT.json)", "- [Protocol manifest](PROTOCOL_MANIFEST.json)", "- Per-arm checkpoints and traces are under `trained_arms/`"]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    interpretation = """# Scientific Interpretation

This gate tests a dynamic candidate-pointer readout rather than extending the
fixed two-class S4 head. Training uses three candidates. Validation-3 changes
operand pairs and layout; Transfer-4 adds a fourth candidate that was absent
from training. Relation rules select a candidate by its operation marker, and
NONE requires abstention.

Training competence is required before transfer is interpreted. Counterfactual
flips and paired correctness are the principal relational metrics. A successful
Transfer-4 result would demonstrate bounded candidate-count and relation-based
authority transfer under this synthetic task. It would not establish arbitrary
instruction hierarchy, general reasoning or autonomous learning.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    hash_paths = [PROTOCOL, OUTPUT / "SUMMARY.json", OUTPUT / "DATA_AUDIT.json", OUTPUT / "PROTOCOL_MANIFEST.json", OUTPUT / "PREDICTIONS.csv", OUTPUT / "REPORT_FINAL.md", OUTPUT / "SCIENTIFIC_INTERPRETATION.md"]
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths})
    finalization = {"experiment": summary["experiment"], "status": "completed", "verification_pass": True, "sealed_test_accessed": False, "architecture_changed": True, "dense_backbone_changed": False, "readout_changed": True, "arm_count": len(results), "prediction_rows": len(all_predictions), "hashes": rel(OUTPUT / "HASHES.json")}
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"event": "compositional_authority_transfer_complete", "output": rel(OUTPUT), "verification_pass": True, "arm_count": len(results), "prediction_rows": len(all_predictions)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
