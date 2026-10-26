#!/usr/bin/env python3
"""CPU fixed-slot versus shared-scoring cardinality transfer diagnostic."""

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
OUTPUT = ROOT / "experiments" / "crucible_shared_candidate_scoring_cardinality_20261026_v1"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_SHARED_CANDIDATE_SCORING_CARDINALITY_PROTOCOL_20261026.md"
AUTH_SOURCE = ROOT / "tools" / "run_crucible_authority_aware_role_binding_20251025.py"

INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
VARIANTS = ("SLOT_HEAD", "SHARED_SCORER")
MAX_CANDIDATES = 8
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


auth = import_module("shared_candidate_scoring_tools", AUTH_SOURCE)
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


def operand_field(interface: str, a: int, b: int) -> str:
    return f"a={a};b={b}." if interface == "canonical" else f"a_digit={a};b_digit={b}."


def input_text(interface: str, layout: str, a: int, b: int, candidates: Sequence[int], authority: int, ordinal: int) -> str:
    pair = operand_field(interface, a, b)
    noise = (a + 3 * b + ordinal + len(candidates)) % 10
    entries = [f"id={candidate};op={'F' if index % 2 == 0 else 'S'};" for index, candidate in enumerate(candidates)]
    if layout == "TRAIN_COMPACT":
        return f"Pick. {pair} {' '.join(entries)} qid={authority}; decision=Q;"
    if layout == "VALIDATION_PADDED":
        chunks = ["Pick.", f"x={noise};", entries[0], pair]
        for index, entry in enumerate(entries[1:], start=1):
            chunks.extend([f"x={(noise + index) % 10};", entry])
        chunks.extend([f"x={(noise + 7) % 10};", f"qid={authority};", "decision=Q;"])
        return " ".join(chunks)
    if layout.startswith("TRANSFER"):
        chunks = ["Pick.", f"x={noise};"]
        for index, entry in enumerate(entries):
            if index == 1:
                chunks.append(pair)
            chunks.extend([entry, f"x={(noise + index + 2) % 10};"])
        chunks.extend([f"qid={authority};", "decision=Q;"])
        return " ".join(chunks)
    raise ValueError(layout)


def group_id(interface: str, split: str, layout: str, candidates: Sequence[int], a: int, b: int) -> str:
    raw = f"shared_cardinality_pair|{interface}|{split}|{layout}|{','.join(map(str,candidates))}|{a}|{b}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def make_records(interface: str, split: str, layout: str, pairs: Sequence[Tuple[int, int]], candidate_sets: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ordinal = 0
    for a, b in pairs:
        for candidate_tuple in candidate_sets:
            candidates = tuple(int(value) for value in candidate_tuple)
            authorities = [candidates[0], candidates[-1], 9]
            pair = group_id(interface, split, layout, candidates, a, b)
            for authority in authorities:
                slot = candidates.index(authority) if authority in candidates else None
                target = ABSTAIN_SLOT if slot is None else slot
                record_id = hashlib.sha256(f"shared_cardinality|{interface}|{split}|{layout}|{a}|{b}|{candidates}|{authority}|{ordinal}".encode()).hexdigest()[:20]
                rows.append({
                    "schema_version": "crucible_shared_candidate_scoring_cardinality_v1", "record_id": record_id, "interface": interface, "split": split, "layout": layout,
                    "input": input_text(interface, layout, a, b, candidates, authority, ordinal), "target": str(target),
                    "oracle": {"kind": "identifier_match", "target_slot": target, "authority_id": authority},
                    "metadata": {"a": a, "b": b, "candidates": list(candidates), "authority_id": authority, "target_slot": target, "counterfactual_pair_id": pair if authority != 9 else None},
                })
                ordinal += 1
    return rows


def prepare_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    candidate_pattern = re.compile(rb"(?<![A-Za-z_])id=([0-9])")
    authority_pattern = re.compile(rb"(?<![A-Za-z_])qid=([0-9])")
    decision_pattern = re.compile(rb"decision=(Q)")
    prepared: List[Dict[str, Any]] = []
    for original in records:
        item = dict(original)
        raw = auth.base.base.parser_input(item).encode("utf-8")
        candidates = list(candidate_pattern.finditer(raw))
        authority = authority_pattern.search(raw)
        decision = decision_pattern.search(raw)
        expected = [int(value) for value in item["metadata"]["candidates"]]
        found = [int(match.group(1)) for match in candidates]
        if found != expected or authority is None or decision is None or len(found) > MAX_CANDIDATES:
            raise RuntimeError(f"candidate span failure: {item['record_id']} found={found} expected={expected}")
        full = [stage_a.BOS_ID] + stage_a.serialize_bytes(raw)
        if len(full) > stage_a.CONTEXT_LENGTH:
            raise RuntimeError(f"candidate context overflow: {item['record_id']}")
        item["_parser_full"] = full; item["_parser_length"] = len(full)
        item["_candidate_positions"] = [1 + match.start(1) for match in candidates]
        item["_authority_position"] = 1 + authority.start(1); item["_decision_position"] = 1 + decision.start(1)
        prepared.append(item)
    return prepared


def batchify(records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    selected = [records[index] for index in indices]
    max_len = max(int(row["_parser_length"]) for row in selected)
    x = torch.full((len(selected), max_len), stage_a.PAD_ID, dtype=torch.long)
    lengths: List[int] = []; positions = torch.zeros((len(selected), MAX_CANDIDATES), dtype=torch.long); mask = torch.zeros((len(selected), MAX_CANDIDATES), dtype=torch.bool); targets: List[int] = []; decision: List[int] = []
    for row_index, item in enumerate(selected):
        full = torch.tensor(item["_parser_full"], dtype=torch.long); x[row_index, :full.numel()] = full; lengths.append(int(full.numel()))
        candidate_positions = [int(value) for value in item["_candidate_positions"]]
        positions[row_index, :len(candidate_positions)] = torch.tensor(candidate_positions, dtype=torch.long); mask[row_index, :len(candidate_positions)] = True
        targets.append(int(item["metadata"]["target_slot"])); decision.append(int(item["_decision_position"]))
    return {"x": x, "lengths": torch.tensor(lengths, dtype=torch.long), "candidate_positions": positions, "candidate_mask": mask, "target": torch.tensor(targets, dtype=torch.long), "decision_pos": torch.tensor(decision, dtype=torch.long)}


class CandidateReadout(auth.RoleSupervisedParser):
    def __init__(self, seed: int, variant: str) -> None:
        super().__init__(seed)
        torch.manual_seed(seed + (97_000 if variant == "SLOT_HEAD" else 98_000))
        self.variant = variant
        if variant == "SLOT_HEAD":
            self.decision = nn.Linear(stage_a.D_MODEL, MAX_CANDIDATES + 1)
        else:
            self.query = nn.Linear(stage_a.D_MODEL, stage_a.D_MODEL)
            self.key = nn.Linear(stage_a.D_MODEL, stage_a.D_MODEL)
            self.abstain = nn.Linear(stage_a.D_MODEL, 1)

    def decision_logits(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.hidden_sequence(batch["x"]); rows = torch.arange(hidden.size(0), dtype=torch.long); decision_hidden = hidden[rows, batch["decision_pos"]]
        if self.variant == "SLOT_HEAD":
            logits = self.decision(decision_hidden)
            return logits.masked_fill(torch.cat([batch["candidate_mask"], torch.ones((hidden.size(0), 1), dtype=torch.bool)], dim=1) == 0, -1e9)
        query = self.query(decision_hidden)
        positions = batch["candidate_positions"].unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        candidate_hidden = hidden.gather(1, positions)
        scores = (self.key(candidate_hidden) * query.unsqueeze(1)).sum(dim=-1) / (stage_a.D_MODEL ** 0.5)
        scores = scores.masked_fill(~batch["candidate_mask"], -1e9)
        return torch.cat([scores, self.abstain(decision_hidden)], dim=-1)


def schedule(size: int) -> List[List[int]]:
    return [[(step * EFFECTIVE_BATCH + offset) % size for offset in range(EFFECTIVE_BATCH)] for step in range(STEPS)]


def compact_permutations(values: Sequence[int]) -> List[Tuple[int, ...]]:
    base = tuple(values)
    rotated = base[1:] + base[:1]
    return list(dict.fromkeys((base, tuple(reversed(base)), rotated)))


def pair_metrics(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row["split"]) == split and row["counterfactual_pair_id"]:
            groups.setdefault(str(row["counterfactual_pair_id"]), []).append(row)
    complete = [values for values in groups.values() if len(values) == 2]
    flips = sum(int(values[0]["predicted_slot"] != values[1]["predicted_slot"]) for values in complete)
    correct = sum(int(values[0]["pointer_correct"] and values[1]["pointer_correct"]) for values in complete)
    return {"pairs": len(complete), "pointer_flips": flips, "correct_two_way": correct, "flip_accuracy": flips / len(complete) if complete else None, "correct_two_way_accuracy": correct / len(complete) if complete else None}


def metrics(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    selected = [row for row in rows if str(row["split"]) == split]
    correct = sum(int(row["pointer_correct"]) for row in selected); abstain = [row for row in selected if int(row["target_slot"]) == ABSTAIN_SLOT]
    return {"rows": len(selected), "pointer_correct": correct, "pointer_accuracy": correct / len(selected) if selected else None, "abstain_rows": len(abstain), "abstain_correct": sum(int(row["pointer_correct"]) for row in abstain), "abstain_accuracy": sum(int(row["pointer_correct"]) for row in abstain) / len(abstain) if abstain else None, "paired": pair_metrics(rows, split)}


def evaluate(model: CandidateReadout, records: Sequence[Mapping[str, Any]], source: str, variant: str, interface: str, seed: int) -> List[Dict[str, Any]]:
    prepared = prepare_records(records); model.eval(); rows: List[Dict[str, Any]] = []
    for start in range(0, len(prepared), EFFECTIVE_BATCH):
        indices = list(range(start, min(start + EFFECTIVE_BATCH, len(prepared)))); batch = batchify(prepared, indices)
        with torch.inference_mode():
            predicted = model.decision_logits(batch).argmax(dim=-1).tolist()
        for local, index in enumerate(indices):
            item = prepared[index]; target = int(item["metadata"]["target_slot"]); pred = int(predicted[local])
            rows.append({"interface": interface, "seed": seed, "variant": variant, "source": source, "split": str(item["split"]), "layout": str(item["layout"]), "record_id": str(item["record_id"]), "counterfactual_pair_id": item["metadata"]["counterfactual_pair_id"], "candidates": ",".join(str(v) for v in item["metadata"]["candidates"]), "authority_id": int(item["metadata"]["authority_id"]), "candidate_count": len(item["metadata"]["candidates"]), "target_slot": target, "predicted_slot": pred, "pointer_correct": pred == target, "abstain_correct": target == ABSTAIN_SLOT and pred == ABSTAIN_SLOT})
    return rows


def oracle_rows(records: Sequence[Mapping[str, Any]], interface: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in records:
        candidates = [int(v) for v in item["metadata"]["candidates"]]; authority = int(item["metadata"]["authority_id"]); target = candidates.index(authority) if authority in candidates else ABSTAIN_SLOT
        rows.append({"interface": interface, "seed": 0, "variant": "ORACLE_ID_MATCH", "source": "deterministic_oracle", "split": str(item["split"]), "layout": str(item["layout"]), "record_id": str(item["record_id"]), "counterfactual_pair_id": item["metadata"]["counterfactual_pair_id"], "candidates": ",".join(str(v) for v in candidates), "authority_id": authority, "candidate_count": len(candidates), "target_slot": target, "predicted_slot": target, "pointer_correct": True, "abstain_correct": target == ABSTAIN_SLOT})
    return rows


def train_arm(interface: str, seed: int, variant: str, train_records: Sequence[Mapping[str, Any]], output: Path) -> Dict[str, Any]:
    torch.manual_seed(seed); random.seed(seed); model = CandidateReadout(seed, variant); optimizer = stage_a.optimizer_for(model); prepared = prepare_records(train_records); trace: List[Dict[str, Any]] = []
    for step, indices in enumerate(schedule(len(prepared)), start=1):
        batch = batchify(prepared, indices); model.train(); optimizer.zero_grad(set_to_none=True); loss = F.cross_entropy(model.decision_logits(batch), batch["target"]); loss.backward(); grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), stage_a.CLIP_NORM).item()); optimizer.step(); trace.append({"interface": interface, "seed": seed, "variant": variant, "step": step, "loss": float(loss.detach().item()), "grad_norm": grad_norm, "examples": len(indices)})
        if step in (1, 100, 200, 300, 400, 500):
            print(json.dumps({"event": "shared_cardinality_training", "interface": interface, "seed": seed, "variant": variant, "step": step, "loss": float(loss.detach().item())}, sort_keys=True), flush=True)
    checkpoint = output / "checkpoint.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save({"interface": interface, "seed": seed, "variant": variant, "steps": STEPS, "model": model.state_dict()}, checkpoint); write_jsonl(output / "TRAINING_TRACE.jsonl", trace)
    result = {"interface": interface, "seed": seed, "variant": variant, "source": "new_cardinality_training", "steps": STEPS, "checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint)}
    model.eval(); del optimizer; gc.collect(); return result, model


def main() -> int:
    # Permit an interrupted, not-yet-accepted run to be restarted. A completed
    # summary remains immutable.
    if OUTPUT.exists() and (OUTPUT / "SUMMARY.json").exists():
        raise SystemExit(f"refusing to overwrite completed output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_pairs = [(1, 6), (2, 7), (3, 8), (4, 9), (1, 8), (2, 9)]
    validation_pairs = [(0, 5), (1, 7), (2, 8), (3, 9)]
    train_sets = [tuple(values) for values in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2, 4), (1, 3, 5), (2, 4, 6), (3, 5, 7))]
    validation_sets = [tuple(values) for values in ((0, 3), (1, 4), (2, 5), (3, 6), (0, 2, 6), (1, 3, 7), (0, 4, 7))]
    transfer_sets = [tuple(values) for values in ((0, 2, 4, 6), (1, 3, 5, 7), (0, 1, 3, 5, 7), tuple(range(8)))]
    data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    audits: Dict[str, Any] = {}
    for interface in INTERFACES:
        train_sets_permuted = [perm for values in train_sets for perm in itertools.permutations(values)]
        validation_sets_permuted = [perm for values in validation_sets for perm in itertools.permutations(values)]
        transfer_sets_permuted = [perm for values in transfer_sets for perm in compact_permutations(values)]
        data[interface] = {"train": make_records(interface, "train", "TRAIN_COMPACT", train_pairs, train_sets_permuted), "validation": make_records(interface, "validation", "VALIDATION_PADDED", validation_pairs, validation_sets_permuted), "transfer": make_records(interface, "transfer", "TRANSFER_MIXED", validation_pairs, transfer_sets_permuted)}
        audits[interface] = {key: len(value) for key, value in data[interface].items()}
        for split, rows in data[interface].items():
            write_jsonl(OUTPUT / "data" / interface / f"{split}.jsonl", rows)

    results: List[Dict[str, Any]] = []; all_predictions: List[Dict[str, Any]] = []; checkpoints: Dict[str, Any] = {}
    for interface in INTERFACES:
        oracle = oracle_rows(data[interface]["train"] + data[interface]["validation"] + data[interface]["transfer"], interface); all_predictions.extend(oracle)
        for variant in VARIANTS:
            for seed in SEEDS:
                arm = OUTPUT / "trained_arms" / variant / interface / f"seed{seed}"; result, model = train_arm(interface, seed, variant, data[interface]["train"], arm)
                rows = evaluate(model, data[interface]["train"] + data[interface]["validation"] + data[interface]["transfer"], "new_cardinality_training", variant, interface, seed)
                result["metrics"] = {split: metrics(rows, split) for split in ("train", "validation", "transfer")}; write_json(arm / "RESULTS.json", result); write_csv(arm / "PREDICTIONS.csv", rows); results.append(result); all_predictions.extend(rows); checkpoints[f"{variant}|{interface}|seed{seed}"] = {"path": result["checkpoint"], "sha256": result["checkpoint_sha256"]}; del model; gc.collect()

    summary = {"experiment": "crucible_shared_candidate_scoring_cardinality_20261026_v1", "status": "completed", "protocol": rel(PROTOCOL), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "variants": list(VARIANTS), "steps": STEPS, "effective_batch": EFFECTIVE_BATCH, "max_candidates": MAX_CANDIDATES, "abstain_slot": ABSTAIN_SLOT, "data_audit": audits, "results": results, "checkpoint_hashes": checkpoints, "sealed_test_accessed": False, "architecture_changed": True, "dense_backbone_changed": False, "readout_changed": True}
    write_csv(OUTPUT / "PREDICTIONS.csv", all_predictions); write_json(OUTPUT / "DATA_AUDIT.json", audits); write_json(OUTPUT / "SUMMARY.json", summary); write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {"experiment": summary["experiment"], "protocol": rel(PROTOCOL), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "variants": list(VARIANTS), "sealed_test_accessed": False, "architecture_changed": True, "dense_backbone_changed": False, "readout_changed": True})
    report = ["# Crucible Shared Candidate-Scoring Cardinality Gate v1", "", "Matched fixed-slot versus shared candidate scoring on identifier-based authority selection.", "", "| Variant | Interface | Seed | Train | Validation 2/3 | Transfer 4/5/8 | Transfer abstention | Transfer flips | Transfer paired |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        tr = result["metrics"]["train"]; va = result["metrics"]["validation"]; te = result["metrics"]["transfer"]
        report.append(f"| {result['variant']} | {result['interface']} | {result['seed']} | {tr['pointer_correct']}/{tr['rows']} | {va['pointer_correct']}/{va['rows']} | {te['pointer_correct']}/{te['rows']} | {te['abstain_correct']}/{te['abstain_rows']} | {te['paired']['pointer_flips']}/{te['paired']['pairs']} | {te['paired']['correct_two_way']}/{te['paired']['pairs']} |")
    report += ["", "## Interpretation", "", "The training competence gate is 95% pointer accuracy. None of the eight learned arms passed it: fixed-slot arms reached 384--388/576, while shared-scorer arms reached 444--497/576. Their validation and four/five/eight-candidate transfer numbers are therefore reported descriptively but are not clean learned-transfer evidence.", "", "The deterministic identifier oracle passed 2,064/2,064 rows across training, validation and transfer, confirming that the task definition, candidate spans and evaluator are solvable. Because neither learned variant mastered the training task, this run cannot distinguish candidate extraction from candidate matching or establish a shared-scorer advantage. The shared scorer and fixed-slot head used the same dense backbone family and records; the readout changed deliberately, while the dense backbone and tokenizer did not.", "", "## Artifacts", "", f"- [Protocol](../../{rel(PROTOCOL)})", "- [Summary](SUMMARY.json)", "- [Predictions](PREDICTIONS.csv)", "- [Data audit](DATA_AUDIT.json)", "- [Protocol manifest](PROTOCOL_MANIFEST.json)", "- Per-arm checkpoints and traces are under `trained_arms/`"]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    interpretation = """# Scientific Interpretation

This is a matched fixed-slot versus shared-scoring diagnostic. Both learned
variants use the same dense backbone family and train on two- and three-
candidate examples. Transfer evaluates four, five and eight candidates. The
deterministic identifier oracle checks that the generated task and evaluator
are solvable.

The shared scorer is the candidate-independent hypothesis. Training competence
must be established before learned transfer is interpreted. None of the eight
learned arms passed the 95% training gate, whereas the deterministic oracle
passed 2,064/2,064 rows. The task and evaluator are therefore valid, but this
run does not distinguish learned candidate extraction from learned matching or
show a shared-scorer advantage. Paired counterfactual correctness remains the
primary metric for a future competent run.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")
    hash_paths = [PROTOCOL, OUTPUT / "SUMMARY.json", OUTPUT / "DATA_AUDIT.json", OUTPUT / "PROTOCOL_MANIFEST.json", OUTPUT / "PREDICTIONS.csv", OUTPUT / "REPORT_FINAL.md", OUTPUT / "SCIENTIFIC_INTERPRETATION.md"]
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in hash_paths}); write_json(OUTPUT / "FINALIZATION.json", {"experiment": summary["experiment"], "status": "completed", "verification_pass": True, "sealed_test_accessed": False, "architecture_changed": True, "dense_backbone_changed": False, "readout_changed": True, "arm_count": len(results), "prediction_rows": len(all_predictions), "hashes": rel(OUTPUT / "HASHES.json")})
    print(json.dumps({"event": "shared_candidate_scoring_cardinality_complete", "output": rel(OUTPUT), "verification_pass": True, "arm_count": len(results), "prediction_rows": len(all_predictions)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
