#!/usr/bin/env python3
"""Read-only implementation and training audit for cardinality gate v1.

This audit never calls an optimizer step, writes checkpoints, changes the
completed experiment, or reads a sealed test set.  It records the data,
training schedule, traces, parameter-use path, and one backward-only gradient
check for each accepted learned checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "crucible_shared_candidate_scoring_cardinality_20261026_v1"
RUNNER = ROOT / "tools" / "run_crucible_shared_candidate_scoring_cardinality_20261026.py"
OUTPUT = EXPERIMENT / "IMPLEMENTATION_AUDIT.json"
INTERFACES = ("canonical", "factorized")
VARIANTS = ("SLOT_HEAD", "SHARED_SCORER")
SEEDS = (42, 1337)
TRAIN_THRESHOLD = 0.95


def import_runner() -> Any:
    spec = importlib.util.spec_from_file_location("cardinality_runner_for_audit", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(RUNNER)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(RUNNER)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def target_for(row: Mapping[str, Any]) -> int:
    candidates = [int(value) for value in row["metadata"]["candidates"]]
    authority = int(row["metadata"]["authority_id"])
    return candidates.index(authority) if authority in candidates else 8


def data_audit(runner: Any, interface: str) -> Dict[str, Any]:
    candidate_pattern = re.compile(rb"(?<![A-Za-z_])id=([0-9])")
    authority_pattern = re.compile(rb"(?<![A-Za-z_])qid=([0-9])")
    decision_pattern = re.compile(rb"decision=(Q)")
    splits = {split: load_jsonl(EXPERIMENT / "data" / interface / f"{split}.jsonl") for split in ("train", "validation", "transfer")}
    result: Dict[str, Any] = {"splits": {}, "cross_split_overlap": {}, "span_checks": {}}
    for split, rows in splits.items():
        target_counts = Counter(int(row["metadata"]["target_slot"]) for row in rows)
        positive_slots = Counter(
            (len(row["metadata"]["candidates"]), int(row["metadata"]["target_slot"]))
            for row in rows
            if int(row["metadata"]["target_slot"]) != 8
        )
        record_ids = [str(row["record_id"]) for row in rows]
        inputs = [str(row["input"]) for row in rows]
        span_failures: List[str] = []
        causal_order_failures: List[str] = []
        lengths: List[int] = []
        for row in rows:
            raw = str(row["input"]).encode("utf-8")
            candidates = list(candidate_pattern.finditer(raw))
            authority = authority_pattern.search(raw)
            decision = decision_pattern.search(raw)
            lengths.append(len(raw) + 1)
            expected = [int(value) for value in row["metadata"]["candidates"]]
            found = [int(match.group(1)) for match in candidates]
            if found != expected or authority is None or decision is None:
                span_failures.append(str(row["record_id"]))
                continue
            candidate_positions = [1 + match.start(1) for match in candidates]
            authority_position = 1 + authority.start(1)
            decision_position = 1 + decision.start(1)
            if not all(position < decision_position for position in candidate_positions + [authority_position]):
                causal_order_failures.append(str(row["record_id"]))
        result["splits"][split] = {
            "rows": len(rows),
            "unique_record_ids": len(set(record_ids)),
            "unique_inputs": len(set(inputs)),
            "candidate_counts": dict(sorted(Counter(len(row["metadata"]["candidates"]) for row in rows).items())),
            "target_slots": dict(sorted(target_counts.items())),
            "positive_slots_by_candidate_count": {f"{count}:{slot}": value for (count, slot), value in sorted(positive_slots.items())},
            "authority_ids": dict(sorted(Counter(int(row["metadata"]["authority_id"]) for row in rows).items())),
            "counterfactual_groups": len({row["metadata"]["counterfactual_pair_id"] for row in rows if row["metadata"]["counterfactual_pair_id"]}),
            "input_byte_length_min": min(lengths),
            "input_byte_length_max": max(lengths),
        }
        result["span_checks"][split] = {
            "candidate_span_failures": len(span_failures),
            "causal_order_failures": len(causal_order_failures),
            "failed_record_ids": span_failures + causal_order_failures,
        }
    for left, right in (("train", "validation"), ("train", "transfer"), ("validation", "transfer")):
        result["cross_split_overlap"][f"{left}__{right}"] = {
            "record_ids": len(set(row["record_id"] for row in splits[left]) & set(row["record_id"] for row in splits[right])),
            "exact_inputs": len(set(row["input"] for row in splits[left]) & set(row["input"] for row in splits[right])),
        }
    train = splits["train"]
    schedule = runner.schedule(len(train))
    presentations = Counter(index for batch in schedule for index in batch)
    profiles = Counter()
    for batch in schedule:
        profile = tuple(sorted(Counter(int(train[index]["metadata"]["target_slot"]) for index in batch).items()))
        profiles[profile] += 1
    result["schedule"] = {
        "steps": len(schedule),
        "batch_size": runner.EFFECTIVE_BATCH,
        "total_presentations": sum(presentations.values()),
        "presentations_per_record": dict(sorted(Counter(presentations.values()).items())),
        "batch_target_profiles": {str(profile): count for profile, count in profiles.items()},
        "candidate_count_blocks_per_cycle": "12 batches of 2-candidate records followed by 36 batches of 3-candidate records",
    }
    result["implementation_facts"] = {
        "candidate_span_extraction": "deterministic regex in prepare_records; no learned span extractor",
        "shared_scorer_candidate_positions": "candidate positions are supplied by the deterministic parser to batchify",
        "candidate_count_mask": "deterministically supplied to both variants; fixed-slot logits are masked for invalid slots",
        "backbone_training": "backbone parameters are trainable; the run does not freeze the dense backbone",
        "training_order": "deterministic cyclic schedule; no shuffle or random sampler",
        "authority_coverage": "authorities are first candidate, last candidate, or absent id=9; interior candidates are never positive targets",
    }
    return result


def trace_audit() -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for variant in VARIANTS:
        for interface in INTERFACES:
            for seed in SEEDS:
                key = f"{variant}|{interface}|seed{seed}"
                path = EXPERIMENT / "trained_arms" / variant / interface / f"seed{seed}" / "TRAINING_TRACE.jsonl"
                rows = load_jsonl(path)
                losses = [float(row["loss"]) for row in rows]
                gradients = [float(row["grad_norm"]) for row in rows]
                result[key] = {
                    "trace_rows": len(rows),
                    "step_min": min(int(row["step"]) for row in rows),
                    "step_max": max(int(row["step"]) for row in rows),
                    "batch_sizes": sorted(set(int(row["examples"]) for row in rows)),
                    "loss_first": losses[0],
                    "loss_min": min(losses),
                    "loss_last": losses[-1],
                    "loss_mean_last_100": sum(losses[-100:]) / min(100, len(losses)),
                    "grad_norm_min": min(gradients),
                    "grad_norm_max": max(gradients),
                    "grad_norm_mean_last_100": sum(gradients[-100:]) / min(100, len(gradients)),
                }
    return result


def gradient_audit(runner: Any) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    torch = runner.torch
    for interface in INTERFACES:
        records = load_jsonl(EXPERIMENT / "data" / interface / "train.jsonl")
        prepared = runner.prepare_records(records)
        batch = runner.batchify(prepared, list(range(runner.EFFECTIVE_BATCH)))
        for variant in VARIANTS:
            for seed in SEEDS:
                key = f"{variant}|{interface}|seed{seed}"
                checkpoint = EXPERIMENT / "trained_arms" / variant / interface / f"seed{seed}" / "checkpoint.pt"
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = runner.CandidateReadout(seed, variant)
                model.load_state_dict(state["model"], strict=True)
                model.train()
                model.zero_grad(set_to_none=True)
                logits = model.decision_logits(batch)
                loss = runner.F.cross_entropy(logits, batch["target"])
                loss.backward()
                parameter_rows = []
                for name, parameter in model.named_parameters():
                    parameter_rows.append({
                        "name": name,
                        "has_gradient": parameter.grad is not None,
                        "gradient_norm": float(parameter.grad.norm().item()) if parameter.grad is not None else 0.0,
                        "nonzero_gradient_elements": int(torch.count_nonzero(parameter.grad).item()) if parameter.grad is not None else 0,
                    })
                output[key] = {
                    "backward_only": True,
                    "batch_rows": len(batch["target"]),
                    "loss": float(loss.detach().item()),
                    "backbone_parameters_with_gradient": sum(row["has_gradient"] for row in parameter_rows if row["name"].startswith("backbone.")),
                    "backbone_parameter_count": sum(row["name"].startswith("backbone.") for row in parameter_rows),
                    "readout_parameters_with_gradient": sum(row["has_gradient"] for row in parameter_rows if not row["name"].startswith("backbone.")),
                    "readout_parameter_count": sum(not row["name"].startswith("backbone.") for row in parameter_rows),
                    "unused_parameter_names": [row["name"] for row in parameter_rows if not row["has_gradient"]],
                    "decision_readout_gradients": [row for row in parameter_rows if any(token in row["name"] for token in ("decision.", "query.", "key.", "abstain."))],
                }
                del model, state
    return output


def main() -> int:
    runner = import_runner()
    audit = {
        "experiment": "crucible_shared_candidate_scoring_cardinality_20261026_v1",
        "audit_type": "read_only_implementation_training_and_gradient_audit",
        "sealed_test_accessed": False,
        "optimizer_step_performed": False,
        "checkpoints_modified": False,
        "data": {interface: data_audit(runner, interface) for interface in INTERFACES},
        "traces": trace_audit(),
        "gradient_flow": gradient_audit(runner),
    }
    OUTPUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "interfaces": list(INTERFACES), "arms": len(audit["gradient_flow"]), "sealed_test_accessed": False, "optimizer_step_performed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
