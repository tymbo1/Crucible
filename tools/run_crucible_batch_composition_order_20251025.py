#!/usr/bin/env python3
"""Run the preregistered batch-composition/update-order control matrix.

The experiment keeps the paired-positional exposure dataset and optimizer
fixed while changing only batch composition and the order in which fixed
batches are presented.  It also performs a read-only evaluation of the prior
A0 checkpoints across the same layout matrix.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "run_crucible_paired_positional_exposure_20251025.py"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_BATCH_COMPOSITION_ORDER_PROTOCOL_20251025.md"
OUTPUT = ROOT / "experiments" / "crucible_batch_composition_order_20251025_v1"
REFERENCE_ROOT = ROOT / "experiments" / "crucible_paired_positional_exposure_20251025_v1"

INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
ARMS = ("B0_DIVERSE_SHUFFLED", "B1_DIVERSE_GROUPED", "B2_SAME_PAIR_SHUFFLED", "B3_SAME_PAIR_GROUPED")
TRAIN_LAYOUTS = ("FIXED_NEAR_PADDED", "EARLY_PADDED", "MIDDLE_POSITION")
EVAL_LAYOUTS = TRAIN_LAYOUTS + ("UNSEEN_BETWEEN_OPERANDS", "UNSEEN_EARLY_CONTEXT_FIRST")
STEPS = 500
EFFECTIVE_BATCH = 12
GROUP_SIZE = 6
SCHEDULE_SEED = 20261025
CURVE_STEPS = (1, 10, 25, 50, 100, 200, 300, 400, 500)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


exposure = import_module("batch_order_exposure_source", SOURCE)
ladder = exposure.ladder
stage_a = exposure.stage_a
torch = stage_a.torch


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(row), sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def group_metadata(records: Sequence[Mapping[str, Any]], groups: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    for group_id, indices in enumerate(groups):
        if len(indices) != GROUP_SIZE:
            raise RuntimeError(f"group {group_id} has {len(indices)} records, expected {GROUP_SIZE}")
        values.append({
            "group_id": group_id,
            "pair": [int(records[indices[0]]["metadata"]["a"]), int(records[indices[0]]["metadata"]["b"])],
            "record_indices": list(indices),
            "record_ids": [str(records[index]["record_id"]) for index in indices],
            "layout_operation_order": [
                [str(records[index]["metadata"]["layout"]), str(records[index]["metadata"]["operation"])]
                for index in indices
            ],
        })
    return values


def diverse_batches(groups: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    """Create 25 balanced cycles of 20 diverse batches.

    There are 40 pair groups in the frozen dataset.  Each cycle presents every
    pair exactly six times, once for each member of its six-record group.  A
    batch contains 12 distinct pair groups.  Thus every record is presented
    exactly 25 times across the 500 updates.
    """
    n_groups = len(groups)
    if n_groups != 40:
        raise RuntimeError(f"expected 40 train pair groups, found {n_groups}")
    batches: List[Dict[str, Any]] = []
    for cycle in range(25):
        occurrences = [0] * n_groups
        cycle_batches: List[Dict[str, Any]] = []
        for batch_in_cycle in range(20):
            pair_groups = [(batch_in_cycle * EFFECTIVE_BATCH + offset) % n_groups for offset in range(EFFECTIVE_BATCH)]
            if len(set(pair_groups)) != EFFECTIVE_BATCH:
                raise RuntimeError("diverse batch contains duplicate pair groups")
            variants: List[int] = []
            record_indices: List[int] = []
            for group_id in pair_groups:
                variant = occurrences[group_id]
                if variant >= GROUP_SIZE:
                    raise RuntimeError("pair group appeared more than six times in a cycle")
                occurrences[group_id] += 1
                variants.append(variant)
                record_indices.append(groups[group_id][variant])
            cycle_batches.append({
                "cycle": cycle,
                "batch_in_cycle": batch_in_cycle,
                "pair_groups": pair_groups,
                "variants": variants,
                "record_indices": record_indices,
                "phase": sum(variants) / len(variants),
            })
        if set(occurrences) != {GROUP_SIZE}:
            raise RuntimeError(f"diverse cycle exposure mismatch: {occurrences}")
        # The natural order is already phase-grouped.  Sort explicitly so the
        # protocol records the intended ordering rather than relying on this
        # construction detail.
        cycle_batches.sort(key=lambda item: (float(item["phase"]), int(item["batch_in_cycle"])))
        batches.extend(cycle_batches)
    return batches


def same_pair_batches(groups: Sequence[Sequence[int]]) -> List[Dict[str, Any]]:
    n_groups = len(groups)
    if n_groups != 40:
        raise RuntimeError(f"expected 40 train pair groups, found {n_groups}")
    batches: List[Dict[str, Any]] = []
    for step in range(STEPS):
        first = (2 * step) % n_groups
        pair_groups = [first, (first + 1) % n_groups]
        record_indices = [index for group_id in pair_groups for index in groups[group_id]]
        batches.append({
            "cycle": step // 20,
            "batch_in_cycle": step % 20,
            "pair_groups": pair_groups,
            "variants": list(range(GROUP_SIZE)) + list(range(GROUP_SIZE)),
            "record_indices": record_indices,
            "phase": float(step % GROUP_SIZE),
        })
    return batches


def reindex_batches(batches: Sequence[Mapping[str, Any]], order_label: str) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for step, batch in enumerate(batches, start=1):
        result.append({**dict(batch), "step": step, "ordering": order_label})
    return result


def build_schedules(groups: Sequence[Sequence[int]]) -> Dict[str, List[Dict[str, Any]]]:
    diverse = diverse_batches(groups)
    same = same_pair_batches(groups)
    diverse_shuffled = [dict(batch) for batch in diverse]
    same_shuffled = [dict(batch) for batch in same]
    random.Random(SCHEDULE_SEED + 1).shuffle(diverse_shuffled)
    random.Random(SCHEDULE_SEED + 2).shuffle(same_shuffled)
    return {
        "B0_DIVERSE_SHUFFLED": reindex_batches(diverse_shuffled, "shuffled_batches"),
        "B1_DIVERSE_GROUPED": reindex_batches(diverse, "phase_grouped_batches"),
        "B2_SAME_PAIR_SHUFFLED": reindex_batches(same_shuffled, "shuffled_batches"),
        "B3_SAME_PAIR_GROUPED": reindex_batches(same, "pair_grouped_batches"),
    }


def schedule_audit(schedules: Mapping[str, Sequence[Mapping[str, Any]]], records: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    audit: Dict[str, Any] = {"arms": {}, "pairwise_schedule_digests": {}, "pass": True}
    expected_counts = {index: 25 for index in range(len(records))}
    expected_tokens = sum(len(item["_full"]) for item in prepared) * 25
    for arm, batches in schedules.items():
        counts: Counter[int] = Counter()
        batch_sizes: List[int] = []
        unique_pairs: List[int] = []
        unique_layouts: List[int] = []
        token_exposure = 0
        for batch in batches:
            indices = [int(index) for index in batch["record_indices"]]
            counts.update(indices)
            batch_sizes.append(len(indices))
            unique_pairs.append(len(set(int(group) for group in batch["pair_groups"])))
            unique_layouts.append(len({str(records[index]["metadata"]["layout"]) for index in indices}))
            token_exposure += sum(len(prepared[index]["_full"]) for index in indices)
        count_map = {str(index): int(counts[index]) for index in range(len(records))}
        passed = (
            len(batches) == STEPS
            and set(batch_sizes) == {EFFECTIVE_BATCH}
            and count_map == {str(index): count for index, count in expected_counts.items()}
            and token_exposure == expected_tokens
        )
        audit["arms"][arm] = {
            "steps": len(batches),
            "batch_sizes": sorted(set(batch_sizes)),
            "record_exposure_counts": count_map,
            "token_exposure": token_exposure,
            "expected_token_exposure": expected_tokens,
            "unique_pair_groups_min": min(unique_pairs),
            "unique_pair_groups_max": max(unique_pairs),
            "unique_layouts_min": min(unique_layouts),
            "unique_layouts_max": max(unique_layouts),
            "schedule_digest": stable_digest(list(batches)),
            "pass": passed,
        }
        audit["pass"] = audit["pass"] and passed
    audit["pairwise_schedule_digests"] = {
        "B0_vs_B1_same_batch_multiset": stable_digest(sorted(stable_digest({"groups": b["pair_groups"], "indices": b["record_indices"]}) for b in schedules["B0_DIVERSE_SHUFFLED"])) == stable_digest(sorted(stable_digest({"groups": b["pair_groups"], "indices": b["record_indices"]}) for b in schedules["B1_DIVERSE_GROUPED"])),
        "B2_vs_B3_same_batch_multiset": stable_digest(sorted(stable_digest({"groups": b["pair_groups"], "indices": b["record_indices"]}) for b in schedules["B2_SAME_PAIR_SHUFFLED"])) == stable_digest(sorted(stable_digest({"groups": b["pair_groups"], "indices": b["record_indices"]}) for b in schedules["B3_SAME_PAIR_GROUPED"])),
        "all_arms_same_record_exposure": len({stable_digest(value["record_exposure_counts"]) for value in audit["arms"].values()}) == 1,
        "all_arms_same_token_exposure": len({int(value["token_exposure"]) for value in audit["arms"].values()}) == 1,
    }
    audit["pass"] = audit["pass"] and all(bool(value) for value in audit["pairwise_schedule_digests"].values())
    return audit


def gradient_vector(model: Any) -> torch.Tensor:
    values = [parameter.grad.detach().reshape(-1).float() for parameter in model.parameters() if parameter.grad is not None]
    if not values:
        raise RuntimeError("no gradients available")
    return torch.cat(values)


def batch_statistics(records: Sequence[Mapping[str, Any]], indices: Sequence[int], pair_groups: Sequence[int]) -> Dict[str, Any]:
    layouts = Counter(str(records[index]["metadata"]["layout"]) for index in indices)
    operations = Counter(str(records[index]["metadata"]["operation"]) for index in indices)
    groups = [int(group) for group in pair_groups]
    return {
        "unique_pair_groups": len(set(groups)),
        "pair_group_count": len(groups),
        "pair_group_diversity_fraction": len(set(groups)) / len(groups),
        "unique_layouts": len(layouts),
        "layout_counts": dict(sorted(layouts.items())),
        "unique_operations": len(operations),
        "operation_counts": dict(sorted(operations.items())),
        "record_count": len(indices),
    }


def train_model(arm: str, interface: str, seed: int, records: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]], output: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = stage_a.make_dense(seed)
    optimizer = stage_a.optimizer_for(model)
    prepared = ladder.prepare(records)
    curves: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    previous_gradient: torch.Tensor | None = None
    started = time.perf_counter()
    for step_spec in schedule:
        step = int(step_spec["step"])
        indices = [int(index) for index in step_spec["record_indices"]]
        batch = exposure.batchify(prepared, indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = stage_a.forward_model(model, batch)
        task_loss, correct, total = stage_a.loss_and_accuracy(logits, batch)
        task_loss.backward()
        current_gradient = gradient_vector(model)
        if previous_gradient is None:
            gradient_cosine = None
        else:
            gradient_cosine = float(torch.nn.functional.cosine_similarity(current_gradient, previous_gradient, dim=0).item())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), stage_a.CLIP_NORM).item())
        optimizer.step()
        previous_gradient = current_gradient.detach().clone()
        trace = {
            "arm": arm,
            "interface": interface,
            "seed": seed,
            "step": step,
            "loss": float(task_loss.detach().item()),
            "task_loss": float(task_loss.detach().item()),
            "correct": int(correct),
            "total": int(total),
            "token_accuracy": correct / total if total else None,
            "grad_norm_pre_clip": grad_norm,
            "gradient_cosine_previous": gradient_cosine,
            **batch_statistics(records, indices, step_spec["pair_groups"]),
        }
        traces.append(trace)
        if step in CURVE_STEPS:
            curves.append({"step": step, **stage_a.evaluate(model, prepared)})
        del current_gradient, logits, batch
    final_eval = stage_a.evaluate(model, prepared)
    checkpoint = output / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    flat_order = [int(index) for step_spec in schedule for index in step_spec["record_indices"]]
    stage_a.save_checkpoint(checkpoint, model, optimizer, STEPS, flat_order)
    result = {
        "arm_id": f"{arm}_{interface}_seed{seed}",
        "arm": arm,
        "interface": interface,
        "seed": seed,
        "steps": STEPS,
        "effective_batch": EFFECTIVE_BATCH,
        "expanded_records": len(records),
        "record_presentations": len(flat_order),
        "train_expanded": final_eval,
        "train_expanded_exact": final_eval["teacher_forced_exact_accuracy"],
        "mean_gradient_cosine_previous": statistics.mean([float(row["gradient_cosine_previous"]) for row in traces if row["gradient_cosine_previous"] is not None]),
        "min_gradient_cosine_previous": min(float(row["gradient_cosine_previous"]) for row in traces if row["gradient_cosine_previous"] is not None),
        "max_gradient_cosine_previous": max(float(row["gradient_cosine_previous"]) for row in traces if row["gradient_cosine_previous"] is not None),
        "mean_loss": statistics.mean(float(row["loss"]) for row in traces),
        "mean_unique_pair_groups_per_batch": statistics.mean(int(row["unique_pair_groups"]) for row in traces),
        "schedule_digest": stable_digest(list(schedule)),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": rel(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
    }
    write_json(output / "RESULTS.json", result)
    write_json(output / "TRAINING_CURVE.json", {"curve_steps": list(CURVE_STEPS), "points": curves})
    write_jsonl(output / "TRAINING_TRACE.jsonl", traces)
    del model, optimizer, previous_gradient
    gc.collect()
    return result, curves, traces


def load_model(path: Path) -> Any:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = stage_a.DenseLM()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def decode_digit(token_id: int) -> Tuple[str, bool]:
    value = int(token_id) - int(stage_a.RAW_BYTE_START)
    if 48 <= value <= 57:
        return chr(value), False
    return f"<token:{token_id}>", True


def prepare_for_prediction(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return ladder.prepare(records)


def paired_metric(rows: Sequence[Mapping[str, Any]], split: str, field: str) -> Dict[str, Any]:
    selected = [row for row in rows if row["split"] == split and int(row["a"]) != int(row["b"])]
    by_pair: Dict[Tuple[int, int], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in selected:
        by_pair[(int(row["a"]), int(row["b"]))][str(row["operation"])] = row
    complete = [value for value in by_pair.values() if set(value) == {"SELECT_FIRST", "SELECT_SECOND"}]
    both = sum(bool(value["SELECT_FIRST"][field] and value["SELECT_SECOND"][field]) for value in complete)
    switched = sum(value["SELECT_FIRST"]["predicted_output"] != value["SELECT_SECOND"]["predicted_output"] for value in complete)
    return {"pair_groups": len(complete), "both_correct": both, "both_correct_rate": both / len(complete) if complete else None, "predictions_switched": switched, "split": split, "correctness_field": field}


def predict_records(model: Any, records: Sequence[Mapping[str, Any]], arm_meta: Mapping[str, Any], layout: str, split: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prepared = prepare_for_prediction(records)
    rows: List[Dict[str, Any]] = []
    model.eval()
    for start in range(0, len(prepared), int(stage_a.BATCH_SIZE)):
        indices = list(range(start, min(start + int(stage_a.BATCH_SIZE), len(prepared))))
        batch = exposure.batchify(prepared, indices)
        with torch.inference_mode():
            logits = stage_a.forward_model(model, batch)
        for local, index in enumerate(indices):
            item = prepared[index]
            output_position = int(item["_prefix_len"]) - 1
            eos_position = output_position + 1
            target_digit_id = int(batch["y"][local, output_position].item())
            predicted_digit_id = int(logits[local, output_position].argmax().item())
            target_eos_id = int(batch["y"][local, eos_position].item()) if eos_position < batch["y"].shape[1] else None
            predicted_eos_id = int(logits[local, eos_position].argmax().item()) if eos_position < logits.shape[1] else None
            predicted_output, output_format_error = decode_digit(predicted_digit_id)
            expected_output = str(item["target"])
            output_token_correct = predicted_digit_id == target_digit_id
            eos_correct = predicted_eos_id == stage_a.EOS_ID if predicted_eos_id is not None else False
            full_sequence_correct = output_token_correct and eos_correct
            rows.append({
                **dict(arm_meta),
                "layout": layout,
                "split": split,
                "record_id": str(item["record_id"]),
                "a": int(item["metadata"]["a"]),
                "b": int(item["metadata"]["b"]),
                "operation": str(item["metadata"]["operation"]),
                "expected_output": expected_output,
                "predicted_output": predicted_output,
                "target_digit_token": target_digit_id,
                "predicted_digit_token": predicted_digit_id,
                "target_eos_token": target_eos_id,
                "predicted_eos_token": predicted_eos_id,
                "output_token_correct": output_token_correct,
                "eos_correct": eos_correct,
                "full_sequence_correct": full_sequence_correct,
                "output_format_error": output_format_error,
                "input_length": len(item["_full"]),
                "output_position": output_position,
            })
    values = [row for row in rows]
    metrics = {
        "rows": len(values),
        "output_token_correct": sum(bool(row["output_token_correct"]) for row in values),
        "output_token_accuracy": sum(bool(row["output_token_correct"]) for row in values) / len(values) if values else None,
        "eos_correct": sum(bool(row["eos_correct"]) for row in values),
        "eos_accuracy": sum(bool(row["eos_correct"]) for row in values) / len(values) if values else None,
        "full_sequence_correct": sum(bool(row["full_sequence_correct"]) for row in values),
        "full_sequence_accuracy": sum(bool(row["full_sequence_correct"]) for row in values) / len(values) if values else None,
        "output_format_errors": sum(bool(row["output_format_error"]) for row in values),
        "paired_output_token": paired_metric(values, split, "output_token_correct"),
        "paired_full_sequence": paired_metric(values, split, "full_sequence_correct"),
    }
    return rows, metrics


def evaluation_records(base: Mapping[str, Sequence[Mapping[str, Any]]], train_rows: Sequence[Mapping[str, Any]], interface: str, split: str, layout: str) -> List[Dict[str, Any]]:
    if split == "train" and layout in TRAIN_LAYOUTS:
        return [row for row in train_rows if row["metadata"]["layout"] == layout]
    return exposure.expanded_eval_records(base, interface, split, layout)


def evaluate_checkpoint(checkpoint: Path, source_label: str, arm: str, interface: str, seed: int, base: Mapping[str, Sequence[Mapping[str, Any]]], train_rows: Sequence[Mapping[str, Any]], checkpoint_hash: str | None = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    model = load_model(checkpoint)
    arm_meta = {
        "arm": arm,
        "arm_id": f"{arm}_{interface}_seed{seed}",
        "interface": interface,
        "seed": seed,
        "checkpoint": rel(checkpoint),
        "checkpoint_sha256": checkpoint_hash or sha256(checkpoint),
        "source_label": source_label,
    }
    predictions: List[Dict[str, Any]] = []
    evaluations: List[Dict[str, Any]] = []
    for layout in EVAL_LAYOUTS:
        for split in ("train", "validation"):
            rows = evaluation_records(base, train_rows, interface, split, layout)
            current, metrics = predict_records(model, rows, arm_meta, layout, split)
            predictions.extend(current)
            evaluations.append({**arm_meta, "layout": layout, "split": split, "metrics": metrics})
    del model
    gc.collect()
    return predictions, evaluations, arm_meta


def cross_layout_analysis(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() == "true"

    grouped: Dict[Tuple[str, str, str, str, str, int, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        if str(row["layout"]) not in TRAIN_LAYOUTS:
            continue
        key = (str(row["source_label"]), str(row["interface"]), str(row["arm"]), str(row["seed"]), str(row["split"]), int(row["a"]), int(row["b"]), str(row["operation"]))
        grouped[key].append(row)
    records: List[Dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        by_layout = {str(row["layout"]): row for row in values}
        if set(by_layout) != set(TRAIN_LAYOUTS):
            continue
        records.append({
            "source_label": key[0], "interface": key[1], "arm": key[2], "seed": int(key[3]), "split": key[4], "a": key[5], "b": key[6], "operation": key[7],
            "all_layouts_output_correct": all(as_bool(by_layout[layout]["output_token_correct"]) for layout in TRAIN_LAYOUTS),
            "all_layouts_full_sequence_correct": all(as_bool(by_layout[layout]["full_sequence_correct"]) for layout in TRAIN_LAYOUTS),
            "layout_prediction_consistent": len({str(by_layout[layout]["predicted_output"]) for layout in TRAIN_LAYOUTS}) == 1,
            "predictions": {layout: str(by_layout[layout]["predicted_output"]) for layout in TRAIN_LAYOUTS},
            "correctness": {layout: bool(by_layout[layout]["output_token_correct"]) for layout in TRAIN_LAYOUTS},
        })
    aggregates: Dict[str, Any] = {}
    group_keys = sorted({(row["source_label"], row["arm"], row["interface"], row["seed"], row["split"]) for row in records})
    for source_label, arm, interface, seed, split in group_keys:
        values = [row for row in records if (row["source_label"], row["arm"], row["interface"], row["seed"], row["split"]) == (source_label, arm, interface, seed, split)]
        aggregates[f"{source_label}__{arm}__{interface}__seed{seed}__{split}"] = {
            "split": split,
            "rows": len(values),
            "all_layouts_output_correct": sum(bool(row["all_layouts_output_correct"]) for row in values),
            "all_layouts_output_rate": sum(bool(row["all_layouts_output_correct"]) for row in values) / len(values) if values else None,
            "layout_prediction_consistent": sum(bool(row["layout_prediction_consistent"]) for row in values),
            "layout_prediction_consistent_rate": sum(bool(row["layout_prediction_consistent"]) for row in values) / len(values) if values else None,
        }
    return {"rows": records, "aggregates": aggregates}


def aggregate_evaluations(evaluations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    keys = sorted({(str(row["source_label"]), str(row["arm"]), str(row["interface"]), str(row["layout"]), str(row["split"])) for row in evaluations})
    for source_label, arm, interface, layout, split in keys:
        values = [row for row in evaluations if (str(row["source_label"]), str(row["arm"]), str(row["interface"]), str(row["layout"]), str(row["split"])) == (source_label, arm, interface, layout, split)]
        metrics = [row["metrics"] for row in values]
        def mean(field: str) -> float | None:
            numbers = [float(metric[field]) for metric in metrics if metric[field] is not None]
            return statistics.mean(numbers) if numbers else None
        def paired_mean(field: str) -> float | None:
            numbers = [float(metric[field]["both_correct_rate"]) for metric in metrics if metric[field]["both_correct_rate"] is not None]
            return statistics.mean(numbers) if numbers else None
        result[f"{source_label}__{arm}__{interface}__{layout}__{split}"] = {
            "arms": len(values),
            "output_token_accuracy_mean": mean("output_token_accuracy"),
            "full_sequence_accuracy_mean": mean("full_sequence_accuracy"),
            "paired_output_rate_mean": paired_mean("paired_output_token"),
            "paired_full_sequence_rate_mean": paired_mean("paired_full_sequence"),
            "output_token_correct_total": sum(int(metric["output_token_correct"]) for metric in metrics),
            "output_token_rows_total": sum(int(metric["rows"]) for metric in metrics),
            "paired_output_correct_total": sum(int(metric["paired_output_token"]["both_correct"]) for metric in metrics),
            "paired_output_groups_total": sum(int(metric["paired_output_token"]["pair_groups"]) for metric in metrics),
        }
    return result


def verify_input_records(base: Mapping[str, Sequence[Mapping[str, Any]]], train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    train_pairs = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in train_rows}
    validation_pairs = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in validation_rows}
    errors: List[str] = []
    if train_pairs & validation_pairs:
        errors.append("train_validation_pair_overlap")
    if len(train_rows) != 240 or len(validation_rows) != 60:
        errors.append(f"expanded_record_counts:{len(train_rows)}:{len(validation_rows)}")
    if len({str(row["record_id"]) for row in train_rows}) != len(train_rows):
        errors.append("duplicate_train_record_ids")
    if len({str(row["record_id"]) for row in validation_rows}) != len(validation_rows):
        errors.append("duplicate_validation_record_ids")
    for row in train_rows + validation_rows:
        a = int(row["metadata"]["a"])
        b = int(row["metadata"]["b"])
        expected = a if row["metadata"]["operation"] == "SELECT_FIRST" else b
        if str(row["target"]) != str(expected):
            errors.append(f"label:{row['record_id']}")
        if row["metadata"]["layout"] not in TRAIN_LAYOUTS:
            errors.append(f"layout:{row['record_id']}")
    # Confirm the source records have the same disjoint pair split used by the
    # prior experiment and that this runner never reads a test file.
    source_train_pairs = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in base["train"]}
    source_validation_pairs = {(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in base["validation"]}
    if source_train_pairs & source_validation_pairs:
        errors.append("source_train_validation_pair_overlap")
    token_counts = Counter(len(item["_full"]) for item in prepared)
    return {
        "errors": errors,
        "pass": not errors,
        "source_train_records": len(base["train"]),
        "source_validation_records": len(base["validation"]),
        "expanded_train_records": len(train_rows),
        "expanded_validation_records": len(validation_rows),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "prepared_token_length_histogram": {str(key): value for key, value in sorted(token_counts.items())},
        "sealed_test_accessed": False,
    }


def build_report(summary: Mapping[str, Any]) -> str:
    arms = summary["arms"]
    evaluations = summary["evaluations"]
    aggregates = summary["aggregates"]
    lines = [
        "# Crucible Batch-Composition and Update-Ordering Control — Final Report",
        "",
        "CPU-only, 500-update, compute-matched comparison. The sealed test set was not accessed.",
        "",
        "## Experimental invariants",
        "",
        f"Schedule audit: **{'PASS' if summary['schedule_audit']['pass'] else 'FAIL'}**. Input audit: **{'PASS' if summary['input_audit']['pass'] else 'FAIL'}**.",
        f"Each arm used {EFFECTIVE_BATCH} records/update, {STEPS} updates and 6,000 record presentations; each expanded training record was presented exactly 25 times.",
        "",
        "## Training fit and gradient trace",
        "",
        "| Arm | Interface | Seed | Train exact | Mean adjacent gradient cosine | Mean unique pair groups/batch |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        lines.append(f"| {arm['arm']} | {arm['interface']} | {arm['seed']} | {arm['train_expanded_exact']:.2%} | {arm['mean_gradient_cosine_previous']:.4f} | {arm['mean_unique_pair_groups_per_batch']:.1f} |")
    lines += [
        "",
        "## Validation paired conditional correctness",
        "",
        "Values are output-token paired correctness; denominators are operand-pair groups, averaged across the two seeds and interfaces where shown.",
        "",
        "| Arm | Fixed-near | Early | Middle | Unseen-between | Unseen-early-context |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        cells: List[str] = []
        for layout in EVAL_LAYOUTS:
            values = [row for row in evaluations if row["source_label"] == "new_B_matrix" and row["arm"] == arm and row["layout"] == layout and row["split"] == "validation"]
            correct = sum(int(row["metrics"]["paired_output_token"]["both_correct"]) for row in values)
            groups = sum(int(row["metrics"]["paired_output_token"]["pair_groups"]) for row in values)
            cells.append(f"{correct}/{groups}" if groups else "n/a")
        lines.append(f"| {arm} | {' | '.join(cells)} |")
    lines += [
        "",
        "## Batch-composition and ordering contrasts",
        "",
        "The new arms use the same batch multiset within B0/B1 and within B2/B3. Therefore B0 versus B1 and B2 versus B3 isolate batch ordering within their respective compositions, subject to stochastic optimizer dynamics. Diverse versus same-pair contrasts test batch composition with the scheduled order held in the corresponding grouped or shuffled family.",
        "",
    ]
    for pair in (("B0_DIVERSE_SHUFFLED", "B1_DIVERSE_GROUPED"), ("B2_SAME_PAIR_SHUFFLED", "B3_SAME_PAIR_GROUPED")):
        left, right = pair
        lines.append(f"- **{left} versus {right}:**")
        for layout in EVAL_LAYOUTS:
            left_values = [row["metrics"]["paired_output_token"] for row in evaluations if row["source_label"] == "new_B_matrix" and row["arm"] == left and row["layout"] == layout and row["split"] == "validation"]
            right_values = [row["metrics"]["paired_output_token"] for row in evaluations if row["source_label"] == "new_B_matrix" and row["arm"] == right and row["layout"] == layout and row["split"] == "validation"]
            left_correct = sum(int(item["both_correct"]) for item in left_values)
            left_groups = sum(int(item["pair_groups"]) for item in left_values)
            right_correct = sum(int(item["both_correct"]) for item in right_values)
            right_groups = sum(int(item["pair_groups"]) for item in right_values)
            lines.append(f"  - `{layout}`: {left} {left_correct}/{left_groups}; {right} {right_correct}/{right_groups}.")
    lines += [
        "",
        "## Historical A0 reference",
        "",
        "The prior A0 checkpoints were evaluated read-only and are not part of the new training arms.",
        "",
        "| Interface | Seed | Fixed-near validation paired | Early validation paired | Middle validation paired |",
        "|---|---:|---:|---:|---:|",
    ]
    for interface in INTERFACES:
        for seed in SEEDS:
            cells = []
            for layout in TRAIN_LAYOUTS:
                values = [row for row in evaluations if row["source_label"] == "historical_A0_reference" and row["interface"] == interface and row["seed"] == seed and row["layout"] == layout and row["split"] == "validation"]
                metric = values[0]["metrics"]["paired_output_token"] if values else None
                cells.append(f"{metric['both_correct']}/{metric['pair_groups']}" if metric else "n/a")
            lines.append(f"| {interface} | {seed} | {' | '.join(cells)} |")
    lines += [
        "",
        "## Cross-layout consistency",
        "",
        "The full machine-readable cross-layout table is in `CROSS_LAYOUT.json`. It separates familiar operands in new layouts from unseen validation operands in trained layouts.",
        "",
        "## Interpretation boundary",
        "",
        "Gradient cosine is a diagnostic of update similarity, not a causal proof of gradient interference. High training fit with low validation transfer remains a memorization/transfer result. The experiment does not establish a position-independent representation or an architectural ceiling.",
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_BATCH_COMPOSITION_ORDER_PROTOCOL_20251025.md)",
        "- [Summary](SUMMARY.json)",
        "- [Predictions](PREDICTIONS.csv)",
        "- [Training traces](TRAINING_TRACE.jsonl)",
        "- [Cross-layout analysis](CROSS_LAYOUT.json)",
        "- [Schedule audit](SCHEDULE_AUDIT.json)",
        "- [Finalization](FINALIZATION.json)",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {
        "experiment": "crucible_batch_composition_order_20251025_v1",
        "status": "frozen_before_training",
        "protocol": rel(PROTOCOL),
        "arms": list(ARMS),
        "interfaces": list(INTERFACES),
        "seeds": list(SEEDS),
        "steps": STEPS,
        "effective_batch": EFFECTIVE_BATCH,
        "schedule_seed": SCHEDULE_SEED,
        "train_layouts": list(TRAIN_LAYOUTS),
        "eval_layouts": list(EVAL_LAYOUTS),
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "consistency_loss": False,
    })

    # The records and source split are loaded from the already-frozen A3 data.
    base_for_audit = exposure.base_records("canonical", 42)
    canonical_train, canonical_groups = exposure.expanded_records(base_for_audit, "canonical", "train")
    canonical_validation, _ = exposure.expanded_records(base_for_audit, "canonical", "validation")
    canonical_prepared = ladder.prepare(canonical_train)
    input_audit = verify_input_records(base_for_audit, canonical_train, canonical_validation, canonical_prepared)
    if not input_audit["pass"]:
        write_json(OUTPUT / "INPUT_AUDIT.json", input_audit)
        raise SystemExit(f"input audit failed: {input_audit['errors']}")

    schedules = build_schedules(canonical_groups)
    schedule_audit_result = schedule_audit(schedules, canonical_train, canonical_prepared)
    write_json(OUTPUT / "INPUT_AUDIT.json", input_audit)
    write_json(OUTPUT / "SCHEDULE_AUDIT.json", schedule_audit_result)
    write_json(OUTPUT / "BATCH_SCHEDULES.json", schedules)
    if not schedule_audit_result["pass"]:
        raise SystemExit("schedule audit failed")

    # Record source data and hashes before training begins.
    source_manifest: Dict[str, Any] = {"protocol": {"path": rel(PROTOCOL), "sha256": sha256(PROTOCOL)}, "source_runner": {"path": rel(SOURCE), "sha256": sha256(SOURCE)}, "reference_summary": {"path": rel(REFERENCE_ROOT / "SUMMARY.json"), "sha256": sha256(REFERENCE_ROOT / "SUMMARY.json")}}
    for interface in INTERFACES:
        base = exposure.base_records(interface, 42)
        train_rows, groups = exposure.expanded_records(base, interface, "train")
        validation_rows, _ = exposure.expanded_records(base, interface, "validation")
        write_jsonl(OUTPUT / "data" / interface / "train.jsonl", train_rows)
        write_jsonl(OUTPUT / "data" / interface / "validation.jsonl", validation_rows)
        write_json(OUTPUT / "data" / interface / "GROUPS.json", {"groups": groups})
        source_manifest[f"{interface}_train"] = {"rows": len(train_rows), "sha256": sha256(OUTPUT / "data" / interface / "train.jsonl")}
        source_manifest[f"{interface}_validation"] = {"rows": len(validation_rows), "sha256": sha256(OUTPUT / "data" / interface / "validation.jsonl")}
    write_json(OUTPUT / "SOURCE_MANIFEST.json", source_manifest)

    arms: List[Dict[str, Any]] = []
    curves: Dict[str, Any] = {}
    predictions: List[Dict[str, Any]] = []
    evaluations: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []

    for arm in ARMS:
        for interface in INTERFACES:
            base = exposure.base_records(interface, 42)
            train_rows, groups = exposure.expanded_records(base, interface, "train")
            schedule = schedules[arm]
            for seed in SEEDS:
                arm_dir = OUTPUT / "arm_outputs" / arm / interface / f"seed{seed}"
                write_json(OUTPUT / "arm_outputs" / arm / interface / f"seed{seed}" / "SCHEDULE_REFERENCE.json", {"arm": arm, "schedule_digest": stable_digest(schedule), "batch_count": len(schedule)})
                result, curve, arm_traces = train_model(arm, interface, seed, train_rows, schedule, arm_dir)
                arms.append(result)
                curves[result["arm_id"]] = curve
                traces.extend(arm_traces)
                checkpoint = ROOT / result["checkpoint"]
                current_predictions, current_evaluations, _ = evaluate_checkpoint(checkpoint, "new_B_matrix", arm, interface, seed, base, train_rows, result["checkpoint_sha256"])
                predictions.extend(current_predictions)
                evaluations.extend(current_evaluations)
                print(json.dumps({"event": "complete", "arm_id": result["arm_id"], "train_exact": result["train_expanded_exact"], "mean_gradient_cosine": result["mean_gradient_cosine_previous"]}, sort_keys=True), flush=True)

    # Historical A0 checkpoints are only loaded and evaluated; they are never
    # used as initializations for the new training arms.
    for interface in INTERFACES:
        base = exposure.base_records(interface, 42)
        train_rows, _ = exposure.expanded_records(base, interface, "train")
        for seed in SEEDS:
            checkpoint = REFERENCE_ROOT / "arm_outputs" / "A0_SHUFFLED_EXPOSURE" / interface / f"seed{seed}" / "checkpoint.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            current_predictions, current_evaluations, _ = evaluate_checkpoint(checkpoint, "historical_A0_reference", "A0_SHUFFLED_EXPOSURE", interface, seed, base, train_rows)
            predictions.extend(current_predictions)
            evaluations.extend(current_evaluations)

    with (OUTPUT / "PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(predictions[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    write_json(OUTPUT / "TRAINING_CURVES.json", curves)
    write_jsonl(OUTPUT / "TRAINING_TRACE.jsonl", traces)
    cross_layout = cross_layout_analysis(predictions)
    write_json(OUTPUT / "CROSS_LAYOUT.json", cross_layout)
    aggregates = aggregate_evaluations(evaluations)
    summary = {
        "experiment": "crucible_batch_composition_order_20251025_v1",
        "status": "completed",
        "arms": arms,
        "evaluations": evaluations,
        "aggregates": aggregates,
        "cross_layout": {"rows": len(cross_layout["rows"]), "aggregate_groups": len(cross_layout["aggregates"])},
        "input_audit": input_audit,
        "schedule_audit": schedule_audit_result,
        "prediction_rows": len(predictions),
        "training_trace_rows": len(traces),
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "retrained": True,
        "historical_reference_retrained": False,
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
    (OUTPUT / "REPORT_FINAL.md").write_text(build_report(summary), encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix != ".pt":
            hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = {
        "status": "completed",
        "verification_pass": bool(input_audit["pass"] and schedule_audit_result["pass"]),
        "sealed_test_accessed": False,
        "architecture_changed": False,
        "retrained": True,
        "historical_reference_retrained": False,
        "new_training_arms": len(arms),
        "evaluations": len(evaluations),
        "prediction_rows": len(predictions),
        "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"),
        "prediction_sha256": sha256(OUTPUT / "PREDICTIONS.csv"),
    }
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "completed", "output": rel(OUTPUT), "new_training_arms": len(arms), "evaluations": len(evaluations), "prediction_rows": len(predictions), "verification_pass": finalization["verification_pass"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
