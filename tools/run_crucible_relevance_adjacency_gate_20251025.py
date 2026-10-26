#!/usr/bin/env python3
"""Frozen-checkpoint relevance/adjacency and attention-pathway gate."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_relevance_adjacency_gate_20251025_v2"
POSITION_ROOT = ROOT / "experiments" / "crucible_operation_position_matrix_20251025_v1"
S3_ROOT = ROOT / "experiments" / "crucible_role_selection_competence_20251025_v5"
A1_ROOT = ROOT / "experiments" / "crucible_operation_marker_interface_20251025_v3"
ROLE_SOURCE = ROOT / "tools" / "run_crucible_role_selection_competence_20251025.py"
POSITION_SOURCE = ROOT / "tools" / "run_crucible_operation_position_matrix_20251025.py"
INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
NATIVE_CONDITIONS = ("A0_WORD_ORIGINAL", "A1_MARKER_ORIGINAL", "A2_WORD_FIXED", "A3_MARKER_FIXED")
VARIANTS = ("A_ADJACENT", "B_EARLY_PADDED", "C_NEAR_PADDED", "D_COMPETING", "E_EXPLICIT_AUTHORITY")
ATTENTION_TOLERANCE = 1.0e-5


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


role = import_module("relevance_role_runner", ROLE_SOURCE)
position = import_module("relevance_position_runner", POSITION_SOURCE)
ladder = role.ladder
stage_a = role.stage_a
torch = stage_a.torch


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(row), sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def marker_field(operation: str) -> str:
    return ("F" if operation == "SELECT_FIRST" else "S") + ("_" * (len(operation.encode("utf-8")) - 1))


def source_arm_dir(condition: str, interface: str, seed: int) -> Path:
    if condition == "A0_WORD_ORIGINAL":
        return S3_ROOT / "arm_outputs" / "S3_JOINT_OPERATION_CONDITIONED" / interface / f"seed{seed}"
    if condition == "A1_MARKER_ORIGINAL":
        return A1_ROOT / "arm_outputs" / interface / f"seed{seed}"
    return POSITION_ROOT / "arm_outputs" / condition / interface / f"seed{seed}"


def records_for(condition: str, interface: str, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    directory = source_arm_dir(condition, interface, seed)
    return {"train": load_jsonl(directory / "train.jsonl"), "validation": load_jsonl(directory / "validation.jsonl")}


def checkpoint_for(condition: str, interface: str, seed: int) -> Path:
    return source_arm_dir(condition, interface, seed) / "checkpoint.pt"


def load_dense(path: Path) -> Any:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = stage_a.DenseLM()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def operation_field(operation: str, marker: bool) -> str:
    return marker_field(operation) if marker else operation


def operands_text(interface: str, a: int, b: int) -> str:
    return f"a={a}; b={b}." if interface == "canonical" else f"a_digit={a}; b_digit={b}."


def variant_input(interface: str, operation: str, a: int, b: int, variant: str) -> str:
    field = operation_field(operation, marker=True)
    operands = operands_text(interface, a, b)
    opposite = operation_field("SELECT_SECOND" if operation == "SELECT_FIRST" else "SELECT_FIRST", marker=True)
    if variant == "A_ADJACENT":
        return f"Select an operand. {operands} Return digit only. operation={field};"
    if variant == "B_EARLY_PADDED":
        return f"Select an operand. operation={field}; context=neutral_17; {operands} Return digit only."
    if variant == "C_NEAR_PADDED":
        return f"Select an operand. context=neutral_17; {operands} Return digit only. operation={field};"
    if variant == "D_COMPETING":
        return f"Select an operand. decoy_operation={opposite}; {operands} Return digit only. operation={field};"
    if variant == "E_EXPLICIT_AUTHORITY":
        return f"Select an operand. authority=PRIMARY; operation={field}; context=neutral_17; {operands} Return digit only. use=PRIMARY;"
    raise ValueError(variant)


def variant_records(base: Mapping[str, Sequence[Mapping[str, Any]]], interface: str, variant: str) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for split, records in base.items():
        result[split] = []
        for row in records:
            cloned = dict(row)
            a = int(row["metadata"]["a"])
            b = int(row["metadata"]["b"])
            operation = str(row["metadata"]["operation"])
            cloned["record_id"] = hashlib.sha256(f"relevance|{interface}|{variant}|{split}|{a}|{b}|{operation}".encode()).hexdigest()[:20]
            cloned["input"] = variant_input(interface, operation, a, b, variant)
            cloned["condition"] = variant
            cloned["metadata"] = {**row["metadata"], "variant": variant}
            result[split].append(cloned)
    return result


def prepare_with_spans(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared = ladder.prepare(records)
    op_pattern = re.compile(rb"(?<![A-Za-z_])operation=([^;]+);")
    decoy_pattern = re.compile(rb"decoy_operation=([^;]+);")
    a_pattern = re.compile(rb"(?<![A-Za-z_])a(?:_digit)?=([0-9])")
    b_pattern = re.compile(rb"(?<![A-Za-z_])b(?:_digit)?=([0-9])")
    pointer_pattern = re.compile(rb"(?:authority|use)=PRIMARY")
    for item in prepared:
        raw = str(item["input"]).encode("utf-8")
        op_match = op_pattern.search(raw)
        a_match = a_pattern.search(raw)
        b_match = b_pattern.search(raw)
        if op_match is None or a_match is None or b_match is None:
            raise RuntimeError(f"missing span in {item['record_id']}")
        def span(match: Any, group: int = 0) -> List[int]:
            return list(range(1 + match.start(group), 1 + match.end(group)))
        item["_output_position"] = int(item["_prefix_len"]) - 1
        item["_spans"] = {
            "operation": span(op_match, 1),
            "a": span(a_match, 1),
            "b": span(b_match, 1),
            "decoy": span(decoy_pattern.search(raw), 1) if decoy_pattern.search(raw) else [],
            "pointer": [index + 1 for match in pointer_pattern.finditer(raw) for index in range(match.start(), match.end())],
        }
        if not 0 <= item["_output_position"] < len(item["_full"]):
            raise RuntimeError(f"invalid answer position in {item['record_id']}")
    return prepared


def shadow_sdpa(capture: List[torch.Tensor], native: Any):
    def function(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Any = None, dropout_p: float = 0.0, is_causal: bool = False, scale: Any = None, enable_gqa: bool = False) -> torch.Tensor:
        if dropout_p or is_causal or enable_gqa:
            raise RuntimeError("unexpected SDPA option in this model")
        factor = float(scale) if scale is not None else q.size(-1) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * factor
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        probabilities = torch.softmax(scores, dim=-1)
        capture.append(probabilities.detach().cpu())
        # Preserve the exact native output path. The shadow probabilities are
        # diagnostic only and never replace native SDPA's returned tensor.
        return native(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    return function


def run_batch(model: Any, records: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> Tuple[torch.Tensor, List[torch.Tensor], float]:
    prepared = prepare_with_spans(records)
    selected = [prepared[index] for index in indices]
    batch = stage_a.batchify(prepared, indices)
    model.eval()
    with torch.inference_mode():
        baseline = stage_a.forward_model(model, batch)
    captured: List[torch.Tensor] = []
    functional = torch.nn.functional
    original = functional.scaled_dot_product_attention
    functional.scaled_dot_product_attention = shadow_sdpa(captured, original)
    try:
        with torch.inference_mode():
            instrumented = stage_a.forward_model(model, batch)
    finally:
        functional.scaled_dot_product_attention = original
    max_abs = float((baseline - instrumented).abs().max().item())
    if len(captured) != len(model.layers):
        raise RuntimeError(f"expected {len(model.layers)} attention calls, got {len(captured)}")
    return instrumented, captured, max_abs


def decode(token_id: int) -> Tuple[str, bool]:
    value = int(token_id) - int(stage_a.RAW_BYTE_START)
    if 48 <= value <= 57:
        return chr(value), False
    return f"<token:{token_id}>", True


def pair_metrics(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    selected = [row for row in rows if row["split"] == split and int(row["a"]) != int(row["b"])]
    by_pair: Dict[Tuple[int, int], Dict[str, Mapping[str, Any]]] = {}
    for row in selected:
        by_pair.setdefault((int(row["a"]), int(row["b"])), {})[str(row["operation"])] = row
    complete = [value for value in by_pair.values() if set(value) == {"SELECT_FIRST", "SELECT_SECOND"}]
    both = sum(value["SELECT_FIRST"]["correct"] and value["SELECT_SECOND"]["correct"] for value in complete)
    switched = sum(value["SELECT_FIRST"]["predicted_output"] != value["SELECT_SECOND"]["predicted_output"] for value in complete)
    return {"pair_groups": len(complete), "both_correct": both, "both_correct_rate": both / len(complete) if complete else None, "predictions_switched": switched, "split": split}


def evaluate(model: Any, records: Mapping[str, Sequence[Mapping[str, Any]]], arm: Mapping[str, Any], eval_condition: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []
    attention_rows: List[Dict[str, Any]] = []
    max_abs = 0.0
    for split in ("train", "validation"):
        values = list(records[split])
        for start in range(0, len(values), int(stage_a.BATCH_SIZE)):
            indices = list(range(start, min(start + int(stage_a.BATCH_SIZE), len(values))))
            logits, attention, diff = run_batch(model, values, indices)
            max_abs = max(max_abs, diff)
            prepared = prepare_with_spans(values)
            batch = stage_a.batchify(prepared, indices)
            for local, index in enumerate(indices):
                item = prepared[index]
                output_position = int(item["_output_position"])
                target_id = int(batch["y"][local, output_position].item())
                predicted_id = int(logits[local, output_position].argmax().item())
                predicted, format_error = decode(predicted_id)
                row = {
                    **arm,
                    "eval_condition": eval_condition,
                    "record_id": item["record_id"],
                    "split": split,
                    "a": int(item["metadata"]["a"]),
                    "b": int(item["metadata"]["b"]),
                    "operation": item["metadata"]["operation"],
                    "expected_output": str(item["target"]),
                    "predicted_output": predicted,
                    "correct": predicted_id == target_id,
                    "output_format_error": format_error,
                }
                all_rows.append(row)
                for layer_index, layer_weights in enumerate(attention):
                    query = layer_weights[local, :, output_position, :].float().mean(dim=0)
                    spans = item["_spans"]
                    known = set(spans["operation"] + spans["a"] + spans["b"] + spans["decoy"] + spans["pointer"])
                    def mass(name: str) -> float:
                        positions = spans[name]
                        return float(query[positions].sum().item()) if positions else 0.0
                    attention_rows.append({
                        **arm,
                        "eval_condition": eval_condition,
                        "record_id": item["record_id"],
                        "split": split,
                        "layer": layer_index,
                        "operation_mass": mass("operation"),
                        "operand_a_mass": mass("a"),
                        "operand_b_mass": mass("b"),
                        "decoy_mass": mass("decoy"),
                        "pointer_mass": mass("pointer"),
                        "other_mass": float(query[[index for index in range(query.numel()) if index not in known]].sum().item()) if len(known) < query.numel() else 0.0,
                        "operation_to_answer_distance": output_position - max(spans["operation"]),
                        "input_tokens": len(item["_full"]),
                    })
    by_split = {}
    for split in ("train", "validation"):
        values = [row for row in all_rows if row["split"] == split]
        by_split[split] = {
            "rows": len(values),
            "accuracy": sum(row["correct"] for row in values) / len(values) if values else None,
            "format_errors": sum(row["output_format_error"] for row in values),
            "paired_unequal": pair_metrics(all_rows, split),
        }
    attention_summary: Dict[str, Any] = {}
    for split in ("train", "validation"):
        attention_summary[split] = {}
        for layer in range(len(model.layers)):
            values = [row for row in attention_rows if row["split"] == split and row["layer"] == layer]
            attention_summary[split][f"layer_{layer}"] = {
                field: sum(float(row[field]) for row in values) / len(values) if values else None
                for field in ("operation_mass", "operand_a_mass", "operand_b_mass", "decoy_mass", "pointer_mass", "other_mass", "operation_to_answer_distance", "input_tokens")
            }
    return all_rows, {"by_split": by_split, "attention_max_abs_logit_diff": max_abs, "instrumentation_pass": max_abs <= ATTENTION_TOLERANCE}, attention_summary, attention_rows


def native_data(condition: str, interface: str, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    return records_for(condition, interface, seed)


def build_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Crucible Relevance and Adjacency Gate — Final Report",
        "",
        "Frozen-checkpoint CPU evaluation with native SDPA output and shadow Q/K attention summaries. No retraining, architecture change or sealed-test access.",
        "",
        "## Native checkpoint comparison",
        "",
        "| Condition | Interface | Seed | Validation | Held-out paired | Instrumentation max abs diff |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in summary["native"]:
        valid = result["metrics"]["by_split"]["validation"]
        lines.append(f"| {result['condition']} | {result['interface']} | {result['seed']} | {valid['accuracy']:.2%} | {valid['paired_unequal']['both_correct']}/{valid['paired_unequal']['pair_groups']} | {result['metrics']['attention_max_abs_logit_diff']:.2e} |")
    lines += [
        "",
        "## A3 frozen prompt variants",
        "",
        "| Variant | Mean validation accuracy | Mean held-out paired rate | Arms |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        values = [result for result in summary["variants"] if result["eval_condition"] == variant]
        accuracies = [result["metrics"]["by_split"]["validation"]["accuracy"] for result in values]
        paired = [result["metrics"]["by_split"]["validation"]["paired_unequal"]["both_correct_rate"] for result in values]
        lines.append(f"| {variant} | {sum(accuracies)/len(accuracies):.2%} | {sum(paired)/len(paired):.2%} | {len(values)} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Native SDPA remains on the output path and the shadow Q/K probabilities are diagnostic only. Attention mass is not proof of a unique causal circuit. The frozen A3 variants measure robustness under prompt distribution shift, not relearning of each variant.",
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_RELEVANCE_ADJACENCY_GATE_PROTOCOL_20251025_V2.md)",
        "- [Summary](SUMMARY.json)",
        "- [Predictions](PREDICTIONS.csv)",
        "- [Attention summaries](ATTENTION_SUMMARIES.json)",
        "- [Attention traces](ATTENTION_TRACES.jsonl)",
        "- [Finalization](FINALIZATION.json)",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {"experiment": "crucible_relevance_adjacency_gate_20251025_v2", "status": "frozen_before_evaluation", "native_conditions": list(NATIVE_CONDITIONS), "variants": list(VARIANTS), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "retrained": False, "sealed_test_accessed": False, "architecture_changed": False, "attention_tolerance": ATTENTION_TOLERANCE, "attention_method": "native_sdpa_output_with_shadow_qk_probabilities"}
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", protocol)
    native_results: List[Dict[str, Any]] = []
    variant_results: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []
    all_attention: List[Dict[str, Any]] = []
    for condition in NATIVE_CONDITIONS:
        for interface in INTERFACES:
            for seed in SEEDS:
                arm_id = f"{condition}_{interface}_seed{seed}"
                checkpoint = checkpoint_for(condition, interface, seed)
                model = load_dense(checkpoint)
                records = native_data(condition, interface, seed)
                rows, metrics, attention, traces = evaluate(model, records, {"arm_id": arm_id, "condition": condition, "interface": interface, "seed": seed, "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint)}, condition)
                native_results.append({"arm_id": arm_id, "condition": condition, "interface": interface, "seed": seed, "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint), "metrics": metrics, "attention": attention})
                all_predictions.extend(rows)
                all_attention.extend(traces)
                write_json(OUTPUT / "native" / f"{arm_id}.json", {"metrics": metrics, "attention": attention})
                del model
                gc.collect()
    for interface in INTERFACES:
        for seed in SEEDS:
            arm_id = f"A3_MARKER_FIXED_{interface}_seed{seed}"
            checkpoint = checkpoint_for("A3_MARKER_FIXED", interface, seed)
            model = load_dense(checkpoint)
            base = native_data("A3_MARKER_FIXED", interface, seed)
            for variant in VARIANTS:
                records = base if variant == "A_ADJACENT" else variant_records(base, interface, variant)
                rows, metrics, attention, traces = evaluate(model, records, {"arm_id": arm_id, "condition": "A3_MARKER_FIXED", "interface": interface, "seed": seed, "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint)}, variant)
                variant_results.append({"arm_id": arm_id, "eval_condition": variant, "interface": interface, "seed": seed, "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint), "metrics": metrics, "attention": attention})
                all_predictions.extend(rows)
                all_attention.extend(traces)
                write_json(OUTPUT / "variants" / f"{arm_id}_{variant}.json", {"metrics": metrics, "attention": attention})
            del model
            gc.collect()
    # Re-run a compact attention trace pass over the metrics already written by
    # evaluate by deriving trace rows from the per-arm JSONs is intentionally
    # avoided; the full record-level rows are retained below by a second,
    # deterministic summary writer from the prediction table.
    write_json(OUTPUT / "ATTENTION_SUMMARIES.json", {"native": {result["arm_id"]: result["attention"] for result in native_results}, "variants": {f"{result['arm_id']}__{result['eval_condition']}": result["attention"] for result in variant_results}})
    with (OUTPUT / "PREDICTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_predictions[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_predictions)
    write_jsonl(OUTPUT / "ATTENTION_TRACES.jsonl", all_attention)
    summary = {"experiment": "crucible_relevance_adjacency_gate_20251025_v2", "status": "completed", "native": native_results, "variants": variant_results, "prediction_rows": len(all_predictions), "attention_trace_rows": len(all_attention), "retrained": False, "sealed_test_accessed": False, "architecture_changed": False, "attention_method": "native_sdpa_output_with_shadow_qk_probabilities"}
    write_json(OUTPUT / "SUMMARY.json", summary)
    (OUTPUT / "REPORT_FINAL.md").write_text(build_report(summary), encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"}:
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = {"status": "completed", "verification_pass": all(result["metrics"]["instrumentation_pass"] for result in native_results + variant_results), "retrained": False, "sealed_test_accessed": False, "architecture_changed": False, "native_arms": len(native_results), "variant_arms": len(variant_results), "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"), "prediction_sha256": sha256(OUTPUT / "PREDICTIONS.csv")}
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    print(json.dumps({"status": "completed", "output": str(OUTPUT.relative_to(ROOT)), "native_arms": len(native_results), "variant_arms": len(variant_results), "prediction_rows": len(all_predictions), "verification_pass": finalization["verification_pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
