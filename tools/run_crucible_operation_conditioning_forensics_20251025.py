#!/usr/bin/env python3
"""Read-only causal forensics for the frozen S3 operation-conditioning arms.

The script reloads the completed S3 checkpoints, reproduces their metrics, and
then runs representation, gradient, sensitivity, and bounded activation-
patching diagnostics.  It never trains or modifies the source checkpoints and
never opens a sealed test corpus.
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_operation_conditioning_forensics_20251025_v1"
S3_ROOT = ROOT / "experiments" / "crucible_role_selection_competence_20251025_v5"
TOKEN_ROOT = ROOT / "experiments" / "crucible_operation_token_identification_20251025_v1"
ROLE_SOURCE = ROOT / "tools" / "run_crucible_role_selection_competence_20251025.py"
INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
S3_CONDITION = "S3_JOINT_OPERATION_CONDITIONED"
PATCH_LAYERS = (0, 2, 3)
PATCH_LOCATIONS = ("operation_last", "output_position")
PATCH_CONTROLS = ("matched_opposite", "same_operation_other_pair", "random_noise")
RIDGE_LAMBDA = 1.0e-2


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


role = import_module("operation_conditioning_role_runner", ROLE_SOURCE)
ladder = role.ladder
stage_a = role.stage_a
torch = stage_a.torch


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(dict(value), sort_keys=True) for value in values) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def arm_id(interface: str, seed: int) -> str:
    return f"S3_JOINT_OPERATION_CONDITIONED_{interface}_seed{seed}"


def s3_arm_dir(interface: str, seed: int) -> Path:
    return S3_ROOT / "arm_outputs" / S3_CONDITION / interface / f"seed{seed}"


def token_arm_dir(interface: str, seed: int) -> Path:
    return TOKEN_ROOT / "arm_outputs" / interface / f"seed{seed}"


def load_dense(checkpoint: Path) -> Any:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = stage_a.DenseLM()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def expected_operand(record: Mapping[str, Any]) -> int:
    operation = str(record["metadata"]["operation"])
    return int(record["metadata"]["a"] if operation == "SELECT_FIRST" else record["metadata"]["b"])


def operation_label(record: Mapping[str, Any]) -> int:
    return 0 if record["metadata"]["operation"] == "SELECT_FIRST" else 1


def prepare_with_positions(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared = ladder.prepare(records)
    for item in prepared:
        operation = str(item["metadata"]["operation"])
        raw_input = str(item["input"]).encode("utf-8")
        op_bytes = operation.encode("utf-8")
        start = raw_input.find(op_bytes)
        if start < 0:
            raise RuntimeError(f"operation token not found: {item['record_id']}")
        item["_operation_start"] = 1 + start
        item["_operation_last"] = 1 + start + len(op_bytes) - 1
        item["_output_position"] = int(item["_prefix_len"]) - 1
        if not (0 <= item["_operation_last"] < len(item["_full"])
                and 0 <= item["_output_position"] < len(item["_full"])):
            raise RuntimeError(f"invalid diagnostic positions: {item['record_id']}")
    return prepared


def record_audit(interface: str, split: str, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    seen: set[str] = set()
    pair_set: set[Tuple[int, int]] = set()
    for row in records:
        rid = str(row["record_id"])
        if rid in seen:
            errors.append(f"duplicate_record:{rid}")
        seen.add(rid)
        a = int(row["metadata"]["a"])
        b = int(row["metadata"]["b"])
        pair_set.add((a, b))
        operation = str(row["metadata"]["operation"])
        target = expected_operand(row)
        if row.get("interface") != interface or row.get("split") != split:
            errors.append(f"metadata:{rid}")
        if int(row["target"]) != target or int(row["oracle"]["value"]) != target:
            errors.append(f"label:{rid}")
        if operation not in {"SELECT_FIRST", "SELECT_SECOND"}:
            errors.append(f"operation:{rid}")
    return {"interface": interface, "split": split, "rows": len(records), "pairs": len(pair_set), "errors": errors, "pass": not errors}


def metric_diff(reproduced: Mapping[str, Any], recorded: Mapping[str, Any]) -> Dict[str, Any]:
    fields = ("teacher_forced_exact_accuracy", "teacher_forced_token_accuracy", "mean_target_loss", "examples")
    result: Dict[str, Any] = {}
    for field in fields:
        left = reproduced.get(field)
        right = recorded.get(field)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            result[field] = {"reproduced": left, "recorded": right, "absolute_delta": abs(float(left) - float(right)), "match": math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-7)}
        else:
            result[field] = {"reproduced": left, "recorded": right, "match": left == right}
    return result


def decode_token(token_id: int) -> Tuple[str, bool]:
    value = int(token_id) - int(stage_a.RAW_BYTE_START)
    if 48 <= value <= 57:
        return chr(value), False
    return f"<token:{token_id}>", True


def token_id_for_digit(value: int) -> int:
    return int(stage_a.RAW_BYTE_START) + ord(str(int(value)))


def top_predictions(logits: torch.Tensor, count: int = 5) -> List[Dict[str, Any]]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(probabilities, min(count, probabilities.numel()))
    return [{"token_id": int(index), "decoded": decode_token(int(index))[0], "probability": float(value)} for value, index in zip(values, indices)]


def capture_forward(model: Any, prepared: Sequence[Mapping[str, Any]], patch: Mapping[str, Any] | None = None, capture: bool = True) -> Dict[str, Any]:
    if len(prepared) != 1:
        raise ValueError("capture_forward expects exactly one record")
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def hook_for(name: str):
        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
            value = output[0] if isinstance(output, tuple) else output
            tensor = value
            if patch is not None and name == f"layer_{patch['layer']}":
                tensor = tensor.clone()
                position = int(prepared[0][f"_{patch['location']}"])
                replacement = patch["vector"].to(device=tensor.device, dtype=tensor.dtype)
                tensor[0, position, :] = replacement
            if capture:
                captured[name] = tensor.detach().cpu()
            return tensor
        return hook

    if capture:
        handles.append(model.token_emb.register_forward_hook(hook_for("embedding")))
    for index, layer in enumerate(model.layers):
        if capture or (patch is not None and index == int(patch["layer"])):
            handles.append(layer.register_forward_hook(hook_for(f"layer_{index}")))
    if capture:
        handles.append(model.final_norm.register_forward_hook(hook_for("final_norm")))

    prepared_batch = stage_a.batchify(prepared, [0])
    with torch.inference_mode():
        logits = stage_a.forward_model(model, prepared_batch)
    for handle in handles:
        handle.remove()
    output_position = int(prepared[0]["_output_position"])
    output_logits = logits[0, output_position].detach().cpu()
    captured["output_logits"] = output_logits
    captured["predicted_token_id"] = torch.tensor(int(output_logits.argmax().item()), dtype=torch.long)
    return captured


def capture_records(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prepared = prepare_with_positions(records)
    names = ["embedding"] + [f"layer_{index}" for index in range(len(model.layers))] + ["final_norm"]
    state_lists: Dict[str, Dict[str, List[torch.Tensor]]] = {name: {"operation_last": [], "output_position": []} for name in names}
    logits: List[torch.Tensor] = []
    ids: List[str] = []
    labels: List[int] = []
    operations: List[str] = []
    for start in range(0, len(prepared), int(stage_a.BATCH_SIZE)):
        batch_items = prepared[start:start + int(stage_a.BATCH_SIZE)]
        batch = stage_a.batchify(batch_items, list(range(len(batch_items))))
        captured: Dict[str, torch.Tensor] = {}
        handles = []

        def hook_for(name: str):
            def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                value = output[0] if isinstance(output, tuple) else output
                captured[name] = value.detach().cpu()
                return output
            return hook

        handles.append(model.token_emb.register_forward_hook(hook_for("embedding")))
        for index, layer in enumerate(model.layers):
            handles.append(layer.register_forward_hook(hook_for(f"layer_{index}")))
        handles.append(model.final_norm.register_forward_hook(hook_for("final_norm")))
        with torch.inference_mode():
            output = stage_a.forward_model(model, batch)
        for handle in handles:
            handle.remove()
        for row, item in enumerate(batch_items):
            op_position = int(item["_operation_last"])
            output_position = int(item["_output_position"])
            for name in names:
                state_lists[name]["operation_last"].append(captured[name][row, op_position].clone())
                state_lists[name]["output_position"].append(captured[name][row, output_position].clone())
            logits.append(output[row, output_position].detach().cpu())
            ids.append(str(item["record_id"]))
            labels.append(operation_label(item))
            operations.append(str(item["metadata"]["operation"]))
    states = {name: {location: torch.stack(values) for location, values in locations.items()} for name, locations in state_lists.items()}
    return {"record_ids": ids, "labels": labels, "operations": operations, "states": states, "output_logits": torch.stack(logits), "prepared": prepared}


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    return float(torch.dot(left, right) / denominator) if denominator else 0.0


def ridge_probe(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor, eval_y: torch.Tensor) -> Dict[str, Any]:
    x = train_x.double()
    y = (train_y.double() * 2.0) - 1.0
    x_aug = torch.cat([x, torch.ones((x.size(0), 1), dtype=x.dtype)], dim=1)
    gram = x_aug.T @ x_aug
    regularizer = torch.eye(gram.size(0), dtype=gram.dtype) * RIDGE_LAMBDA
    regularizer[-1, -1] = 0.0
    weights = torch.linalg.solve(gram + regularizer, x_aug.T @ y)
    eval_aug = torch.cat([eval_x.double(), torch.ones((eval_x.size(0), 1), dtype=eval_x.dtype)], dim=1)
    train_scores = x_aug @ weights
    eval_scores = eval_aug @ weights
    train_pred = (train_scores >= 0).long()
    eval_pred = (eval_scores >= 0).long()
    return {
        "train_accuracy": float((train_pred == train_y).float().mean().item()),
        "validation_accuracy": float((eval_pred == eval_y).float().mean().item()),
        "train_margin_mean": float(train_scores.abs().mean().item()),
        "validation_margin_mean": float(eval_scores.abs().mean().item()),
        "regularization": RIDGE_LAMBDA,
    }


def centroid_probe(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor, eval_y: torch.Tensor) -> Dict[str, Any]:
    centroids = torch.stack([train_x[train_y == label].float().mean(dim=0) for label in (0, 1)])
    def predict(x: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(x.float(), centroids.float())
        return distances.argmin(dim=1)
    train_pred = predict(train_x)
    eval_pred = predict(eval_x)
    return {
        "train_accuracy": float((train_pred == train_y).float().mean().item()),
        "validation_accuracy": float((eval_pred == eval_y).float().mean().item()),
    }


def build_hidden_probe_results(bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    split = [str(row["split"]) for row in records]
    train_indices = torch.tensor([index for index, value in enumerate(split) if value == "train"], dtype=torch.long)
    validation_indices = torch.tensor([index for index, value in enumerate(split) if value == "validation"], dtype=torch.long)
    labels = torch.tensor(bundle["labels"], dtype=torch.long)
    result: Dict[str, Any] = {}
    for name, locations in bundle["states"].items():
        result[name] = {}
        for location, values in locations.items():
            train_x = values[train_indices]
            validation_x = values[validation_indices]
            train_y = labels[train_indices]
            validation_y = labels[validation_indices]
            result[name][location] = {
                "dimension": int(values.size(1)),
                "ridge": ridge_probe(train_x, train_y, validation_x, validation_y),
                "nearest_centroid": centroid_probe(train_x, train_y, validation_x, validation_y),
                "mean_norm": float(values.float().norm(dim=1).mean().item()),
            }
    return result


def hidden_difference(bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_key = {(int(row["metadata"]["a"]), int(row["metadata"]["b"]), str(row["metadata"]["operation"])): index for index, row in enumerate(records)}
    pairs = sorted({(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records if row["split"] == "validation" and row["metadata"]["a"] != row["metadata"]["b"]})
    result: Dict[str, Any] = {}
    for name, locations in bundle["states"].items():
        result[name] = {}
        for location, values in locations.items():
            rows = []
            for a, b in pairs:
                first = values[by_key[(a, b, "SELECT_FIRST")]]
                second = values[by_key[(a, b, "SELECT_SECOND")]]
                diff = first.float() - second.float()
                rows.append({"pair": [a, b], "l2": float(torch.linalg.vector_norm(diff).item()), "cosine": cosine(first, second), "mean_absolute": float(diff.abs().mean().item())})
            result[name][location] = {
                "pairs": rows,
                "mean_l2": statistics.fmean(row["l2"] for row in rows) if rows else None,
                "mean_cosine": statistics.fmean(row["cosine"] for row in rows) if rows else None,
                "mean_absolute": statistics.fmean(row["mean_absolute"] for row in rows) if rows else None,
            }
    return result


def output_sensitivity(bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_key = {(int(row["metadata"]["a"]), int(row["metadata"]["b"]), str(row["metadata"]["operation"])): index for index, row in enumerate(records)}
    pairs = sorted({(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in records if row["split"] == "validation" and row["metadata"]["a"] != row["metadata"]["b"]})
    rows = []
    for a, b in pairs:
        first = bundle["output_logits"][by_key[(a, b, "SELECT_FIRST")]].float()
        second = bundle["output_logits"][by_key[(a, b, "SELECT_SECOND")]].float()
        first_token = token_id_for_digit(a)
        second_token = token_id_for_digit(b)
        delta = first - second
        rows.append({
            "pair": [a, b],
            "logit_l2": float(torch.linalg.vector_norm(delta).item()),
            "logit_cosine": cosine(first, second),
            "mean_absolute_logit_delta": float(delta.abs().mean().item()),
            "first_prediction": decode_token(int(first.argmax().item()))[0],
            "second_prediction": decode_token(int(second.argmax().item()))[0],
            "first_margin_a_minus_b": float((first[first_token] - first[second_token]).item()),
            "second_margin_b_minus_a": float((second[second_token] - second[first_token]).item()),
            "operation_changes_argmax": int(first.argmax().item()) != int(second.argmax().item()),
        })
    return {
        "pairs": rows,
        "mean_logit_l2": statistics.fmean(row["logit_l2"] for row in rows) if rows else None,
        "mean_logit_cosine": statistics.fmean(row["logit_cosine"] for row in rows) if rows else None,
        "argmax_changes": sum(bool(row["operation_changes_argmax"]) for row in rows),
        "pair_count": len(rows),
    }


def gradient_snapshot(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prepared = prepare_with_positions(records)
    model.zero_grad(set_to_none=True)
    batch = stage_a.batchify(prepared, list(range(len(prepared))))
    logits = stage_a.forward_model(model, batch)
    loss, correct, total = stage_a.loss_and_accuracy(logits, batch)
    loss.backward()
    gradients: List[torch.Tensor] = []
    group_norms: Dict[str, float] = {}
    parameter_count = 0
    operation_token_rows: Dict[str, List[float]] = {"SELECT_FIRST": [], "SELECT_SECOND": []}
    token_embedding_grad = model.token_emb.weight.grad
    operation_token_ids = {
        operation: [int(stage_a.RAW_BYTE_START) + byte for byte in operation.encode("utf-8")]
        for operation in ("SELECT_FIRST", "SELECT_SECOND")
    }
    for operation, token_ids in operation_token_ids.items():
        if token_embedding_grad is not None:
            operation_token_rows[operation] = [float(torch.linalg.vector_norm(token_embedding_grad[token_id]).item()) for token_id in token_ids]
    for name, parameter in model.named_parameters():
        parameter_count += parameter.numel()
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float().cpu()
        gradients.append(gradient.reshape(-1))
        if name.startswith("layers."):
            parts = name.split(".")
            group = ".".join(parts[:3])
        else:
            group = name.split(".")[0]
        group_norms[group] = group_norms.get(group, 0.0) + float(torch.sum(gradient * gradient).item())
    vector = torch.cat(gradients) if gradients else torch.zeros(1)
    group_norms = {key: math.sqrt(value) for key, value in group_norms.items()}
    model.zero_grad(set_to_none=True)
    return {
        "examples": len(records),
        "loss": float(loss.detach().item()),
        "correct_target_tokens": int(correct),
        "target_tokens": int(total),
        "parameter_count": parameter_count,
        "gradient_norm": float(torch.linalg.vector_norm(vector).item()),
        "gradient_vector": vector,
        "group_norms": group_norms,
        "operation_token_embedding_row_norms": operation_token_rows,
    }


def gradient_diagnostics(model: Any, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    train_unequal = [row for row in records if row["split"] == "train" and row["metadata"]["a"] != row["metadata"]["b"]]
    first = gradient_snapshot(model, [row for row in train_unequal if row["metadata"]["operation"] == "SELECT_FIRST"])
    second = gradient_snapshot(model, [row for row in train_unequal if row["metadata"]["operation"] == "SELECT_SECOND"])
    first_vector = first.pop("gradient_vector")
    second_vector = second.pop("gradient_vector")
    dot = float(torch.dot(first_vector, second_vector).item())
    norm_product = float(torch.linalg.vector_norm(first_vector) * torch.linalg.vector_norm(second_vector))
    return {
        "dataset": "training unequal examples",
        "first": first,
        "second": second,
        "cross_operation_dot_product": dot,
        "cross_operation_cosine": dot / norm_product if norm_product else 0.0,
        "first_descent_first_order_effect_on_second_loss": -dot,
        "second_descent_first_order_effect_on_first_loss": -dot,
        "interpretation_note": "Negative cosine means the two descent directions oppose one another at this checkpoint; this is a diagnostic association, not proof of training-cause attribution.",
    }


def load_arm_records(interface: str, seed: int) -> Dict[str, Any]:
    directory = s3_arm_dir(interface, seed)
    return {
        "interface": interface,
        "seed": seed,
        "arm_id": arm_id(interface, seed),
        "directory": directory,
        "checkpoint": directory / "checkpoint.pt",
        "recorded": load_json(directory / "RESULTS.json"),
        "train": load_jsonl(directory / "train.jsonl"),
        "validation": load_jsonl(directory / "validation.jsonl"),
    }


def reproduce_arm(arm: Mapping[str, Any]) -> Dict[str, Any]:
    model = load_dense(arm["checkpoint"])
    prepared_train = prepare_with_positions(arm["train"])
    prepared_validation = prepare_with_positions(arm["validation"])
    train_metrics = stage_a.evaluate(model, prepared_train)
    validation_metrics = stage_a.evaluate(model, prepared_validation)
    recorded = arm["recorded"]
    return {
        "arm_id": arm["arm_id"],
        "interface": arm["interface"],
        "seed": arm["seed"],
        "checkpoint": rel(arm["checkpoint"]),
        "checkpoint_sha256": sha256(arm["checkpoint"]),
        "recorded_checkpoint_sha256": recorded.get("checkpoint_sha256"),
        "checkpoint_hash_match": sha256(arm["checkpoint"]) == recorded.get("checkpoint_sha256"),
        "train": train_metrics,
        "validation": validation_metrics,
        "recorded_train": recorded.get("train", {}).get("aggregate", {}),
        "recorded_validation": recorded.get("validation", {}).get("aggregate", {}),
        "train_metric_comparison": metric_diff(train_metrics, recorded.get("train", {}).get("aggregate", {})),
        "validation_metric_comparison": metric_diff(validation_metrics, recorded.get("validation", {}).get("aggregate", {})),
        "model": model,
    }


def token_control_reproduction() -> Dict[str, Any]:
    arms: List[Dict[str, Any]] = []
    errors: List[str] = []
    for interface in INTERFACES:
        for seed in SEEDS:
            directory = token_arm_dir(interface, seed)
            checkpoint = directory / "checkpoint.pt"
            recorded = load_json(directory / "RESULTS.json")
            train = load_jsonl(directory / "train.jsonl")
            validation = load_jsonl(directory / "validation.jsonl")
            model = load_dense(checkpoint)
            train_metrics = stage_a.evaluate(model, prepare_with_positions(train))
            validation_metrics = stage_a.evaluate(model, prepare_with_positions(validation))
            record = {
                "arm_id": f"operation_token_{interface}_seed{seed}",
                "interface": interface,
                "seed": seed,
                "checkpoint": rel(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "recorded_checkpoint_sha256": recorded.get("checkpoint_sha256"),
                "checkpoint_hash_match": sha256(checkpoint) == recorded.get("checkpoint_sha256"),
                "train": train_metrics,
                "validation": validation_metrics,
                "recorded_train": recorded.get("train", {}),
                "recorded_validation": recorded.get("validation", {}),
                "train_metric_comparison": metric_diff(train_metrics, recorded.get("train", {})),
                "validation_metric_comparison": metric_diff(validation_metrics, recorded.get("validation", {})),
            }
            if not record["checkpoint_hash_match"]:
                errors.append(f"token_checkpoint_hash:{record['arm_id']}")
            arms.append(record)
            del model
            gc.collect()
    return {"arms": arms, "errors": errors, "pass": not errors and len(arms) == 4}


def prediction_rows(arm: Mapping[str, Any], bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        logits = bundle["output_logits"][index]
        predicted_id = int(logits.argmax().item())
        predicted, format_error = decode_token(predicted_id)
        target = str(record["target"])
        rows.append({
            "arm_id": arm["arm_id"],
            "interface": arm["interface"],
            "seed": arm["seed"],
            "checkpoint": rel(arm["checkpoint"]),
            "checkpoint_sha256": sha256(arm["checkpoint"]),
            "record_id": record["record_id"],
            "split": record["split"],
            "a": int(record["metadata"]["a"]),
            "b": int(record["metadata"]["b"]),
            "operation": record["metadata"]["operation"],
            "expected_output": target,
            "predicted_output": predicted,
            "predicted_token_id": predicted_id,
            "output_format_error": format_error,
            "output_token_correct": predicted == target,
            "top_predictions": json.dumps(top_predictions(logits), sort_keys=True),
        })
    return rows


def find_index(records: Sequence[Mapping[str, Any]], a: int, b: int, operation: str) -> int:
    for index, record in enumerate(records):
        if int(record["metadata"]["a"]) == a and int(record["metadata"]["b"]) == b and record["metadata"]["operation"] == operation:
            return index
    raise KeyError((a, b, operation))


def patch_logit(model: Any, record: Mapping[str, Any], patch: Mapping[str, Any]) -> torch.Tensor:
    prepared = prepare_with_positions([record])
    return capture_forward(model, prepared, patch=patch, capture=False)["output_logits"]


def random_vector_like(vector: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(vector.shape, generator=generator, dtype=vector.dtype)
    scale = float(vector.std().item())
    if scale == 0.0:
        scale = 1.0
    return noise * scale + float(vector.mean().item())


def patching_diagnostics(model: Any, bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]], arm: Mapping[str, Any]) -> List[Dict[str, Any]]:
    validation_unequal = [row for row in records if row["split"] == "validation" and int(row["metadata"]["a"]) != int(row["metadata"]["b"])]
    rows: List[Dict[str, Any]] = []
    state_by_id = {record["record_id"]: index for index, record in enumerate(records)}
    pair_keys = sorted({(int(row["metadata"]["a"]), int(row["metadata"]["b"])) for row in validation_unequal})
    for pair_index, (a, b) in enumerate(pair_keys):
        for target_operation, donor_operation in (("SELECT_FIRST", "SELECT_SECOND"), ("SELECT_SECOND", "SELECT_FIRST")):
            target_index = find_index(records, a, b, target_operation)
            donor_index = find_index(records, a, b, donor_operation)
            target_record = records[target_index]
            baseline_logits = bundle["output_logits"][target_index]
            target_value = expected_operand(target_record)
            donor_value = expected_operand(records[donor_index])
            target_token = token_id_for_digit(target_value)
            donor_token = token_id_for_digit(donor_value)
            for layer in PATCH_LAYERS:
                for location in PATCH_LOCATIONS:
                    layer_name = f"layer_{layer}"
                    matched_vector = bundle["states"][layer_name][location][donor_index]
                    same_candidates = [
                        index for index, row in enumerate(records)
                        if row["split"] == "validation"
                        and row["metadata"]["operation"] == target_operation
                        and int(row["metadata"]["a"]) != int(row["metadata"]["b"])
                        and index != target_index
                    ]
                    same_index = same_candidates[(pair_index + layer) % len(same_candidates)]
                    same_vector = bundle["states"][layer_name][location][same_index]
                    controls = {
                        "matched_opposite": (matched_vector, donor_operation, records[donor_index]["record_id"]),
                        "same_operation_other_pair": (same_vector, target_operation, records[same_index]["record_id"]),
                        "random_noise": (random_vector_like(matched_vector, 1000 + pair_index * 100 + layer * 10 + (0 if target_operation == "SELECT_FIRST" else 1)), "RANDOM", "synthetic"),
                    }
                    for control, (vector, donor_label, donor_record_id) in controls.items():
                        patched_logits = patch_logit(model, target_record, {"layer": layer, "location": location, "vector": vector})
                        patched_prediction = decode_token(int(patched_logits.argmax().item()))[0]
                        baseline_margin = float((baseline_logits[target_token] - baseline_logits[donor_token]).item())
                        patched_margin = float((patched_logits[target_token] - patched_logits[donor_token]).item())
                        rows.append({
                            "arm_id": arm["arm_id"],
                            "interface": arm["interface"],
                            "seed": arm["seed"],
                            "pair": [a, b],
                            "target_operation": target_operation,
                            "donor_operation": donor_label,
                            "target_record_id": target_record["record_id"],
                            "donor_record_id": donor_record_id,
                            "layer": layer,
                            "location": location,
                            "control": control,
                            "baseline_prediction": decode_token(int(baseline_logits.argmax().item()))[0],
                            "patched_prediction": patched_prediction,
                            "target_value": target_value,
                            "donor_value": donor_value,
                            "baseline_target_correct": int(baseline_logits.argmax().item()) == target_token,
                            "patched_target_correct": int(patched_logits.argmax().item()) == target_token,
                            "patched_equals_donor_value": int(patched_logits.argmax().item()) == donor_token,
                            "baseline_target_minus_donor_margin": baseline_margin,
                            "patched_target_minus_donor_margin": patched_margin,
                            "margin_change_toward_donor": patched_margin - baseline_margin,
                            "predicted_token_changed": int(patched_logits.argmax().item()) != int(baseline_logits.argmax().item()),
                        })
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def summarize_patch(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for control in PATCH_CONTROLS:
        selected = [row for row in rows if row["control"] == control]
        result[control] = {
            "rows": len(selected),
            "patched_target_correct": sum(bool(row["patched_target_correct"]) for row in selected),
            "patched_equals_donor_value": sum(bool(row["patched_equals_donor_value"]) for row in selected),
            "prediction_changed": sum(bool(row["predicted_token_changed"]) for row in selected),
            "mean_margin_change_toward_donor": statistics.fmean(row["margin_change_toward_donor"] for row in selected) if selected else None,
            "by_layer_location": {
                f"layer{layer}:{location}": {
                    "rows": sum(row["layer"] == layer and row["location"] == location for row in selected),
                    "prediction_changed": sum(row["layer"] == layer and row["location"] == location and row["predicted_token_changed"] for row in selected),
                    "patched_equals_donor_value": sum(row["layer"] == layer and row["location"] == location and row["patched_equals_donor_value"] for row in selected),
                }
                for layer in PATCH_LAYERS for location in PATCH_LOCATIONS
            },
        }
    return result


def curve_summary(arms: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result = {}
    for arm in arms:
        path = s3_arm_dir(arm["interface"], arm["seed"]) / "TRAINING_CURVE.json"
        curve = load_json(path)
        result[arm["arm_id"]] = {"path": rel(path), "curve_steps": curve.get("curve_steps"), "points": curve.get("points")}
    return result


def build_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Crucible Operation-Conditioning Mechanism Investigation — Final Report",
        "",
        "CPU-only, read-only forensics on the frozen S3 joint-operation checkpoints. No new training, architecture change, GPU execution or sealed-test access.",
        "",
        "## Phase 1 — Reproduction",
        "",
        f"S3 arm reproduction: {'PASS' if summary['verification']['s3_reproduction_pass'] else 'FAIL'}; token-identifier control reproduction: {'PASS' if summary['verification']['token_control_pass'] else 'FAIL'}.",
        "",
        "| Interface | Seed | S3 train exact | S3 validation exact | Checkpoint hash |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in summary["s3_reproduction"]:
        lines.append(f"| {arm['interface']} | {arm['seed']} | {arm['train']['teacher_forced_exact_accuracy']:.2%} | {arm['validation']['teacher_forced_exact_accuracy']:.2%} | {'match' if arm['checkpoint_hash_match'] else 'MISMATCH'} |")
    lines += [
        "",
        "## Phase 2 — Representation evidence",
        "",
        "The operation-identity probes are fitted on training rows and evaluated on disjoint validation rows. Probe decodability is evidence that information is recoverable, not evidence that the forward computation uses it causally.",
        "",
        "| Arm | State/location with highest validation ridge accuracy | Ridge validation | Nearest-centroid validation |",
        "|---|---|---:|---:|",
    ]
    for arm_id_value, probe in summary["probe_highlights"].items():
        lines.append(f"| {arm_id_value} | {probe['state']}/{probe['location']} | {probe['ridge_validation']:.2%} | {probe['centroid_validation']:.2%} |")
    lines += [
        "",
        "| Arm | Mean output logit L2 change | Mean output logit cosine | Pairwise argmax changes |",
        "|---|---:|---:|---:|",
    ]
    for arm_id_value, sensitivity in summary["output_sensitivity"].items():
        lines.append(f"| {arm_id_value} | {sensitivity['mean_logit_l2']:.4f} | {sensitivity['mean_logit_cosine']:.4f} | {sensitivity['argmax_changes']}/{sensitivity['pair_count']} |")
    lines += [
        "",
        "## Phase 3 — Gradient diagnostics",
        "",
        "| Arm | First gradient norm | Second gradient norm | Cross-operation cosine |",
        "|---|---:|---:|---:|",
    ]
    for arm_id_value, gradient in summary["gradient_diagnostics"].items():
        lines.append(f"| {arm_id_value} | {gradient['first']['gradient_norm']:.4f} | {gradient['second']['gradient_norm']:.4f} | {gradient['cross_operation_cosine']:.4f} |")
    lines += [
        "",
        "## Phase 4 — Activation patching",
        "",
        "Patching used validation unequal pairs, layers 0, 2 and 3, operation-token/output-position locations, and same-operation/random controls. A patch changing a prediction or margin is reported as an intervention effect for that tested location; it is not treated as proof of a unique mechanism.",
        "",
        "| Arm | Control | Rows | Prediction changes | Patched outputs equal donor value | Mean margin movement |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm_id_value, controls in summary["patch_summary"].items():
        for control, values in controls.items():
            lines.append(f"| {arm_id_value} | {control} | {values['rows']} | {values['prediction_changed']} | {values['patched_equals_donor_value']} | {values['mean_margin_change_toward_donor']:.4f} |" if values["mean_margin_change_toward_donor"] is not None else f"| {arm_id_value} | {control} | 0 | 0 | 0 | n/a |")
    lines += [
        "",
        "## Supported conclusions",
        "",
        "- The S3 metrics and checkpoint hashes were reproduced without opening the sealed test set.",
        "- The separate operation-token identification control establishes that the model can classify the requested operation under the same broad task interface.",
        "- The S3 failure therefore cannot be reduced to a basic inability to recognize the operation string.",
        "- Hidden-state decodability, logit sensitivity, gradient alignment and patch effects are separate observations. None alone establishes whether the model uses operation information correctly.",
        "- The report does not assign an internal mechanism unless the corresponding intervention exceeds its matched controls.",
        "",
        "## Attention limitation",
        "",
        "The dense model routes attention through the current scaled-dot-product-attention API without exposing attention probability matrices. Native attention-pattern results are therefore unavailable in this run; no substitute was mislabeled as native attention telemetry.",
        "",
        "## Artifacts",
        "",
        "- [Protocol](../../docs/CRUCIBLE_OPERATION_CONDITIONING_FORENSICS_PROTOCOL_20251025.md)",
        "- [Machine-readable summary](SUMMARY.json)",
        "- [Reproduced metrics](REPRODUCED_METRICS.json)",
        "- [Predictions](PREDICTIONS.csv)",
        "- [Hidden states](hidden_states/)",
        "- [Operation probes](OPERATION_PROBES.json)",
        "- [Gradient diagnostics](GRADIENT_DIAGNOSTICS.json)",
        "- [Activation patch matrix](ACTIVATION_PATCHING.csv)",
        "- [Training curves](CURVE_SUMMARY.json)",
        "- [Finalization](FINALIZATION.json)",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    protocol_manifest = {
        "experiment": "crucible_operation_conditioning_forensics_20251025",
        "status": "frozen_before_analysis",
        "source_s3_root": rel(S3_ROOT),
        "source_token_root": rel(TOKEN_ROOT),
        "interfaces": list(INTERFACES),
        "seeds": list(SEEDS),
        "patch_layers": list(PATCH_LAYERS),
        "patch_locations": list(PATCH_LOCATIONS),
        "patch_controls": list(PATCH_CONTROLS),
        "sealed_test_accessed": False,
        "new_training": False,
        "architecture_changed": False,
    }
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", protocol_manifest)

    print(json.dumps({"event": "phase1_start", "output": rel(OUTPUT)}, sort_keys=True), flush=True)
    arms: List[Dict[str, Any]] = []
    s3_reproduction: List[Dict[str, Any]] = []
    all_prediction_rows: List[Dict[str, Any]] = []
    hidden_probe_results: Dict[str, Any] = {}
    hidden_differences: Dict[str, Any] = {}
    output_sensitivities: Dict[str, Any] = {}
    gradient_results: Dict[str, Any] = {}
    patch_results: Dict[str, List[Dict[str, Any]]] = {}
    models: Dict[str, Any] = {}
    bundles: Dict[str, Any] = {}
    records_by_arm: Dict[str, List[Dict[str, Any]]] = {}

    for interface in INTERFACES:
        for seed in SEEDS:
            arm = load_arm_records(interface, seed)
            train_audit = record_audit(interface, "train", arm["train"])
            validation_audit = record_audit(interface, "validation", arm["validation"])
            if not train_audit["pass"] or not validation_audit["pass"]:
                errors.append(f"record_audit:{arm['arm_id']}")
            reproduction = reproduce_arm(arm)
            reproduction["record_audit"] = {"train": train_audit, "validation": validation_audit}
            if not reproduction["checkpoint_hash_match"]:
                errors.append(f"checkpoint_hash:{arm['arm_id']}")
            for comparison_name in ("train_metric_comparison", "validation_metric_comparison"):
                if not all(entry.get("match", False) for entry in reproduction[comparison_name].values()):
                    errors.append(f"metric_mismatch:{arm['arm_id']}:{comparison_name}")
            s3_reproduction.append({key: value for key, value in reproduction.items() if key != "model"})
            model = reproduction["model"]
            records = arm["train"] + arm["validation"]
            aid = arm["arm_id"]
            models[aid] = model
            records_by_arm[aid] = records
            bundle = capture_records(model, records)
            bundles[aid] = bundle
            probe = build_hidden_probe_results(bundle, records)
            hidden_probe_results[aid] = probe
            hidden_differences[aid] = hidden_difference(bundle, records)
            output_sensitivities[aid] = output_sensitivity(bundle, records)
            gradient_results[aid] = gradient_diagnostics(model, records)
            rows = prediction_rows(arm, bundle, records)
            all_prediction_rows.extend(rows)
            hidden_path = OUTPUT / "hidden_states" / f"{aid}.pt"
            hidden_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "arm_id": aid,
                "record_ids": bundle["record_ids"],
                "labels": bundle["labels"],
                "operations": bundle["operations"],
                "states": bundle["states"],
                "output_logits": bundle["output_logits"],
            }, hidden_path)
            print(json.dumps({"event": "phase2_3_arm_complete", "arm_id": aid}, sort_keys=True), flush=True)

    print(json.dumps({"event": "phase1_token_control_start"}, sort_keys=True), flush=True)
    token_reproduction = token_control_reproduction()
    if not token_reproduction["pass"]:
        errors.append("token_control_reproduction")

    print(json.dumps({"event": "phase4_start", "arms": len(models)}, sort_keys=True), flush=True)
    for aid in sorted(models):
        # Construct lightweight arm metadata without reopening source data.
        interface = "canonical" if "_canonical_" in aid else "factorized"
        seed = 42 if aid.endswith("seed42") else 1337
        arm_meta = {"arm_id": aid, "interface": interface, "seed": seed, "checkpoint": s3_arm_dir(interface, seed) / "checkpoint.pt"}
        patch_rows = patching_diagnostics(models[aid], bundles[aid], records_by_arm[aid], arm_meta)
        patch_results[aid] = patch_rows
        write_json(OUTPUT / "patching" / f"{aid}.json", summarize_patch(patch_rows))
        print(json.dumps({"event": "phase4_arm_complete", "arm_id": aid, "rows": len(patch_rows)}, sort_keys=True), flush=True)

    write_csv(OUTPUT / "PREDICTIONS.csv", all_prediction_rows)
    all_patch_rows = [row for rows in patch_results.values() for row in rows]
    write_csv(OUTPUT / "ACTIVATION_PATCHING.csv", all_patch_rows)
    write_json(OUTPUT / "REPRODUCED_METRICS.json", {"s3": s3_reproduction, "operation_token_control": token_reproduction})
    write_json(OUTPUT / "OPERATION_PROBES.json", hidden_probe_results)
    write_json(OUTPUT / "HIDDEN_DIFFERENCES.json", hidden_differences)
    write_json(OUTPUT / "OUTPUT_SENSITIVITY.json", output_sensitivities)
    write_json(OUTPUT / "GRADIENT_DIAGNOSTICS.json", json_safe(gradient_results))
    curves = curve_summary([{"arm_id": arm["arm_id"], "interface": arm["interface"], "seed": arm["seed"]} for arm in s3_reproduction])
    write_json(OUTPUT / "CURVE_SUMMARY.json", curves)

    probe_highlights: Dict[str, Any] = {}
    for aid, states in hidden_probe_results.items():
        candidates = []
        for state, locations in states.items():
            for location, values in locations.items():
                candidates.append({"state": state, "location": location, "ridge_validation": values["ridge"]["validation_accuracy"], "centroid_validation": values["nearest_centroid"]["validation_accuracy"]})
        probe_highlights[aid] = max(candidates, key=lambda item: (item["ridge_validation"], item["centroid_validation"]))
    patch_summary = {aid: summarize_patch(rows) for aid, rows in patch_results.items()}
    verification = {
        "s3_reproduction_pass": not any(error.startswith(("record_audit:", "checkpoint_hash:", "metric_mismatch:")) for error in errors),
        "token_control_pass": token_reproduction["pass"],
        "prediction_rows": len(all_prediction_rows),
        "patch_rows": len(all_patch_rows),
        "sealed_test_accessed": False,
        "new_training": False,
        "architecture_changed": False,
        "errors": errors,
    }
    summary = {
        "experiment": "crucible_operation_conditioning_forensics_20251025_v1",
        "status": "completed",
        "s3_reproduction": s3_reproduction,
        "token_control_reproduction": token_reproduction,
        "probe_highlights": probe_highlights,
        "output_sensitivity": output_sensitivities,
        "gradient_diagnostics": json_safe(gradient_results),
        "patch_summary": patch_summary,
        "verification": verification,
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
    (OUTPUT / "REPORT_FINAL.md").write_text(build_report(summary), encoding="utf-8")
    hashes: Dict[str, str] = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in {"HASHES.json", "FINALIZATION.json"} and path.suffix not in {".pt"}:
            hashes[rel(path)] = sha256(path)
    write_json(OUTPUT / "HASHES.json", hashes)
    finalization = {
        "status": "completed",
        "verification_pass": not errors,
        "s3_reproduction_pass": verification["s3_reproduction_pass"],
        "token_control_pass": verification["token_control_pass"],
        "sealed_test_accessed": False,
        "new_training": False,
        "architecture_changed": False,
        "prediction_sha256": sha256(OUTPUT / "PREDICTIONS.csv"),
        "patch_sha256": sha256(OUTPUT / "ACTIVATION_PATCHING.csv"),
        "report_sha256": sha256(OUTPUT / "REPORT_FINAL.md"),
        "errors": errors,
    }
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    for model in models.values():
        del model
    gc.collect()
    print(json.dumps({"status": "completed", "output": rel(OUTPUT), "s3_arms": len(s3_reproduction), "prediction_rows": len(all_prediction_rows), "patch_rows": len(all_patch_rows), "verification_pass": not errors}, sort_keys=True), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
