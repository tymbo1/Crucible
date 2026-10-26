#!/usr/bin/env python3
"""Read-only authority representation-versus-use diagnostic for A3 checkpoints."""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "crucible_authority_representation_use_20261026_v1"
PROTOCOL = ROOT / "docs" / "CRUCIBLE_AUTHORITY_REPRESENTATION_USE_PROTOCOL_20261026.md"
AUTH_SOURCE = ROOT / "tools" / "run_crucible_authority_aware_role_binding_20251025.py"
CF_SOURCE = ROOT / "tools" / "run_crucible_authority_counterfactual_20261026_v2.py"
V2_ROOT = ROOT / "experiments" / "crucible_authority_aware_role_binding_20251025_v2"
CF_ROOT = ROOT / "experiments" / "crucible_authority_counterfactual_20261026_v2"

INTERFACES = ("canonical", "factorized")
SEEDS = (42, 1337)
FAMILIAR_LAYOUTS = ("AUTHORITY_NEAR", "AUTHORITY_EARLY", "AUTHORITY_MIDDLE")
UNSEEN_LAYOUT = "UNSEEN_REVERSED_AUTHORITY"
PATCH_SITES = ("embedding", "layer_0", "layer_1", "layer_2", "layer_3", "final_norm")
PROBE_LOCATIONS = ("authority", "op_first", "op_second", "output_end", "mean_valid")
RIDGE_LAMBDA = 1.0e-2
NUMERICAL_TOLERANCE = 2.0e-6


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__file__"] = str(path)
    spec.loader.exec_module(module)
    return module


auth = import_module("authority_role_binding_for_representation", AUTH_SOURCE)
cf = import_module("authority_counterfactual_for_representation", CF_SOURCE)
torch = auth.torch


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_a3_model(path: Path, seed: int) -> Any:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = auth.RoleSupervisedParser(seed)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def source_checkpoint(interface: str, seed: int) -> Path:
    return V2_ROOT / "trained_arms" / "A3_AUTHORITY_POINTER_BALANCED" / interface / f"seed{seed}" / "checkpoint.pt"


def decode_operation(full: Sequence[int], position: int) -> str | None:
    if position < 0 or position >= len(full):
        return None
    return auth.decode_operation(int(full[position]))


def pointer_summary(prepared: Sequence[Mapping[str, Any]], op_logits: Sequence[torch.Tensor]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item, logits in zip(prepared, op_logits):
        pos = int(logits.argmax().item())
        predicted = decode_operation(item["_parser_full"], pos)
        target = str(item["metadata"]["operation"])
        rows.append({
            "record_id": str(item["record_id"]),
            "a": int(item["metadata"]["a"]),
            "b": int(item["metadata"]["b"]),
            "authority_side": str(item["metadata"].get("authority_side", "")),
            "operation_order": str(item["metadata"].get("operation_order", "")),
            "target_operation": target,
            "pointer_position": pos,
            "pointer_operation": predicted,
            "pointer_correct": predicted == target,
        })
    return rows


def capture_bundle(model: Any, records: Sequence[Mapping[str, Any]], verify_first_batch: bool = True) -> Dict[str, Any]:
    prepared = auth.prepare_authority_parser(records)
    names = list(PATCH_SITES)
    captured_states: Dict[str, Dict[str, List[torch.Tensor]]] = {
        name: {location: [] for location in PROBE_LOCATIONS} for name in names
    }
    op_logits: List[torch.Tensor] = []
    fidelity: Dict[str, Any] = {"batches": 0, "max_abs_delta_by_key": {}, "pass": True}

    for start in range(0, len(prepared), int(auth.stage_a.BATCH_SIZE)):
        batch_items = prepared[start:start + int(auth.stage_a.BATCH_SIZE)]
        indices = list(range(len(batch_items)))
        batch = auth.authority_batchify(batch_items, indices)
        captured: Dict[str, torch.Tensor] = {}
        handles = []

        def hook_for(name: str):
            def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
                value = output[0] if isinstance(output, tuple) else output
                captured[name] = value.detach().cpu()
                return output
            return hook

        handles.append(model.backbone.token_emb.register_forward_hook(hook_for("embedding")))
        for index, layer in enumerate(model.backbone.layers):
            handles.append(layer.register_forward_hook(hook_for(f"layer_{index}")))
        handles.append(model.backbone.final_norm.register_forward_hook(hook_for("final_norm")))
        with torch.inference_mode():
            instrumented = model.forward_all(batch)
            ordinary = model.forward_all(batch) if verify_first_batch and start == 0 else None
        for handle in handles:
            handle.remove()

        fidelity["batches"] += 1
        if ordinary is not None:
            for key in sorted(instrumented):
                delta = float((instrumented[key] - ordinary[key]).abs().max().item())
                fidelity["max_abs_delta_by_key"][key] = max(fidelity["max_abs_delta_by_key"].get(key, 0.0), delta)
                if delta > NUMERICAL_TOLERANCE:
                    fidelity["pass"] = False

        for local, item in enumerate(batch_items):
            authority_pos = int(item["_authority_position"])
            end_pos = int(item["_parser_length"]) - 1
            valid_slice = slice(0, int(item["_parser_length"]))
            for name in names:
                value = captured[name][local]
                captured_states[name]["authority"].append(value[authority_pos].clone())
                captured_states[name]["op_first"].append(value[int(item["_op_positions"][0])].clone())
                captured_states[name]["op_second"].append(value[int(item["_op_positions"][1])].clone())
                captured_states[name]["output_end"].append(value[end_pos].clone())
                captured_states[name]["mean_valid"].append(value[valid_slice].float().mean(dim=0).clone())
            op_logits.append(instrumented["op_pos"][local, : int(item["_parser_length"])].detach().cpu().clone())

    states = {
        name: {location: torch.stack(values) for location, values in locations.items()}
        for name, locations in captured_states.items()
    }
    pointer_rows = pointer_summary(prepared, op_logits)
    return {"prepared": prepared, "states": states, "op_logits": op_logits, "pointer_rows": pointer_rows, "fidelity": fidelity}


def ridge_fit(train_x: torch.Tensor, train_y: torch.Tensor) -> torch.Tensor:
    x = train_x.double()
    y = train_y.double() * 2.0 - 1.0
    x_aug = torch.cat([x, torch.ones((x.size(0), 1), dtype=x.dtype)], dim=1)
    gram = x_aug.T @ x_aug
    regularizer = torch.eye(gram.size(0), dtype=gram.dtype) * RIDGE_LAMBDA
    regularizer[-1, -1] = 0.0
    return torch.linalg.solve(gram + regularizer, x_aug.T @ y)


def ridge_predict(weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x_aug = torch.cat([x.double(), torch.ones((x.size(0), 1), dtype=torch.float64)], dim=1)
    return (x_aug @ weights >= 0).long()


def probe_one(train_values: torch.Tensor, train_labels: torch.Tensor, eval_values: torch.Tensor, eval_labels: torch.Tensor) -> Dict[str, Any]:
    weights = ridge_fit(train_values, train_labels)
    train_pred = ridge_predict(weights, train_values)
    eval_pred = ridge_predict(weights, eval_values)
    return {
        "train_rows": int(train_labels.numel()),
        "train_correct": int((train_pred == train_labels).sum().item()),
        "train_accuracy": float((train_pred == train_labels).float().mean().item()),
        "eval_rows": int(eval_labels.numel()),
        "eval_correct": int((eval_pred == eval_labels).sum().item()),
        "eval_accuracy": float((eval_pred == eval_labels).float().mean().item()) if eval_labels.numel() else None,
        "prediction": eval_pred.tolist(),
        "margin_mean": float((torch.cat([eval_values.double(), torch.ones((eval_values.size(0), 1), dtype=torch.float64)], dim=1) @ weights).abs().mean().item()) if eval_values.size(0) else None,
    }


def labels_for(records: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    values = []
    for row in records:
        side = str(row["metadata"].get("authority_side", ""))
        if side not in {"EARLIER", "LATER"}:
            raise RuntimeError(f"missing authority label: {row['record_id']}")
        values.append(0 if side == "EARLIER" else 1)
    return torch.tensor(values, dtype=torch.long)


def build_datasets(interface: str) -> Dict[str, Any]:
    base_records = auth.base.exposure.base_records(interface, 42)
    train_ids = {str(row["record_id"]) for row in base_records["train"]}
    validation_ids = {str(row["record_id"]) for row in base_records["validation"]}
    if train_ids & validation_ids:
        raise RuntimeError(f"source split overlap for {interface}")
    train_records = auth.build_training_records(base_records, interface)
    familiar = {layout: auth.build_eval_records(base_records, interface, "validation", layout) for layout in FAMILIAR_LAYOUTS}
    unseen = cf.build_records(interface, base_records["validation"])
    cf.audit_records(unseen)
    return {
        "base": base_records,
        "train": train_records,
        "familiar": familiar,
        "unseen": unseen,
        "audit": {
            "base_train_rows": len(base_records["train"]),
            "base_validation_rows": len(base_records["validation"]),
            "base_id_overlap": len(train_ids & validation_ids),
            "authority_train_rows": len(train_records),
            "familiar_rows": {layout: len(rows) for layout, rows in familiar.items()},
            "unseen_rows": len(unseen),
            "unseen_pairs": len(unseen) // 2,
        },
    }


def compare_to_accepted_export(interface: str, seed: int, unseen_rows: Sequence[Mapping[str, Any]], familiar_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    accepted_cf = list(csv.DictReader((CF_ROOT / "COUNTERFACTUAL_PREDICTIONS.csv").open(newline="", encoding="utf-8")))
    accepted = {
        str(row["record_id"]): row for row in accepted_cf
        if row["arm"] == "A3_AUTHORITY_POINTER_BALANCED" and row["interface"] == interface and int(row["seed"]) == seed
    }
    cf_matches = 0
    cf_mismatches: List[Dict[str, Any]] = []
    for row in unseen_rows:
        old = accepted.get(str(row["record_id"]))
        if old is None:
            cf_mismatches.append({"record_id": row["record_id"], "reason": "missing_accepted_counterfactual_row"})
            continue
        cf_matches += 1
        if str(old["pointer_operation"]) != str(row["pointer_operation"]) or int(old["pointer_operation_position"]) != int(row["pointer_position"]):
            cf_mismatches.append({"record_id": row["record_id"], "accepted": {"pointer_operation": old["pointer_operation"], "pointer_position": old["pointer_operation_position"]}, "reproduced": {"pointer_operation": row["pointer_operation"], "pointer_position": row["pointer_position"]}})

    accepted_familiar = list(csv.DictReader((V2_ROOT / "PARSER_PREDICTIONS.csv").open(newline="", encoding="utf-8")))
    familiar_matches = 0
    familiar_mismatches: List[Dict[str, Any]] = []
    for layout, rows in familiar_rows.items():
        accepted_map = {
            str(row["record_id"]): row for row in accepted_familiar
            if row["arm"] == "A3_AUTHORITY_POINTER_BALANCED" and row["interface"] == interface and int(row["seed"]) == seed and row["layout"] == layout and row["split"] == "validation"
        }
        for row in rows:
            reproduced = row
            old = accepted_map.get(str(row["record_id"]))
            if old is None:
                familiar_mismatches.append({"layout": layout, "record_id": row["record_id"], "reason": "missing_accepted_familiar_row"})
                continue
            familiar_matches += 1
            if str(old["pointer_operation"]) != str(reproduced["pointer_operation"]) or int(old["op_pointer_position"]) != int(reproduced["pointer_position"]):
                familiar_mismatches.append({"layout": layout, "record_id": row["record_id"], "accepted": {"pointer_operation": old["pointer_operation"], "pointer_position": old["op_pointer_position"]}, "reproduced": {"pointer_operation": reproduced["pointer_operation"], "pointer_position": reproduced["pointer_position"]}})
    return {
        "counterfactual_rows_compared": cf_matches,
        "counterfactual_mismatches": cf_mismatches,
        "counterfactual_pass": not cf_mismatches and cf_matches == len(unseen_rows),
        "familiar_rows_compared": familiar_matches,
        "familiar_mismatches": familiar_mismatches,
        "familiar_pass": not familiar_mismatches and familiar_matches == sum(len(rows) for rows in familiar_rows.values()),
    }


def probe_diagnostics(model_key: str, bundles: Mapping[str, Dict[str, Any]], datasets: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_records = datasets["train"]
    train_bundle = bundles["train"]
    train_labels = labels_for(train_records)
    metrics: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    for layer_name, locations in train_bundle["states"].items():
        for location in PROBE_LOCATIONS:
            weights = ridge_fit(locations[location], train_labels)
            train_pred = ridge_predict(weights, locations[location])
            train_accuracy = float((train_pred == train_labels).float().mean().item())
            for domain, records, bundle in [("train", train_records, train_bundle)] + [(layout, datasets["familiar"][layout], bundles[layout]) for layout in FAMILIAR_LAYOUTS] + [("unseen_reversed_authority", datasets["unseen"], bundles["unseen"])]:
                eval_labels = labels_for(records)
                eval_values = bundle["states"][layer_name][location]
                eval_pred = ridge_predict(weights, eval_values)
                correct = eval_pred == eval_labels
                metrics.append({
                    "model": model_key, "layer": layer_name, "location": location, "domain": domain,
                    "rows": int(eval_labels.numel()), "correct": int(correct.sum().item()),
                    "accuracy": float(correct.float().mean().item()) if eval_labels.numel() else None,
                    "probe_train_rows": int(train_labels.numel()), "probe_train_correct": int((train_pred == train_labels).sum().item()),
                    "probe_train_accuracy": train_accuracy,
                    "ridge_lambda": RIDGE_LAMBDA,
                })
                for row, prediction, label, is_correct in zip(records, eval_pred.tolist(), eval_labels.tolist(), correct.tolist()):
                    predictions.append({
                        "model": model_key, "layer": layer_name, "location": location, "domain": domain,
                        "record_id": str(row["record_id"]), "authority_side": str(row["metadata"]["authority_side"]),
                        "predicted_authority_side": "EARLIER" if int(prediction) == 0 else "LATER",
                        "correct": bool(is_correct), "label": "EARLIER" if int(label) == 0 else "LATER",
                    })
    return metrics, predictions


def run_forward(model: Any, batch: Mapping[str, Any], patch_site: str | None = None, positions: Sequence[int] | None = None, vectors: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    handles = []
    if patch_site is not None:
        if positions is None or vectors is None:
            raise ValueError("patch requires positions and vectors")
        position_tensor = torch.tensor(list(positions), dtype=torch.long)
        donor = vectors.detach().cpu()

        def patch_hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> Any:
            value = output[0] if isinstance(output, tuple) else output
            patched = value.clone()
            rows = torch.arange(patched.size(0), dtype=torch.long)
            patched[rows, position_tensor, :] = donor.to(dtype=patched.dtype)
            return patched

        if patch_site == "embedding":
            handles.append(model.backbone.token_emb.register_forward_hook(patch_hook))
        elif patch_site == "final_norm":
            handles.append(model.backbone.final_norm.register_forward_hook(patch_hook))
        elif patch_site.startswith("layer_"):
            handles.append(model.backbone.layers[int(patch_site.split("_")[1])].register_forward_hook(patch_hook))
        else:
            raise ValueError(patch_site)
    try:
        with torch.inference_mode():
            return model.forward_all(batch)
    finally:
        for handle in handles:
            handle.remove()


def patch_batch(model: Any, prepared: Sequence[Mapping[str, Any]], indices: Sequence[int], donor_vectors: torch.Tensor, patch_site: str | None) -> List[torch.Tensor]:
    selected = [prepared[index] for index in indices]
    local_batch = auth.authority_batchify(selected, list(range(len(selected))))
    positions = [int(item["_authority_position"]) for item in selected]
    logits = run_forward(model, local_batch, patch_site, positions, donor_vectors)
    return [logits["op_pos"][row, : int(item["_parser_length"])].detach().cpu().clone() for row, item in enumerate(selected)]


def patch_diagnostics(model_key: str, model: Any, bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    prepared = bundle["prepared"]
    by_pair: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_pair[str(row["metadata"]["counterfactual_pair_id"])].append(index)
    directions: List[Tuple[int, int, str]] = []
    for pair_id, indices in sorted(by_pair.items()):
        if len(indices) != 2:
            raise RuntimeError(f"counterfactual pair cardinality: {pair_id}")
        sides = {str(records[index]["metadata"]["authority_side"]): index for index in indices}
        if set(sides) != {"EARLIER", "LATER"}:
            raise RuntimeError(f"counterfactual side failure: {pair_id}")
        directions.extend([(sides["EARLIER"], sides["LATER"], "recipient_earlier"), (sides["LATER"], sides["EARLIER"], "recipient_later")])

    opposite_rows: List[Dict[str, Any]] = []
    self_rows: List[Dict[str, Any]] = []
    patch_summary: Dict[str, Any] = {"model": model_key, "sites": {}}
    all_indices = list(range(len(records)))
    for site in PATCH_SITES:
        own_vectors = bundle["states"][site]["authority"]
        self_logits = patch_batch(model, prepared, all_indices, own_vectors, site)
        self_deltas = []
        self_pointer_changes = 0
        for index, (patched, baseline) in enumerate(zip(self_logits, bundle["op_logits"])):
            length = int(prepared[index]["_parser_length"])
            base = baseline[:length].float(); altered = patched.float()
            delta = float((altered - base).abs().max().item())
            base_pos = int(base.argmax().item()); altered_pos = int(altered.argmax().item())
            self_deltas.append(delta)
            self_pointer_changes += int(base_pos != altered_pos)
            self_rows.append({"model": model_key, "site": site, "record_id": str(records[index]["record_id"]), "max_abs_op_pointer_logit_delta": delta, "baseline_pointer_position": base_pos, "patched_pointer_position": altered_pos, "pointer_changed": base_pos != altered_pos})

        recipient_indices = [item[0] for item in directions]
        donor_indices = [item[1] for item in directions]
        donor_vectors = own_vectors[torch.tensor(donor_indices, dtype=torch.long)]
        patched_logits = patch_batch(model, prepared, recipient_indices, donor_vectors, site)
        changed = 0; correct = 0; donor_match = 0; deltas = []
        for (recipient, donor, direction), patched in zip(directions, patched_logits):
            item = prepared[recipient]
            length = int(item["_parser_length"])
            base = bundle["op_logits"][recipient][:length].float(); altered = patched.float()
            base_pos = int(base.argmax().item()); altered_pos = int(altered.argmax().item())
            base_operation = decode_operation(item["_parser_full"], base_pos)
            altered_operation = decode_operation(item["_parser_full"], altered_pos)
            donor_item = prepared[donor]
            donor_base = bundle["op_logits"][donor][: int(donor_item["_parser_length"])].float()
            donor_operation = decode_operation(donor_item["_parser_full"], int(donor_base.argmax().item()))
            target = str(item["metadata"]["operation"])
            delta = float((altered - base).abs().max().item())
            changed += int(altered_pos != base_pos or altered_operation != base_operation)
            correct += int(altered_operation == target)
            donor_match += int(altered_operation == donor_operation)
            deltas.append(delta)
            opposite_rows.append({
                "model": model_key, "site": site, "direction": direction,
                "pair_id": str(item["metadata"]["counterfactual_pair_id"]),
                "recipient_record_id": str(item["record_id"]), "donor_record_id": str(donor_item["record_id"]),
                "recipient_authority_side": str(item["metadata"]["authority_side"]),
                "baseline_pointer_operation": base_operation, "patched_pointer_operation": altered_operation,
                "donor_baseline_pointer_operation": donor_operation, "target_operation": target,
                "baseline_pointer_position": base_pos, "patched_pointer_position": altered_pos,
                "pointer_changed": bool(altered_pos != base_pos or altered_operation != base_operation),
                "patched_pointer_correct": altered_operation == target,
                "patched_pointer_matches_donor": altered_operation == donor_operation,
                "max_abs_op_pointer_logit_delta": delta,
            })
        patch_summary["sites"][site] = {
            "opposite_authority_rows": len(patched_logits), "opposite_pointer_changed": changed,
            "opposite_pointer_changed_rate": changed / len(patched_logits) if patched_logits else None,
            "opposite_patched_pointer_correct": correct, "opposite_patched_pointer_accuracy": correct / len(patched_logits) if patched_logits else None,
            "opposite_matches_donor_baseline": donor_match, "mean_abs_op_pointer_logit_delta": statistics.fmean(deltas) if deltas else None,
            "self_rows": len(self_logits), "self_pointer_changed": self_pointer_changes,
            "self_max_abs_op_pointer_logit_delta": max(self_deltas) if self_deltas else None,
            "self_mean_abs_op_pointer_logit_delta": statistics.fmean(self_deltas) if self_deltas else None,
            "self_fidelity_pass": max(self_deltas, default=0.0) <= NUMERICAL_TOLERANCE and self_pointer_changes == 0,
        }
    return opposite_rows, self_rows, patch_summary


def causal_access_diagnostics(model_key: str, bundle: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Verify whether a later authority field changes earlier pointer states."""
    by_pair: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_pair[str(row["metadata"]["counterfactual_pair_id"])].append(index)
    rows: List[Dict[str, Any]] = []
    for pair_id, indices in sorted(by_pair.items()):
        if len(indices) != 2:
            raise RuntimeError(f"causal-access pair cardinality: {pair_id}")
        left, right = indices
        left_item = bundle["prepared"][left]
        right_item = bundle["prepared"][right]
        if left_item["_op_positions"] != right_item["_op_positions"]:
            raise RuntimeError(f"causal-access operation positions differ: {pair_id}")
        op_positions = [int(value) for value in left_item["_op_positions"]]
        op_logit_deltas = [float((bundle["op_logits"][left][pos] - bundle["op_logits"][right][pos]).abs().item()) for pos in op_positions]
        hidden_deltas: Dict[str, Dict[str, float]] = {}
        for site in PATCH_SITES:
            hidden_deltas[site] = {
                "op_first": float((bundle["states"][site]["op_first"][left] - bundle["states"][site]["op_first"][right]).abs().max().item()),
                "op_second": float((bundle["states"][site]["op_second"][left] - bundle["states"][site]["op_second"][right]).abs().max().item()),
            }
        rows.append({"pair_id": pair_id, "authority_position": int(left_item["_authority_position"]), "operation_positions": op_positions, "max_operation_pointer_logit_delta": max(op_logit_deltas, default=0.0), "operation_pointer_logit_deltas": op_logit_deltas, "hidden_deltas_at_operation_positions": hidden_deltas})
    max_pointer_delta = max((row["max_operation_pointer_logit_delta"] for row in rows), default=0.0)
    max_hidden_delta = max((value for row in rows for site in row["hidden_deltas_at_operation_positions"].values() for value in site.values()), default=0.0)
    return {"model": model_key, "pairs": len(rows), "rows": rows, "max_operation_pointer_logit_delta": max_pointer_delta, "max_hidden_delta_at_operation_positions": max_hidden_delta, "position_path_unchanged_within_tolerance": max_pointer_delta <= NUMERICAL_TOLERANCE and max_hidden_delta <= NUMERICAL_TOLERANCE}


def aggregate_probe(metrics: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for domain in ["train", *FAMILIAR_LAYOUTS, "unseen_reversed_authority"]:
        subset = [row for row in metrics if row["domain"] == domain]
        if not subset:
            continue
        result[domain] = {}
        for layer in sorted({str(row["layer"]) for row in subset}):
            result[domain][layer] = {}
            for location in PROBE_LOCATIONS:
                rows = [row for row in subset if row["layer"] == layer and row["location"] == location]
                result[domain][layer][location] = [{"model": row["model"], "correct": row["correct"], "rows": row["rows"], "accuracy": row["accuracy"]} for row in rows]
    return result


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_summary = load_json(V2_ROOT / "SUMMARY.json")
    source_map = {(str(row["interface"]), int(row["seed"])): row for row in source_summary["trained_arms"] if row["arm"] == "A3_AUTHORITY_POINTER_BALANCED"}
    models: Dict[Tuple[str, int], Any] = {}
    datasets: Dict[str, Any] = {}
    bundles: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    all_probe_metrics: List[Dict[str, Any]] = []
    all_probe_predictions: List[Dict[str, Any]] = []
    all_patch_rows: List[Dict[str, Any]] = []
    all_self_rows: List[Dict[str, Any]] = []
    patch_summaries: List[Dict[str, Any]] = []
    causal_access: List[Dict[str, Any]] = []
    integrity: Dict[str, Any] = {
        "retrained": False, "architecture_changed": False, "sealed_test_accessed": False,
        "checkpoint_hashes": {}, "source_split_audits": {}, "capture_fidelity": {},
        "accepted_export_reproduction": {}, "unseen_pair_invariants": {},
    }

    for interface in INTERFACES:
        datasets[interface] = build_datasets(interface)
        integrity["source_split_audits"][interface] = datasets[interface]["audit"]
        for seed in SEEDS:
            key = (interface, seed)
            checkpoint = source_checkpoint(interface, seed)
            actual_hash = sha256(checkpoint)
            expected_hash = str(source_map[key]["checkpoint_sha256"])
            if actual_hash != expected_hash:
                raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
            integrity["checkpoint_hashes"][f"{interface}|seed{seed}"] = {"path": rel(checkpoint), "sha256": actual_hash, "accepted": expected_hash}
            model = load_a3_model(checkpoint, seed)
            models[key] = model
            data = datasets[interface]
            record_groups: Dict[str, Sequence[Mapping[str, Any]]] = {"train": data["train"], **data["familiar"], "unseen": data["unseen"]}
            model_bundles: Dict[str, Dict[str, Any]] = {}
            all_pointer_rows: List[Dict[str, Any]] = []
            for domain, records in record_groups.items():
                model_bundles[domain] = capture_bundle(model, records, verify_first_batch=True)
                for row in model_bundles[domain]["pointer_rows"]:
                    row.update({"model": f"{interface}_seed{seed}", "domain": domain})
                    all_pointer_rows.append(row)
                integrity["capture_fidelity"][f"{interface}|seed{seed}|{domain}"] = model_bundles[domain]["fidelity"]
            bundles[key] = model_bundles
            integrity["accepted_export_reproduction"][f"{interface}|seed{seed}"] = compare_to_accepted_export(interface, seed, model_bundles["unseen"]["pointer_rows"], {layout: model_bundles[layout]["pointer_rows"] for layout in FAMILIAR_LAYOUTS})
            if not integrity["accepted_export_reproduction"][f"{interface}|seed{seed}"]["counterfactual_pass"] or not integrity["accepted_export_reproduction"][f"{interface}|seed{seed}"]["familiar_pass"]:
                raise RuntimeError(f"accepted baseline reproduction failed: {interface} seed {seed}")
            probe_metrics, probe_predictions = probe_diagnostics(f"{interface}_seed{seed}", model_bundles, data)
            all_probe_metrics.extend(probe_metrics); all_probe_predictions.extend(probe_predictions)
            opposite, self_rows, patch_summary = patch_diagnostics(f"{interface}_seed{seed}", model, model_bundles["unseen"], data["unseen"])
            all_patch_rows.extend(opposite); all_self_rows.extend(self_rows); patch_summaries.append(patch_summary)
            causal_access.append(causal_access_diagnostics(f"{interface}_seed{seed}", model_bundles["unseen"], data["unseen"]))
            del models[key]
            del bundles[key]
            del model_bundles
            del model
            gc.collect()
            print(json.dumps({"event": "authority_representation_model_complete", "interface": interface, "seed": seed, "probe_rows": len(probe_metrics), "patch_rows": len(opposite)}, sort_keys=True), flush=True)

    unseen_invariants = []
    for interface in INTERFACES:
        records = datasets[interface]["unseen"]
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in records:
            grouped[str(row["metadata"]["counterfactual_pair_id"])].append(row)
        for pair_id, rows in grouped.items():
            if len(rows) != 2:
                raise RuntimeError(f"unseen pair cardinality failure: {pair_id}")
            prepared = auth.prepare_authority_parser(rows)
            invariant = {
                "interface": interface, "pair_id": pair_id, "rows": len(rows),
                "operation_positions_identical": prepared[0]["_op_positions"] == prepared[1]["_op_positions"],
                "operand_positions_identical": prepared[0]["_a_pos"] == prepared[1]["_a_pos"] and prepared[0]["_b_pos"] == prepared[1]["_b_pos"],
                "authority_position_identical": prepared[0]["_authority_position"] == prepared[1]["_authority_position"],
                "target_operation_flipped": prepared[0]["metadata"]["operation"] != prepared[1]["metadata"]["operation"],
            }
            unseen_invariants.append(invariant)
    integrity["unseen_pair_invariants"] = {"pairs": len(unseen_invariants), "all_pass": all(all(value for key, value in row.items() if key not in {"interface", "pair_id", "rows"}) for row in unseen_invariants), "by_interface": {interface: sum(1 for row in unseen_invariants if row["interface"] == interface) for interface in INTERFACES}}
    if not integrity["unseen_pair_invariants"]["all_pass"]:
        raise RuntimeError("unseen pair invariant failure")
    integrity["causal_access"] = causal_access

    write_csv(OUTPUT / "AUTHORITY_PROBE_PREDICTIONS.csv", all_probe_predictions)
    write_csv(OUTPUT / "ACTIVATION_PATCH_ROWS.csv", all_patch_rows)
    write_csv(OUTPUT / "SELF_PATCH_ROWS.csv", all_self_rows)
    write_json(OUTPUT / "AUTHORITY_PROBE_RESULTS.json", {"rows": all_probe_metrics, "aggregate": aggregate_probe(all_probe_metrics)})
    write_json(OUTPUT / "ACTIVATION_PATCH_RESULTS.json", {"summaries": patch_summaries, "causal_access": causal_access})
    write_json(OUTPUT / "CAUSAL_ACCESS.json", {"models": causal_access})
    write_json(OUTPUT / "FIDELITY.json", integrity)
    write_json(OUTPUT / "PROTOCOL_MANIFEST.json", {"experiment": "crucible_authority_representation_use_20261026_v1", "protocol": rel(PROTOCOL), "interfaces": list(INTERFACES), "seeds": list(SEEDS), "familiar_layouts": list(FAMILIAR_LAYOUTS), "unseen_layout": UNSEEN_LAYOUT, "patch_sites": list(PATCH_SITES), "probe_locations": list(PROBE_LOCATIONS), "ridge_lambda": RIDGE_LAMBDA, "numerical_tolerance": NUMERICAL_TOLERANCE, "sealed_test_accessed": False, "retrained": False, "architecture_changed": False})

    # Summarize the strongest, directly comparable probe rows and patch controls.
    headline: Dict[str, Any] = {"probe": {}, "patch": {}}
    for domain in [*FAMILIAR_LAYOUTS, "unseen_reversed_authority"]:
        subset = [row for row in all_probe_metrics if row["domain"] == domain and row["layer"] == "final_norm" and row["location"] == "authority"]
        headline["probe"][domain] = [{"model": row["model"], "correct": row["correct"], "rows": row["rows"], "accuracy": row["accuracy"]} for row in subset]
    for summary in patch_summaries:
        headline["patch"][summary["model"]] = summary["sites"]

    write_json(OUTPUT / "ANALYSIS.json", {"experiment": "crucible_authority_representation_use_20261026_v1", "status": "completed", "headline": headline, "integrity": integrity, "source_checkpoints": integrity["checkpoint_hashes"]})
    report_lines = [
        "# Crucible Authority Representation-versus-Use Diagnostic v1", "",
        "Read-only hidden-state probe and activation-patching analysis of the verified A3 checkpoints.", "",
        "## Scope", "",
        "No weights were changed, no training was run, and no sealed test set was accessed. The clean counterfactual baseline and familiar v2 validation predictions were reproduced before analysis.", "",
        "## Baseline reproduction", "",
        "All four A3 checkpoint/interface/seed combinations matched the accepted pointer predictions on the clean unseen counterfactual export and on familiar validation layouts.", "",
        "## Authority-cue probe", "",
        "The complete machine-readable probe table is in `AUTHORITY_PROBE_RESULTS.json`. The probe was trained on frozen authority-training representations; probe accuracy is not evidence of causal use.", "",
        "| Domain | Hidden location | Model rows | Accuracy range |", "|---|---|---:|---:|",
    ]
    for domain in [*FAMILIAR_LAYOUTS, "unseen_reversed_authority"]:
        rows = [row for row in all_probe_metrics if row["domain"] == domain and row["layer"] == "final_norm" and row["location"] == "authority"]
        if rows:
            report_lines.append(f"| {domain} | final_norm / authority token | {sum(row['rows'] for row in rows)}/{len(rows)} model evaluations | {min(row['accuracy'] for row in rows):.1%}–{max(row['accuracy'] for row in rows):.1%} |" )
    report_lines += ["", "The operation-position probe is the more diagnostic comparison because the pointer head scores these positions:", "", "| Domain | Location | Per-model accuracy range |", "|---|---|---:|"]
    for domain in [*FAMILIAR_LAYOUTS, "unseen_reversed_authority"]:
        for location in ("op_first", "op_second"):
            rows = [row for row in all_probe_metrics if row["domain"] == domain and row["layer"] == "final_norm" and row["location"] == location]
            if rows:
                report_lines.append(f"| {domain} | final_norm / {location} | {min(row['correct'] for row in rows)}/{rows[0]['rows']}–{max(row['correct'] for row in rows)}/{rows[0]['rows']} |" )
    report_lines += [
        "", "## Activation patching", "",
        "Opposite-authority patches replace the authority-position hidden vector with its paired counterfactual counterpart. In this clean layout the authority token is later than both operation candidates, so the causal-access audit is the primary interpretation; the patch is a bounded negative control. Self-patching is the same-input numerical control.", "",
        "| Model | Site | Opposite pointer changes | Opposite patched correct | Self max logit delta | Self control |", "|---|---|---:|---:|---:|---|",
    ]
    for summary in patch_summaries:
        for site in PATCH_SITES:
            row = summary["sites"][site]
            report_lines.append(f"| {summary['model']} | {site} | {row['opposite_pointer_changed']}/{row['opposite_authority_rows']} | {row['opposite_patched_pointer_correct']}/{row['opposite_authority_rows']} | {row['self_max_abs_op_pointer_logit_delta']:.3e} | {'PASS' if row['self_fidelity_pass'] else 'FAIL'} |" )
    report_lines += ["", "## Causal-access audit", "", "| Model | Pairs | Max pointer-logit delta at operation positions | Max hidden delta at operation positions |", "|---|---:|---:|---:|"]
    for row in causal_access:
        report_lines.append(f"| {row['model']} | {row['pairs']} | {row['max_operation_pointer_logit_delta']:.3e} | {row['max_hidden_delta_at_operation_positions']:.3e} |" )
    report_lines += [
        "", "## Interpretation boundary", "",
        "A decodable authority cue would show representation, not use. A patch effect would show sensitivity under the intervention, not a unique mechanism. Familiar/unseen differences remain conditional on these layouts and cannot be generalized to real-world instruction hierarchy.", "",
        "## Artifacts", "",
        f"- [Protocol](../../{rel(PROTOCOL)})", "- [Probe results](AUTHORITY_PROBE_RESULTS.json)", "- [Probe predictions](AUTHORITY_PROBE_PREDICTIONS.csv)", "- [Activation patch results](ACTIVATION_PATCH_RESULTS.json)", "- [Activation patch rows](ACTIVATION_PATCH_ROWS.csv)", "- [Self-patch controls](SELF_PATCH_ROWS.csv)", "- [Causal-access audit](CAUSAL_ACCESS.json)", "- [Fidelity and invariants](FIDELITY.json)", "- [Analysis](ANALYSIS.json)",
    ]
    (OUTPUT / "REPORT_FINAL.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    interpretation = """# Scientific Interpretation

This report separates three claims:

1. **Representation:** whether the authority label is recoverable from frozen
   hidden states by a held-out linear probe.
2. **Behavioral use:** whether the A3 pointer changes under the clean paired
   authority reversal.
3. **Causal sensitivity:** whether replacing the authority-position hidden
   vector with the counterfactual vector changes pointer logits or selection.

The first claim does not entail the second. The third is an intervention result,
not a complete mechanistic explanation. In particular, a probe can exploit
layout or other correlated features, and a patch can alter more than one
semantically isolated factor in a nonlinear network.

The clean unseen layout places the authority field after both operation
candidates. The operation-pointer head scores hidden states at those earlier
candidate positions. Therefore, under a causal attention mask, a later
authority token has no route to those pointer states. The causal-access audit
tests this directly. Zero authority-pointer flips in this layout should not be
described by itself as proof that the model represents and ignores authority;
it is consistent with an information-access limitation imposed by sequence
ordering.

The clean counterfactual baseline is the accepted position-safe result. The
earlier padded authority export is intentionally excluded from this report.
"""
    (OUTPUT / "SCIENTIFIC_INTERPRETATION.md").write_text(interpretation, encoding="utf-8")

    manifest_paths = [PROTOCOL, OUTPUT / "PROTOCOL_MANIFEST.json", OUTPUT / "AUTHORITY_PROBE_RESULTS.json", OUTPUT / "AUTHORITY_PROBE_PREDICTIONS.csv", OUTPUT / "ACTIVATION_PATCH_RESULTS.json", OUTPUT / "ACTIVATION_PATCH_ROWS.csv", OUTPUT / "SELF_PATCH_ROWS.csv", OUTPUT / "CAUSAL_ACCESS.json", OUTPUT / "FIDELITY.json", OUTPUT / "ANALYSIS.json", OUTPUT / "REPORT_FINAL.md", OUTPUT / "SCIENTIFIC_INTERPRETATION.md"]
    write_json(OUTPUT / "HASHES.json", {rel(path): sha256(path) for path in manifest_paths})
    finalization = {"experiment": "crucible_authority_representation_use_20261026_v1", "status": "completed", "verification_pass": bool(integrity["unseen_pair_invariants"]["all_pass"] and all(value["counterfactual_pass"] and value["familiar_pass"] for value in integrity["accepted_export_reproduction"].values()) and all(value["pass"] for value in integrity["capture_fidelity"].values()) and all(row["sites"][site]["self_fidelity_pass"] for row in patch_summaries for site in PATCH_SITES) and all(row["position_path_unchanged_within_tolerance"] for row in causal_access)), "sealed_test_accessed": False, "retrained": False, "architecture_changed": False, "hashes": rel(OUTPUT / "HASHES.json")}
    write_json(OUTPUT / "FINALIZATION.json", finalization)
    # Hash finalization after all content is present; it is intentionally not in HASHES.
    print(json.dumps({"event": "authority_representation_use_complete", "output": rel(OUTPUT), "verification_pass": finalization["verification_pass"], "probe_rows": len(all_probe_metrics), "patch_rows": len(all_patch_rows)}, sort_keys=True), flush=True)
    return 0 if finalization["verification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
