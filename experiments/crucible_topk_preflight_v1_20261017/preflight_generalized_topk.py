"""Disposable, CPU-only preflight for generalized Crucible top-k routing.

This script does not load a training manifest, write checkpoints, construct an
optimizer, or perform an optimizer step.  It exercises only the versioned
development copy of model_core_generalized.py and writes validation results
beside that copy.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import torch


ROOT = Path(__file__).resolve().parent
GENERALIZED_PATH = ROOT / "model_core_generalized.py"
REFERENCE_PATH = Path("/home/timbushnell/model_core.py")
RESULTS_PATH = ROOT / "preflight_results.json"
HASHES_PATH = ROOT / "preflight_hashes.json"

VOCAB = 259
LAYERS = 4
D_MODEL = 256
HEADS = 4
LATENT_KV = 64
LATENT_Q = 64
PE_DIM = 32
SEQ_LEN = 8
BATCH = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(module: Any, experts: int, d_ff: int, top_k: int):
    return module.BraidPrototypeLM(
        VOCAB,
        LAYERS,
        D_MODEL,
        HEADS,
        LATENT_KV,
        LATENT_Q,
        PE_DIM,
        experts,
        d_ff,
        top_k=top_k,
    )


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise AssertionError(f"nonfinite tensor: {name}")


def run_training_path_probe(
    module: Any,
    experts: int,
    d_ff: int,
    top_k: int,
) -> Dict[str, Any]:
    torch.manual_seed(9000 + experts + top_k)
    model = build_model(module, experts, d_ff, top_k)
    model.train()

    tokens = torch.randint(
        3,
        VOCAB,
        (BATCH, SEQ_LEN),
        dtype=torch.long,
    )
    seq_idx = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    seq_idx = seq_idx.expand(BATCH, -1)
    valid_tokens = int(tokens.ne(0).sum().item())
    before = parameter_digest(model)

    module.ResearchMoETelemetry.reset_step_buffer()
    logits = model(
        tokens,
        seq_idx,
        current_branch="B",
        telemetry_class=module.ResearchMoETelemetry,
    )
    assert_finite_tensor("logits", logits)

    auxiliary_losses = [layer.moe.last_aux_loss for layer in model.layers]
    for layer_index, value in enumerate(auxiliary_losses):
        assert_finite_tensor(f"layer_{layer_index}_auxiliary_loss", value)

    loss = logits.float().square().mean()
    loss = loss + 0.01 * torch.stack(auxiliary_losses).mean()
    assert_finite_tensor("probe_loss", loss)
    loss.backward()

    gradient_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient_count += 1
        assert_finite_tensor(f"gradient:{name}", parameter.grad)

    after = parameter_digest(model)
    if before != after:
        raise AssertionError("parameters changed without an optimizer step")

    summary = module.ResearchMoETelemetry.get_summary("B")
    layer_checks: Dict[str, Any] = {}
    for layer_index in range(LAYERS):
        layer_id = f"layer_{layer_index}_moe"
        accumulator = module.ResearchMoETelemetry._accumulators["B"][layer_id]
        counts = accumulator["hard_assignments"]
        count_total = int(counts.sum().item())
        expected_count_total = valid_tokens * top_k
        if count_total != expected_count_total:
            raise AssertionError(
                f"{layer_id}: count total {count_total} != "
                f"{expected_count_total}"
            )
        if counts.numel() != experts:
            raise AssertionError(
                f"{layer_id}: count width {counts.numel()} != {experts}"
            )
        point = summary[layer_id]
        if point["top_k"] != top_k:
            raise AssertionError(
                f"{layer_id}: telemetry top_k {point['top_k']} != {top_k}"
            )
        if not all(
            value is not None and value == value
            for value in (
                point["global_hard_cv"],
                point["max_min_hard_ratio"],
                point["normalized_entropy"],
            )
        ):
            raise AssertionError(f"{layer_id}: nonfinite routing summary")
        layer_checks[layer_id] = {
            "top_k": top_k,
            "valid_tokens": valid_tokens,
            "hard_assignment_count": count_total,
            "expected_hard_assignment_count": expected_count_total,
            "count_shape": list(counts.shape),
            "router_cv": float(point["global_hard_cv"]),
            "load_ratio": float(point["max_min_hard_ratio"]),
            "normalized_entropy": float(point["normalized_entropy"]),
        }

    return {
        "experts": experts,
        "top_k": top_k,
        "d_ff": d_ff,
        "parameter_count": parameter_count(model),
        "parameter_digest_unchanged_without_optimizer": before == after,
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().item()),
        "gradient_parameter_count": gradient_count,
        "layers": layer_checks,
    }


def run_reference_compatibility(
    reference_module: Any,
    generalized_module: Any,
) -> Dict[str, Any]:
    torch.manual_seed(12345)
    reference = reference_module.BraidPrototypeLM(
        VOCAB,
        LAYERS,
        D_MODEL,
        HEADS,
        LATENT_KV,
        LATENT_Q,
        PE_DIM,
        4,
        3447,
    ).eval()
    generalized = build_model(generalized_module, 4, 3447, 2).eval()
    generalized.load_state_dict(reference.state_dict(), strict=True)

    tokens = torch.randint(3, VOCAB, (BATCH, SEQ_LEN), dtype=torch.long)
    seq_idx = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    seq_idx = seq_idx.expand(BATCH, -1)
    with torch.no_grad():
        reference_logits = reference(tokens, seq_idx, current_branch="A")
        generalized_logits = generalized(tokens, seq_idx, current_branch="A")
    difference = (reference_logits - generalized_logits).abs()
    max_abs = float(difference.max().item())
    if max_abs > 1e-6:
        raise AssertionError(
            f"E=4/top-2 compatibility mismatch: max_abs={max_abs}"
        )
    return {
        "reference_parameter_count": parameter_count(reference),
        "generalized_parameter_count": parameter_count(generalized),
        "max_abs_logit_difference": max_abs,
        "compatible_within_tolerance": True,
    }


def main() -> int:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    generalized = load_module(GENERALIZED_PATH, "crucible_generalized_model")
    reference = load_module(REFERENCE_PATH, "crucible_reference_model")

    results: Dict[str, Any] = {
        "protocol": "crucible_generalized_topk_preflight_v1",
        "device": "cpu",
        "training_started": False,
        "optimizer_constructed": False,
        "optimizer_step_count": 0,
        "checkpoint_created": False,
        "original_files_modified": False,
        "source_hashes": {
            "reference_model_core_sha256": sha256_file(REFERENCE_PATH),
            "generalized_model_core_sha256": sha256_file(GENERALIZED_PATH),
        },
        "reference_compatibility": run_reference_compatibility(
            reference, generalized
        ),
        "probes": [],
    }

    # Exact parameter geometries for the first proposed fixed-active gate.
    for experts, d_ff, top_k in (
        (1, 6894, 1),
        (2, 3447, 2),
        (4, 3447, 2),
    ):
        results["probes"].append(
            run_training_path_probe(generalized, experts, d_ff, top_k)
        )

    # A small E=8 path confirms that the implementation is not accidentally
    # restricted to the original four-expert width.
    small = generalized.BraidPrototypeLM(
        32,
        2,
        32,
        4,
        16,
        16,
        8,
        8,
        64,
        top_k=2,
    ).eval()
    small_tokens = torch.randint(3, 32, (2, 8), dtype=torch.long)
    small_seq_idx = torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, -1)
    with torch.no_grad():
        small_logits = small(
            small_tokens,
            small_seq_idx,
            current_branch="A",
        )
    assert_finite_tensor("small_E8_logits", small_logits)
    results["small_E8_top2_forward"] = {
        "parameter_count": parameter_count(small),
        "logits_shape": list(small_logits.shape),
        "finite": True,
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {
        "preflight_generalized_topk.py": sha256_file(Path(__file__).resolve()),
        "model_core_generalized.py": sha256_file(GENERALIZED_PATH),
        "reference_model_core.py": sha256_file(REFERENCE_PATH),
        "preflight_results.json": sha256_file(RESULTS_PATH),
    }
    HASHES_PATH.write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    print("preflight_hashes", json.dumps(hashes, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
