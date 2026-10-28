#!/usr/bin/env python3
"""Evaluate one trained Crucible foundation checkpoint with one-pass and OP."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
THROUGHPUT = ROOT / "experiments" / "crucible_heterogeneous_training_throughput_20261029"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(THROUGHPUT))
sys.path.insert(0, str(TOOLS))

from benchmark import make_model, state_digest  # noqa: E402
from run_crucible_op_reasoning_foundation_pilot_20261029 import evaluate_policy, load_jsonl  # noqa: E402
from run_crucible_foundation_training_20261029 import foundation_model_config  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, default=Path("/home/timbushnell/crucible_reasoning_pilot_20261029/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=("tiny", "small", "120m"), default="small")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-records", type=int, default=304)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use host access or --device cpu")
    device = torch.device(args.device)
    cfg = foundation_model_config(args.model_size, activation_checkpointing=False)
    model = make_model(cfg, device, seed=42)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = load_jsonl(args.test_jsonl.resolve())[: args.max_records]
    policies = [
        evaluate_policy(model, rows, "one_pass", device, args.context, len(rows)),
        evaluate_policy(model, rows, "fixed_depth_2", device, args.context, len(rows)),
        evaluate_policy(model, rows, "op_external", device, args.context, len(rows)),
    ]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "crucible-foundation-checkpoint-eval-v1-20261029",
        "classification": "post_foundation_checkpoint_op_comparison",
        "checkpoint": {"sha256": sha256_file(args.checkpoint.resolve()), "step": checkpoint.get("step"), "state_sha256": state_digest(model)},
        "data": {"test_jsonl_sha256": sha256_file(args.test_jsonl.resolve()), "records": len(rows), "gold_used_only_after_training": True},
        "model": {"family": "Crucible latent-attention + SwiGLU dense", "size": args.model_size, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "context": args.context},
        "policies": [{key: value for key, value in policy.items() if key != "traces"} for policy in policies],
        "limitations": [
            "The OP mechanism is external and bounded; it is not internalized in model weights.",
            "This is a single checkpoint and seed, not a preregistered scientific comparison.",
            "Gold answers are used by the evaluator after training for verification and are not input to proposal generation.",
        ],
    }
    (output / "CHECKPOINT_OP_EVAL.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "CHECKPOINT_OP_TRACES.jsonl").open("w", encoding="utf-8") as handle:
        for policy in policies:
            for trace in policy["traces"]:
                handle.write(json.dumps(trace, sort_keys=True) + "\n")
    hashes = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    (output / "HASHES.json").write_text(json.dumps({"files": hashes}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
