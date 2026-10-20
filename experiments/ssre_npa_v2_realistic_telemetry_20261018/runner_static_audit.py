#!/usr/bin/env python3
"""Read-only AST/source audit for the live Crucible E16 runner path.

This script deliberately does not import Torch, import the runner, construct a
model, touch CUDA, or execute training. It records source hashes and structural
signals that identify likely data/dispatch/synchronization boundaries for the
future SSRE-NPA adapter.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_RUNNER = Path(
    "/home/timbushnell/crucible_context_gate_v1_20261018/shared/context_runner.py"
)
DEFAULT_MODEL = Path(
    "/home/timbushnell/crucible_context_gate_v1_20261018/shared/model_core.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return "<dynamic>"


class SignalVisitor(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.calls: Dict[str, int] = {}
        self.call_lines: Dict[str, List[int]] = {}
        self.data_loader_calls: List[Dict[str, Any]] = []
        self.device_transfer_calls: List[Dict[str, Any]] = []
        self.loops: List[Dict[str, Any]] = []
        self.method_names: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.method_names.append(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> Any:
        name = dotted(node.func)
        self.calls[name] = self.calls.get(name, 0) + 1
        self.call_lines.setdefault(name, []).append(node.lineno)
        if name.endswith("DataLoader"):
            keywords = {item.arg: ast.unparse(item.value) for item in node.keywords if item.arg}
            self.data_loader_calls.append(
                {
                    "line": node.lineno,
                    "keywords": keywords,
                    "num_workers_explicit": "num_workers" in keywords,
                    "pin_memory_explicit": "pin_memory" in keywords,
                    "persistent_workers_explicit": "persistent_workers" in keywords,
                }
            )
        if name.endswith(".to"):
            keywords = {item.arg: ast.unparse(item.value) for item in node.keywords if item.arg}
            self.device_transfer_calls.append(
                {
                    "line": node.lineno,
                    "keywords": keywords,
                    "non_blocking_explicit": "non_blocking" in keywords,
                }
            )
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        target = ast.unparse(node.target)
        if any(term in target.lower() for term in ("expert", "rank", "microbatch", "validation")):
            self.loops.append(
                {
                    "line": node.lineno,
                    "target": target,
                    "iterable": ast.unparse(node.iter),
                }
            )
        self.generic_visit(node)


def audit_file(path: Path) -> Dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = SignalVisitor(source)
    visitor.visit(tree)
    lines = source.count("\n") + (0 if source.endswith("\n") else 1)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "lines": lines,
        "calls": visitor.calls,
        "call_lines": visitor.call_lines,
        "data_loader_calls": visitor.data_loader_calls,
        "device_transfer_calls": visitor.device_transfer_calls,
        "relevant_loops": visitor.loops,
        "method_count": len(visitor.method_names),
        "methods": visitor.method_names,
    }


def build_report(runner: Path, model: Path) -> Dict[str, Any]:
    runner_audit = audit_file(runner)
    model_audit = audit_file(model)
    runner_calls = runner_audit["calls"]
    model_calls = model_audit["calls"]

    observations = [
        {
            "id": "dataset_full_read",
            "severity": "medium",
            "evidence": "ImmutableBinaryDataset.__init__ calls read() on the complete binary payload before exposing samples.",
            "source": str(runner),
            "signals": ["handle.read", "torch.frombuffer", "bytearray"],
            "implication": "A prepared indexed/mmap adapter can remove repeated sample slicing and reduce host-copy pressure, but must preserve hashes and geometry.",
        },
        {
            "id": "per_sample_clone_and_stack",
            "severity": "high",
            "evidence": "Training constructs each microbatch with a Python list of dataset[index] calls followed by torch.stack; __getitem__ clones both x and y.",
            "source": str(runner),
            "signals": ["torch.stack", "Tensor.clone"],
            "implication": "This is a primary candidate for precomputed contiguous batch assembly or bounded worker preparation.",
        },
        {
            "id": "synchronous_device_transfer",
            "severity": "medium",
            "evidence": "Training and validation call .to(device) without an explicit non_blocking argument.",
            "source": str(runner),
            "signals": ["Tensor.to"],
            "implication": "Pinned staging and nonblocking handoff are future adapter questions; they must be measured, not assumed beneficial.",
        },
        {
            "id": "validation_single_process_loader",
            "severity": "medium",
            "evidence": "The validation DataLoader omits num_workers, pin_memory and persistent_workers.",
            "source": str(runner),
            "signals": ["DataLoader"],
            "implication": "Validation timing may include serial host assembly and should be separated from training throughput.",
        },
        {
            "id": "source_integrity_reread",
            "severity": "medium",
            "evidence": "Each trial re-reads and validates source JSONL records and provenance counts before constructing binary datasets.",
            "source": str(runner),
            "signals": ["read_jsonl_source_lengths", "assert_file_sha256_integrity"],
            "implication": "This is appropriate for fail-closed launch validation but should be excluded from per-step throughput claims.",
        },
        {
            "id": "explicit_cuda_barriers",
            "severity": "medium",
            "evidence": "The runner explicitly synchronizes CUDA before checkpoint telemetry and after validation.",
            "source": str(runner),
            "signals": ["torch.cuda.synchronize"],
            "implication": "These barriers are valid for interval timing but must be accounted for separately from steady-state step timing.",
        },
        {
            "id": "durable_metrics_flush",
            "severity": "low",
            "evidence": "The runner calls os.fsync after writing each metric row.",
            "source": str(runner),
            "signals": ["os.fsync"],
            "implication": "Durability is desirable, but metric-write latency should be recorded outside the model-step critical path where possible.",
        },
        {
            "id": "expert_dispatch_loop",
            "severity": "high",
            "evidence": "Each MoE layer loops over every expert and top-k rank, selects tokens with boolean masks, and evaluates an expert only when selected.",
            "source": str(model),
            "signals": ["for expert_index", "for rank", "record_routing", "record_expert_output"],
            "implication": "Routing/dispatch and telemetry overhead need independent timing fields; CPU data scheduling alone may not solve GPU under-utilization.",
        },
        {
            "id": "gpu_telemetry_accumulation",
            "severity": "high",
            "evidence": "Routing telemetry uses bincount/index_add_ accumulation and expert-output norm collection during the model forward path.",
            "source": str(model),
            "signals": ["torch.bincount", "index_add_", "record_expert_output"],
            "implication": "A representative telemetry-heavy workload is required before deciding whether telemetry belongs on the critical path.",
        },
    ]

    return {
        "schema_version": "ssre-npa-runner-static-audit-v2",
        "scope": "read-only structural audit; no Torch/CUDA import or execution",
        "runner": runner_audit,
        "model": model_audit,
        "observations": observations,
        "profile_boundary": {
            "measured": False,
            "reason": "This artifact intentionally performs no live model or GPU benchmark.",
            "next_measurement": "Attach the v2 event collector to deterministic scheduler packet boundaries and separately time preparation, H2D, forward/backward, synchronization, telemetry and commit phases.",
        },
        "source_summary": {
            "runner_cuda_sync_calls": runner_calls.get("torch.cuda.synchronize", 0),
            "runner_stack_calls": runner_calls.get("torch.stack", 0),
            "runner_fsync_calls": runner_calls.get("os.fsync", 0),
            "model_bincount_calls": sum(
                count for name, count in model_calls.items() if name.endswith("bincount")
            ),
            "model_index_add_calls": sum(
                count for name, count in model_calls.items() if name.endswith("index_add_")
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Crucible runner static audit")
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.runner, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = build_report(args.runner.resolve(), args.model.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "runner_sha256": report["runner"]["sha256"], "model_sha256": report["model"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

