#!/usr/bin/env python3
"""CPU-only ARC portfolio expansion and math identifiability audit.

Track A expands the deterministic ARC solver portfolio and measures its
bounded-search oracle ceiling. Track B reuses C1d's frozen math tasks and
solvers to measure how much of the apparent selection gap is observable before
the solver is chosen. No neural model or learned selector is trained here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
C1D_ROOT = ROOT / "experiments" / "crucible_c1d_arc_logic_l100000_20261021"
DEFAULT_ARC_ROOT = C1D_ROOT / "arc_data"
DEFAULT_OUTPUT = ROOT / "experiments" / "crucible_c1e_arc_portfolio_math_20261021"
PROTOCOL = ROOT / "experiments" / "crucible_c1e_arc_portfolio_math_20261021" / "PROTOCOL.md"
EXPERIMENT = "crucible_c1e_arc_portfolio_math_20261021"

BASE_ACTIONS = (
    "arc_identity",
    "arc_color_map",
    "arc_rot90",
    "arc_rot180",
    "arc_rot270",
    "arc_flip_h",
    "arc_flip_v",
    "arc_transpose",
    "arc_crop_nonzero",
    "arc_scale",
)
OBJECT_ACTIONS = (
    "arc_crop_largest",
    "arc_crop_smallest",
    "arc_crop_tallest",
    "arc_crop_widest",
    "arc_remove_largest",
    "arc_remove_smallest",
    "arc_fill_bbox_largest",
    "arc_outline_bbox_largest",
    "arc_duplicate_h",
    "arc_duplicate_v",
    "arc_tile_2x2",
    "arc_recolor_nonzero",
    *(f"arc_crop_color_{color}" for color in range(1, 10)),
)
STRUCTURAL_ACTIONS = tuple(action for action in BASE_ACTIONS if action not in {"arc_color_map", "arc_scale"}) + OBJECT_ACTIONS
COMPOSABLE_ACTIONS = tuple(action for action in STRUCTURAL_ACTIONS if action != "arc_identity")
LEVELS = ("baseline", "objects", "compose2")

Grid = List[List[int]]


@dataclass(frozen=True)
class Problem:
    problem_id: str
    payload: Mapping[str, Any]
    split: str
    source: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Program:
    name: str
    operations: Tuple[str, ...]
    depth: int
    cost: float


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def grid_area(grid: Grid) -> int:
    return len(grid) * (len(grid[0]) if grid else 0)


def rotate90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])]


def rotate180(grid: Grid) -> Grid:
    return [list(row[::-1]) for row in grid[::-1]]


def rotate270(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)][::-1]


def transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def crop_nonzero(grid: Grid) -> Optional[Grid]:
    cells = [(row, col) for row, values in enumerate(grid) for col, value in enumerate(values) if value != 0]
    if not cells:
        return None
    row0 = min(row for row, _ in cells)
    row1 = max(row for row, _ in cells)
    col0 = min(col for _, col in cells)
    col1 = max(col for _, col in cells)
    return [row[col0:col1 + 1] for row in grid[row0:row1 + 1]]


def infer_color_map(pairs: Sequence[Mapping[str, Any]]) -> Optional[Dict[int, int]]:
    mapping: Dict[int, int] = {}
    for pair in pairs:
        source = pair["input"]
        target = pair["output"]
        if len(source) != len(target) or any(len(left) != len(right) for left, right in zip(source, target)):
            return None
        for source_row, target_row in zip(source, target):
            for source_value, target_value in zip(source_row, target_row):
                previous = mapping.get(source_value)
                if previous is not None and previous != target_value:
                    return None
                mapping[source_value] = target_value
    return mapping or None


def apply_color_map(grid: Grid, mapping: Mapping[int, int]) -> Grid:
    return [[mapping.get(value, value) for value in row] for row in grid]


def infer_scale(pairs: Sequence[Mapping[str, Any]]) -> Optional[int]:
    scale: Optional[int] = None
    for pair in pairs:
        source = pair["input"]
        target = pair["output"]
        if not source or not source[0] or len(target) % len(source) or len(target[0]) % len(source[0]):
            return None
        row_scale = len(target) // len(source)
        col_scale = len(target[0]) // len(source[0])
        if row_scale != col_scale or row_scale < 1 or any(len(row) != len(target[0]) for row in target):
            return None
        if scale is not None and scale != row_scale:
            return None
        scale = row_scale
        for row, source_row in enumerate(source):
            for col, value in enumerate(source_row):
                for target_row in range(row * scale, (row + 1) * scale):
                    if any(target[target_row][target_col] != value for target_col in range(col * scale, (col + 1) * scale)):
                        return None
    return scale


def components(grid: Grid) -> List[Dict[str, Any]]:
    """Return four-connected same-colour non-background components."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    seen: set[Tuple[int, int]] = set()
    found: List[Dict[str, Any]] = []
    for row in range(height):
        for col in range(width):
            if grid[row][col] == 0 or (row, col) in seen:
                continue
            color = grid[row][col]
            stack = [(row, col)]
            seen.add((row, col))
            cells: List[Tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                cells.append((current_row, current_col))
                for next_row, next_col in ((current_row - 1, current_col), (current_row + 1, current_col), (current_row, current_col - 1), (current_row, current_col + 1)):
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in seen or grid[next_row][next_col] != color:
                        continue
                    seen.add((next_row, next_col))
                    stack.append((next_row, next_col))
            row0 = min(item[0] for item in cells)
            row1 = max(item[0] for item in cells)
            col0 = min(item[1] for item in cells)
            col1 = max(item[1] for item in cells)
            found.append({"cells": cells, "color": color, "row0": row0, "row1": row1, "col0": col0, "col1": col1})
    return found


def selected_component(grid: Grid, selector: str) -> Optional[Dict[str, Any]]:
    items = components(grid)
    if not items:
        return None
    def key(item: Mapping[str, Any]) -> Tuple[int, int, int, int]:
        area = len(item["cells"])
        height = item["row1"] - item["row0"] + 1
        width = item["col1"] - item["col0"] + 1
        if selector == "smallest":
            return (area, height * width, item["row0"], item["col0"])
        if selector == "tallest":
            return (-height, -area, item["row0"], item["col0"])
        if selector == "widest":
            return (-width, -area, item["row0"], item["col0"])
        return (-area, -(height * width), item["row0"], item["col0"])
    return sorted(items, key=key)[0]


def crop_component(grid: Grid, item: Optional[Mapping[str, Any]]) -> Optional[Grid]:
    if item is None:
        return None
    row0, row1 = int(item["row0"]), int(item["row1"])
    col0, col1 = int(item["col0"]), int(item["col1"])
    output = [[0 for _ in range(col1 - col0 + 1)] for _ in range(row1 - row0 + 1)]
    for row, col in item["cells"]:
        output[row - row0][col - col0] = grid[row][col]
    return output


def remove_component(grid: Grid, item: Optional[Mapping[str, Any]]) -> Grid:
    output = [row[:] for row in grid]
    if item is not None:
        for row, col in item["cells"]:
            output[row][col] = 0
    return output


def bbox_component(grid: Grid, item: Optional[Mapping[str, Any]], mode: str) -> Optional[Grid]:
    if item is None:
        return None
    row0, row1 = int(item["row0"]), int(item["row1"])
    col0, col1 = int(item["col0"]), int(item["col1"])
    color = int(item["color"])
    output = [[0 for _ in range(col1 - col0 + 1)] for _ in range(row1 - row0 + 1)]
    for row in range(len(output)):
        for col in range(len(output[0])):
            if mode == "fill":
                output[row][col] = color
            elif row in {0, len(output) - 1} or col in {0, len(output[0]) - 1}:
                output[row][col] = color
    return output


def apply_primitive(action: str, pairs: Sequence[Mapping[str, Any]], grid: Grid) -> Optional[Grid]:
    if action == "arc_identity":
        return [row[:] for row in grid]
    if action == "arc_color_map":
        mapping = infer_color_map(pairs)
        return None if mapping is None else apply_color_map(grid, mapping)
    if action == "arc_rot90":
        return rotate90(grid)
    if action == "arc_rot180":
        return rotate180(grid)
    if action == "arc_rot270":
        return rotate270(grid)
    if action == "arc_flip_h":
        return [row[::-1] for row in grid]
    if action == "arc_flip_v":
        return grid[::-1]
    if action == "arc_transpose":
        return transpose(grid)
    if action == "arc_crop_nonzero":
        return crop_nonzero(grid)
    if action == "arc_scale":
        scale = infer_scale(pairs)
        return None if scale is None else [[value for value in row for _ in range(scale)] for row in grid for _ in range(scale)]
    if action.startswith("arc_crop_color_"):
        color = int(action.rsplit("_", 1)[-1])
        cells = [(row, col) for row, values in enumerate(grid) for col, value in enumerate(values) if value == color]
        if not cells:
            return None
        item = {"cells": cells, "row0": min(row for row, _ in cells), "row1": max(row for row, _ in cells), "col0": min(col for _, col in cells), "col1": max(col for _, col in cells)}
        return crop_component(grid, item)
    if action.startswith("arc_crop_") and action.rsplit("_", 1)[-1] in {"largest", "smallest", "tallest", "widest"}:
        return crop_component(grid, selected_component(grid, action.rsplit("_", 1)[-1]))
    if action == "arc_remove_largest":
        return remove_component(grid, selected_component(grid, "largest"))
    if action == "arc_remove_smallest":
        return remove_component(grid, selected_component(grid, "smallest"))
    if action == "arc_fill_bbox_largest":
        return bbox_component(grid, selected_component(grid, "largest"), "fill")
    if action == "arc_outline_bbox_largest":
        return bbox_component(grid, selected_component(grid, "largest"), "outline")
    if action == "arc_duplicate_h":
        return [row + row for row in grid]
    if action == "arc_duplicate_v":
        return grid + [row[:] for row in grid]
    if action == "arc_tile_2x2":
        doubled = [row + row for row in grid]
        return doubled + [row[:] for row in doubled]
    if action == "arc_recolor_nonzero":
        return [[1 if value != 0 else 0 for value in row] for row in grid]
    raise ValueError(action)


def apply_program(program: Program, pairs: Sequence[Mapping[str, Any]], grid: Grid) -> Optional[Grid]:
    current: Optional[Grid] = [row[:] for row in grid]
    for operation in program.operations:
        if current is None:
            return None
        current = apply_primitive(operation, pairs, current)
    return current


def programs_for_level(level: str) -> List[Program]:
    if level == "baseline":
        return [Program(action, (action,), 1, 1.0) for action in BASE_ACTIONS]
    one_step = list(BASE_ACTIONS) + list(OBJECT_ACTIONS)
    programs = [Program(action, (action,), 1, 1.0 if action in BASE_ACTIONS else 1.5) for action in one_step]
    if level == "objects":
        return programs
    if level != "compose2":
        raise ValueError(level)
    programs.extend(
        Program(f"compose:{first}+{second}", (first, second), 2, 2.0)
        for first in COMPOSABLE_ACTIONS
        for second in COMPOSABLE_ACTIONS
        if first != second or first not in {"arc_crop_nonzero", "arc_identity"}
    )
    names = set()
    unique: List[Program] = []
    for program in programs:
        if program.name not in names:
            names.add(program.name)
            unique.append(program)
    return unique


def load_arc_tasks(arc_root: Path, output: Path) -> Tuple[List[Problem], List[Problem], List[Problem], Dict[str, Any]]:
    training_root = arc_root / "training"
    evaluation_root = arc_root / "evaluation"
    if not training_root.is_dir() or not evaluation_root.is_dir():
        raise RuntimeError(f"ARC copied data not found under {arc_root}")
    training_files = sorted(training_root.glob("*.json"))
    evaluation_files = sorted(evaluation_root.glob("*.json"))
    if len(training_files) != 400 or len(evaluation_files) != 400:
        raise RuntimeError(f"expected 400/400 ARC files, got {len(training_files)}/{len(evaluation_files)}")
    destination_root = output / "arc_data"
    index: Dict[str, Any] = {"source": "https://github.com/fchollet/ARC-AGI", "source_commit": None, "training": [], "evaluation": []}
    c1d_manifest = C1D_ROOT / "MANIFEST.json"
    if c1d_manifest.is_file():
        index["source_commit"] = json.loads(c1d_manifest.read_text(encoding="utf-8"))["arc_source"]["source_commit"]
    groups: Dict[str, List[Problem]] = {"train": [], "validation": [], "evaluation": []}
    for split_name, files in (("training", training_files), ("evaluation", evaluation_files)):
        for position, source_file in enumerate(files):
            task = json.loads(source_file.read_text(encoding="utf-8"))
            target = destination_root / split_name / source_file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target)
            index[split_name].append({"task_id": source_file.stem, "path": str(target.relative_to(output)), "sha256": sha256_file(target), "train_pairs": len(task["train"]), "test_pairs": len(task["test"])})
            problem = Problem(f"arc-{split_name}-{source_file.stem}", task, "test" if split_name == "evaluation" else "train", "official_arc_agi_1", {"task_id": source_file.stem, "train_pairs": len(task["train"]), "test_pairs": len(task["test"])})
            if split_name == "evaluation":
                groups["evaluation"].append(problem)
            elif position % 5 == 0:
                groups["validation"].append(Problem(problem.problem_id, problem.payload, "validation", problem.source, problem.metadata))
            else:
                groups["train"].append(problem)
    return groups["train"], groups["validation"], groups["evaluation"], index


def score_programs(problem: Problem, programs: Sequence[Program]) -> Dict[str, Any]:
    pairs = problem.payload["train"]
    test_pairs = problem.payload["test"]
    candidates: List[Dict[str, Any]] = []
    for program in programs:
        demo_fit = True
        for pair in pairs:
            prediction = apply_program(program, pairs, pair["input"])
            if prediction != pair["output"]:
                demo_fit = False
                break
        correct = False
        if demo_fit:
            correct = all(apply_program(program, pairs, pair["input"]) == pair["output"] for pair in test_pairs)
        candidates.append({"name": program.name, "depth": program.depth, "cost": program.cost, "demo_fit": demo_fit, "correct": correct})
    consistent = [candidate for candidate in candidates if candidate["demo_fit"]]
    correct = [candidate for candidate in consistent if candidate["correct"]]
    return {"problem_id": problem.problem_id, "candidate_count": len(candidates), "consistent_count": len(consistent), "coverage": bool(consistent), "solved": bool(correct), "best_correct_cost": min((item["cost"] for item in correct), default=None), "correct_programs": [item["name"] for item in correct]}


def portfolio_report(problems: Sequence[Problem], level: str, programs: Sequence[Program]) -> Dict[str, Any]:
    rows = [score_programs(problem, programs) for problem in problems]
    return {
        "level": level,
        "program_count": len(programs),
        "tasks": len(rows),
        "coverage": sum(row["coverage"] for row in rows) / max(1, len(rows)),
        "oracle_accuracy": sum(row["solved"] for row in rows) / max(1, len(rows)),
        "mean_candidate_count": sum(row["candidate_count"] for row in rows) / max(1, len(rows)),
        "mean_consistent_count": sum(row["consistent_count"] for row in rows) / max(1, len(rows)),
        "mean_best_correct_cost": sum(row["best_correct_cost"] or 0.0 for row in rows) / max(1, sum(row["best_correct_cost"] is not None for row in rows)),
        "solved_task_ids": [row["problem_id"] for row in rows if row["solved"]],
    }


def import_c1d() -> Any:
    sys.path.insert(0, str(ROOT))
    return importlib.import_module("tools.run_crucible_c1d_arc_logic_l100000")


def math_observable_signature(problem: Any) -> Tuple[Any, ...]:
    cases = problem.payload["train_cases"]
    operation_names = tuple(sorted({operation for case in cases for operation, _ in case["operations"]}))
    operation_lengths = tuple(sorted(len(case["operations"]) for case in cases))
    return (len(cases), operation_names, operation_lengths, bool(problem.metadata.get("mixed_test")))


def math_c1d_feature_signature(c1d: Any, problem: Any, candidates: Mapping[str, Any]) -> Tuple[float, ...]:
    return tuple(c1d.feature_vector(problem, candidates))


def math_group_report(items: Sequence[Mapping[str, Any]], key_name: str) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        groups[canonical(item[key_name])].append(item)
    total = len(items)
    oracle_success = sum(bool(item["oracle_correct"]) for item in items)
    observable_upper = 0
    fully_identifiable = 0
    group_rows = []
    for encoded, group in groups.items():
        action_counts: Counter[str] = Counter()
        for item in group:
            for action in item["correct_eligible_actions"]:
                action_counts[action] += 1
        best_action, best_count = (action_counts.most_common(1)[0] if action_counts else (None, 0))
        group_oracle = sum(bool(item["oracle_correct"]) for item in group)
        observable_upper += best_count
        if group_oracle == len(group) and best_count == len(group):
            fully_identifiable += len(group)
        group_rows.append({"key": encoded, "tasks": len(group), "oracle_correct": group_oracle, "best_action": best_action, "best_action_correct": best_count, "action_counts": dict(sorted(action_counts.items()))})
    return {
        "group_count": len(groups),
        "tasks": total,
        "portfolio_oracle_accuracy": oracle_success / max(1, total),
        "observable_policy_upper_bound": observable_upper / max(1, total),
        "selection_gap_unavoidable_from_features": (oracle_success - observable_upper) / max(1, total),
        "fully_identifiable_task_fraction": fully_identifiable / max(1, total),
        "groups": group_rows,
    }


def math_identifiability(c1d: Any) -> Dict[str, Any]:
    problems = c1d.generated_math(20261032, "test", 300)
    items: List[Dict[str, Any]] = []
    for problem in problems:
        candidates = c1d.candidates_for(problem)
        frozen = {action: candidates[action] for action in c1d.MATH_ACTIONS}
        correct_eligible = [action for action, candidate in frozen.items() if candidate.demo_fit and candidate.correct]
        item = {
            "problem_id": problem.problem_id,
            "semantics": problem.metadata["semantics"],
            "correct_eligible_actions": correct_eligible,
            "oracle_correct": bool(correct_eligible),
            "c1d_features": list(math_c1d_feature_signature(c1d, problem, candidates)),
            "observable_signature": list(math_observable_signature(problem)),
            "eligible_actions": [action for action, candidate in frozen.items() if candidate.demo_fit],
        }
        items.append(item)
    return {
        "tasks": len(items),
        "semantics_counts": dict(sorted(Counter(item["semantics"] for item in items).items())),
        "solver_oracle_accuracy": sum(item["oracle_correct"] for item in items) / len(items),
        "c1d_feature_grouping": math_group_report(items, "c1d_features"),
        "demonstration_signature_grouping": math_group_report(items, "observable_signature"),
        "interpretation": "Training demonstrations contain only add/sub operations, so left-to-right and precedence semantics are observationally identical before the held-out multiplication cases.",
        "items": items,
    }


def source_hashes() -> Dict[str, str]:
    paths = [
        ROOT / "tools" / "run_crucible_c1e_arc_portfolio_math_20261021.py",
        PROTOCOL,
        C1D_ROOT / "RESULTS_SUMMARY.md",
        C1D_ROOT / "HASHES.json",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def summary(result: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1e ARC portfolio expansion and math identifiability",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; no GPU, cloud or Crucible neural training.",
        "",
        "## Verdict",
        "",
        "- Track A measures solver capability before any controller retraining.",
        "- Track B measures the observable ceiling of the frozen C1d math solver choice.",
        "- The C1d-L100000 experiment and artifacts remain unchanged.",
        "",
        "## Track A — ARC public evaluation",
        "",
        "| Search level | Programs | Demonstration coverage | Portfolio oracle accuracy | Mean candidates |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for level in LEVELS:
        item = result["arc"]["evaluation"][level]
        lines.append(f"| `{level}` | {item['program_count']} | {item['coverage']:.4f} | {item['oracle_accuracy']:.4f} | {item['mean_candidate_count']:.1f} |")
    lines.extend([
        "",
        "The oracle only counts exact programs that fit every demonstration. No selector was trained in this track.",
        "",
        "## Track B — frozen math identifiability",
        "",
        f"- Frozen solver oracle accuracy: **{result['math']['solver_oracle_accuracy']:.4f}**.",
        f"- C1d-feature observable upper bound: **{result['math']['c1d_feature_grouping']['observable_policy_upper_bound']:.4f}**.",
        f"- Demonstration-signature observable upper bound: **{result['math']['demonstration_signature_grouping']['observable_policy_upper_bound']:.4f}**.",
        f"- Semantics counts: `{json.dumps(result['math']['semantics_counts'], sort_keys=True)}`.",
        "- The generated demonstrations expose only addition and subtraction; multiplication is introduced only in held-out cases. This makes left-versus-precedence selection unidentifiable from the supplied demonstrations.",
        "",
        "## Decision",
        "",
        "Do not increase controller capacity or run another 100k-step selector until the ARC portfolio oracle rises and the math task generator exposes the strategy distinction it asks the controller to learn.",
        "",
        "## Provenance",
        "",
        f"- ARC source: [official ARC-AGI-1 repository](https://github.com/fchollet/ARC-AGI), commit `{result['arc']['source_commit']}`.",
        f"- ARC counts: `{json.dumps(result['arc']['counts'], sort_keys=True)}`.",
        "- Public ARC outputs were used only for final portfolio scoring; no learned checkpoint or selector was selected from them.",
        "",
        "## Source hashes",
        "",
    ])
    for path, digest in sorted(hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-root", type=Path, default=DEFAULT_ARC_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train, validation, evaluation, arc_index = load_arc_tasks(args.arc_root, args.output)
    for split, problems in (("train", train), ("validation", validation), ("evaluation", evaluation)):
        write_json(args.output / "problem_manifests" / f"{split}.json", [{"problem_id": p.problem_id, "split": p.split, "metadata": p.metadata} for p in problems])
    all_ids = [problem.problem_id for problem in train + validation + evaluation]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("duplicate ARC problem IDs")
    arc_result: Dict[str, Any] = {"source_commit": arc_index["source_commit"], "counts": {"train": len(train), "validation": len(validation), "evaluation": len(evaluation)}, "train": {}, "validation": {}, "evaluation": {}}
    for level in LEVELS:
        programs = programs_for_level(level)
        arc_result["train"][level] = portfolio_report(train, level, programs)
        arc_result["validation"][level] = portfolio_report(validation, level, programs)
        arc_result["evaluation"][level] = portfolio_report(evaluation, level, programs)
    c1d = import_c1d()
    math_result = math_identifiability(c1d)
    result = {"experiment": EXPERIMENT, "status": "completed", "arc": arc_result, "math": math_result, "source_c1d": {"results_summary": str((C1D_ROOT / "RESULTS_SUMMARY.md").relative_to(ROOT)), "hashes": str((C1D_ROOT / "HASHES.json").relative_to(ROOT))}}
    write_json(args.output / "ARC_PORTFOLIO_RESULTS.json", {"source": arc_result, "train": arc_result["train"], "validation": arc_result["validation"], "evaluation": arc_result["evaluation"]})
    write_json(args.output / "MATH_IDENTIFIABILITY.json", math_result)
    write_json(args.output / "RESULTS.json", result)
    write_json(args.output / "RUN_CONFIG.json", {"experiment": EXPERIMENT, "levels": list(LEVELS), "base_actions": list(BASE_ACTIONS), "object_actions": list(OBJECT_ACTIONS), "max_depth": 2, "execution": "CPU-only", "source_c1d": str(C1D_ROOT.relative_to(ROOT))})
    hashes = source_hashes()
    (args.output / "RESULTS_SUMMARY.md").write_text(summary(result, hashes), encoding="utf-8")
    file_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(args.output.rglob("*")) if path.is_file() and path.name != "HASHES.json"}
    write_json(args.output / "HASHES.json", file_hashes)
    print(json.dumps({"status": "completed", "output": str(args.output), "arc_evaluation_tasks": len(evaluation), "math_tasks": len(math_result["items"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
