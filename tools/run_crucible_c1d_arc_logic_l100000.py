#!/usr/bin/env python3
"""CPU-only 100k-step strategy selection over logic, arithmetic and ARC tasks.

The learner chooses among a small portfolio of deterministic, exact-checkable
solvers.  It is intentionally not a language model and never sees held-out
answers at decision time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARC_ROOT = Path("/tmp/arc_agi_official_20261021")
DEFAULT_OUTPUT = ROOT / "experiments" / "crucible_c1d_arc_logic_l100000_20261021"
PROTOCOL = ROOT / "experiments" / "crucible_c1d_arc_logic_l100000_20261021" / "PROTOCOL.md"
EXPERIMENT = "crucible_c1d_arc_logic_l100000_20261021"
SEEDS = (42, 1337, 2026)
UPDATES = 100000
CHECKPOINTS = (0, 5000, 10000, 25000, 50000, 75000, 90000, 100000)
WALL_CAP_SECONDS = 20.0 * 60.0

ARC_ACTIONS = (
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
LOGIC_ACTIONS = (
    "logic_direct",
    "logic_shallow",
    "logic_forward",
    "logic_reverse",
)
MATH_ACTIONS = (
    "math_left",
    "math_precedence",
    "math_reverse",
)
ACTIONS = ARC_ACTIONS + LOGIC_ACTIONS + MATH_ACTIONS + ("fallback",)
FAMILIES = ("arc", "logic", "math")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_int(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


Grid = List[List[int]]


@dataclass(frozen=True)
class Problem:
    problem_id: str
    family: str
    payload: Mapping[str, Any]
    split: str
    source: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Candidate:
    action: str
    available: bool
    demo_fit: bool
    prediction: Any
    correct: bool
    cost: float
    complexity: float
    output_area: float


def grid_equal(left: Any, right: Any) -> bool:
    return left == right


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
    cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value != 0]
    if not cells:
        return None
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    return [row[c0:c1 + 1] for row in grid[r0:r1 + 1]]


def infer_color_map(pairs: Sequence[Mapping[str, Any]]) -> Optional[Dict[int, int]]:
    mapping: Dict[int, int] = {}
    for pair in pairs:
        source = pair["input"]
        target = pair["output"]
        if len(source) != len(target) or any(len(a) != len(b) for a, b in zip(source, target)):
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
        for r, source_row in enumerate(source):
            for c, value in enumerate(source_row):
                for rr in range(r * scale, (r + 1) * scale):
                    if any(target[rr][cc] != value for cc in range(c * scale, (c + 1) * scale)):
                        return None
    return scale


def arc_transform(action: str, pairs: Sequence[Mapping[str, Any]], grid: Grid) -> Optional[Grid]:
    if action == "arc_identity":
        return [row[:] for row in grid]
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
    if action == "arc_color_map":
        mapping = infer_color_map(pairs)
        return None if mapping is None else apply_color_map(grid, mapping)
    if action == "arc_scale":
        scale = infer_scale(pairs)
        if scale is None:
            return None
        return [[value for value in row for _ in range(scale)] for row in grid for _ in range(scale)]
    return None


def arc_candidate(problem: Problem, action: str) -> Candidate:
    pairs = problem.payload["train"]
    tests = problem.payload["test"]
    demo_fit = True
    for pair in pairs:
        prediction = arc_transform(action, pairs, pair["input"])
        if prediction is None or not grid_equal(prediction, pair["output"]):
            demo_fit = False
            break
    predictions = []
    available = False
    for pair in tests:
        prediction = arc_transform(action, pairs, pair["input"])
        if prediction is None:
            predictions.append(None)
        else:
            available = True
            predictions.append(prediction)
    correct = available and all(grid_equal(prediction, pair.get("output")) for prediction, pair in zip(predictions, tests))
    area = sum(grid_area(prediction) for prediction in predictions if prediction is not None) / max(1, len(predictions))
    complexity = {
        "arc_identity": 1.0, "arc_color_map": 2.0, "arc_rot90": 1.5,
        "arc_rot180": 1.5, "arc_rot270": 1.5, "arc_flip_h": 1.2,
        "arc_flip_v": 1.2, "arc_transpose": 1.2, "arc_crop_nonzero": 2.0,
        "arc_scale": 2.5,
    }[action]
    costs = {
        "arc_identity": 0.4, "arc_color_map": 1.2, "arc_rot90": 0.7,
        "arc_rot180": 0.7, "arc_rot270": 0.7, "arc_flip_h": 0.6,
        "arc_flip_v": 0.6, "arc_transpose": 0.6, "arc_crop_nonzero": 1.0,
        "arc_scale": 1.4,
    }
    return Candidate(action, available, demo_fit and available, predictions, correct, costs[action], complexity, area)


def closure(facts: Iterable[str], rules: Iterable[Tuple[str, str]], limit: Optional[int] = None) -> set[str]:
    result = set(facts)
    for _ in range(100 if limit is None else limit):
        changed = False
        for premise, conclusion in rules:
            if premise in result and conclusion not in result:
                result.add(conclusion)
                changed = True
        if not changed:
            break
    return result


def logic_answer(facts: Sequence[str], rules: Sequence[Tuple[str, str]], query: str, limit: Optional[int]) -> str:
    known = closure(facts, rules, limit)
    if query.startswith("not "):
        return "contradicted" if query[4:] in known else "unknown"
    if query in known:
        return "entailed"
    return "contradicted" if f"not {query}" in known else "unknown"


def prove(atom: str, facts: set[str], rules: Sequence[Tuple[str, str]], seen: set[str]) -> bool:
    if atom in facts:
        return True
    if atom in seen:
        return False
    seen.add(atom)
    return any(conclusion == atom and prove(premise, facts, rules, seen) for premise, conclusion in rules)


def logic_candidate(problem: Problem, action: str) -> Candidate:
    payload = problem.payload
    limits = {"logic_direct": 0, "logic_shallow": 2, "logic_forward": None, "logic_reverse": None}
    predictions: List[str] = []
    for query in payload["train_queries"]:
        if action == "logic_reverse":
            value = "contradicted" if query.startswith("not ") and prove(query[4:], set(payload["facts"]), payload["rules"], set()) else (
                "entailed" if not query.startswith("not ") and prove(query, set(payload["facts"]), payload["rules"], set()) else "unknown"
            )
        else:
            value = logic_answer(payload["facts"], payload["rules"], query, limits[action])
        predictions.append(value)
    demo_fit = predictions == payload["train_answers"]
    test_predictions: List[str] = []
    for query in payload["test_queries"]:
        if action == "logic_reverse":
            positive = query[4:] if query.startswith("not ") else query
            proved = prove(positive, set(payload["facts"]), payload["rules"], set())
            if query.startswith("not "):
                value = "contradicted" if proved else "unknown"
            else:
                value = "entailed" if proved else "unknown"
        else:
            value = logic_answer(payload["facts"], payload["rules"], query, limits[action])
        test_predictions.append(value)
    correct = test_predictions == payload["test_answers"]
    costs = {"logic_direct": 0.5, "logic_shallow": 1.0, "logic_forward": 1.5, "logic_reverse": 2.0}
    return Candidate(action, True, demo_fit, test_predictions, correct, costs[action], float(limits[action] or 5), float(len(test_predictions)))


def math_left(start: int, operations: Sequence[Tuple[str, int]]) -> int:
    value = start
    for operation, operand in operations:
        if operation == "add":
            value += operand
        elif operation == "sub":
            value -= operand
        else:
            value *= operand
    return value


def math_precedence(start: int, operations: Sequence[Tuple[str, int]]) -> int:
    terms: List[Tuple[int, int]] = [(1, start)]
    for operation, operand in operations:
        if operation == "mul":
            sign, value = terms[-1]
            terms[-1] = (sign, value * operand)
        else:
            terms.append((1 if operation == "add" else -1, operand))
    return sum(sign * value for sign, value in terms)


def math_reverse(start: int, operations: Sequence[Tuple[str, int]]) -> int:
    return math_left(start, list(reversed(operations)))


def math_candidate(problem: Problem, action: str) -> Candidate:
    function = {"math_left": math_left, "math_precedence": math_precedence, "math_reverse": math_reverse}[action]
    payload = problem.payload
    train_predictions = [function(case["start"], case["operations"]) for case in payload["train_cases"]]
    demo_fit = train_predictions == payload["train_answers"]
    test_predictions = [function(case["start"], case["operations"]) for case in payload["test_cases"]]
    correct = test_predictions == payload["test_answers"]
    costs = {"math_left": 0.8, "math_precedence": 1.0, "math_reverse": 1.4}
    return Candidate(action, True, demo_fit, test_predictions, correct, costs[action], 1.0, float(len(test_predictions)))


def fallback_candidate(problem: Problem) -> Candidate:
    if problem.family == "arc":
        predictions = [pair["input"] for pair in problem.payload["test"]]
    elif problem.family == "logic":
        predictions = ["unknown" for _ in problem.payload["test_queries"]]
    else:
        predictions = [case["start"] for case in problem.payload["test_cases"]]
    expected = problem.payload.get("test_answers")
    if expected is None:
        expected = [pair["output"] for pair in problem.payload["test"]]
    # Fallback is a recovery action, not a demonstration-consistent solver.
    # It is selected only when no domain solver fits the demonstrations.
    return Candidate("fallback", True, False, predictions, predictions == expected, 0.2, 0.5, float(len(predictions)))


def candidates_for(problem: Problem) -> Dict[str, Candidate]:
    result: Dict[str, Candidate] = {"fallback": fallback_candidate(problem)}
    actions = ARC_ACTIONS if problem.family == "arc" else LOGIC_ACTIONS if problem.family == "logic" else MATH_ACTIONS
    for action in actions:
        if problem.family == "arc":
            result[action] = arc_candidate(problem, action)
        elif problem.family == "logic":
            result[action] = logic_candidate(problem, action)
        else:
            result[action] = math_candidate(problem, action)
    for action in ACTIONS:
        result.setdefault(action, Candidate(action, False, False, None, False, 99.0, 99.0, 0.0))
    return result


def generated_logic(seed: int, split: str, count: int) -> List[Problem]:
    problems: List[Problem] = []
    for index in range(count):
        depth = {"train": 2 + index % 2, "validation": 3 + index % 2, "test": 4 + index % 2}[split]
        facts = [f"P{index}_0"]
        rules = [(f"P{index}_{position}", f"P{index}_{position + 1}") for position in range(depth)]
        if index % 3 == 0:
            facts.append(f"D{index}_0")
            rules.append((f"D{index}_0", f"D{index}_1"))
        train_queries = [f"P{index}_{min(1 + j, depth - 1)}" for j in range(2)]
        test_queries = [f"P{index}_{depth}", f"not P{index}_{depth}"]
        train_answers = [logic_answer(facts, rules, query, None) for query in train_queries]
        test_answers = [logic_answer(facts, rules, query, None) for query in test_queries]
        payload = {"facts": facts, "rules": rules, "train_queries": train_queries, "train_answers": train_answers, "test_queries": test_queries, "test_answers": test_answers}
        problems.append(Problem(f"logic-{seed}-{split}-{index}", "logic", payload, split, "generated", {"depth": depth, "seed": seed}))
    return problems


def generated_math(seed: int, split: str, count: int) -> List[Problem]:
    problems: List[Problem] = []
    rng = random.Random(stable_int("math", seed, split))
    for index in range(count):
        semantics = "left" if index % 2 == 0 else "precedence"
        train_cases = []
        for _ in range(3):
            start = rng.randint(1, 9)
            operations = [("add", rng.randint(1, 4)), ("sub", rng.randint(1, 3))]
            train_cases.append({"start": start, "operations": operations})
        test_cases = []
        for _ in range(2):
            start = rng.randint(1, 9)
            operations = [("add", rng.randint(1, 4)), ("mul", rng.randint(2, 5)), ("sub", rng.randint(1, 3))]
            test_cases.append({"start": start, "operations": operations})
        target_function = math_left if semantics == "left" else math_precedence
        payload = {
            "train_cases": train_cases,
            "train_answers": [target_function(c["start"], c["operations"]) for c in train_cases],
            "test_cases": test_cases,
            "test_answers": [target_function(c["start"], c["operations"]) for c in test_cases],
        }
        problems.append(Problem(f"math-{seed}-{split}-{index}", "math", payload, split, "generated", {"semantics": semantics, "mixed_test": True}))
    return problems


def load_arc_tasks(arc_root: Path, output: Path) -> Tuple[List[Problem], List[Problem], List[Problem], Dict[str, Any]]:
    if not (arc_root / "data" / "training").is_dir() or not (arc_root / "data" / "evaluation").is_dir():
        raise RuntimeError(f"ARC-AGI-1 data not found under {arc_root}")
    arc_copy = output / "arc_data"
    training_files = sorted((arc_root / "data" / "training").glob("*.json"))
    evaluation_files = sorted((arc_root / "data" / "evaluation").glob("*.json"))
    if len(training_files) != 400 or len(evaluation_files) != 400:
        raise RuntimeError(f"expected 400/400 ARC tasks, got {len(training_files)}/{len(evaluation_files)}")
    arc_train: List[Problem] = []
    arc_validation: List[Problem] = []
    arc_eval: List[Problem] = []
    index: Dict[str, Any] = {"source": "https://github.com/fchollet/ARC-AGI", "source_commit": "399030444e0ab0cc8b4e199870fb20b863846f34", "training": [], "evaluation": []}
    for split_name, files in (("training", training_files), ("evaluation", evaluation_files)):
        for position, source_file in enumerate(files):
            task = json.loads(source_file.read_text(encoding="utf-8"))
            task_id = source_file.stem
            destination = arc_copy / split_name / source_file.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, destination)
            entry = {"task_id": task_id, "path": str(destination.relative_to(output)), "sha256": sha256_file(destination), "train_pairs": len(task["train"]), "test_pairs": len(task["test"])}
            index[split_name].append(entry)
            problem = Problem(f"arc-{split_name}-{task_id}", "arc", task, "test" if split_name == "evaluation" else "train", "official_arc_agi_1", {"task_id": task_id, "train_pairs": len(task["train"]), "test_pairs": len(task["test"])})
            if split_name == "evaluation":
                arc_eval.append(problem)
            elif position % 5 == 0:
                arc_validation.append(Problem(problem.problem_id, problem.family, problem.payload, "validation", problem.source, problem.metadata))
            else:
                arc_train.append(problem)
    return arc_train, arc_validation, arc_eval, index


def feature_vector(problem: Problem, candidates: Mapping[str, Candidate]) -> List[float]:
    family = [1.0 if problem.family == item else 0.0 for item in FAMILIES]
    if problem.family == "arc":
        pairs = problem.payload["train"]
        areas = [grid_area(pair["input"]) for pair in pairs]
        colors = {value for pair in pairs for row in pair["input"] for value in row}
        stats = [min(1.0, len(pairs) / 8.0), min(1.0, (sum(areas) / max(1, len(areas))) / 900.0), min(1.0, len(colors) / 10.0), min(1.0, len(problem.payload["test"]) / 4.0)]
    elif problem.family == "logic":
        stats = [min(1.0, len(problem.payload["rules"]) / 8.0), min(1.0, max((int(rule[1].rsplit("_", 1)[-1]) for rule in problem.payload["rules"]), default=0) / 8.0), min(1.0, len(problem.payload["train_queries"]) / 4.0), 0.0]
    else:
        stats = [min(1.0, len(problem.payload["train_cases"]) / 8.0), 1.0 if problem.metadata.get("mixed_test") else 0.0, min(1.0, len(problem.payload["test_cases"]) / 4.0), 0.0]
    features = [1.0, *family, *stats]
    for action in ACTIONS:
        candidate = candidates[action]
        features.extend([
            1.0 if candidate.available else 0.0,
            1.0 if candidate.demo_fit else 0.0,
            min(1.0, candidate.cost / 10.0),
            min(1.0, candidate.complexity / 10.0),
            min(1.0, candidate.output_area / 900.0),
        ])
    return features


class Bandit:
    def __init__(self, seed: int, feature_count: int) -> None:
        self.rng = random.Random(seed)
        self.weights = {action: [0.0] * feature_count for action in ACTIONS}
        self.counts = {action: 0 for action in ACTIONS}

    def snapshot(self) -> Dict[str, Any]:
        return {"weights": {a: list(w) for a, w in self.weights.items()}, "counts": dict(self.counts)}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.weights = {a: [float(x) for x in snapshot["weights"][a]] for a in ACTIONS}
        self.counts = {a: int(snapshot["counts"][a]) for a in ACTIONS}

    def score(self, action: str, features: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(self.weights[action], features))

    def select(self, features: Sequence[float], candidates: Mapping[str, Candidate], update: int, explore: bool) -> str:
        eligible = [action for action in ACTIONS if candidates[action].demo_fit]
        if not eligible:
            eligible = ["fallback"]
        epsilon = max(0.05, 0.25 * (1.0 - (update - 1) / UPDATES))
        if explore and self.rng.random() < epsilon:
            return eligible[self.rng.randrange(len(eligible))]
        return max(eligible, key=lambda action: (self.score(action, features), -ACTIONS.index(action)))

    def update(self, action: str, features: Sequence[float], reward: float) -> None:
        prediction = self.score(action, features)
        error = reward - prediction
        rate = 0.05 / math.sqrt(1.0 + self.counts[action] * 0.01)
        for index, value in enumerate(features):
            self.weights[action][index] += rate * error * value
        self.counts[action] += 1


def metric(results: Sequence[Tuple[str, Candidate]]) -> Dict[str, Any]:
    if not results:
        return {"cases": 0, "accuracy": 0.0, "coverage": 0.0, "mean_cost": 0.0, "action_counts": {}}
    counts = {action: sum(chosen == action for chosen, _ in results) for action in ACTIONS}
    return {
        "cases": len(results),
        "accuracy": sum(candidate.correct for _, candidate in results) / len(results),
        "coverage": sum(candidate.demo_fit for _, candidate in results) / len(results),
        "mean_cost": sum(candidate.cost for _, candidate in results) / len(results),
        "action_counts": counts,
    }


def choose(problem: Problem, candidates: Mapping[str, Candidate], policy: str, bandit: Optional[Bandit], features: Sequence[float], seed: int) -> str:
    eligible = [action for action in ACTIONS if candidates[action].demo_fit]
    if not eligible:
        return "fallback"
    if policy == "learned" and bandit is not None:
        return bandit.select(features, candidates, UPDATES, False)
    if policy == "first_consistent":
        return eligible[0]
    if policy == "cheapest_consistent":
        return min(eligible, key=lambda action: (candidates[action].cost, ACTIONS.index(action)))
    if policy == "random_consistent":
        return eligible[stable_int("random-policy", seed, problem.problem_id) % len(eligible)]
    if policy == "family_heuristic":
        preferred = {"arc": ("arc_color_map", "arc_identity", "arc_rot90", "arc_flip_h"), "logic": ("logic_forward", "logic_reverse", "logic_shallow"), "math": ("math_precedence", "math_left", "math_reverse")}[problem.family]
        return next((action for action in preferred if action in eligible), eligible[0])
    if policy == "oracle":
        correct = [action for action in eligible if candidates[action].correct]
        return min(correct or eligible, key=lambda action: (candidates[action].cost, ACTIONS.index(action)))
    raise ValueError(policy)


def evaluate(problems: Sequence[Problem], policy: str, *, bandit: Optional[Bandit], seed: int, cache: Mapping[str, Mapping[str, Candidate]]) -> Dict[str, Any]:
    chosen: List[Tuple[str, Candidate]] = []
    by_family: Dict[str, List[Tuple[str, Candidate]]] = {family: [] for family in FAMILIES}
    for problem in problems:
        candidates = cache[problem.problem_id]
        features = feature_vector(problem, candidates)
        action = choose(problem, candidates, policy, bandit, features, seed)
        item = (action, candidates[action])
        chosen.append(item)
        by_family[problem.family].append(item)
    return {
        **metric(chosen),
        "by_family": {family: metric(items) for family, items in by_family.items() if items},
    }


def validation_score(item: Mapping[str, Any]) -> float:
    return round(float(item["accuracy"]) - 0.01 * float(item["mean_cost"]), 9)


def prepare_problems(arc_root: Path, output: Path) -> Tuple[List[Problem], List[Problem], List[Problem], Dict[str, Any]]:
    arc_train, arc_validation, arc_eval, arc_index = load_arc_tasks(arc_root, output)
    train = arc_train + generated_logic(20261031, "train", 600) + generated_math(20261032, "train", 600)
    validation = arc_validation + generated_logic(20261031, "validation", 200) + generated_math(20261032, "validation", 200)
    test = arc_eval + generated_logic(20261031, "test", 300) + generated_math(20261032, "test", 300)
    ids = [problem.problem_id for problem in train + validation + test]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate problem IDs")
    manifest = {
        "experiment": EXPERIMENT,
        "arc_source": arc_index,
        "counts": {"train": len(train), "validation": len(validation), "test": len(test), "arc_public_evaluation": len(arc_eval)},
        "generated_seeds": {"logic": 20261031, "math": 20261032},
        "split_policy": "ARC training task IDs: sorted position modulo 5 equals validation; ARC evaluation is final test only",
        "held_out_outputs": "never included in learner features or checkpoint selection",
    }
    write_json(output / "MANIFEST.json", manifest)
    for split, problems in (("train", train), ("validation", validation), ("test", test)):
        write_json(output / "problem_manifests" / f"{split}.json", [{"problem_id": p.problem_id, "family": p.family, "source": p.source, "metadata": p.metadata} for p in problems])
    return train, validation, test, manifest


def build_cache(problems: Iterable[Problem]) -> Dict[str, Dict[str, Candidate]]:
    return {problem.problem_id: candidates_for(problem) for problem in problems}


def run_seed(train: Sequence[Problem], validation: Sequence[Problem], test: Sequence[Problem], train_validation_cache: Mapping[str, Mapping[str, Candidate]], seed: int, started: float) -> Dict[str, Any]:
    feature_count = len(feature_vector(train[0], train_validation_cache[train[0].problem_id]))
    bandit = Bandit(seed, feature_count)
    checkpoints: List[Dict[str, Any]] = []
    curves: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    rewards: List[float] = []
    validation_zero = evaluate(validation, "learned", bandit=bandit, seed=seed, cache=train_validation_cache)
    checkpoints.append({"update": 0, "validation": validation_zero, "validation_score": validation_score(validation_zero), "learner": bandit.snapshot()})
    for update in range(1, UPDATES + 1):
        if perf_counter() - started > WALL_CAP_SECONDS:
            raise TimeoutError("C1d-L100000 exceeded 20-minute CPU cap")
        problem = train[rng.randrange(len(train))]
        candidates = train_validation_cache[problem.problem_id]
        features = feature_vector(problem, candidates)
        action = bandit.select(features, candidates, update, True)
        candidate = candidates[action]
        reward = (1.0 if candidate.correct else 0.0) - 0.02 * min(1.0, candidate.cost / 10.0)
        if action == "fallback" and not candidate.correct:
            reward -= 0.10
        bandit.update(action, features, reward)
        rewards.append(reward)
        if update % 1000 == 0 or update in CHECKPOINTS:
            recent = rewards[-1000:]
            curves.append({"seed": seed, "update": update, "mean_reward_last_1000": sum(recent) / len(recent), "epsilon": max(0.05, 0.25 * (1.0 - (update - 1) / UPDATES))})
        if update in CHECKPOINTS:
            validation_item = evaluate(validation, "learned", bandit=bandit, seed=seed, cache=train_validation_cache)
            checkpoints.append({"update": update, "validation": validation_item, "validation_score": validation_score(validation_item), "learner": bandit.snapshot()})
    selected = max(checkpoints, key=lambda item: (item["validation_score"], -item["update"]))
    selected_bandit = Bandit(seed, feature_count)
    selected_bandit.restore(selected["learner"])
    # Do not load or score ARC public-evaluation outputs until after the
    # validation-selected checkpoint has been fixed.
    test_cache = build_cache(test)
    policies = ("first_consistent", "cheapest_consistent", "family_heuristic", "random_consistent", "oracle", "learned")
    results = {policy: evaluate(test, policy, bandit=selected_bandit if policy == "learned" else None, seed=seed, cache=test_cache) for policy in policies}
    return {"seed": seed, "updates_completed": UPDATES, "selected_checkpoint": selected["update"], "selected_validation_score": selected["validation_score"], "checkpoints": checkpoints, "curves": curves, "test": results}


def summary(result: Mapping[str, Any], manifest: Mapping[str, Any], elapsed: float, source_hashes: Mapping[str, str]) -> str:
    lines = [
        "# C1d-L100000 logic, arithmetic and ARC strategy selection",
        "",
        "Date: 2026-10-21",
        "Execution: CPU-only; no GPU, cloud or Crucible model training.",
        "",
        "## Verdict",
        "",
        f"- Seeds: **{len(result['seeds'])}/3**; updates per seed: **{UPDATES}**.",
        f"- Wall time: **{elapsed:.3f} s**; cap: **{WALL_CAP_SECONDS:.1f} s**.",
        "- The public ARC evaluation split was used only after validation checkpoint selection.",
        "- This tests solver selection, not general intelligence or neural reasoning.",
        "",
        "## Final test results",
        "",
        "| Policy | All tasks | ARC public evaluation | Logic | Math | Mean cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in result["aggregate_test"]:
        item = result["aggregate_test"][policy]
        arc = result["aggregate_test_by_family"][policy].get("arc", {})
        logic = result["aggregate_test_by_family"][policy].get("logic", {})
        math_item = result["aggregate_test_by_family"][policy].get("math", {})
        lines.append(f"| `{policy}` | {item['accuracy']:.4f} | {arc.get('accuracy', 0.0):.4f} | {logic.get('accuracy', 0.0):.4f} | {math_item.get('accuracy', 0.0):.4f} | {item['mean_cost']:.3f} |")
    lines.extend([
        "",
        "## Data separation",
        "",
        f"- ARC source: [official ARC-AGI-1 repository](https://github.com/fchollet/ARC-AGI), commit `{manifest['arc_source']['source_commit']}`.",
        f"- Problem counts: `{json.dumps(manifest['counts'], sort_keys=True)}`.",
        "- ARC training tasks were split into train/validation; ARC public evaluation tasks were not used for training or checkpoint selection.",
        "- Generated logic/math tasks use independent seeds and disjoint IDs.",
        "- Held-out outputs were evaluator-only.",
        "",
        "## Checkpoint selection",
        "",
    ])
    for seed_result in result["seeds"]:
        lines.append(f"- Seed {seed_result['seed']}: selected update `{seed_result['selected_checkpoint']}`; validation score `{seed_result['selected_validation_score']:.6f}`.")
    lines.extend([
        "",
        "## Limitations",
        "",
        "The solver portfolio is intentionally small. A low ARC score can mean that no included transformation generated the correct output, not that the selector failed to choose among available correct candidates. ARC results are public-evaluation measurements, not private leaderboard results.",
        "",
        "## Source hashes",
        "",
    ])
    for path, digest in sorted(source_hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-root", type=Path, default=DEFAULT_ARC_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    train, validation, test, manifest = prepare_problems(args.arc_root, args.output)
    # Training and validation candidates are constructed before learning.
    # Public ARC evaluation candidates (including their outputs) are delayed
    # until each seed has selected its checkpoint on validation only.
    cache = build_cache(train + validation)
    write_json(args.output / "RUN_CONFIG.json", {"experiment": EXPERIMENT, "seeds": list(SEEDS), "updates": UPDATES, "checkpoints": list(CHECKPOINTS), "actions": list(ACTIONS), "wall_cap_seconds": WALL_CAP_SECONDS, "execution": "CPU-only", "arc_root": str(args.arc_root), "protocol": str(PROTOCOL.relative_to(ROOT))})
    seed_results = [run_seed(train, validation, test, cache, seed, started) for seed in SEEDS]
    aggregate_test = {}
    aggregate_family = {}
    policies = list(seed_results[0]["test"])
    for policy in policies:
        aggregate_test[policy] = {metric_name: sum(item["test"][policy][metric_name] for item in seed_results) / len(seed_results) for metric_name in ("accuracy", "coverage", "mean_cost")}
        aggregate_family[policy] = {}
        for family in FAMILIES:
            family_items = [item["test"][policy]["by_family"].get(family) for item in seed_results]
            family_items = [item for item in family_items if item]
            if family_items:
                aggregate_family[policy][family] = {metric_name: sum(item[metric_name] for item in family_items) / len(family_items) for metric_name in ("accuracy", "coverage", "mean_cost")}
    result = {"experiment": EXPERIMENT, "status": "completed", "elapsed_seconds": round(perf_counter() - started, 6), "seeds": seed_results, "aggregate_test": aggregate_test, "aggregate_test_by_family": aggregate_family}
    write_json(args.output / "RESULTS.json", result)
    with (args.output / "TRAINING_CURVES.jsonl").open("w", encoding="utf-8") as handle:
        for seed_result in seed_results:
            for curve in seed_result["curves"]:
                handle.write(json.dumps(curve, sort_keys=True) + "\n")
    source_paths = [ROOT / "tools" / "run_crucible_c1d_arc_logic_l100000.py", PROTOCOL]
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}
    (args.output / "RESULTS_SUMMARY.md").write_text(summary(result, manifest, result["elapsed_seconds"], source_hashes), encoding="utf-8")
    hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(args.output.rglob("*")) if path.is_file() and path.name != "HASHES.json"}
    write_json(args.output / "HASHES.json", hashes)
    print(json.dumps({"status": "completed", "output": str(args.output), "elapsed_seconds": result["elapsed_seconds"], "seeds": len(seed_results), "updates_per_seed": UPDATES}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
