#!/usr/bin/env python3
"""Small direct-generation B3 proposer and lifecycle interface primitives.

This module is intentionally independent of the frozen Lifecycle v0 rule
filter.  B3 emits prefix-serialized ASTs directly from neural logits.  The
grammar mask makes emitted sequences parseable, but it never supplies a target
AST or an enumerated catalogue of candidate procedures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


OPERATORS: Tuple[str, ...] = (
    "add_mod10",
    "sub_mod10",
    "mul_mod10",
    "xor_bits_mod10",
    "min",
    "max",
)
COMMUTATIVE = {"add_mod10", "mul_mod10", "xor_bits_mod10", "min", "max"}
EOS = "<eos>"
TOKEN_LIST: Tuple[str, ...] = (
    "VAR_A",
    "VAR_B",
    *(f"CONST_{value}" for value in range(10)),
    *(f"OP_{operator}" for operator in OPERATORS),
    EOS,
)
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKEN_LIST)}
ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}


def canonical_ast(ast: Mapping[str, Any]) -> Dict[str, Any]:
    op = str(ast["op"])
    if op == "var":
        name = str(ast["name"])
        if name not in {"a", "b"}:
            raise ValueError(f"invalid variable: {name}")
        return {"op": "var", "name": name}
    if op == "const":
        value = int(ast["value"])
        if not 0 <= value <= 9:
            raise ValueError(f"invalid constant: {value}")
        return {"op": "const", "value": value}
    if op not in OPERATORS:
        raise ValueError(f"invalid operator: {op}")
    args = ast.get("args")
    if not isinstance(args, list) or len(args) != 2:
        raise ValueError(f"operator {op} requires two args")
    children = [canonical_ast(args[0]), canonical_ast(args[1])]
    if op in COMMUTATIVE:
        children.sort(key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
    return {"op": op, "args": children}


def ast_json_key(ast: Mapping[str, Any]) -> str:
    return json.dumps(canonical_ast(ast), sort_keys=True, separators=(",", ":"))


def ast_hash(ast: Mapping[str, Any]) -> str:
    return hashlib.sha256(ast_json_key(ast).encode("utf-8")).hexdigest()


def evaluate_ast(ast: Mapping[str, Any], a: int, b: int) -> int:
    node = canonical_ast(ast)
    op = node["op"]
    if op == "var":
        return int(a if node["name"] == "a" else b)
    if op == "const":
        return int(node["value"])
    left = evaluate_ast(node["args"][0], a, b)
    right = evaluate_ast(node["args"][1], a, b)
    if op == "add_mod10":
        return (left + right) % 10
    if op == "sub_mod10":
        return (left - right) % 10
    if op == "mul_mod10":
        return (left * right) % 10
    if op == "xor_bits_mod10":
        return (left ^ right) % 10
    if op == "min":
        return min(left, right)
    if op == "max":
        return max(left, right)
    raise ValueError(f"unknown operator: {op}")


def ast_depth(ast: Mapping[str, Any]) -> int:
    node = canonical_ast(ast)
    if node["op"] in {"var", "const"}:
        return 0
    return 1 + max(ast_depth(node["args"][0]), ast_depth(node["args"][1]))


def ast_nodes(ast: Mapping[str, Any]) -> int:
    node = canonical_ast(ast)
    if node["op"] in {"var", "const"}:
        return 1
    return 1 + ast_nodes(node["args"][0]) + ast_nodes(node["args"][1])


def ast_to_tokens(ast: Mapping[str, Any]) -> List[str]:
    node = canonical_ast(ast)
    op = node["op"]
    if op == "var":
        return ["VAR_A" if node["name"] == "a" else "VAR_B"]
    if op == "const":
        return [f"CONST_{node['value']}"]
    return [f"OP_{op}"] + ast_to_tokens(node["args"][0]) + ast_to_tokens(node["args"][1])


def tokens_to_ast(tokens: Sequence[str]) -> Dict[str, Any]:
    position = 0

    def parse_expression() -> Dict[str, Any]:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("truncated AST token sequence")
        token = tokens[position]
        position += 1
        if token == "VAR_A":
            return {"op": "var", "name": "a"}
        if token == "VAR_B":
            return {"op": "var", "name": "b"}
        if token.startswith("CONST_"):
            return {"op": "const", "value": int(token.split("_", 1)[1])}
        if token.startswith("OP_"):
            op = token[3:]
            if op not in OPERATORS:
                raise ValueError(f"invalid operator token: {token}")
            return {"op": op, "args": [parse_expression(), parse_expression()]}
        raise ValueError(f"invalid expression token: {token}")

    ast = parse_expression()
    if position >= len(tokens) or tokens[position] != EOS:
        raise ValueError("AST token sequence lacks EOS")
    position += 1
    if position != len(tokens):
        raise ValueError("tokens appear after EOS")
    return canonical_ast(ast)


class GrammarConstraint:
    """Prefix-expression grammar state used to mask every decoder step."""

    def __init__(self, max_depth: int = 3, max_nodes: int = 7) -> None:
        self.max_depth = int(max_depth)
        self.max_nodes = int(max_nodes)
        self.max_tokens = self.max_nodes + 1

    def initial_state(self) -> Tuple[List[int], int]:
        return [0], 0

    def valid_token_ids(self, pending_depths: Sequence[int], node_count: int) -> List[int]:
        if not pending_depths:
            return [TOKEN_TO_ID[EOS]]
        ids: List[int] = []
        if node_count < self.max_nodes:
            ids.extend(TOKEN_TO_ID[token] for token in ("VAR_A", "VAR_B"))
            ids.extend(TOKEN_TO_ID[f"CONST_{value}"] for value in range(10))
            # An operator consumes one node and necessarily creates two child
            # expression slots. Reserve those minimum two child nodes so the
            # decoder cannot enter an unfinished state at max_nodes.
            if pending_depths[-1] < self.max_depth and node_count + 3 <= self.max_nodes:
                ids.extend(TOKEN_TO_ID[f"OP_{operator}"] for operator in OPERATORS)
        return ids

    def advance(self, pending_depths: List[int], node_count: int, token: str) -> Tuple[List[int], int]:
        if token == EOS:
            if pending_depths:
                raise ValueError("EOS before expression completion")
            return pending_depths, node_count
        if not pending_depths:
            raise ValueError("expression token after completion")
        depth = pending_depths.pop()
        if token.startswith("OP_"):
            if depth >= self.max_depth or node_count >= self.max_nodes:
                raise ValueError("operator exceeds grammar limits")
            pending_depths.extend((depth + 1, depth + 1))
        elif token.startswith("VAR_") or token.startswith("CONST_"):
            if node_count >= self.max_nodes:
                raise ValueError("leaf exceeds grammar node limit")
        else:
            raise ValueError(f"invalid grammar token: {token}")
        return pending_depths, node_count + 1

    def validate(self, ast: Mapping[str, Any]) -> Dict[str, Any]:
        canonical = canonical_ast(ast)
        if ast_depth(canonical) > self.max_depth or ast_nodes(canonical) > self.max_nodes:
            raise ValueError("AST exceeds frozen grammar limits")
        tokens = ast_to_tokens(canonical) + [EOS]
        pending, count = self.initial_state()
        for token in tokens:
            valid = self.valid_token_ids(pending, count)
            if TOKEN_TO_ID[token] not in valid and not (token == EOS and not pending):
                raise ValueError(f"token is not valid under grammar state: {token}")
            pending, count = self.advance(pending, count, token)
        if pending or count != ast_nodes(canonical):
            raise ValueError("grammar validation did not terminate cleanly")
        return canonical


def encode_demonstrations(demonstrations: Sequence[Mapping[str, int]]) -> Tensor:
    if len(demonstrations) != 8:
        raise ValueError(f"B3 preflight expects eight demonstrations, got {len(demonstrations)}")
    values = torch.zeros(8 * 30, dtype=torch.float32)
    for row_index, row in enumerate(demonstrations):
        a, b, output = int(row["a"]), int(row["b"]), int(row["output"])
        if not all(0 <= value <= 9 for value in (a, b, output)):
            raise ValueError("digit demonstration outside 0..9")
        offset = row_index * 30
        values[offset + a] = 1.0
        values[offset + 10 + b] = 1.0
        values[offset + 20 + output] = 1.0
    return values


def target_token_ids(ast: Mapping[str, Any], grammar: GrammarConstraint) -> Tensor:
    tokens = ast_to_tokens(ast) + [EOS]
    if len(tokens) > grammar.max_tokens:
        raise ValueError("target AST exceeds decoder length")
    tokens += [EOS] * (grammar.max_tokens - len(tokens))
    return torch.tensor([TOKEN_TO_ID[token] for token in tokens], dtype=torch.long)


class DirectASTProposer(nn.Module):
    """Tiny direct AST-token decoder used only for the compatibility gate."""

    def __init__(self, grammar: GrammarConstraint, hidden_size: int = 128) -> None:
        super().__init__()
        self.grammar = grammar
        self.encoder = nn.Sequential(
            nn.Linear(8 * 30, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.decoder = nn.Linear(hidden_size, grammar.max_tokens * len(TOKEN_LIST))

    def forward(self, encoded: Tensor) -> Tensor:
        hidden = self.encoder(encoded)
        return self.decoder(hidden).view(encoded.shape[0], self.grammar.max_tokens, len(TOKEN_LIST))

    @torch.no_grad()
    def generate(self, logits: Tensor, candidate_count: int = 1) -> List[Dict[str, Any]]:
        if logits.ndim != 2:
            raise ValueError("generate expects one task's [tokens, vocabulary] logits")
        candidates: List[Dict[str, Any]] = []
        for candidate_index in range(candidate_count):
            pending, node_count = self.grammar.initial_state()
            selected_tokens: List[str] = []
            confidence_terms: List[float] = []
            for position in range(self.grammar.max_tokens):
                valid_ids = self.grammar.valid_token_ids(pending, node_count)
                if not valid_ids:
                    raise RuntimeError("grammar produced no valid token")
                valid_logits = logits[position, valid_ids]
                order = torch.argsort(valid_logits, descending=True).tolist()
                rank = min(candidate_index, len(order) - 1)
                selected_offset = int(order[rank])
                token_id = valid_ids[selected_offset]
                token = ID_TO_TOKEN[token_id]
                probabilities = torch.softmax(valid_logits, dim=0)
                confidence_terms.append(float(probabilities[selected_offset]))
                selected_tokens.append(token)
                pending, node_count = self.grammar.advance(pending, node_count, token)
                if token == EOS:
                    break
            ast = tokens_to_ast(selected_tokens)
            ast = self.grammar.validate(ast)
            candidates.append({
                "candidate_index": candidate_index,
                "candidate_source": "neural_generated",
                "tokens": selected_tokens,
                "ast": ast,
                "canonical_ast_hash": ast_hash(ast),
                "confidence": sum(confidence_terms) / max(1, len(confidence_terms)),
                "decoder_tokens": len(selected_tokens),
            })
        return candidates


@dataclass
class RuntimeCounters:
    candidate_source: str = "neural_generated"
    candidate_outputs: int = 0
    unique_canonical_ast_outputs: int = 0
    candidate_evaluations: int = 0
    verifier_calls: int = 0
    oracle_calls: int = 0
    proposer_forward_passes: int = 0
    decoder_tokens: int = 0
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    wall_clock_seconds: float = 0.0
    peak_memory_bytes: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ASTVerifier:
    """Deterministic verifier/executor boundary for canonical AST proposals."""

    def __init__(self, grammar: GrammarConstraint) -> None:
        self.grammar = grammar

    def verify(self, proposal_ast: Mapping[str, Any], examples: Sequence[Mapping[str, int]]) -> Tuple[bool, int]:
        ast = self.grammar.validate(proposal_ast)
        checks = 0
        for row in examples:
            checks += 1
            if evaluate_ast(ast, int(row["a"]), int(row["b"])) != int(row["output"]):
                return False, checks
        return True, checks


class ASTProcedureStore:
    """Minimal versioned AST store for the compatibility lifecycle smoke test."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self.state: Dict[str, Any] = {"schema_version": "crucible.procedure.v1", "procedures": {}, "events": []}
        if path is not None and path.exists():
            self.state = json.loads(path.read_text(encoding="utf-8"))

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def promote(self, alias: str, ast: Mapping[str, Any], provenance: Mapping[str, Any]) -> Dict[str, Any]:
        canonical = canonical_ast(ast)
        versions = self.state.setdefault("procedures", {}).setdefault(alias, [])
        record = {
            "alias": alias,
            "version": len(versions) + 1,
            "language": "digit_expr_mod10_v1",
            "ast": canonical,
            "canonical_ast_hash": ast_hash(canonical),
            "status": "validated",
            "provenance": dict(provenance),
        }
        versions.append(record)
        self.state.setdefault("events", []).append({"kind": "promotion", "alias": alias, "version": record["version"]})
        self.save()
        return record

    def active(self, alias: str) -> Optional[Dict[str, Any]]:
        for record in reversed(self.state.get("procedures", {}).get(alias, [])):
            if record.get("status") == "validated":
                return record
        return None

    def execute(self, alias: str, a: int, b: int) -> Optional[int]:
        record = self.active(alias)
        return None if record is None else evaluate_ast(record["ast"], a, b)

    def invalidate(self, alias: str, reason: str) -> bool:
        record = self.active(alias)
        if record is None:
            return False
        record["status"] = "invalidated"
        record.setdefault("revision_history", []).append({"reason": reason})
        self.state.setdefault("events", []).append({"kind": "invalidation", "alias": alias, "reason": reason})
        self.save()
        return True


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def truth_table_hash(ast: Mapping[str, Any]) -> str:
    values = [evaluate_ast(ast, a, b) for a in range(10) for b in range(10)]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()

