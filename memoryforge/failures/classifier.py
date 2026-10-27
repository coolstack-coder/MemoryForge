"""Rule-based failure classifier.

Maps one failed execution (test outcomes + source code) onto the shared
pattern vocabulary. Signals come from three places: the exception or timeout,
the shape of the failing input vs expected vs actual, and cheap AST checks on
the code. Confidences from different rules for the same pattern combine by
noisy-or.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

from ..sandbox.runner import ExecResult, TestOutcome


@dataclass(frozen=True)
class Diagnosis:
    pattern: str
    confidence: float
    ranked: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict:
        return {"pattern": self.pattern, "confidence": round(self.confidence, 3),
                "ranked": [[p, round(c, 3)] for p, c in self.ranked]}


def _loads(text: str | None):
    if text is None:
        return None
    try:
        return json.loads(text.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false"))
    except ValueError:
        return None


def _is_empty_like(value) -> bool:
    if value in ([], "", {}, 0, None):
        return True
    return isinstance(value, (list, str, dict)) and len(value) <= 1


def _has_negative(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value < 0
    if isinstance(value, list):
        return any(_has_negative(v) for v in value)
    return False


def _flatten_numbers(value):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _flatten_numbers(v)


def _text_differs_only_in_spacing(a, e) -> bool:
    if not (isinstance(a, (str, list)) and isinstance(e, (str, list))):
        return False
    norm = lambda x: re.sub(r"\s+", "", json.dumps(x, ensure_ascii=False).lower())  # noqa: E731
    return a != e and norm(a) == norm(e)


def _same_multiset(a, e) -> bool:
    if not (isinstance(a, list) and isinstance(e, list)) or a == e or len(a) != len(e):
        return False
    key = lambda x: json.dumps(x, sort_keys=True)  # noqa: E731
    return sorted(map(key, a)) == sorted(map(key, e))


def _off_by_one_shape(a, e) -> bool:
    na, ne = list(_flatten_numbers(a)), list(_flatten_numbers(e))
    if isinstance(a, (int, float)) and isinstance(e, (int, float)) and not isinstance(a, bool):
        return abs(a - e) == 1
    if isinstance(a, list) and isinstance(e, list):
        if abs(len(a) - len(e)) == 1:
            return True
        if len(na) == len(ne) and na != ne:
            diffs = [abs(x - y) for x, y in zip(na, ne) if x != y]
            return bool(diffs) and all(d == 1 for d in diffs)
    return False


def _static_flags(code: str) -> set[str]:
    flags: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return flags
    func = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    params = {a.arg for a in func.args.args} if func else set()
    loops = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = node.func.value
            if node.func.attr in {"sort", "pop", "append", "extend", "remove", "insert", "reverse", "clear", "update"} \
                    and isinstance(target, ast.Name) and target.id in params:
                flags.add("mutates_param")
        if isinstance(node, (ast.For, ast.While)):
            inner = [n for n in ast.walk(node) if isinstance(n, (ast.For, ast.While)) and n is not node]
            if inner:
                loops += 1
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            flags.add("float_div")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"round", "float"}:
            flags.add("float_round")
    if loops:
        flags.add("nested_loops")
    return flags


def classify(result: ExecResult, code: str) -> Diagnosis:
    scores: dict[str, float] = {}

    def add(pattern: str, conf: float) -> None:
        scores[pattern] = 1 - (1 - scores.get(pattern, 0.0)) * (1 - conf)

    flags = _static_flags(code)
    failures: list[TestOutcome] = result.failures

    if result.crashed:
        add("inefficient" if "time limit" in result.crashed else "crash", 0.8)
    if result.load_error:
        add("crash", 0.9)

    for t in failures:
        inp, exp = _loads(t.input), _loads(t.expected)
        act = t.actual_json if t.actual_json is not None else _loads(t.actual)
        args = inp if isinstance(inp, list) else [inp]
        if t.timeout:
            add("inefficient", 0.9 if "nested_loops" in flags else 0.75)
        if t.mutated:
            add("mutates_input", 0.95)
        if t.aliased:
            add("shared_reference", 0.95)
        if t.expects_raise:
            add("error_contract", 0.9)
            continue
        if t.exc_type:
            if t.exc_type in {"IndexError", "ZeroDivisionError", "StopIteration", "ValueError"} and any(
                _is_empty_like(a) for a in args
            ):
                add("empty_input", 0.8)
            elif t.exc_type == "IndexError":
                add("off_by_one", 0.5)
            elif t.exc_type in {"ValueError", "TypeError", "KeyError", "AttributeError"}:
                add("crash", 0.4)
            else:
                add("crash", 0.5)
            continue
        if t.mutated or t.aliased:
            continue
        if any(a in ([], "", {}) for a in args) or (isinstance(exp, list) and exp == [] ) or act in ([[]], [""]):
            add("empty_input", 0.6)
        if _same_multiset(act, exp):
            add("tie_ordering", 0.85)
        if _text_differs_only_in_spacing(act, exp):
            add("whitespace_case", 0.8)
        if any(isinstance(a, str) and (a != a.strip() or "\t" in a or "  " in a or a != a.lower()) for a in args):
            add("whitespace_case", 0.35)
        if _off_by_one_shape(act, exp):
            add("off_by_one", 0.6)
        if isinstance(act, float) and isinstance(exp, (int, float)) and abs(act - exp) < 1e-2 * max(1, abs(exp)):
            add("float_rounding", 0.8)
        if isinstance(act, str) and isinstance(exp, str) and re.sub(r"\d", "", act) == re.sub(r"\d", "", exp):
            add("float_rounding", 0.5 if {"float_div", "float_round"} & flags else 0.25)
        if any(_has_negative(a) for a in args):
            add("negative_numbers", 0.55)

    if "mutates_param" in flags and failures and not scores:
        add("mutates_input", 0.4)
    if not scores:
        add("logic_error", 0.4)

    ranked = tuple(sorted(scores.items(), key=lambda kv: -kv[1]))
    return Diagnosis(pattern=ranked[0][0], confidence=ranked[0][1], ranked=ranked[:3])
