"""Lesson lifecycle: leak guard, de-duplication, verification and deprecation.

    candidate --replay passes--> verified --online lift < 0--> deprecated
        |                                  
        +--replay fails--> rejected        near-duplicates --> merged
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from ..bench.task import Task
from .utility import Utility, lesson_utility

MERGE_COSINE = 0.9
MAX_WORDS = 45
DEPRECATE_MIN_INJECTED = 6
DEPRECATE_P_NEGATIVE = 0.9


@dataclass(frozen=True)
class Policy:
    """Knobs that differ between experimental conditions."""

    retrieve_statuses: tuple[str, ...] = ("verified",)
    verify: bool = True
    replay_seeds: int = 10  # upper bound; replay stops early once decisive
    holdout: float = 0.15


RAW_POLICY = Policy(retrieve_statuses=("candidate", "verified"), verify=False, holdout=0.0)
VERIFIED_POLICY = Policy()


def _literals(value, out: set[str]) -> set[str]:
    if isinstance(value, dict):
        for v in value.values():
            _literals(v, out)
    elif isinstance(value, list):
        for v in value:
            _literals(v, out)
    elif isinstance(value, str) and len(value.strip()) >= 3:
        out.add(value.strip().lower())
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and abs(value) >= 10:
        out.add(str(value))
    return out


def leak_reason(text: str, task: Task) -> str | None:
    """Return why a lesson would leak task-specific answers, or None if it is clean."""
    lowered = text.lower()
    if len(text.split()) > MAX_WORDS:
        return "too long to be a general rule"
    for name in {task.id, task.entry_point}:
        if name.lower() in lowered:
            return f"mentions the task name `{name}`"
    literals: set[str] = set()
    for c in task.tests:
        _literals(c["args"], literals)
        _literals(c["expected"], literals)
    for lit in literals:
        if not lit.startswith("{") and re.search(rf"(?<![\w]){re.escape(lit)}(?![\w])", lowered):
            return f"quotes a test literal {lit!r}"
    return None


def find_duplicate(store, namespace: str, pattern: str, embedding: np.ndarray) -> dict | None:
    rows = store.lessons(with_vectors=True, namespace=namespace, statuses=("candidate", "verified"))
    best, best_cos = None, MERGE_COSINE
    for r in rows:
        if r["pattern"] != pattern:
            continue
        cos = float(np.dot(r["embedding"], embedding))
        if cos >= best_cos:
            best, best_cos = r, cos
    return best


def record_online_evidence(store, events: list[dict]) -> None:
    """Fold resolved retrieval events into each lesson's Beta counts."""
    for e in events:
        lesson = e["lesson"]
        avoided = not e["pattern_hit"]
        if e["withheld"]:
            fields = {"alpha0": lesson["alpha0"] + avoided, "beta0": lesson["beta0"] + (not avoided)}
        else:
            fields = {"alpha": lesson["alpha"] + avoided, "beta": lesson["beta"] + (not avoided),
                      "uses": lesson["uses"] + 1}
        store.update_lesson(lesson["lesson_id"], **fields)


def refresh_status(store, namespace: str, step: int) -> list[tuple[int, str, Utility]]:
    """Recompute utilities, append history, deprecate lessons that demonstrably hurt."""
    rows = store.lessons(namespace=namespace)
    changes = []
    for l in rows:
        if l["status"] not in ("verified", "candidate"):
            continue
        u = lesson_utility(l, rows)
        history = list(l["history"]) + [[step, round(u.lift_mean, 4)]]
        fields: dict = {"history": history[-100:]}
        if (
            l["status"] == "verified"
            and u.n_injected >= DEPRECATE_MIN_INJECTED
            and (1 - u.p_positive) >= DEPRECATE_P_NEGATIVE
        ):
            fields["status"] = "deprecated"
            changes.append((l["lesson_id"], "deprecated", u))
        store.update_lesson(l["lesson_id"], **fields)
    return changes


def summarize(lesson: dict, namespace_lessons: list[dict]) -> dict:
    u = lesson_utility(lesson, namespace_lessons)
    out = {k: v for k, v in lesson.items() if k not in ("embedding", "context")}
    out["utility"] = u.as_dict()
    if isinstance(out.get("replay"), str):
        out["replay"] = json.loads(out["replay"])
    return out
