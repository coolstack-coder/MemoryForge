"""Sanity-check benchmark tasks: the reference passes, every bug fails.

    python -m memoryforge.bench.validate [module]
"""

from __future__ import annotations

import importlib
import sys

from ..failures_vocab import PATTERNS
from ..sandbox import extract_function, run_tests
from .task import Task


def check_task(task: Task) -> list[str]:
    problems: list[str] = []
    if f"def {task.entry_point}(" not in task.signature:
        problems.append("signature does not define the entry point")
    if len(task.tests) < 4:
        problems.append("needs at least 4 tests")
    if not task.bugs:
        problems.append("needs at least one buggy variant")
    try:
        extract_function(task.reference, task.entry_point)
    except Exception as e:  # noqa: BLE001
        problems.append(f"reference rejected by extractor: {e}")
    ref = run_tests(task.reference, task.entry_point, task.tests, timeout=20)
    if not ref.passed:
        problems.append(f"reference fails:\n{ref.feedback(5)}")
    for pattern, code in task.bugs.items():
        if pattern not in PATTERNS:
            problems.append(f"unknown bug pattern {pattern!r}")
        res = run_tests(code, task.entry_point, task.tests, timeout=20)
        if res.passed:
            problems.append(f"bug {pattern!r} passes every test")
    return problems


def main(argv: list[str]) -> int:
    if argv:
        tasks = importlib.import_module(f"memoryforge.bench.{argv[0]}").TASKS
    else:
        from .suite import all_tasks

        tasks = all_tasks()
    bad = 0
    for t in tasks:
        problems = check_task(t)
        status = "ok " if not problems else "BAD"
        print(f"[{status}] {t.split:5} {t.id:28} bugs={','.join(t.bugs)}")
        for p in problems:
            bad += 1
            print("      " + p.replace("\n", "\n      "))
    print(f"{len(tasks)} tasks, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
