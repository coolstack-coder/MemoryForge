"""Benchmark task definition.

Every task ships hidden tests, a reference solution, and a few hand-written
buggy variants, each tagged with the failure pattern it exhibits. The variants
power the offline simulator and let us measure classifier accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


def case(*args: Any, expected: Any, check_alias: bool = False) -> dict:
    """A test case. Arguments and expected value must be JSON-serialisable.

    Use expr("...") for large generated inputs and raises("ValueError") for
    expected exceptions.
    """
    c = {"args": list(args), "expected": expected}
    if check_alias:
        c["check_alias"] = True
    return c


def expr(source: str) -> dict:
    return {"__expr__": source}


def raises(exc_name: str) -> dict:
    return {"__raises__": exc_name}


@dataclass(frozen=True)
class Task:
    id: str
    split: Literal["train", "eval"]
    entry_point: str
    signature: str
    spec: str
    tests: list[dict]
    reference: str
    bugs: dict[str, str]  # failure pattern -> buggy implementation
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pitfalls(self) -> tuple[str, ...]:
        return tuple(self.bugs)
