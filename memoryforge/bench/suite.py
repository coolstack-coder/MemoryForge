"""The MemoryForge benchmark: 24 experience (train) tasks and 12 held-out eval tasks."""

from __future__ import annotations

from .task import Task


def all_tasks() -> list[Task]:
    from . import tasks_eval, tasks_train_a, tasks_train_b

    tasks = [*tasks_train_a.TASKS, *tasks_train_b.TASKS, *tasks_eval.TASKS]
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids in the benchmark")
    return tasks


def train_tasks() -> list[Task]:
    return [t for t in all_tasks() if t.split == "train"]


def eval_tasks() -> list[Task]:
    return [t for t in all_tasks() if t.split == "eval"]


def get_task(task_id: str) -> Task:
    for t in all_tasks():
        if t.id == task_id:
            return t
    raise KeyError(task_id)
