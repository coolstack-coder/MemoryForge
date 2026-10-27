"""Paired statistics for memory-on vs memory-off comparisons.

The unit of analysis is a task: outcomes are averaged over seeds first, then
compared task-by-task between two conditions. Paired bootstrap gives a CI on
the difference; McNemar's exact test on (seed, task) pairs gives a p-value for
pass@1. Seeds are not independent draws of tasks, so treat the McNemar p-value
as descriptive and lead with the bootstrap interval.
"""

from __future__ import annotations

from collections import defaultdict
from math import comb

import numpy as np


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b (A pass, B fail) vs c (A fail, B pass)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def _by_task(rows: list[dict], metric: str) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        acc[r["task_id"]].append(float(r[metric]))
    return {t: float(np.mean(v)) for t, v in acc.items()}


def summarize_condition(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for metric in ("pass1", "solved"):
        per_task = np.array(list(_by_task(rows, metric).values()))
        lo, hi = bootstrap_ci(per_task)
        out[metric] = {"mean": float(per_task.mean()) if len(per_task) else 0.0, "ci95": [lo, hi]}
    solved = [r["trials"] for r in rows if r["solved"]]
    out["mean_trials_solved"] = float(np.mean(solved)) if solved else None
    out["mean_trials_censored"] = float(np.mean([r["trials"] if r["solved"] else r["trials"] + 1 for r in rows])) if rows else None
    out["tokens_per_task"] = float(np.mean([r["tokens"] for r in rows])) if rows else 0.0
    n_solved = sum(r["solved"] for r in rows)
    out["tokens_per_solved"] = float(sum(r["tokens"] for r in rows) / n_solved) if n_solved else None
    return out


def contrast(a_rows: list[dict], b_rows: list[dict], metric: str) -> dict:
    a_task, b_task = _by_task(a_rows, metric), _by_task(b_rows, metric)
    tasks = sorted(set(a_task) & set(b_task))
    diffs = np.array([a_task[t] - b_task[t] for t in tasks])
    lo, hi = bootstrap_ci(diffs, seed=1)
    a_pair = {(r["seed"], r["task_id"]): bool(r[metric]) for r in a_rows}
    b_pair = {(r["seed"], r["task_id"]): bool(r[metric]) for r in b_rows}
    keys = set(a_pair) & set(b_pair)
    b = sum(a_pair[k] and not b_pair[k] for k in keys)
    c = sum(b_pair[k] and not a_pair[k] for k in keys)
    return {
        "metric": metric,
        "diff": float(diffs.mean()) if len(diffs) else 0.0,
        "ci95": [lo, hi],
        "mcnemar": {"b": b, "c": c, "n_pairs": len(keys), "p": mcnemar_exact(b, c)},
        "n_tasks": len(tasks),
    }


def learning_curve(rows: list[dict], window: int = 6) -> dict:
    """Rolling pass@1 over the experience stream, averaged over seeds."""
    by_seed: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append((r["step"], float(r["pass1"])))
    if not by_seed:
        return {"x": [], "y": [], "lo": [], "hi": []}
    n = max(len(v) for v in by_seed.values())
    curves = []
    for seq in by_seed.values():
        y = np.array([v for _, v in sorted(seq)])
        cum = np.cumsum(y)
        roll = np.array([(cum[i] - (cum[i - window] if i >= window else 0)) / min(i + 1, window) for i in range(len(y))])
        curves.append(np.pad(roll, (0, n - len(roll)), constant_values=np.nan))
    M = np.array(curves)
    mean = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0)
    k = np.sqrt(np.sum(~np.isnan(M), axis=0))
    return {"x": list(range(1, n + 1)), "y": mean.round(4).tolist(),
            "lo": (mean - 1.96 * sd / k).round(4).tolist(), "hi": (mean + 1.96 * sd / k).round(4).tolist()}


def by_pattern(rows: list[dict], conditions: list[str]) -> dict:
    """pass@1 per condition on eval tasks grouped by each pitfall they contain."""
    out: dict[str, dict] = {}
    for r in rows:
        for p in r["pitfalls"]:
            cell = out.setdefault(p, {c: [] for c in conditions})
            if r["condition"] in cell:
                cell[r["condition"]].append(float(r["pass1"]))
    return {
        p: {"n_tasks": len({r["task_id"] for r in rows if p in r["pitfalls"]}),
            **{c: (float(np.mean(v)) if v else None) for c, v in cell.items()}}
        for p, cell in sorted(out.items())
    }
