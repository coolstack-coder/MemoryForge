"""Run the memory-on vs memory-off benchmark.

Protocol (per seed):
  1. Experience phase: every condition that owns a memory works through the
     24 training tasks in a seed-specific order, writing lessons as it goes.
  2. Freeze memory.
  3. Evaluation phase: every condition attempts the 12 held-out tasks with
     no memory writes. `verified_rerank` and `shuffled` reuse the memory that
     `verified` built, so they differ only in how lessons are retrieved.
"""

from __future__ import annotations

import json
import random
import subprocess
import time
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..agent import Agent, TaskResult
from ..bench import eval_tasks, train_tasks
from ..conditions import CONDITIONS, DEFAULT_CONDITIONS
from ..failures_vocab import PATTERNS
from ..llm import LLM, SimulatedLLM, make_llm
from ..llm.simulated import HARMFUL_TEMPLATES, SimConfig
from ..memory.embeddings import get_embedder
from ..memory.reranker import train_reranker
from ..memory.store import Store, open_store
from . import stats

Progress = Callable[[dict], None]


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _row(seed: int, cond: str, phase: str, step: int, task, res: TaskResult) -> dict:
    return {"seed": seed, "condition": cond, "phase": phase, "step": step, "task_id": task.id,
            "pass1": res.pass1, "solved": res.passed, "trials": res.trials, "tokens": res.tokens,
            "pitfalls": list(task.pitfalls)}


def run_experiment(
    *,
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    seeds: int = 5,
    backend: str = "simulated",
    sim_config: SimConfig | None = None,
    max_trials: int = 3,
    top_k: int = 3,
    embedder: str | None = None,
    store: Store | None = None,
    out_dir: str | Path = "runs",
    run_id: str | None = None,
    n_train: int | None = None,
    n_eval: int | None = None,
    progress: Progress | None = None,
) -> dict:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    store = store or open_store()
    emb = get_embedder(embedder)
    llm: LLM = SimulatedLLM(sim_config) if backend == "simulated" else make_llm(backend)
    conds = [CONDITIONS[c] for c in conditions]
    builders = []
    for c in conds:
        key = c.memory_from or c.key
        if key not in builders:
            builders.append(key)
    train_set = train_tasks()[:n_train] if n_train else train_tasks()
    eval_set = eval_tasks()[:n_eval] if n_eval else eval_tasks()

    meta = {
        "run_id": run_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": llm.name,
        "simulated": llm.is_simulation,
        "sim_config": asdict(llm.config) if isinstance(llm, SimulatedLLM) else None,
        "conditions": [{"key": c.key, "label": c.label, "description": c.description} for c in conds],
        "seeds": list(range(seeds)),
        "max_trials": max_trials,
        "top_k": top_k,
        "n_train": len(train_set),
        "n_eval": len(eval_set),
        "embedder": emb.name,
        "store": store.dialect,
        "git_sha": _git_sha(),
        "status": "running",
    }
    store.create_run(run_id, meta)
    total_units = seeds * (len(builders) * len(train_set) + len(conds) * len(eval_set))
    done = 0
    t0 = time.time()

    def tick(**info) -> None:
        nonlocal done
        done += 1
        if progress:
            progress({"run_id": run_id, "done": done, "total": total_units, **info})

    rows: list[dict] = []
    rerank_reports: list[dict] = []
    for seed in range(seeds):
        order = list(train_set)
        random.Random(seed).shuffle(order)
        agents: dict[str, Agent] = {}
        for key in builders:
            ns = f"{run_id}/{key}/s{seed}"
            agent = Agent(llm, store, emb, CONDITIONS[key], run_id=run_id, namespace=ns, seed=seed,
                          max_trials=max_trials, top_k=top_k)
            agents[key] = agent
            for step, task in enumerate(order):
                res = agent.solve(task, step=step, phase="train", learn=True)
                rows.append(_row(seed, key, "train", step, task, res))
                tick(seed=seed, condition=key, phase="train", task=task.id, passed=res.passed)

        for c in conds:
            ns = f"{run_id}/{c.memory_from or c.key}/s{seed}"
            reranker = None
            if c.rerank:
                events = []
                for key in builders:
                    if CONDITIONS[key].cross_task:
                        events += store.retrieval_events(namespace=f"{run_id}/{key}/s{seed}")
                reranker = train_reranker(events, seed=seed)
                rerank_reports.append({"seed": seed, "trained": reranker is not None,
                                       **(reranker.report() if reranker else {"n_events": len(events)})})
            agent = Agent(llm, store, emb, c, run_id=run_id, namespace=ns, seed=seed,
                          max_trials=max_trials, top_k=top_k, reranker=reranker)
            for i, task in enumerate(eval_set):
                res = agent.solve(task, step=len(order) + i, phase="eval", learn=False)
                rows.append(_row(seed, c.key, "eval", len(order) + i, task, res))
                tick(seed=seed, condition=c.key, phase="eval", task=task.id, passed=res.passed)

    results = analyze(store, run_id, meta, rows, rerank_reports, llm)
    results["meta"]["status"] = "done"
    results["meta"]["duration_s"] = round(time.time() - t0, 1)
    results["meta"]["usage"] = llm.meter.snapshot()
    store.update_run(run_id, results["meta"])
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    return results


CONTRASTS = [
    ("verified", "none"),
    ("verified", "reflexion"),
    ("raw_memory", "none"),
    ("verified", "raw_memory"),
    ("verified", "shuffled"),
    ("verified_rerank", "verified"),
    ("reflexion", "none"),
]


def analyze(store: Store, run_id: str, meta: dict, rows: list[dict], rerank_reports: list[dict], llm) -> dict:
    cond_keys = [c["key"] for c in meta["conditions"]]
    eval_rows = [r for r in rows if r["phase"] == "eval"]
    train_rows = [r for r in rows if r["phase"] == "train"]
    by_cond = {c: [r for r in eval_rows if r["condition"] == c] for c in cond_keys}

    contrasts = []
    for a, b in CONTRASTS:
        if a in by_cond and b in by_cond and by_cond[a] and by_cond[b]:
            for metric in ("pass1", "solved"):
                contrasts.append({"a": a, "b": b, **stats.contrast(by_cond[a], by_cond[b], metric)})
    adj = stats.holm([c["mcnemar"]["p"] for c in contrasts]) if contrasts else []
    for c, p in zip(contrasts, adj):
        c["mcnemar"]["p_holm"] = p

    return {
        "meta": meta,
        "summary": {c: stats.summarize_condition(v) for c, v in by_cond.items()},
        "contrasts": contrasts,
        "learning_curve": {
            c: stats.learning_curve([r for r in train_rows if r["condition"] == c])
            for c in sorted({r["condition"] for r in train_rows})
        },
        "by_pattern": stats.by_pattern(eval_rows, cond_keys),
        "memory": memory_report(store, run_id, llm.is_simulation),
        "classifier": classifier_report(store, run_id),
        "reranker": rerank_reports,
        "rows": rows,
    }


def memory_report(store: Store, run_id: str, simulated: bool) -> dict:
    from ..bench import get_task

    harmful = set(HARMFUL_TEMPLATES.values())
    out: dict[str, dict] = {}
    for l in store.lessons():
        if not l["namespace"].startswith(run_id + "/"):
            continue
        key = l["namespace"].split("/")[1]
        cell = out.setdefault(key, {"total": 0, "by_status": {}, "on_target": 0, "harmful_total": 0,
                                    "harmful_live": 0, "seeds": set()})
        cell["total"] += 1
        cell["seeds"].add(l["namespace"])
        cell["by_status"][l["status"]] = cell["by_status"].get(l["status"], 0) + 1
        try:
            cell["on_target"] += l["pattern"] in get_task(l["source_task"]).bugs
        except KeyError:
            pass
        if simulated and l["text"] in harmful:
            cell["harmful_total"] += 1
            cell["harmful_live"] += l["status"] in ("verified", "candidate") and key == "raw_memory" or l["status"] == "verified"
    for cell in out.values():
        n = max(1, len(cell.pop("seeds")))
        cell["per_seed"] = round(cell["total"] / n, 2)

    # retrieval precision on eval: retrieved lesson targets a pitfall the task really has
    precision: dict[str, list[int]] = {}
    harmful_retrieved: dict[str, int] = {}
    for a in store.attempts(run_id=run_id, phase="eval"):
        task = get_task(a["task_id"])
        for r in a["retrieved"]:
            precision.setdefault(a["condition"], []).append(int(r["pattern"] in task.bugs))
            if simulated and r["text"] in harmful:
                harmful_retrieved[a["condition"]] = harmful_retrieved.get(a["condition"], 0) + 1
    return {
        "lessons": out,
        "retrieval_precision": {k: round(sum(v) / len(v), 3) for k, v in precision.items() if v},
        "retrievals": {k: len(v) for k, v in precision.items()},
        "harmful_retrieved": harmful_retrieved,
    }


def classifier_report(store: Store, run_id: str) -> dict:
    rows = [a for a in store.attempts(run_id=run_id) if not a["passed"] and a["true_pattern"]]
    if not rows:
        return {"n": 0}
    confusion: dict[str, dict[str, int]] = {}
    for a in rows:
        confusion.setdefault(a["true_pattern"], {})
        confusion[a["true_pattern"]][a["pattern"]] = confusion[a["true_pattern"]].get(a["pattern"], 0) + 1
    correct = sum(a["pattern"] == a["true_pattern"] for a in rows)
    per_pattern = {p: round(confusion.get(p, {}).get(p, 0) / max(1, sum(confusion.get(p, {}).values())), 3)
                   for p in PATTERNS if p in confusion}
    return {"n": len(rows), "accuracy": round(correct / len(rows), 3), "per_pattern": per_pattern,
            "confusion": confusion}


def sim_config_from(overrides: dict) -> SimConfig:
    return replace(SimConfig(), **overrides)
