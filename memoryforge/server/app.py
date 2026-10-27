"""Dashboard backend: a thin JSON API over results files and the memory store."""

from __future__ import annotations

import difflib
import json
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..bench import all_tasks, get_task
from ..config import settings
from ..memory.lessons import summarize
from ..memory.store import open_store

STATIC = Path(__file__).with_name("static")
app = FastAPI(title="MemoryForge", docs_url="/api/docs")
_store = None
_jobs: dict[str, dict] = {}


def store():
    global _store
    if _store is None:
        _store = open_store()
    return _store


def runs_dir() -> Path:
    return Path(settings.runs_dir)


@lru_cache(maxsize=16)
def _load(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def load_results(run_id: str) -> dict:
    path = runs_dir() / run_id / "results.json"
    if not path.exists():
        raise HTTPException(404, f"no results for run {run_id}")
    return _load(str(path), path.stat().st_mtime)


@app.get("/api/runs")
def list_runs() -> list[dict]:
    out = []
    for p in sorted(runs_dir().glob("*/results.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta = _load(str(p), p.stat().st_mtime)["meta"]
        except (ValueError, KeyError):
            continue
        out.append({k: meta.get(k) for k in ("run_id", "created", "backend", "simulated", "seeds", "n_train",
                                             "n_eval", "max_trials", "duration_s", "embedder", "status")})
    return out


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    res = load_results(run_id)
    return {k: v for k, v in res.items() if k != "rows"}


@app.get("/api/runs/{run_id}/rows")
def get_rows(run_id: str) -> list[dict]:
    return load_results(run_id)["rows"]


@app.get("/api/runs/{run_id}/lessons")
def get_lessons(run_id: str, memory: str = "verified", seed: int | None = None, status: str | None = None,
                q: str | None = None) -> dict:
    rows = []
    for l in store().lessons():
        parts = l["namespace"].split("/")
        if len(parts) != 3 or parts[0] != run_id or parts[1] != memory:
            continue
        if seed is not None and parts[2] != f"s{seed}":
            continue
        rows.append(l)
    by_ns: dict[str, list[dict]] = {}
    for l in rows:
        by_ns.setdefault(l["namespace"], []).append(l)
    items = []
    for l in rows:
        if status and l["status"] != status:
            continue
        if q and q.lower() not in l["text"].lower() and q.lower() not in l["pattern"]:
            continue
        s = summarize(l, by_ns[l["namespace"]])
        s["seed"] = int(l["namespace"].split("/s")[-1])
        items.append(s)
    counts: dict[str, int] = {}
    for l in rows:
        counts[l["status"]] = counts.get(l["status"], 0) + 1
    return {"total": len(items), "counts": counts, "items": items}


@app.get("/api/tasks")
def list_tasks() -> list[dict]:
    return [{"id": t.id, "split": t.split, "signature": t.signature, "pitfalls": list(t.pitfalls)} for t in all_tasks()]


@app.get("/api/tasks/{task_id}")
def task_detail(task_id: str) -> dict:
    try:
        t = get_task(task_id)
    except KeyError:
        raise HTTPException(404, "unknown task") from None
    return {"id": t.id, "split": t.split, "spec": t.spec, "signature": t.signature, "pitfalls": list(t.pitfalls),
            "n_tests": len(t.tests)}


@app.get("/api/runs/{run_id}/attempts")
def get_attempts(run_id: str, task_id: str, condition: str, seed: int = 0, phase: str = "eval") -> dict:
    attempts = [a for a in store().attempts(run_id=run_id, task_id=task_id, condition=condition, phase=phase)
                if a["seed"] == seed]
    trials, prev = [], None
    for a in attempts:
        diff = None
        if prev is not None:
            diff = "\n".join(difflib.unified_diff(prev.splitlines(), a["code"].splitlines(),
                                                  "previous", "this trial", lineterm="", n=2))
        trials.append({k: a[k] for k in ("trial", "passed", "n_passed", "n_tests", "feedback", "pattern",
                                         "pattern_conf", "true_pattern", "reflection", "retrieved", "code", "tokens")}
                      | {"diff": diff})
        prev = a["code"]
    return {"task": task_detail(task_id), "trials": trials}


class LiveRequest(BaseModel):
    seeds: int = Field(2, ge=1, le=10)
    n_train: int = Field(24, ge=2, le=24)
    n_eval: int = Field(12, ge=2, le=12)
    harmful_rate: float = Field(0.2, ge=0.0, le=0.9)


@app.post("/api/live")
def start_live(req: LiveRequest) -> dict:
    if any(j["status"] == "running" for j in _jobs.values()):
        raise HTTPException(409, "a run is already in progress")
    job_id = uuid.uuid4().hex[:8]
    job = {"id": job_id, "status": "running", "done": 0, "total": 0, "log": [], "run_id": None,
           "passed": 0, "attempted": 0, "started": time.time()}
    _jobs[job_id] = job

    def progress(info: dict) -> None:
        job.update(done=info["done"], total=info["total"], run_id=info["run_id"])
        job["attempted"] += 1
        job["passed"] += bool(info.get("passed"))
        mark = "pass" if info.get("passed") else "fail"
        job["log"].append(f"seed {info['seed']} · {info['phase']:5} · {info['condition']:<16} {info['task']:<26} {mark}")
        job["log"] = job["log"][-200:]

    def work() -> None:
        from ..eval.experiment import run_experiment, sim_config_from

        try:
            res = run_experiment(seeds=req.seeds, n_train=req.n_train, n_eval=req.n_eval, store=store(),
                                 sim_config=sim_config_from({"harmful_rate": req.harmful_rate}),
                                 out_dir=runs_dir(), progress=progress)
            job.update(status="done", run_id=res["meta"]["run_id"])
        except Exception as e:  # noqa: BLE001
            job.update(status="error", error=str(e))

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/live/{job_id}")
def live_status(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    return _jobs[job_id]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
