"""Command line entry point.

    memoryforge bench                 # run the memory on/off benchmark
    memoryforge solve <task_id>       # watch one task being solved
    memoryforge validate              # check the benchmark tasks
    memoryforge serve                 # open the dashboard
    memoryforge sweep                 # sensitivity of verification to harmful lessons
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .conditions import CONDITIONS, DEFAULT_CONDITIONS

console = Console()


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def print_results(res: dict) -> None:
    meta = res["meta"]
    if meta["simulated"]:
        console.print("[bold yellow]SIMULATION[/] - lesson effects are parameters of the simulator, "
                      "not measurements of a real model.\n")
    t = Table(title=f"Held-out evaluation - run {meta['run_id']} ({len(meta['seeds'])} seeds x {meta['n_eval']} tasks)")
    for col in ("Condition", "pass@1", "95% CI", f"solved@{meta['max_trials']}", "tokens / solved"):
        t.add_column(col, justify="right" if col != "Condition" else "left")
    for c in meta["conditions"]:
        s = res["summary"][c["key"]]
        lo, hi = s["pass1"]["ci95"]
        t.add_row(c["label"], _pct(s["pass1"]["mean"]), f"[{_pct(lo)}, {_pct(hi)}]",
                  _pct(s["solved"]["mean"]), f"{s['tokens_per_solved']:.0f}" if s["tokens_per_solved"] else "-")
    console.print(t)
    ct = Table(title="Paired contrasts (pass@1)")
    for col in ("A vs B", "diff", "95% CI", "McNemar b/c", "p (Holm)"):
        ct.add_column(col)
    for c in res["contrasts"]:
        if c["metric"] != "pass1":
            continue
        lo, hi = c["ci95"]
        ct.add_row(f"{c['a']} vs {c['b']}", f"{100 * c['diff']:+.1f} pp", f"[{100 * lo:+.1f}, {100 * hi:+.1f}]",
                   f"{c['mcnemar']['b']}/{c['mcnemar']['c']}", f"{c['mcnemar']['p_holm']:.3f}")
    console.print(ct)
    mem = res["memory"]["lessons"]
    for key, cell in mem.items():
        console.print(f"memory ({key}): {cell['per_seed']} lessons/seed, status {cell['by_status']}, "
                      f"harmful still retrievable: {cell['harmful_live']}")
    console.print(f"retrieval precision: {res['memory']['retrieval_precision']}")
    if res["classifier"].get("n"):
        console.print(f"failure classifier accuracy: {_pct(res['classifier']['accuracy'])} on {res['classifier']['n']} failures")


def cmd_bench(args: argparse.Namespace) -> int:
    from .eval.experiment import run_experiment, sim_config_from

    overrides = json.loads(args.sim) if args.sim else {}
    with Progress(TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console, transient=True) as bar:
        job = bar.add_task("benchmark", total=None)

        def progress(info: dict) -> None:
            bar.update(job, total=info["total"], completed=info["done"],
                       description=f"seed {info['seed']} {info['phase']:5} {info['condition']}")

        res = run_experiment(
            conditions=tuple(args.conditions.split(",")), seeds=args.seeds, backend=args.backend,
            sim_config=sim_config_from(overrides), max_trials=args.trials, top_k=args.top_k,
            embedder=args.embedder, out_dir=args.out, n_train=args.n_train, n_eval=args.n_eval,
            progress=progress,
        )
    print_results(res)
    console.print(f"\nresults written to [bold]{args.out}/{res['meta']['run_id']}/results.json[/]")
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    from .agent import Agent
    from .bench import get_task
    from .llm import make_llm
    from .memory.embeddings import get_embedder
    from .memory.store import open_store

    task = get_task(args.task_id)
    store = open_store()
    agent = Agent(make_llm(args.backend), store, get_embedder(args.embedder), CONDITIONS[args.condition],
                  run_id="adhoc", namespace=args.namespace, seed=args.seed, max_trials=args.trials)
    res = agent.solve(task, step=0, phase="train", learn=args.learn)
    for rec in res.trials_log:
        status = "[green]PASS[/]" if rec.passed else f"[red]FAIL[/] ({rec.pattern})"
        console.print(f"trial {rec.trial}: {status}  lessons={rec.retrieved}")
        if rec.reflection:
            console.print(f"   reflection: {rec.reflection}")
    for l in res.new_lessons:
        console.print(f"lesson [{l['pattern']}] {l['status']}: {l['text']}")
    return 0 if res.passed else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    from .eval.experiment import run_experiment, sim_config_from

    rows = []
    for rate in [float(x) for x in args.rates.split(",")]:
        res = run_experiment(conditions=("none", "raw_memory", "verified"), seeds=args.seeds,
                             sim_config=sim_config_from({"harmful_rate": rate}), embedder=args.embedder,
                             out_dir=args.out, run_id=None)
        s = res["summary"]
        rows.append((rate, s["none"]["pass1"]["mean"], s["raw_memory"]["pass1"]["mean"], s["verified"]["pass1"]["mean"]))
        console.print(f"harmful_rate={rate:.2f}  none={_pct(rows[-1][1])}  raw={_pct(rows[-1][2])}  verified={_pct(rows[-1][3])}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    console.print(f"MemoryForge dashboard on [bold]http://{args.host}:{args.port}[/]")
    uvicorn.run("memoryforge.server.app:app", host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from .bench.validate import main

    return main([])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="memoryforge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="run the memory on/off benchmark")
    b.add_argument("--seeds", type=int, default=5)
    b.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    b.add_argument("--backend", default="simulated", choices=["simulated", "openai_compatible", "hf_local"])
    b.add_argument("--trials", type=int, default=3)
    b.add_argument("--top-k", type=int, default=3)
    b.add_argument("--embedder", default=None, choices=[None, "auto", "minilm", "hash"])
    b.add_argument("--n-train", type=int, default=None, help="use only the first N training tasks")
    b.add_argument("--n-eval", type=int, default=None, help="use only the first N eval tasks")
    b.add_argument("--sim", default=None, help='JSON overrides for the simulator, e.g. \'{"harmful_rate": 0.4}\'')
    b.add_argument("--out", default="runs")
    b.set_defaults(fn=cmd_bench)

    s = sub.add_parser("solve", help="solve one task and show each trial")
    s.add_argument("task_id")
    s.add_argument("--backend", default=None)
    s.add_argument("--condition", default="verified", choices=list(CONDITIONS))
    s.add_argument("--namespace", default="adhoc")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--trials", type=int, default=3)
    s.add_argument("--embedder", default=None)
    s.add_argument("--learn", action="store_true", help="write lessons to memory")
    s.set_defaults(fn=cmd_solve)

    w = sub.add_parser("sweep", help="vary the simulator's harmful-lesson rate")
    w.add_argument("--rates", default="0,0.2,0.4,0.6")
    w.add_argument("--seeds", type=int, default=5)
    w.add_argument("--embedder", default=None)
    w.add_argument("--out", default="runs")
    w.set_defaults(fn=cmd_sweep)

    v = sub.add_parser("serve", help="start the dashboard")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8765)
    v.set_defaults(fn=cmd_serve)

    sub.add_parser("validate", help="check every benchmark task").set_defaults(fn=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
