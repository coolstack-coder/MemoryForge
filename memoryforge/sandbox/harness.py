"""Executed inside the sandbox subprocess. Standard library only.

usage: python -I harness.py <workdir> <sentinel>
Reads candidate.py and tests.json from <workdir>, prints one JSON report
wrapped in the sentinel so anything the candidate prints cannot spoof it.
"""

import copy
import importlib.util
import json
import math
import os
import signal
import sys
import traceback


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b if isinstance(a, bool) and isinstance(b, bool) else False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) or isinstance(b, float):
            return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return a == b


def _materialize(value):
    """Expand {"__expr__": "..."} placeholders (trusted, from the task file)."""
    if isinstance(value, dict) and set(value) == {"__expr__"}:
        return eval(value["__expr__"], {})  # noqa: S307 - task-authored, read before candidate loads
    if isinstance(value, list):
        return [_materialize(v) for v in value]
    return value


def _mutable_ids(value, acc):
    if isinstance(value, (list, dict, set)):
        acc.add(id(value))
        for v in value.values() if isinstance(value, dict) else value:
            _mutable_ids(v, acc)
    return acc


def _short(value, limit=300):
    r = repr(value)
    return r if len(r) <= limit else r[:limit] + "..."


def main():
    workdir, sentinel = sys.argv[1], sys.argv[2]
    with open(f"{workdir}/tests.json") as f:
        spec = json.load(f)
    os.unlink(f"{workdir}/tests.json")  # the candidate must not read expectations from disk
    for case in spec["tests"]:
        case["args"] = _materialize(case["args"])
        case["expected"] = _materialize(case["expected"])
    entry, per_test = spec["entry_point"], float(spec.get("per_test_timeout", 2.0))
    report = {"load_error": None, "tests": []}
    try:
        mod_spec = importlib.util.spec_from_file_location("candidate", f"{workdir}/candidate.py")
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        fn = getattr(mod, entry)
    except BaseException as e:  # noqa: BLE001 - report anything
        report["load_error"] = f"{type(e).__name__}: {e}"
        fn = None

    if fn is not None:
        signal.signal(signal.SIGALRM, _alarm)
        for case in spec["tests"]:
            args, expected = case["args"], case["expected"]
            raises = expected.get("__raises__") if isinstance(expected, dict) else None
            before = copy.deepcopy(args)
            call_args = copy.deepcopy(args)
            row = {"input": _short(args), "expected": _short(expected), "actual": None,
                   "passed": False, "exc_type": None, "traceback": None,
                   "timeout": False, "mutated": False, "actual_is_none": False,
                   "aliased": False, "expects_raise": raises}
            try:
                signal.setitimer(signal.ITIMER_REAL, per_test)
                actual = fn(*call_args)
                signal.setitimer(signal.ITIMER_REAL, 0)
                row["actual"] = _short(actual)
                row["actual_is_none"] = actual is None
                row["passed"] = raises is None and _equal(actual, expected)
                if row["passed"] and case.get("check_alias"):
                    if _mutable_ids(actual, set()) & _mutable_ids(call_args, set()):
                        row["passed"] = False
                        row["aliased"] = True
                try:
                    row["actual_json"] = json.loads(json.dumps(actual))
                except (TypeError, ValueError):
                    row["actual_json"] = None
            except _Timeout:
                row["timeout"] = True
                row["exc_type"] = "Timeout"
            except BaseException as e:  # noqa: BLE001
                signal.setitimer(signal.ITIMER_REAL, 0)
                row["exc_type"] = type(e).__name__
                if raises is not None and type(e).__name__ == raises:
                    row["passed"] = True
                    row["exc_type"] = None
                row["traceback"] = "".join(traceback.format_exception_only(type(e), e)).strip()[-500:]
            row["mutated"] = not _equal(call_args, before)
            if row["mutated"]:
                row["passed"] = False
            report["tests"].append(row)

    sys.stdout.write(f"\n{sentinel}{json.dumps(report)}{sentinel}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
