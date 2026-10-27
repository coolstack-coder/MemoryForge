"""Run candidate code against tests in an isolated, resource-limited subprocess."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HARNESS = Path(__file__).with_name("harness.py")


@dataclass
class TestOutcome:
    input: str
    expected: str
    actual: str | None
    passed: bool
    exc_type: str | None = None
    traceback: str | None = None
    timeout: bool = False
    mutated: bool = False
    actual_is_none: bool = False
    actual_json: object = None
    aliased: bool = False
    expects_raise: str | None = None


@dataclass
class ExecResult:
    passed: bool
    tests: list[TestOutcome] = field(default_factory=list)
    load_error: str | None = None
    crashed: str | None = None  # whole-process failure (timeout, OOM, killed)

    @property
    def n_passed(self) -> int:
        return sum(t.passed for t in self.tests)

    @property
    def failures(self) -> list[TestOutcome]:
        return [t for t in self.tests if not t.passed]

    def feedback(self, max_cases: int = 3) -> str:
        """Human/LLM-readable summary of what went wrong."""
        if self.crashed:
            return f"The run was aborted: {self.crashed}."
        if self.load_error:
            return f"The code failed to load: {self.load_error}"
        lines = [f"{self.n_passed}/{len(self.tests)} tests passed."]
        for t in self.failures[:max_cases]:
            if t.timeout:
                lines.append(f"- input {t.input}: timed out")
            elif t.exc_type:
                lines.append(f"- input {t.input}: raised {t.traceback or t.exc_type}")
            elif t.mutated:
                lines.append(f"- input {t.input}: the function modified its input argument")
            elif t.aliased:
                lines.append(f"- input {t.input}: the result shares mutable objects with the input")
            elif t.expects_raise:
                lines.append(f"- input {t.input}: expected {t.expects_raise} to be raised, got {t.actual}")
            else:
                lines.append(f"- input {t.input}: expected {t.expected}, got {t.actual}")
        return "\n".join(lines)


def _limits(cpu_seconds: int, memory_mb: int):
    def apply() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        if sys.platform.startswith("linux"):  # RLIMIT_AS is unreliable on macOS
            limit = memory_mb << 20
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        os.setsid()

    return apply


def run_tests(
    code: str,
    entry_point: str,
    tests: list[dict],
    *,
    timeout: float = 6.0,
    per_test_timeout: float = 2.0,
    memory_mb: int = 512,
) -> ExecResult:
    sentinel = "@@MF" + secrets.token_hex(8) + "@@"
    with tempfile.TemporaryDirectory(prefix="mf-sbx-") as tmp:
        Path(tmp, "candidate.py").write_text(code)
        Path(tmp, "tests.json").write_text(
            json.dumps({"entry_point": entry_point, "tests": tests, "per_test_timeout": per_test_timeout})
        )
        env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(HARNESS), tmp, sentinel],
                cwd=tmp,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limits(int(timeout) + 1, memory_mb) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(passed=False, crashed=f"exceeded the {timeout:.0f}s time limit")

    parts = proc.stdout.split(sentinel)
    if len(parts) < 3:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
        return ExecResult(passed=False, crashed=tail[0][:300])
    report = json.loads(parts[-2])
    outcomes = [TestOutcome(**row) for row in report["tests"]]
    passed = report["load_error"] is None and bool(outcomes) and all(t.passed for t in outcomes)
    return ExecResult(passed=passed, tests=outcomes, load_error=report["load_error"])
