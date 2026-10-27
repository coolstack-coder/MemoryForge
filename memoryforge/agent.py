"""The MemoryForge agent: a Reflexion-style trial loop with persistent, verified memory.

For each task:
  1. retrieve lessons relevant to the task (cross-task conditions only)
  2. generate code, run the hidden tests in the sandbox
  3. on failure: classify the failure pattern, reflect verbally, retry
  4. after the task (learning phase only): distill each diagnosed failure into
     a general lesson, guard it against answer leakage, de-duplicate it, and
     verify it by counterfactual replay before it becomes retrievable
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np

from .bench.task import Task
from .conditions import Condition
from .failures import classify
from .failures_vocab import ALL_PATTERNS
from .llm import LLM
from .llm import prompts as P
from .memory import lessons as L
from .memory.retrieval import Retrieved, Retriever
from .sandbox import ExtractionError, ExecResult, extract_function, run_tests

log = logging.getLogger(__name__)
VOCAB = ", ".join(ALL_PATTERNS)


@dataclass
class TrialRecord:
    trial: int
    attempt_id: int
    passed: bool
    pattern: str | None
    true_pattern: str | None
    retrieved: list[dict]
    reflection: str | None


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    pass1: bool
    trials: int
    tokens: int
    trials_log: list[TrialRecord] = field(default_factory=list)
    new_lessons: list[dict] = field(default_factory=list)


def _norm(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def true_bug_pattern(task: Task, code: str) -> str | None:
    """Ground-truth pattern when the code is one of the task's known buggy variants."""
    c = _norm(code)
    for pattern, bug in task.bugs.items():
        if _norm(bug) == c:
            return pattern
    return None


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


class Agent:
    def __init__(
        self,
        llm: LLM,
        store,
        embedder,
        condition: Condition,
        *,
        run_id: str,
        namespace: str,
        seed: int = 0,
        max_trials: int = 3,
        top_k: int = 3,
        reranker=None,
        sandbox_timeout: float = 6.0,
    ) -> None:
        self.llm, self.store, self.embedder, self.condition = llm, store, embedder, condition
        self.run_id, self.namespace, self.seed = run_id, namespace, seed
        self.max_trials, self.top_k, self.sandbox_timeout = max_trials, top_k, sandbox_timeout
        policy = condition.policy or L.VERIFIED_POLICY
        self.policy = policy
        import random

        self.retriever = Retriever(
            store, namespace, policy.retrieve_statuses, k=top_k, holdout=0.0,
            shuffled=condition.shuffled, reranker=reranker,
            rng=random.Random(_seed("retrieval", seed, condition.key)),
        )

    # ------------------------------------------------------------------
    def _generate(self, task: Task, lessons: list[Retrieved], reflections: list[str],
                  feedback: str | None, seed: int) -> tuple[str, int]:
        lesson_block = P.render_lessons(
            [P.PromptLesson(str(r.lesson["lesson_id"]), r.lesson["pattern"], r.lesson["text"])
             for r in lessons if not r.withheld]
        )
        reflection_block = P.render_reflection(reflections) if self.condition.reflexion else ""
        user = P.SOLVE_USER.format(
            task_id=task.id, spec=task.spec, signature=task.signature, entry_point=task.entry_point,
            lessons_block=lesson_block, reflection_block=reflection_block + P.render_feedback(feedback),
        )
        out = self.llm.complete(
            [{"role": "system", "content": P.SOLVE_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.2, max_tokens=1500, seed=seed,
        )
        return out.text, out.usage.total

    def _evaluate(self, task: Task, raw: str) -> tuple[str, ExecResult]:
        try:
            code = extract_function(raw, task.entry_point)
        except ExtractionError as e:
            return raw, ExecResult(passed=False, load_error=f"ExtractionError: {e}")
        return code, run_tests(code, task.entry_point, task.tests, timeout=self.sandbox_timeout)

    def _reflect(self, task: Task, code: str, feedback: str, seed: int) -> tuple[str, int]:
        user = P.REFLECT_USER.format(task_id=task.id, spec=task.spec, code=code, feedback=feedback, vocab=VOCAB)
        out = self.llm.complete(
            [{"role": "system", "content": P.REFLECT_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=300, seed=seed,
        )
        return out.text.strip(), out.usage.total

    # ------------------------------------------------------------------
    def solve(self, task: Task, *, step: int, phase: str, learn: bool) -> TaskResult:
        query = self.embedder.embed([task.spec])[0]
        reflections: list[str] = []
        feedback: str | None = None
        last_pattern: str | None = None
        tokens = 0
        records: list[TrialRecord] = []
        failures: list[tuple[int, str, str | None]] = []  # (attempt_id, reflection, diagnosed pattern)
        self.retriever.holdout = self.policy.holdout if learn else 0.0

        for trial in range(1, self.max_trials + 1):
            retrieved = self.retriever.retrieve(query, step, last_pattern) if self.condition.cross_task else []
            seed = _seed(self.seed, task.id, trial)
            raw, used = self._generate(task, retrieved, reflections, feedback, seed)
            tokens += used
            code, result = self._evaluate(task, raw)
            diag = None if result.passed else classify(result, code)
            true_pattern = true_bug_pattern(task, code) if self.llm.is_simulation else None
            attempt_id = self.store.add_attempt(
                run_id=self.run_id, namespace=self.namespace, condition=self.condition.key, seed=self.seed,
                phase=phase, step=step, task_id=task.id, trial=trial, code=code, passed=result.passed,
                n_passed=result.n_passed, n_tests=len(task.tests), feedback=result.feedback(),
                pattern=diag.pattern if diag else None, pattern_conf=diag.confidence if diag else None,
                true_pattern=true_pattern,
                retrieved=[{"lesson_id": r.lesson["lesson_id"], "pattern": r.lesson["pattern"],
                            "text": r.lesson["text"], "score": round(r.score, 4), "withheld": r.withheld}
                           for r in retrieved],
                tokens=used,
            )
            for r in retrieved:
                self.store.add_retrieval(
                    attempt_id=attempt_id, lesson_id=r.lesson["lesson_id"], namespace=self.namespace,
                    rank=r.rank, score=r.score, features=r.features, withheld=r.withheld,
                )
            if retrieved:
                events = self.store.resolve_retrievals(attempt_id, result.passed, diag.pattern if diag else None)
                if learn:
                    L.record_online_evidence(self.store, events)

            reflection = None
            if not result.passed and (self.condition.reflexion or (learn and self.condition.cross_task)):
                reflection, used = self._reflect(task, code, result.feedback(), seed)
                tokens += used
                self.store.update_attempt(attempt_id, reflection=reflection)
                reflections.append(reflection)
                failures.append((attempt_id, reflection, diag.pattern if diag else None))
            records.append(TrialRecord(trial, attempt_id, result.passed, diag.pattern if diag else None,
                                       true_pattern, [r.lesson["lesson_id"] for r in retrieved], reflection))
            if result.passed:
                break
            feedback = result.feedback()
            last_pattern = (P.parse_pattern(reflection) if reflection else None) or (diag.pattern if diag else None)

        new_lessons: list[dict] = []
        if learn and self.condition.cross_task and failures:
            new_lessons = self._learn(task, step, failures)
            tokens += sum(l.pop("_tokens", 0) for l in new_lessons)
        if learn and self.condition.cross_task:
            L.refresh_status(self.store, self.namespace, step)

        passed = any(r.passed for r in records)
        return TaskResult(task.id, passed, records[0].passed, len(records), tokens, records, new_lessons)

    # ------------------------------------------------------------------
    def _learn(self, task: Task, step: int, failures: list[tuple[int, str, str | None]]) -> list[dict]:
        out: list[dict] = []
        context = self.embedder.embed([task.spec])[0]
        seen: set[str] = set()
        for attempt_id, reflection, diagnosed in failures:
            user = P.DISTILL_USER.format(task_id=task.id, reflection=reflection, vocab=VOCAB)
            comp = self.llm.complete(
                [{"role": "system", "content": P.DISTILL_SYSTEM}, {"role": "user", "content": user}],
                temperature=0.0, max_tokens=200, seed=_seed(self.seed, task.id, "distill", attempt_id),
            )
            pattern, text = P.parse_distilled(comp.text)
            pattern = pattern if pattern in ALL_PATTERNS else (diagnosed or "logic_error")
            info = {"pattern": pattern, "text": text, "_tokens": comp.usage.total, "attempt_id": attempt_id}
            if not text or pattern in seen:
                continue
            seen.add(pattern)
            leak = L.leak_reason(text, task)
            if leak:
                info["status"] = f"rejected: {leak}"
                out.append(info)
                continue
            emb = self.embedder.embed([text])[0]
            dup = L.find_duplicate(self.store, self.namespace, pattern, emb)
            if dup is not None:
                self.store.update_lesson(dup["lesson_id"], context=_merge_context(dup["context"], context))
                info["status"], info["lesson_id"] = "merged", dup["lesson_id"]
                out.append(info)
                continue
            lesson_id = self.store.add_lesson(
                namespace=self.namespace, text=text, pattern=pattern, embedding=emb, context=context,
                embedder=self.embedder.name, status="candidate", source_task=task.id,
                source_attempt=attempt_id, created_step=step,
            )
            info["lesson_id"] = lesson_id
            if self.policy.verify:
                verdict, replay, used = self.verify(task, lesson_id, step)
                info["_tokens"] += used
                self.store.update_lesson(lesson_id, status=verdict, replay=replay)
                info["status"] = verdict
            else:
                info["status"] = "candidate"
            out.append(info)
        return out

    def verify(self, task: Task, lesson_id: int, step: int) -> tuple[str, dict, int]:
        """Counterfactual replay: same task, paired seeds, with vs without the lesson."""
        lesson = self.store.get_lesson(lesson_id)
        r = Retrieved(lesson=lesson, score=1.0, features={})
        b = c = with_pass = without_pass = tokens = 0
        n = self.policy.replay_seeds
        for i in range(n):
            seed = _seed(self.seed, task.id, "replay", lesson_id, i)
            raw_with, t1 = self._generate(task, [r], [], None, seed)
            raw_without, t2 = self._generate(task, [], [], None, seed)
            tokens += t1 + t2
            ok_with = self._evaluate(task, raw_with)[1].passed
            ok_without = self._evaluate(task, raw_without)[1].passed
            with_pass += ok_with
            without_pass += ok_without
            b += ok_with and not ok_without
            c += ok_without and not ok_with
        verdict = "verified" if b >= 1 and b > c else "rejected"
        return verdict, {"n": n, "b": b, "c": c, "with_pass": with_pass, "without_pass": without_pass,
                         "step": step}, tokens


def _merge_context(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    v = np.asarray(old, dtype=np.float32) + np.asarray(new, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-9)
