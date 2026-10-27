"""An offline, deterministic "simulated developer".

It exists so the whole pipeline (sandbox, classifier, memory, verification,
retrieval, statistics, dashboard) can run in CI and on a laptop with no model
and no network. It is NOT evidence that memory helps a real model: the size of
the lesson effect is an input (`SimConfig.lesson_effect`), so a simulated run
only shows that the machinery measures what it should.

How it behaves:
  * Each benchmark task has hand-written buggy variants, one per pitfall.
  * For every pitfall the simulator draws a uniform u (common random numbers,
    keyed on seed + task) and writes that bug if u < p.
  * p starts at `base_p`; a lesson tagged with the same pattern multiplies it
    by (1 - lesson_effect); a within-task reflection naming the pattern
    multiplies it by (1 - reflection_effect); each irrelevant lesson adds a
    small distraction penalty.
  * Reflections misdiagnose the pattern with probability `misdiagnosis`, which
    produces lessons aimed at the wrong pitfall.
  * With probability `harmful_rate` distillation writes a plausible-sounding
    but wrong rule (HARMFUL_TEMPLATES). When such a rule is in the prompt the
    pitfall it talks about becomes *more* likely (`harm_effect`). Verification
    and online deprecation have to catch those.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Sequence

from ..failures_vocab import PATTERNS
from . import prompts
from .base import Completion, Message, Usage, UsageMeter

LESSON_TEMPLATES: dict[str, list[str]] = {
    "empty_input": [
        "When a function indexes, reduces or divides by the length of a sequence, handle the empty and single-element cases explicitly first.",
        "When input can be an empty list or string, return the documented empty result before touching the first element.",
    ],
    "off_by_one": [
        "When a spec says 'inclusive' or describes both ends of a range, check every loop bound and slice end against the boundary values.",
        "When iterating windows or ranges, verify the first and last index by hand with a tiny example before trusting range().",
    ],
    "tie_ordering": [
        "When sorting or ranking with possible ties, rely on a stable sort and add only the tie-break key the spec names.",
        "When output order matters, never pass results through a set or dict that loses the required first-occurrence order.",
    ],
    "whitespace_case": [
        "When parsing text, use split() without arguments, strip each token, and normalise case exactly as the spec says.",
        "When comparing user-provided strings, normalise whitespace runs, tabs and case before matching.",
    ],
    "negative_numbers": [
        "When numbers can be negative or zero, test the sign cases explicitly; Python's % and // round toward negative infinity.",
        "When a parameter may be negative, decide its meaning from the spec instead of clamping or relying on truthiness.",
    ],
    "mutates_input": [
        "When a function receives a list or dict, work on a copy; never sort, pop or assign into the caller's argument.",
        "When you need sorted data, use sorted(x) rather than x.sort() unless the spec says to modify the input.",
    ],
    "shared_reference": [
        "When returning containers derived from the input, build fresh lists and dicts so the result never aliases the argument.",
        "When copying nested structures, copy every level that the caller might later mutate.",
    ],
    "float_rounding": [
        "When the spec fixes a rounding rule, use integer or decimal arithmetic rather than float formatting or round().",
        "When exact integers are involved, avoid float division entirely; floats lose precision beyond 2**53.",
    ],
    "inefficient": [
        "When inputs can reach hundreds of thousands of items, replace nested loops with a single pass using counters, prefix sums or sliding windows.",
        "When a window or pair count is required, maintain running state instead of recomputing each slice.",
    ],
    "error_contract": [
        "When the spec says to raise an exception for invalid arguments, validate them up front and raise exactly that exception type.",
        "When a parameter has a documented invalid range, check it before any computation so the right error surfaces.",
    ],
}

# Plausible but wrong advice: each one sounds like a lesson and makes the bug likelier.
HARMFUL_TEMPLATES: dict[str, str] = {
    "empty_input": "When a sequence might be empty, let the natural exception propagate instead of special-casing it.",
    "off_by_one": "When computing an upper bound, Python ranges and slices already include the end, so no adjustment is needed.",
    "tie_ordering": "When results contain ties, sort them alphabetically so the output is deterministic.",
    "whitespace_case": "When splitting text, split on a single space so empty tokens keep their positions.",
    "negative_numbers": "When a count or offset is negative, clamp it to zero to keep indices valid.",
    "mutates_input": "When performance matters, sort the input list in place instead of allocating a copy.",
    "shared_reference": "When returning data you were given, return it directly; copying wastes memory.",
    "float_rounding": "When a spec asks for one decimal place, format with f'{x:.1f}', which rounds half up.",
    "inefficient": "When inputs are lists, a straightforward nested loop is fast enough in practice.",
    "error_contract": "When arguments are invalid, return a neutral value such as 0 or an empty list rather than raising.",
}

REFLECTION_TEMPLATES: dict[str, str] = {
    "empty_input": "The implementation assumed the input is non-empty and broke on the degenerate case.",
    "off_by_one": "A boundary was off by one: an inclusive end was treated as exclusive (or vice versa).",
    "tie_ordering": "Equal elements came out in the wrong order; the tie-break did not follow the spec.",
    "whitespace_case": "The parser did not normalise whitespace or case the way the spec requires.",
    "negative_numbers": "Negative values were handled incorrectly by the arithmetic or a truthiness check.",
    "mutates_input": "The function modified the caller's argument in place.",
    "shared_reference": "The result reused mutable objects from the input instead of fresh copies.",
    "float_rounding": "Float arithmetic or the rounding mode produced a slightly wrong value.",
    "inefficient": "The algorithm was quadratic and timed out on the large input.",
    "error_contract": "The required exception was not raised for invalid arguments.",
}


@dataclass(frozen=True)
class SimConfig:
    base_p: float = 0.42
    lesson_effect: float = 0.8
    reflection_effect: float = 0.6
    distraction: float = 0.03
    misdiagnosis: float = 0.3
    harmful_rate: float = 0.2
    harm_effect: float = 0.35


def _rng(*parts: object) -> random.Random:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _norm(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


class SimulatedLLM:
    is_simulation = True

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()
        self.name = "simulated"
        self.meter = UsageMeter()

    def describe(self) -> dict:
        return {"backend": "simulated", **asdict(self.config)}

    # -- public -----------------------------------------------------------
    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> Completion:
        prompt = messages[-1]["content"]
        seed = 0 if seed is None else seed
        if prompt.startswith("MODE: REFLECT"):
            text = self._reflect(prompt, seed)
        elif prompt.startswith("MODE: DISTILL"):
            text = self._distill(prompt, seed)
        else:
            text = self._solve(prompt, seed)
        usage = Usage(
            prompt_tokens=sum(len(m["content"].split()) for m in messages),
            completion_tokens=len(text.split()),
        )
        self.meter.add(usage)
        return Completion(text=text, usage=usage, backend="simulated", simulated=True)

    def bug_probability(self, pattern: str, lessons: list[tuple[str, str]], reflection_patterns: list[str]) -> float:
        c = self.config
        harmful = {HARMFUL_TEMPLATES[lp] == text for lp, text in lessons if lp == pattern and lp in HARMFUL_TEMPLATES}
        helpful = any(lp == pattern and text != HARMFUL_TEMPLATES.get(lp) for lp, text in lessons)
        p = c.base_p
        if helpful:
            p *= 1 - c.lesson_effect
        if pattern in reflection_patterns:
            p *= 1 - c.reflection_effect
        if True in harmful:
            p += c.harm_effect
        irrelevant = sum(1 for lp, _ in lessons if lp != pattern)
        return min(0.95, p + c.distraction * irrelevant)

    # -- modes ------------------------------------------------------------
    def _task(self, prompt: str):
        from ..bench import get_task

        task_id = prompts.parse_task_id(prompt)
        return get_task(task_id) if task_id else None

    def _solve(self, prompt: str, seed: int) -> str:
        task = self._task(prompt)
        if task is None:
            return "```python\ndef solution(*args):\n    raise NotImplementedError\n```"
        lessons = prompts.parse_lessons(prompt)
        reflections = prompts.parse_reflection_patterns(prompt)
        rng = _rng("solve", seed, task.id)
        code = task.reference
        for pattern in sorted(task.bugs):  # fixed order, one draw per pitfall
            u = rng.random()
            if u < self.bug_probability(pattern, lessons, reflections):
                code = task.bugs[pattern]
                break
        return f"```python\n{code.strip()}\n```"

    def _true_pattern(self, prompt: str) -> tuple[str | None, str]:
        task = self._task(prompt)
        code = _norm(prompt.split("```python", 1)[-1].split("```", 1)[0])
        if task is not None:
            for pattern, bug in task.bugs.items():
                if _norm(bug) == code:
                    return pattern, code
        return None, code

    def _reflect(self, prompt: str, seed: int) -> str:
        pattern, code = self._true_pattern(prompt)
        rng = _rng("reflect", seed, prompts.parse_task_id(prompt), code)
        if pattern is None:
            return "pattern=logic_error. The logic does not match the specification; re-read it carefully."
        if rng.random() < self.config.misdiagnosis:
            pattern = rng.choice([p for p in PATTERNS if p != pattern])
        return f"pattern={pattern}. {REFLECTION_TEMPLATES[pattern]} Fix that specific case next time."

    def _distill(self, prompt: str, seed: int) -> str:
        reflection = prompt.split("## Diagnosis of the failed attempt", 1)[-1]
        pattern = prompts.parse_pattern(reflection)
        if pattern not in LESSON_TEMPLATES:
            return "pattern=logic_error\nlesson=When a solution fails, re-read the specification before changing code."
        rng = _rng("distill", seed, prompts.parse_task_id(prompt))
        if rng.random() < self.config.harmful_rate:
            return f"pattern={pattern}\nlesson={HARMFUL_TEMPLATES[pattern]}"
        return f"pattern={pattern}\nlesson={rng.choice(LESSON_TEMPLATES[pattern])}"
