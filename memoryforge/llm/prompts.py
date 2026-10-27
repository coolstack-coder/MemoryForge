"""Prompt templates.

Lessons and reflections are wrapped in sentinel-delimited blocks. Real models
read them as ordinary text; the simulated backend parses the `pattern=` tags so
its behaviour depends only on *which* lessons were injected, never on wording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LESSON_BEGIN, LESSON_END = "<<<MF_LESSONS>>>", "<<<END_MF_LESSONS>>>"
REFLECT_BEGIN, REFLECT_END = "<<<MF_REFLECTION>>>", "<<<END_MF_REFLECTION>>>"

SOLVE_SYSTEM = (
    "You are a careful senior Python developer. You write a single, complete, "
    "dependency-free function that satisfies the specification exactly. "
    "Reply with exactly one ```python fenced block and nothing else."
)

SOLVE_USER = """TASK_ID: {task_id}
## Task
{spec}

## Signature
```python
{signature}
```
{lessons_block}{reflection_block}
Write the complete implementation of `{entry_point}`."""

REFLECT_SYSTEM = (
    "You are debugging your own code. Be concrete and brief. Never write code."
)

REFLECT_USER = """MODE: REFLECT
TASK_ID: {task_id}
## Task
{spec}

## Your attempt
```python
{code}
```

## Test feedback
{feedback}

In at most three sentences, state the root cause and what to change next time.
Begin with `pattern=<name>` where <name> is one of: {vocab}."""

DISTILL_SYSTEM = (
    "You turn one debugging experience into a reusable engineering rule. "
    "The rule must be general: never mention the task, its name, or literal test values."
)

DISTILL_USER = """MODE: DISTILL
TASK_ID: {task_id}
## Diagnosis of the failed attempt
{reflection}

Write ONE reusable lesson in exactly this format:
pattern=<name from: {vocab}>
lesson=<one imperative sentence, at most 30 words, of the form 'When ..., ...'>"""


@dataclass(frozen=True)
class PromptLesson:
    id: str
    pattern: str
    text: str


def render_lessons(lessons: list[PromptLesson]) -> str:
    if not lessons:
        return ""
    body = "\n".join(f"- [id={l.id}] [pattern={l.pattern}] {l.text}" for l in lessons)
    return (
        "\n## Lessons from past experience (apply the ones that are relevant)\n"
        f"{LESSON_BEGIN}\n{body}\n{LESSON_END}\n"
    )


def render_reflection(reflections: list[str]) -> str:
    if not reflections:
        return ""
    body = "\n".join(f"- {r}" for r in reflections)
    return (
        "\n## What went wrong in your previous attempts on this task\n"
        f"{REFLECT_BEGIN}\n{body}\n{REFLECT_END}\n"
    )


_LESSON_RE = re.compile(r"\[id=(\S+)\] \[pattern=([a-z_]+)\]")
_PATTERN_RE = re.compile(r"pattern\s*=\s*([a-z_]+)")
_LESSON_LINE_RE = re.compile(r"lesson\s*=\s*(.+)")
_TASK_RE = re.compile(r"TASK_ID:\s*(\S+)")


def _between(text: str, begin: str, end: str) -> str:
    i, j = text.find(begin), text.find(end)
    return text[i + len(begin) : j] if 0 <= i < j else ""


def parse_lesson_patterns(prompt: str) -> list[str]:
    return [m.group(2) for m in _LESSON_RE.finditer(_between(prompt, LESSON_BEGIN, LESSON_END))]


def parse_reflection_patterns(prompt: str) -> list[str]:
    return _PATTERN_RE.findall(_between(prompt, REFLECT_BEGIN, REFLECT_END))


def parse_task_id(prompt: str) -> str | None:
    m = _TASK_RE.search(prompt)
    return m.group(1) if m else None


def parse_pattern(text: str) -> str | None:
    m = _PATTERN_RE.search(text)
    return m.group(1) if m else None


def parse_distilled(text: str) -> tuple[str | None, str | None]:
    m = _LESSON_LINE_RE.search(text)
    lesson = m.group(1).strip() if m else None
    return parse_pattern(text), lesson


def render_feedback(feedback: str | None) -> str:
    if not feedback:
        return ""
    return f"\n## Test feedback from your previous attempt\n{feedback}\n"
