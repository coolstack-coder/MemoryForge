"""Experimental conditions compared by the benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from .memory.lessons import RAW_POLICY, VERIFIED_POLICY, Policy


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    reflexion: bool  # verbal self-reflection between trials of the same task
    cross_task: bool  # lessons persist across tasks
    policy: Policy | None = None
    rerank: bool = False
    shuffled: bool = False
    memory_from: str | None = None  # reuse another condition's memory at eval time
    description: str = ""


CONDITIONS: dict[str, Condition] = {
    c.key: c
    for c in [
        Condition("none", "No memory", False, False,
                  description="Retries see raw test feedback only. Nothing persists."),
        Condition("reflexion", "Reflexion (in-task)", True, False,
                  description="Verbal reflection between trials; discarded when the task ends."),
        Condition("raw_memory", "Raw cross-task memory", True, True, RAW_POLICY,
                  description="Every distilled reflection is stored and retrievable, unverified."),
        Condition("verified", "MemoryForge (verified)", True, True, VERIFIED_POLICY,
                  description="Only lessons that survive counterfactual replay are retrieved; harmful ones are retired."),
        Condition("verified_rerank", "MemoryForge + reranker", True, True, VERIFIED_POLICY, rerank=True,
                  memory_from="verified",
                  description="Verified memory, ranked with the learned PyTorch reranker."),
        Condition("shuffled", "Shuffled-lesson control", True, True, VERIFIED_POLICY, shuffled=True,
                  memory_from="verified",
                  description="Same memory and prompt length, but lessons drawn at random."),
    ]
}

DEFAULT_CONDITIONS = tuple(CONDITIONS)
