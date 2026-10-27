"""Hybrid lesson retrieval.

    score = 0.55 * sim + 0.20 * pattern_match + 0.15 * utility + 0.10 * recency

where sim = max(cos(task, lesson text), cos(task, lesson's source task)),
pattern_match = 1 if the lesson targets the pattern just diagnosed on this
task (retry trials only), utility = 0.5 + posterior mean lift (clipped), and
recency decays with the number of tasks since the lesson was written.

At most one lesson per pattern is selected so the prompt is not padded with
redundant advice. Once enough retrieval outcomes exist, the learned reranker
blends in.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .utility import lesson_utility

WEIGHTS = {"sim": 0.55, "pattern": 0.20, "utility": 0.15, "recency": 0.10}
FEATURES = ("sim", "cos_text", "cos_ctx", "pattern", "utility", "recency", "log_uses", "lift_lo")
MIN_SIM = 0.12
RECENCY_TAU = 40.0


@dataclass
class Retrieved:
    lesson: dict
    score: float
    features: dict
    withheld: bool = False
    rank: int = 0


@dataclass
class Retriever:
    store: object
    namespace: str
    statuses: tuple[str, ...]
    k: int = 3
    holdout: float = 0.0
    shuffled: bool = False
    reranker: object | None = None
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def features(self, lesson: dict, pool: list[dict], step: int, pattern: str | None) -> dict:
        u = lesson_utility(lesson, pool)
        sim = max(lesson["cos_text"], lesson["cos_ctx"])
        return {
            "sim": sim,
            "cos_text": lesson["cos_text"],
            "cos_ctx": lesson["cos_ctx"],
            "pattern": 1.0 if pattern and lesson["pattern"] == pattern else 0.0,
            "utility": min(1.0, max(0.0, 0.5 + u.lift_mean)),
            "recency": math.exp(-max(0, step - lesson["created_step"]) / RECENCY_TAU),
            "log_uses": math.log1p(lesson["uses"]),
            "lift_lo": u.lift_lo,
        }

    @staticmethod
    def formula(f: dict) -> float:
        return sum(WEIGHTS[k] * f[k] for k in WEIGHTS)

    def retrieve(self, query: np.ndarray, step: int, pattern: str | None = None) -> list[Retrieved]:
        if self.shuffled:
            return self._shuffled(step)
        cands = self.store.candidates(self.namespace, self.statuses, query)
        if not cands:
            return []
        pool = self.store.lessons(namespace=self.namespace)
        scored: list[Retrieved] = []
        for c in cands:
            f = self.features(c, pool, step, pattern)
            if f["sim"] < MIN_SIM and not f["pattern"]:
                continue
            score = self.formula(f)
            if self.reranker is not None:
                score = self.reranker.blend(f, score)
            scored.append(Retrieved(lesson=c, score=score, features=f))
        scored.sort(key=lambda r: -r.score)

        chosen: list[Retrieved] = []
        seen_patterns: set[str] = set()
        for r in scored:
            if r.lesson["pattern"] in seen_patterns:
                continue
            seen_patterns.add(r.lesson["pattern"])
            r.rank = len(chosen)
            r.withheld = self.holdout > 0 and self.rng.random() < self.holdout
            chosen.append(r)
            if len(chosen) == self.k:
                break
        return chosen

    def _shuffled(self, step: int) -> list[Retrieved]:
        """Control condition: same number of lessons, chosen at random."""
        pool = self.store.lessons(namespace=self.namespace, statuses=self.statuses)
        picks = self.rng.sample(pool, min(self.k, len(pool)))
        zero = {k: 0.0 for k in FEATURES}
        return [Retrieved(lesson=p, score=0.0, features=zero, rank=i) for i, p in enumerate(picks)]
