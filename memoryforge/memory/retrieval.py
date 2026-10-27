"""Hybrid lesson retrieval.

    score = 0.35 * sim + 0.25 * prior + 0.20 * pattern_match + 0.12 * utility + 0.08 * recency

where
  sim           = max(cos(task, lesson text), cos(task, lesson's source task))
  prior         = how often tasks similar to this one failed with the lesson's
                  pattern in past experience (a kernel-weighted vote over every
                  failed attempt in memory, not just the lesson's source task)
  pattern_match = 1 if the lesson targets the pattern just diagnosed on this
                  task (retry trials only)
  utility       = 0.5 + posterior mean lift, clipped to [0, 1]
  recency       = exp(-tasks since the lesson was written / 40)

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

WEIGHTS = {"sim": 0.35, "prior": 0.25, "pattern": 0.20, "utility": 0.12, "recency": 0.08}
FEATURES = ("sim", "cos_text", "cos_ctx", "prior", "pattern", "utility", "recency", "log_uses", "lift_lo")
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
    embedder: object | None = None
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    _spec_cache: dict = field(default_factory=dict)

    def pattern_prior(self, query: np.ndarray) -> dict[str, float]:
        """Kernel-weighted vote of past failures: P(pattern | similar task)."""
        if self.embedder is None:
            return {}
        from ..bench import get_task

        failed = [a for a in self.store.attempts(namespace=self.namespace, phase="train")
                  if not a["passed"] and a["pattern"]]
        if not failed:
            return {}
        missing = sorted({a["task_id"] for a in failed} - set(self._spec_cache))
        if missing:
            vecs = self.embedder.embed([get_task(t).spec for t in missing])
            self._spec_cache.update(zip(missing, vecs))
        votes: dict[str, float] = {}
        total = 0.0
        for a in failed:
            w = max(0.0, float(np.dot(self._spec_cache[a["task_id"]], query))) ** 3
            votes[a["pattern"]] = votes.get(a["pattern"], 0.0) + w
            total += w
        return {p: v / total for p, v in votes.items()} if total > 0 else {}

    def features(self, lesson: dict, pool: list[dict], step: int, pattern: str | None,
                 prior: dict[str, float] | None = None) -> dict:
        u = lesson_utility(lesson, pool)
        sim = max(lesson["cos_text"], lesson["cos_ctx"])
        return {
            "sim": sim,
            "cos_text": lesson["cos_text"],
            "cos_ctx": lesson["cos_ctx"],
            "prior": (prior or {}).get(lesson["pattern"], 0.0),
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
        prior = self.pattern_prior(query)
        scored: list[Retrieved] = []
        for c in cands:
            f = self.features(c, pool, step, pattern, prior)
            if f["sim"] < MIN_SIM and not f["pattern"] and f["prior"] < 0.05:
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
