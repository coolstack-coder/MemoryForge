"""Bayesian bookkeeping for lessons.

Each lesson keeps two Beta posteriors over "the lesson's failure pattern was
avoided on this attempt":
  * p1 ~ Beta(alpha, beta)   when the lesson was injected into the prompt
  * p0 ~ Beta(alpha0, beta0) when it was retrieved but deliberately withheld
    (a randomised control arm, MF_HOLDOUT of retrievals)
Lift = p1 - p0. Because withheld data per lesson is thin, p0 is pooled over
every lesson with the same pattern in the namespace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_DRAWS = 4000


@dataclass(frozen=True)
class Utility:
    p1_mean: float
    p0_mean: float
    lift_mean: float
    lift_lo: float  # 10th percentile
    lift_hi: float  # 90th percentile
    p_positive: float
    n_injected: int
    n_withheld: int

    def as_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def utility(alpha: float, beta: float, alpha0: float, beta0: float, seed: int = 0) -> Utility:
    rng = np.random.default_rng(seed)
    p1 = rng.beta(alpha, beta, N_DRAWS)
    p0 = rng.beta(alpha0, beta0, N_DRAWS)
    lift = p1 - p0
    return Utility(
        p1_mean=float(alpha / (alpha + beta)),
        p0_mean=float(alpha0 / (alpha0 + beta0)),
        lift_mean=float(lift.mean()),
        lift_lo=float(np.quantile(lift, 0.1)),
        lift_hi=float(np.quantile(lift, 0.9)),
        p_positive=float((lift > 0).mean()),
        n_injected=int(round(alpha + beta - 2)),
        n_withheld=int(round(alpha0 + beta0 - 2)),
    )


def pooled_baseline(lessons: list[dict], pattern: str) -> tuple[float, float]:
    """Sum the withheld-arm evidence of every lesson sharing `pattern`."""
    a0, b0 = 1.0, 1.0
    for l in lessons:
        if l["pattern"] == pattern:
            a0 += l["alpha0"] - 1
            b0 += l["beta0"] - 1
    return a0, b0


def lesson_utility(lesson: dict, namespace_lessons: list[dict]) -> Utility:
    a0, b0 = pooled_baseline(namespace_lessons, lesson["pattern"])
    return utility(lesson["alpha"], lesson["beta"], a0, b0, seed=int(lesson["lesson_id"]))
