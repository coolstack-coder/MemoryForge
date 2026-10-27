import numpy as np

from memoryforge.eval.stats import bootstrap_ci, contrast, holm, mcnemar_exact


def test_mcnemar_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(10, 0) - 2 / 1024) < 1e-12
    assert mcnemar_exact(5, 5) == 1.0


def test_bootstrap_ci_contains_mean():
    v = np.array([0, 1] * 50, dtype=float)
    lo, hi = bootstrap_ci(v)
    assert lo < 0.5 < hi


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03])
    assert all(0 <= p <= 1 for p in adj) and adj[0] <= adj[2] <= adj[1]


def test_contrast_direction():
    a = [{"seed": 0, "task_id": f"t{i}", "pass1": True} for i in range(10)]
    b = [{"seed": 0, "task_id": f"t{i}", "pass1": i < 3} for i in range(10)]
    c = contrast(a, b, "pass1")
    assert c["diff"] == 0.7 and c["mcnemar"]["b"] == 7 and c["mcnemar"]["c"] == 0
