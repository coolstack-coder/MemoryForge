from memoryforge.bench import all_tasks, eval_tasks, train_tasks
from memoryforge.bench.validate import check_task
from memoryforge.failures_vocab import PATTERNS


def test_split_sizes():
    assert len(train_tasks()) == 24 and len(eval_tasks()) == 12


def test_eval_pitfalls_are_covered_by_training():
    seen = {p for t in train_tasks() for p in t.bugs}
    for t in eval_tasks():
        assert set(t.bugs) <= seen, t.id


def test_every_pattern_appears():
    used = {p for t in all_tasks() for p in t.bugs}
    assert used == set(PATTERNS)


def test_tasks_are_valid():
    problems = {t.id: check_task(t) for t in all_tasks()}
    assert not {k: v for k, v in problems.items() if v}
