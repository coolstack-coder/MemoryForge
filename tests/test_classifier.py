from memoryforge.bench import all_tasks
from memoryforge.failures import classify
from memoryforge.sandbox import run_tests


def test_classifier_recovers_most_bug_patterns():
    total = correct = 0
    for task in all_tasks():
        for pattern, code in task.bugs.items():
            res = run_tests(code, task.entry_point, task.tests, timeout=20)
            total += 1
            correct += classify(res, code).pattern == pattern
    assert total >= 75
    assert correct / total >= 0.6, f"classifier accuracy {correct}/{total}"


def test_timeout_is_inefficient():
    code = "def f(xs):\n    return sum(1 for a in xs for b in xs)\n"
    res = run_tests(code, "f", [{"args": [{"__expr__": "list(range(30000))"}], "expected": 0}], per_test_timeout=0.5)
    assert classify(res, code).pattern == "inefficient"
