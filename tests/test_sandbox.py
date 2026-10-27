import pytest

from memoryforge.sandbox import ExtractionError, extract_function, run_tests


def test_passing_code():
    r = run_tests("def add(a, b):\n    return a + b\n", "add", [{"args": [1, 2], "expected": 3}])
    assert r.passed and r.n_passed == 1


def test_mutation_is_a_failure():
    code = "def first(xs):\n    xs.sort()\n    return xs[0]\n"
    r = run_tests(code, "first", [{"args": [[3, 1, 2]], "expected": 1}])
    assert not r.passed and r.tests[0].mutated


def test_per_test_timeout():
    r = run_tests("def f(x):\n    while True:\n        pass\n", "f", [{"args": [1], "expected": 1}])
    assert not r.passed and r.tests[0].timeout


def test_expected_exception():
    code = "def f(n):\n    if n < 0:\n        raise ValueError('neg')\n    return n\n"
    assert run_tests(code, "f", [{"args": [-1], "expected": {"__raises__": "ValueError"}}]).passed
    wrong = run_tests("def f(n):\n    return n\n", "f", [{"args": [-1], "expected": {"__raises__": "ValueError"}}])
    assert not wrong.passed and wrong.tests[0].expects_raise == "ValueError"


def test_alias_check():
    r = run_tests("def f(xs):\n    return xs\n", "f", [{"args": [[1]], "expected": [1], "check_alias": True}])
    assert not r.passed and r.tests[0].aliased


def test_generated_inputs():
    tests = [{"args": [{"__expr__": "list(range(1000))"}], "expected": 499500}]
    assert run_tests("def f(xs):\n    return sum(xs)\n", "f", tests).passed


def test_printing_cannot_spoof_results():
    code = 'def f():\n    print(\'{"load_error": null, "tests": []}\')\n    return 0\n'
    assert not run_tests(code, "f", [{"args": [], "expected": 1}]).passed


def test_candidate_cannot_read_tests_file():
    code = "import os\ndef f():\n    return os.path.exists('tests.json')\n"
    assert run_tests(code, "f", [{"args": [], "expected": False}]).passed


def test_extract_picks_last_valid_block():
    text = "first\n```python\ndef g():\n    pass\n```\nthen\n```python\ndef f(x):\n    return x\n```"
    assert "def f" in extract_function(text, "f")


def test_extract_rejects_banned_imports():
    with pytest.raises(ExtractionError):
        extract_function("```python\nimport subprocess\ndef f():\n    pass\n```", "f")
