"""The closed vocabulary of failure patterns shared by tasks, classifier and lessons."""

PATTERNS: dict[str, str] = {
    "empty_input": "Fails on empty or degenerate input (empty list/string, single element, zero).",
    "off_by_one": "A boundary is shifted by one: inclusive vs exclusive ranges, loop bounds, slice ends.",
    "tie_ordering": "Elements are right but ties or equal keys come out in the wrong (unstable) order.",
    "whitespace_case": "Text parsing breaks on extra whitespace, tabs, case differences or unicode.",
    "negative_numbers": "Wrong result for negative numbers or zero (modulo, division, sign handling).",
    "mutates_input": "The function modifies an argument the caller still owns.",
    "shared_reference": "The result aliases mutable objects from the input instead of returning fresh copies.",
    "float_rounding": "Floating point or rounding mode produces a slightly wrong number.",
    "inefficient": "Correct but too slow for large inputs (quadratic or worse).",
    "error_contract": "Raises the wrong exception, or fails to raise when the spec demands it.",
}

# labels the classifier may emit when nothing in PATTERNS fits
FALLBACK_PATTERNS = ("logic_error", "crash")

ALL_PATTERNS = tuple(PATTERNS) + FALLBACK_PATTERNS
