"""Train split, batch B: twelve experience tasks."""

from __future__ import annotations

from .task import Task, case, expr

TASKS: list[Task] = []


# ---------------------------------------------------------------- 1
TASKS.append(Task(
    id="dedupe_keep_last",
    split="train",
    entry_point="dedupe_keep_last",
    signature="def dedupe_keep_last(items: list) -> list:",
    spec=(
        "Return a new list with duplicates removed, keeping only the LAST occurrence of each value. "
        "Items are hashable scalars (numbers or strings). Survivors are ordered by the position of "
        "their last occurrence in the input. An empty list gives []. The input list must not be modified."
    ),
    tests=[
        case([1, 2, 1, 3, 2], expected=[1, 3, 2]),
        case([5, 1, 5, 2, 1], expected=[5, 2, 1]),
        case(["b", "a", "b", "a"], expected=["b", "a"]),
        case(["x", "y", "x", "z", "y", "x"], expected=["z", "y", "x"]),
        case([], expected=[]),
        case(["a"], expected=["a"]),
        case([3, 3, 3], expected=[3]),
        case(["p", "q", "r"], expected=["p", "q", "r"]),
    ],
    reference='''
def dedupe_keep_last(items: list) -> list:
    last = {}
    for i, x in enumerate(items):
        last[x] = i
    return [x for i, x in enumerate(items) if last[x] == i]
''',
    bugs={
        "tie_ordering": '''
def dedupe_keep_last(items: list) -> list:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
''',
        "mutates_input": '''
def dedupe_keep_last(items: list) -> list:
    items.reverse()
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    out.reverse()
    return out
''',
    },
    tags=("tie_ordering", "mutates_input", "empty_input"),
))


# ---------------------------------------------------------------- 2
TASKS.append(Task(
    id="group_anagrams_ci",
    split="train",
    entry_point="group_anagrams_ci",
    signature="def group_anagrams_ci(words: list[str]) -> list[list[str]]:",
    spec=(
        "Group words that are anagrams of each other after removing leading/trailing whitespace "
        "(str.strip()) and lowercasing; two words belong together when their stripped, lowercased "
        "characters are the same multiset. Return a list of groups; groups are ordered by the first "
        "appearance of any of their members, and each group lists its members in input order using the "
        "ORIGINAL, unmodified strings. An empty input gives []."
    ),
    tests=[
        case(["listen", "silent", "enlist", "google", "gooegl"],
             expected=[["listen", "silent", "enlist"], ["google", "gooegl"]]),
        case(["Tea", "eat", " ate ", "tan", "Nat"], expected=[["Tea", "eat", " ate "], ["tan", "Nat"]]),
        case(["ab", "BA", "\tba\n", "c"], expected=[["ab", "BA", "\tba\n"], ["c"]]),
        case(["zz", "yx", "xy", "ZZ"], expected=[["zz", "ZZ"], ["yx", "xy"]]),
        case(["", "  ", "x"], expected=[["", "  "], ["x"]]),
        case([], expected=[]),
        case(["a"], expected=[["a"]]),
        case(["dog", "cat", "god", "act", "bird"], expected=[["dog", "god"], ["cat", "act"], ["bird"]]),
    ],
    reference='''
def group_anagrams_ci(words: list[str]) -> list[list[str]]:
    groups = {}
    for w in words:
        key = "".join(sorted(w.strip().lower()))
        groups.setdefault(key, []).append(w)
    return list(groups.values())
''',
    bugs={
        "whitespace_case": '''
def group_anagrams_ci(words: list[str]) -> list[list[str]]:
    groups = {}
    for w in words:
        key = "".join(sorted(w))
        groups.setdefault(key, []).append(w)
    return list(groups.values())
''',
        "tie_ordering": '''
def group_anagrams_ci(words: list[str]) -> list[list[str]]:
    groups = {}
    for w in words:
        key = "".join(sorted(w.strip().lower()))
        groups.setdefault(key, []).append(w)
    return [groups[k] for k in sorted(groups)]
''',
    },
    tags=("whitespace_case", "tie_ordering", "empty_input"),
))


# ---------------------------------------------------------------- 3
TASKS.append(Task(
    id="top_k_frequent",
    split="train",
    entry_point="top_k_frequent",
    signature="def top_k_frequent(words: list[str], k: int) -> list[str]:",
    spec=(
        "Return the k most frequent distinct words, ordered by count descending. Words with equal "
        "counts are ordered by their first appearance in the input (earlier first). If k exceeds the "
        "number of distinct words, return all of them in that order. If k <= 0 (including negative k), "
        "return []."
    ),
    tests=[
        case(["a", "b", "a", "c", "b", "a"], 2, expected=["a", "b"]),
        case(["pear", "apple", "pear", "apple", "fig"], 2, expected=["pear", "apple"]),
        case(["d", "c", "b", "a"], 3, expected=["d", "c", "b"]),
        case(["x", "y", "x"], 5, expected=["x", "y"]),
        case(["b", "a"], 1, expected=["b"]),
        case(["a", "b", "c"], 0, expected=[]),
        case(["a", "b", "a", "c"], -1, expected=[]),
        case([], 3, expected=[]),
        case(["m", "n", "n", "o", "o", "o"], 3, expected=["o", "n", "m"]),
    ],
    reference='''
def top_k_frequent(words: list[str], k: int) -> list[str]:
    if k <= 0:
        return []
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    order = sorted(counts, key=lambda w: -counts[w])
    return order[:k]
''',
    bugs={
        "tie_ordering": '''
def top_k_frequent(words: list[str], k: int) -> list[str]:
    if k <= 0:
        return []
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    order = sorted(counts, key=lambda w: (-counts[w], w))
    return order[:k]
''',
        "off_by_one": '''
def top_k_frequent(words: list[str], k: int) -> list[str]:
    if k <= 0:
        return []
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    order = sorted(counts, key=lambda w: -counts[w])
    out = []
    for i, w in enumerate(order):
        if i > k:
            break
        out.append(w)
    return out
''',
        "negative_numbers": '''
def top_k_frequent(words: list[str], k: int) -> list[str]:
    if k == 0:
        return []
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    order = sorted(counts, key=lambda w: -counts[w])
    return order[:k]
''',
    },
    tags=("tie_ordering", "off_by_one", "negative_numbers", "empty_input"),
))


# ---------------------------------------------------------------- 4
TASKS.append(Task(
    id="days_inclusive",
    split="train",
    entry_point="days_inclusive",
    signature="def days_inclusive(start: str, end: str) -> int:",
    spec=(
        "Both arguments are valid Gregorian dates formatted \"YYYY-MM-DD\". Return the number of "
        "calendar days from start to end counting BOTH endpoints, so equal dates give 1. Leap years "
        "must be handled correctly. If end is earlier than start, return 0."
    ),
    tests=[
        case("2024-03-15", "2024-03-15", expected=1),
        case("2024-01-01", "2024-01-31", expected=31),
        case("2024-02-28", "2024-03-01", expected=3),
        case("2023-02-28", "2023-03-01", expected=2),
        case("2023-12-31", "2024-01-01", expected=2),
        case("2000-01-01", "2000-12-31", expected=366),
        case("2024-01-02", "2024-01-01", expected=0),
        case("2024-05-10", "2024-05-01", expected=0),
    ],
    reference='''
def days_inclusive(start: str, end: str) -> int:
    from datetime import date
    diff = (date.fromisoformat(end) - date.fromisoformat(start)).days
    return diff + 1 if diff >= 0 else 0
''',
    bugs={
        "off_by_one": '''
def days_inclusive(start: str, end: str) -> int:
    from datetime import date
    diff = (date.fromisoformat(end) - date.fromisoformat(start)).days
    return max(diff, 0)
''',
        "negative_numbers": '''
def days_inclusive(start: str, end: str) -> int:
    from datetime import date
    diff = (date.fromisoformat(end) - date.fromisoformat(start)).days
    return diff + 1
''',
    },
    tags=("off_by_one", "negative_numbers"),
))


# ---------------------------------------------------------------- 5
TASKS.append(Task(
    id="bucket_counts",
    split="train",
    entry_point="bucket_counts",
    signature="def bucket_counts(vals: list[float], edges: list[float]) -> list[int]:",
    spec=(
        "edges is strictly increasing; m+1 edges define m bins. Bin i covers the half-open interval "
        "[edges[i], edges[i+1]), except the LAST bin, which is closed on both sides: "
        "[edges[m-1], edges[m]]. Return the list of m counts of how many values fall in each bin; values "
        "outside [edges[0], edges[-1]] are ignored. If edges has fewer than 2 elements, return []."
    ),
    tests=[
        case([0, 1, 1.5, 2, 3, 4, -1, 5], [0, 1, 2, 4], expected=[1, 2, 3]),
        case([10], [0, 5, 10], expected=[0, 1]),
        case([0.5, 0.25, 0.75, 1.0], [0, 0.5, 1], expected=[1, 3]),
        case([-3, -2.5, -1, 0], [-3, -1, 0], expected=[2, 2]),
        case([], [0, 1], expected=[0]),
        case([1, 2], [], expected=[]),
        case([3], [3], expected=[]),
        case([100, -100], [0, 1, 2], expected=[0, 0]),
    ],
    reference='''
def bucket_counts(vals: list[float], edges: list[float]) -> list[int]:
    import bisect
    m = len(edges) - 1
    if m < 1:
        return []
    counts = [0] * m
    for v in vals:
        if v < edges[0] or v > edges[-1]:
            continue
        if v == edges[-1]:
            counts[-1] += 1
            continue
        counts[bisect.bisect_right(edges, v) - 1] += 1
    return counts
''',
    bugs={
        "off_by_one": '''
def bucket_counts(vals: list[float], edges: list[float]) -> list[int]:
    import bisect
    m = len(edges) - 1
    if m < 1:
        return []
    counts = [0] * m
    for v in vals:
        if v < edges[0] or v >= edges[-1]:
            continue
        counts[bisect.bisect_right(edges, v) - 1] += 1
    return counts
''',
        "empty_input": '''
def bucket_counts(vals: list[float], edges: list[float]) -> list[int]:
    import bisect
    m = len(edges) - 1
    counts = [0] * m
    for v in vals:
        if v < edges[0] or v > edges[-1]:
            continue
        if v == edges[-1]:
            counts[-1] += 1
            continue
        counts[bisect.bisect_right(edges, v) - 1] += 1
    return counts
''',
    },
    tags=("off_by_one", "empty_input"),
))


# ---------------------------------------------------------------- 6
TASKS.append(Task(
    id="flatten_depth",
    split="train",
    entry_point="flatten_depth",
    signature="def flatten_depth(nested: list, depth: int) -> list:",
    spec=(
        "Flatten nested lists by at most depth levels (depth >= 0): each level splices the elements of "
        "list elements into their parent. Only lists are flattened; strings and other values are kept "
        "as single elements. depth 0 performs no flattening but still returns a new list. The input "
        "must not be modified, and the result must share no list object with the input (any lists left "
        "nested in the result are fresh copies)."
    ),
    tests=[
        case([1, [2, [3, [4]]]], 1, expected=[1, 2, [3, [4]]], check_alias=True),
        case([1, [2, [3, [4]]]], 2, expected=[1, 2, 3, [4]], check_alias=True),
        case([1, [2, [3, [4]]]], 10, expected=[1, 2, 3, 4], check_alias=True),
        case([1, [2, [3]]], 0, expected=[1, [2, [3]]], check_alias=True),
        case([1, 2, 3], 0, expected=[1, 2, 3], check_alias=True),
        case(["ab", ["cd", ["e"]]], 1, expected=["ab", "cd", ["e"]], check_alias=True),
        case([[], [1, []], 2], 1, expected=[1, [], 2], check_alias=True),
        case([], 3, expected=[], check_alias=True),
    ],
    reference='''
def flatten_depth(nested: list, depth: int) -> list:
    def clone(x):
        return [clone(y) for y in x] if isinstance(x, list) else x

    def go(xs, d):
        out = []
        for x in xs:
            if isinstance(x, list) and d > 0:
                out.extend(go(x, d - 1))
            else:
                out.append(clone(x))
        return out

    return go(nested, depth)
''',
    bugs={
        "shared_reference": '''
def flatten_depth(nested: list, depth: int) -> list:
    def clone(x):
        return [clone(y) for y in x] if isinstance(x, list) else x

    def go(xs, d):
        out = []
        for x in xs:
            if isinstance(x, list) and d > 0:
                out.extend(go(x, d - 1))
            else:
                out.append(clone(x))
        return out

    if depth == 0:
        return nested
    return go(nested, depth)
''',
        "off_by_one": '''
def flatten_depth(nested: list, depth: int) -> list:
    def clone(x):
        return [clone(y) for y in x] if isinstance(x, list) else x

    def go(xs, d):
        out = []
        for x in xs:
            if isinstance(x, list) and d >= 0:
                out.extend(go(x, d - 1))
            else:
                out.append(clone(x))
        return out

    return go(nested, depth)
''',
    },
    tags=("shared_reference", "off_by_one", "mutates_input"),
))


# ---------------------------------------------------------------- 7
TASKS.append(Task(
    id="count_gap_pairs",
    split="train",
    entry_point="count_gap_pairs",
    signature="def count_gap_pairs(nums: list[int], d: int) -> int:",
    spec=(
        "Count index pairs (i, j) with i < j such that abs(nums[i] - nums[j]) == d, where d >= 0. "
        "Each unordered pair of positions is counted exactly once, and an element is never paired with "
        "itself (so d == 0 counts pairs of distinct positions holding equal values). An empty or "
        "single-element list gives 0. Must run in roughly linear time for len(nums) up to 200_000."
    ),
    tests=[
        case([1, 5, 3, 4, 2], 2, expected=3),
        case([1, 1, 1], 0, expected=3),
        case([1, 1, 2, 2], 1, expected=4),
        case([-2, 0, 2, -2], 2, expected=3),
        case([1, 2, 3], 5, expected=0),
        case([], 3, expected=0),
        case([7], 0, expected=0),
        case(expr("[0] * 200000"), 0, expected=19999900000),
        case(expr("list(range(200000))"), 1, expected=199999),
    ],
    reference='''
def count_gap_pairs(nums: list[int], d: int) -> int:
    from collections import Counter
    c = Counter(nums)
    if d == 0:
        return sum(v * (v - 1) // 2 for v in c.values())
    return sum(v * c.get(x + d, 0) for x, v in c.items())
''',
    bugs={
        "inefficient": '''
def count_gap_pairs(nums: list[int], d: int) -> int:
    n = len(nums)
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(nums[i] - nums[j]) == d:
                total += 1
    return total
''',
        "negative_numbers": '''
def count_gap_pairs(nums: list[int], d: int) -> int:
    from collections import Counter
    c = Counter(nums)
    return sum(v * c.get(x + d, 0) for x, v in c.items())
''',
    },
    tags=("inefficient", "negative_numbers", "empty_input"),
))


# ---------------------------------------------------------------- 8
TASKS.append(Task(
    id="merge_by_key",
    split="train",
    entry_point="merge_by_key",
    signature="def merge_by_key(a: list[list], b: list[list]) -> list[list]:",
    spec=(
        "Each input is a list of [key, value] items already sorted by key (item[0]) ascending; keys are "
        "mutually comparable (all numbers or all strings). Merge them into a new list sorted by key. "
        "When keys are equal, all of a's items with that key come before b's, and each input keeps its "
        "own internal order (a stable merge). Either or both inputs may be empty; the inputs must not "
        "be modified."
    ),
    tests=[
        case([[1, "a"], [3, "c"]], [[2, "b"], [4, "d"]],
             expected=[[1, "a"], [2, "b"], [3, "c"], [4, "d"]]),
        case([[1, "a1"], [2, "a2"], [2, "a3"]], [[2, "b1"], [2, "b2"], [3, "b3"]],
             expected=[[1, "a1"], [2, "a2"], [2, "a3"], [2, "b1"], [2, "b2"], [3, "b3"]]),
        case([["apple", 1], ["cherry", 2]], [["apple", 9], ["banana", 3]],
             expected=[["apple", 1], ["apple", 9], ["banana", 3], ["cherry", 2]]),
        case([[5, "a"]], [[1, "b"], [2, "c"]], expected=[[1, "b"], [2, "c"], [5, "a"]]),
        case([[0, "x"]], [[0, "y"]], expected=[[0, "x"], [0, "y"]]),
        case([], [[1, "x"]], expected=[[1, "x"]]),
        case([[1, "x"], [1, "z"]], [], expected=[[1, "x"], [1, "z"]]),
        case([], [], expected=[]),
    ],
    reference='''
def merge_by_key(a: list[list], b: list[list]) -> list[list]:
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] <= b[j][0]:
            out.append(list(a[i]))
            i += 1
        else:
            out.append(list(b[j]))
            j += 1
    out.extend(list(x) for x in a[i:])
    out.extend(list(x) for x in b[j:])
    return out
''',
    bugs={
        "tie_ordering": '''
def merge_by_key(a: list[list], b: list[list]) -> list[list]:
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] < b[j][0]:
            out.append(list(a[i]))
            i += 1
        else:
            out.append(list(b[j]))
            j += 1
    out.extend(list(x) for x in a[i:])
    out.extend(list(x) for x in b[j:])
    return out
''',
        "empty_input": '''
def merge_by_key(a: list[list], b: list[list]) -> list[list]:
    if a[-1][0] <= b[0][0]:
        return [list(x) for x in a + b]
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] <= b[j][0]:
            out.append(list(a[i]))
            i += 1
        else:
            out.append(list(b[j]))
            j += 1
    out.extend(list(x) for x in a[i:])
    out.extend(list(x) for x in b[j:])
    return out
''',
    },
    tags=("tie_ordering", "empty_input"),
))


# ---------------------------------------------------------------- 9
TASKS.append(Task(
    id="rescale",
    split="train",
    entry_point="rescale",
    signature="def rescale(vals: list[float], lo: float, hi: float) -> list[float]:",
    spec=(
        "Linearly map vals so that the minimum value maps to lo and the maximum maps to hi (hi may be "
        "smaller than lo, which reverses the direction). Round each result with round(x, 3). If all "
        "values are equal (including a single value), every output is round(lo, 3). An empty list "
        "gives []."
    ),
    tests=[
        case([0, 5, 10], 0, 1, expected=[0.0, 0.5, 1.0]),
        case([1, 2, 3, 4], 0, 1, expected=[0.0, 0.333, 0.667, 1.0]),
        case([0, 1, 2, 3], -1, 0, expected=[-1.0, -0.667, -0.333, 0.0]),
        case([-5, 0, 5], 10, 20, expected=[10.0, 15.0, 20.0]),
        case([0, 1, 2], 1, 0, expected=[1.0, 0.5, 0.0]),
        case([3, 3, 3], 2, 9, expected=[2.0, 2.0, 2.0]),
        case([4, 4], 0.12345, 1, expected=[0.123, 0.123]),
        case([7], -1, 1, expected=[-1.0]),
        case([], 0, 1, expected=[]),
    ],
    reference='''
def rescale(vals: list[float], lo: float, hi: float) -> list[float]:
    if not vals:
        return []
    mn, mx = min(vals), max(vals)
    if mn == mx:
        return [round(lo, 3) for _ in vals]
    scale = (hi - lo) / (mx - mn)
    return [round(lo + (v - mn) * scale, 3) for v in vals]
''',
    bugs={
        "empty_input": '''
def rescale(vals: list[float], lo: float, hi: float) -> list[float]:
    mn, mx = min(vals), max(vals)
    if mn == mx:
        return [round(lo, 3) for _ in vals]
    scale = (hi - lo) / (mx - mn)
    return [round(lo + (v - mn) * scale, 3) for v in vals]
''',
        "float_rounding": '''
def rescale(vals: list[float], lo: float, hi: float) -> list[float]:
    if not vals:
        return []
    mn, mx = min(vals), max(vals)
    if mn == mx:
        return [int(lo * 1000) / 1000 for _ in vals]
    scale = (hi - lo) / (mx - mn)
    return [int((lo + (v - mn) * scale) * 1000) / 1000 for v in vals]
''',
    },
    tags=("empty_input", "float_rounding"),
))


# ---------------------------------------------------------------- 10
TASKS.append(Task(
    id="wrap_greedy",
    split="train",
    entry_point="wrap_greedy",
    signature="def wrap_greedy(text: str, width: int) -> list[str]:",
    spec=(
        "Split text into words on any run of whitespace (spaces, tabs, newlines), then pack words "
        "greedily into lines: each line is words joined by single spaces, and a word is added to the "
        "current line if the resulting length is <= width (inclusive); otherwise it starts a new line. "
        "A word longer than width sits alone on its own line. width >= 1. Empty or whitespace-only "
        "text gives []."
    ),
    tests=[
        case("the quick brown fox jumps over the lazy dog", 10,
             expected=["the quick", "brown fox", "jumps over", "the lazy", "dog"]),
        case("ab cd", 5, expected=["ab cd"]),
        case("a extraordinarily b", 5, expected=["a", "extraordinarily", "b"]),
        case("  hello\tworld\n\nfoo  ", 11, expected=["hello world", "foo"]),
        case("a b c", 1, expected=["a", "b", "c"]),
        case("one", 1, expected=["one"]),
        case("   \t\n", 5, expected=[]),
        case("", 3, expected=[]),
    ],
    reference='''
def wrap_greedy(text: str, width: int) -> list[str]:
    lines = []
    cur = ""
    for w in text.split():
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
''',
    bugs={
        "whitespace_case": '''
def wrap_greedy(text: str, width: int) -> list[str]:
    lines = []
    cur = ""
    for w in text.split(" "):
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
''',
        "off_by_one": '''
def wrap_greedy(text: str, width: int) -> list[str]:
    lines = []
    cur = ""
    for w in text.split():
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) < width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
''',
    },
    tags=("whitespace_case", "off_by_one", "empty_input"),
))


# ---------------------------------------------------------------- 11
TASKS.append(Task(
    id="caesar",
    split="train",
    entry_point="caesar",
    signature="def caesar(text: str, k: int) -> str:",
    spec=(
        "Shift every ASCII letter (a-z, A-Z) forward by k positions in the alphabet, wrapping around; "
        "k may be any integer, including negative values and values whose magnitude exceeds 26. "
        "Letter case is preserved. Every other character, including digits, punctuation, whitespace "
        "and non-ASCII letters such as 'é' or 'ß', is returned unchanged."
    ),
    tests=[
        case("abc", 1, expected="bcd"),
        case("xyz", 3, expected="abc"),
        case("Hello, World!", 13, expected="Uryyb, Jbeyq!"),
        case("abc", -1, expected="zab"),
        case("Zebra", -30, expected="Vaxnw"),
        case("abc", -1000, expected="opq"),
        case("abc", 52, expected="abc"),
        case("café ß 1", 1, expected="dbgé ß 1"),
        case("Ünïcode", 2, expected="Üpïeqfg"),
        case("", 5, expected=""),
    ],
    reference='''
def caesar(text: str, k: int) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + k) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + k) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)
''',
    bugs={
        "negative_numbers": '''
def caesar(text: str, k: int) -> str:
    if k >= 0:
        k %= 26
    out = []
    for ch in text:
        if "a" <= ch <= "z" or "A" <= ch <= "Z":
            base = 97 if ch.islower() else 65
            idx = ord(ch) - base + k
            if idx < 0:
                idx += 26
            elif idx >= 26:
                idx -= 26
            out.append(chr(idx + base))
        else:
            out.append(ch)
    return "".join(out)
''',
        "whitespace_case": '''
def caesar(text: str, k: int) -> str:
    out = []
    for ch in text:
        if ch.isalpha():
            base = 97 if ch.islower() else 65
            out.append(chr((ord(ch) - base + k) % 26 + base))
        else:
            out.append(ch)
    return "".join(out)
''',
    },
    tags=("negative_numbers", "whitespace_case"),
))


# ---------------------------------------------------------------- 12
TASKS.append(Task(
    id="deep_merge",
    split="train",
    entry_point="deep_merge",
    signature="def deep_merge(base: dict, override: dict) -> dict:",
    spec=(
        "Return a new dict combining base and override recursively: for a key present in both, if both "
        "values are dicts they are merged recursively, otherwise the override value wins (lists are "
        "replaced, not merged). Keys present in only one input are kept. Neither input may be modified, "
        "and the result must share no dict or list object with either input (copy nested values)."
    ),
    tests=[
        case({"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"c": 20, "e": 5}, "f": 6},
             expected={"a": 1, "b": {"c": 20, "d": 3, "e": 5}, "f": 6}, check_alias=True),
        case({"a": {"b": {"c": 1, "d": 2}}}, {"a": {"b": {"d": 3}}},
             expected={"a": {"b": {"c": 1, "d": 3}}}, check_alias=True),
        case({"a": {"x": 1}}, {"a": 5}, expected={"a": 5}),
        case({"a": 5}, {"a": {"x": 1}}, expected={"a": {"x": 1}}, check_alias=True),
        case({"l": [1, 2]}, {"l": [3]}, expected={"l": [3]}, check_alias=True),
        case({"k": {"n": {"m": 1}}, "z": 0}, {"z": 1}, expected={"k": {"n": {"m": 1}}, "z": 1}, check_alias=True),
        case({}, {"k": [1, 2]}, expected={"k": [1, 2]}, check_alias=True),
        case({}, {}, expected={}),
    ],
    reference='''
def deep_merge(base: dict, override: dict) -> dict:
    import copy
    result = {k: copy.deepcopy(v) for k, v in base.items()}
    for k, v in override.items():
        if isinstance(result.get(k), dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result
''',
    bugs={
        "mutates_input": '''
def deep_merge(base: dict, override: dict) -> dict:
    import copy
    for k, v in override.items():
        if isinstance(base.get(k), dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return copy.deepcopy(base)
''',
        "shared_reference": '''
def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(result.get(k), dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result
''',
    },
    tags=("mutates_input", "shared_reference"),
))
