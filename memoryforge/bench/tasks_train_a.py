"""Training (experience) tasks, batch A."""

from __future__ import annotations

from .task import Task, case, expr, raises

SPLIT = "train"

TASKS: list[Task] = [
    # 1 ------------------------------------------------------------------
    Task(
        id="merge_inclusive_spans",
        split=SPLIT,
        entry_point="merge_inclusive_spans",
        signature="def merge_inclusive_spans(spans: list[list[int]]) -> list[list[int]]:",
        spec=(
            "Each span is a two-element list [start, end] of integers with start <= end, describing the "
            "inclusive integer range start..end. Merge spans that overlap or touch (a span ending at e touches "
            "one starting at e + 1) and return the merged spans as new [start, end] lists sorted by start. "
            "The input may be in any order and must not be modified. An empty input returns []."
        ),
        tests=[
            case([[1, 3], [2, 6], [8, 10], [15, 18]], expected=[[1, 6], [8, 10], [15, 18]]),
            case([[1, 2], [3, 5]], expected=[[1, 5]]),
            case([[1, 2], [4, 5]], expected=[[1, 2], [4, 5]]),
            case([[5, 7], [1, 2], [10, 11]], expected=[[1, 2], [5, 7], [10, 11]]),
            case([[6, 8], [1, 10], [2, 3]], expected=[[1, 10]]),
            case([], expected=[]),
            case([[4, 4]], expected=[[4, 4]]),
            case([[0, 2], [-3, -1]], expected=[[-3, 2]]),
            case([[9, 9], [7, 7], [8, 8]], expected=[[7, 9]]),
        ],
        reference='''def merge_inclusive_spans(spans):
    out = []
    for start, end in sorted(spans, key=lambda p: p[0]):
        if out and start <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
''',
        bugs={
            "empty_input": '''def merge_inclusive_spans(spans):
    ordered = sorted(spans, key=lambda p: p[0])
    out = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
''',
            "off_by_one": '''def merge_inclusive_spans(spans):
    out = []
    for start, end in sorted(spans, key=lambda p: p[0]):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
''',
            "mutates_input": '''def merge_inclusive_spans(spans):
    spans.sort(key=lambda p: p[0])
    out = []
    for start, end in spans:
        if out and start <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
''',
        },
        tags=("empty_input", "off_by_one", "mutates_input", "shared_reference"),
    ),
    # 2 ------------------------------------------------------------------
    Task(
        id="count_in_window",
        split=SPLIT,
        entry_point="count_in_window",
        signature="def count_in_window(nums: list[int], lo: int, hi: int) -> int:",
        spec=(
            "Return how many elements x of nums satisfy lo <= x <= hi (both bounds inclusive). "
            "Values and bounds may be negative or zero, and duplicates are each counted. "
            "If lo > hi the window is empty and the result is 0; an empty nums also gives 0."
        ),
        tests=[
            case([1, 5, 3, 7, 9], 3, 7, expected=3),
            case([2, 2, 2], 2, 2, expected=3),
            case([1, 2, 3], 5, 1, expected=0),
            case([], 0, 10, expected=0),
            case([-5, -2, 0, 3], -5, -2, expected=2),
            case([-3, -1, 0, 2, 5], 0, 3, expected=2),
            case([10, 20, 30], 0, 100, expected=3),
            case([4, 8, 15, 16, 23, 42], 15, 23, expected=3),
        ],
        reference='''def count_in_window(nums, lo, hi):
    if lo > hi:
        return 0
    return sum(1 for x in nums if lo <= x <= hi)
''',
        bugs={
            "off_by_one": '''def count_in_window(nums, lo, hi):
    if lo > hi:
        return 0
    return sum(1 for x in nums if lo <= x < hi)
''',
            "negative_numbers": '''def count_in_window(nums, lo, hi):
    if not nums or lo > hi:
        return 0
    lo = lo or min(nums)
    return sum(1 for x in nums if lo <= x <= hi)
''',
        },
        tags=("off_by_one", "negative_numbers", "empty_input"),
    ),
    # 3 ------------------------------------------------------------------
    Task(
        id="rank_with_ties",
        split=SPLIT,
        entry_point="rank_with_ties",
        signature="def rank_with_ties(scores: list[int]) -> list[int]:",
        spec=(
            "Return the competition rank of each score, in the same order as the input. The highest score "
            "has rank 1; equal scores share the same rank, and the next lower score's rank skips accordingly "
            "(rank = 1 + number of scores strictly greater), e.g. [10, 20, 20, 5] -> [3, 1, 1, 4]. "
            "An empty list returns []."
        ),
        tests=[
            case([10, 20, 20, 5], expected=[3, 1, 1, 4]),
            case([], expected=[]),
            case([7], expected=[1]),
            case([5, 5, 5], expected=[1, 1, 1]),
            case([90, 80, 80, 80, 70], expected=[1, 2, 2, 2, 5]),
            case([-1, -3, -1, 0], expected=[2, 4, 2, 1]),
            case([1, 2, 3, 4], expected=[4, 3, 2, 1]),
        ],
        reference='''def rank_with_ties(scores):
    first = {}
    for i, s in enumerate(sorted(scores, reverse=True)):
        first.setdefault(s, i + 1)
    return [first[s] for s in scores]
''',
        bugs={
            "tie_ordering": '''def rank_with_ties(scores):
    distinct = sorted(set(scores), reverse=True)
    rank = {s: i + 1 for i, s in enumerate(distinct)}
    return [rank[s] for s in scores]
''',
            "empty_input": '''def rank_with_ties(scores):
    order = sorted(scores, reverse=True)
    first = {order[0]: 1}
    for i in range(1, len(order)):
        if order[i] != order[i - 1]:
            first[order[i]] = i + 1
    return [first[s] for s in scores]
''',
        },
        tags=("tie_ordering", "empty_input", "negative_numbers"),
    ),
    # 4 ------------------------------------------------------------------
    Task(
        id="sort_words_by_length",
        split=SPLIT,
        entry_point="sort_words_by_length",
        signature="def sort_words_by_length(words: list[str]) -> list[str]:",
        spec=(
            "Return a new list containing the words sorted by length, shortest first. Words of equal length "
            "keep their original relative order (no alphabetical tie-break). The input list must not be "
            "modified. An empty list returns []."
        ),
        tests=[
            case(["pear", "fig", "apple", "kiwi", "date"], expected=["fig", "pear", "kiwi", "date", "apple"]),
            case([], expected=[]),
            case(["solo"], expected=["solo"]),
            case(["ccc", "bb", "a"], expected=["a", "bb", "ccc"]),
            case(["zz", "yy", "xx", "w"], expected=["w", "zz", "yy", "xx"]),
            case(["banana", "", "cherry", "ab"], expected=["", "ab", "banana", "cherry"]),
            case(["Bob", "amy", "Al", "cy"], expected=["Al", "cy", "Bob", "amy"]),
        ],
        reference='''def sort_words_by_length(words):
    return sorted(words, key=len)
''',
        bugs={
            "tie_ordering": '''def sort_words_by_length(words):
    return sorted(words, key=lambda w: (len(w), w))
''',
            "mutates_input": '''def sort_words_by_length(words):
    words.sort(key=len)
    return words
''',
        },
        tags=("tie_ordering", "mutates_input", "empty_input"),
    ),
    # 5 ------------------------------------------------------------------
    Task(
        id="normalize_tags",
        split=SPLIT,
        entry_point="normalize_tags",
        signature="def normalize_tags(raw: str) -> list[str]:",
        spec=(
            "Split raw on commas, strip surrounding whitespace (spaces, tabs, newlines) from each piece and "
            "lowercase it. Drop pieces that are empty after stripping, and drop duplicates, keeping only the "
            "first occurrence. Return the tags in order of first occurrence; an empty or blank string returns []."
        ),
        tests=[
            case("Python, rust ,GO", expected=["python", "rust", "go"]),
            case("zeta,alpha,Zeta,beta", expected=["zeta", "alpha", "beta"]),
            case("", expected=[]),
            case("  ,  ,", expected=[]),
            case("a,,b", expected=["a", "b"]),
            case("\tWeb\n, web ,API", expected=["web", "api"]),
            case("single", expected=["single"]),
            case("c,b,a", expected=["c", "b", "a"]),
        ],
        reference='''def normalize_tags(raw):
    out = []
    seen = set()
    for piece in raw.split(","):
        tag = piece.strip().lower()
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out
''',
        bugs={
            "whitespace_case": '''def normalize_tags(raw):
    out = []
    seen = set()
    for piece in raw.split(","):
        tag = piece.lower()
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out
''',
            "tie_ordering": '''def normalize_tags(raw):
    tags = {piece.strip().lower() for piece in raw.split(",")}
    return sorted(t for t in tags if t)
''',
            "empty_input": '''def normalize_tags(raw):
    out = []
    seen = set()
    for piece in raw.split(","):
        tag = piece.strip().lower()
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out
''',
        },
        tags=("whitespace_case", "tie_ordering", "empty_input"),
    ),
    # 6 ------------------------------------------------------------------
    Task(
        id="parse_kv_line",
        split=SPLIT,
        entry_point="parse_kv_line",
        signature="def parse_kv_line(line: str) -> dict[str, str]:",
        spec=(
            "Parse a line of key=value pairs separated by ';' into a dict of strings. Split each segment at "
            "its FIRST '=' only (the value may itself contain '='); strip surrounding whitespace from key and "
            "value and lowercase the key. Ignore segments that contain no '=' (including empty segments) and "
            "segments whose key is empty after stripping. If a key repeats, the later value wins; an empty "
            "line returns {}."
        ),
        tests=[
            case("a=1;b=2", expected={"a": "1", "b": "2"}),
            case(" Name = Bob ; AGE=30 ", expected={"name": "Bob", "age": "30"}),
            case("x=1=2", expected={"x": "1=2"}),
            case("k=1;K=2", expected={"k": "2"}),
            case("", expected={}),
            case("a=1;;junk;b=", expected={"a": "1", "b": ""}),
            case("=v;c=3", expected={"c": "3"}),
            case("a=1;", expected={"a": "1"}),
        ],
        reference='''def parse_kv_line(line):
    out = {}
    for segment in line.split(";"):
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        key = key.strip().lower()
        if key:
            out[key] = value.strip()
    return out
''',
        bugs={
            "whitespace_case": '''def parse_kv_line(line):
    out = {}
    for segment in line.split(";"):
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        key = key.lower()
        if key:
            out[key] = value.strip()
    return out
''',
            "empty_input": '''def parse_kv_line(line):
    out = {}
    for segment in line.split(";"):
        key, value = segment.split("=", 1)
        key = key.strip().lower()
        if key:
            out[key] = value.strip()
    return out
''',
        },
        tags=("whitespace_case", "empty_input"),
    ),
    # 7 ------------------------------------------------------------------
    Task(
        id="window_means",
        split=SPLIT,
        entry_point="window_means",
        signature="def window_means(vals: list[float], k: int) -> list[float]:",
        spec=(
            "Return the mean of every window of k consecutive values of vals, in order, each rounded with "
            "round(x, 2); there are len(vals) - k + 1 windows, and the result is [] when k > len(vals). "
            "Raise ValueError if k <= 0. It must handle len(vals) = 200_000 with large k (e.g. 10_000) in "
            "well under a second, so do not re-sum each window."
        ),
        tests=[
            case([1, 2, 3, 4, 5], 2, expected=[1.5, 2.5, 3.5, 4.5]),
            case([1, 2, 3], 3, expected=[2.0]),
            case([1, 2], 3, expected=[]),
            case([], 1, expected=[]),
            case([1, 2, 2, 4], 3, expected=[1.67, 2.67]),
            case([-1, -2, 3], 2, expected=[-1.5, 0.5]),
            case([5, 6, 7], 1, expected=[5.0, 6.0, 7.0]),
            case([1, 2, 3], 0, expected=raises("ValueError")),
            case([1, 2, 3], -2, expected=raises("ValueError")),
            case(expr("[float(i % 10) for i in range(200_000)]"), 10_000, expected=expr("[4.5] * 190_001")),
        ],
        reference='''def window_means(vals, k):
    if k <= 0:
        raise ValueError("k must be positive")
    n = len(vals)
    if k > n:
        return []
    total = sum(vals[:k])
    out = [round(total / k, 2)]
    for i in range(k, n):
        total += vals[i] - vals[i - k]
        out.append(round(total / k, 2))
    return out
''',
        bugs={
            "inefficient": '''def window_means(vals, k):
    if k <= 0:
        raise ValueError("k must be positive")
    n = len(vals)
    if k > n:
        return []
    return [round(sum(vals[i:i + k]) / k, 2) for i in range(n - k + 1)]
''',
            "off_by_one": '''def window_means(vals, k):
    if k <= 0:
        raise ValueError("k must be positive")
    n = len(vals)
    if k >= n:
        return []
    total = sum(vals[:k])
    out = [round(total / k, 2)]
    for i in range(k, n):
        total += vals[i] - vals[i - k]
        out.append(round(total / k, 2))
    return out
''',
            "error_contract": '''def window_means(vals, k):
    n = len(vals)
    if k > n:
        return []
    total = sum(vals[:k])
    out = [round(total / k, 2)]
    for i in range(k, n):
        total += vals[i] - vals[i - k]
        out.append(round(total / k, 2))
    return out
''',
        },
        tags=("inefficient", "off_by_one", "error_contract", "float_rounding", "empty_input"),
    ),
    # 8 ------------------------------------------------------------------
    Task(
        id="split_cents",
        split=SPLIT,
        entry_point="split_cents",
        signature="def split_cents(total: int, parts: int) -> list[int]:",
        spec=(
            "Split an integer amount of cents into parts integer shares that differ by at most one cent and "
            "sum to total. Every share gets abs(total) // parts, and the first abs(total) % parts shares get "
            "one extra cent. A negative total is split exactly like its absolute value with every share "
            "negated (e.g. -7 into 3 -> [-3, -2, -2]). Raise ValueError if parts < 1."
        ),
        tests=[
            case(100, 3, expected=[34, 33, 33]),
            case(10, 5, expected=[2, 2, 2, 2, 2]),
            case(7, 1, expected=[7]),
            case(2, 5, expected=[1, 1, 0, 0, 0]),
            case(0, 3, expected=[0, 0, 0]),
            case(-7, 3, expected=[-3, -2, -2]),
            case(-100, 3, expected=[-34, -33, -33]),
            case(5, 0, expected=raises("ValueError")),
            case(5, -2, expected=raises("ValueError")),
        ],
        reference='''def split_cents(total, parts):
    if parts < 1:
        raise ValueError("parts must be at least 1")
    sign = -1 if total < 0 else 1
    base, extra = divmod(abs(total), parts)
    return [sign * (base + (1 if i < extra else 0)) for i in range(parts)]
''',
        bugs={
            "negative_numbers": '''def split_cents(total, parts):
    if parts < 1:
        raise ValueError("parts must be at least 1")
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]
''',
            "error_contract": '''def split_cents(total, parts):
    sign = -1 if total < 0 else 1
    base, extra = divmod(abs(total), parts)
    return [sign * (base + (1 if i < extra else 0)) for i in range(parts)]
''',
        },
        tags=("negative_numbers", "error_contract", "empty_input"),
    ),
    # 9 ------------------------------------------------------------------
    Task(
        id="format_percent",
        split=SPLIT,
        entry_point="format_percent",
        signature="def format_percent(num: int, den: int) -> str:",
        spec=(
            "Format the exact value 100 * num / den as a string with exactly one decimal place followed by "
            "'%', rounding halves away from zero (ROUND_HALF_UP, e.g. 1/16 = 6.25 -> '6.3%', -1/16 -> "
            "'-6.3%'). num may be any integer and den is never negative; if den == 0 return 'n/a'. "
            "A value that rounds to zero is written '0.0%' (never '-0.0%')."
        ),
        tests=[
            case(1, 2, expected="50.0%"),
            case(1, 3, expected="33.3%"),
            case(2, 3, expected="66.7%"),
            case(1, 16, expected="6.3%"),
            case(1, 80, expected="1.3%"),
            case(-1, 16, expected="-6.3%"),
            case(3, 2, expected="150.0%"),
            case(0, 7, expected="0.0%"),
            case(5, 0, expected="n/a"),
            case(0, 0, expected="n/a"),
        ],
        reference='''def format_percent(num, den):
    if den == 0:
        return "n/a"
    # tenths of a percent = 1000 * |num| / den, rounded half up using integers only
    tenths = (2000 * abs(num) + den) // (2 * den)
    sign = "-" if num < 0 and tenths > 0 else ""
    return f"{sign}{tenths // 10}.{tenths % 10}%"
''',
        bugs={
            "float_rounding": '''def format_percent(num, den):
    if den == 0:
        return "n/a"
    return f"{100 * num / den:.1f}%"
''',
            "error_contract": '''def format_percent(num, den):
    tenths = (2000 * abs(num) + den) // (2 * den)
    sign = "-" if num < 0 and tenths > 0 else ""
    return f"{sign}{tenths // 10}.{tenths % 10}%"
''',
        },
        tags=("float_rounding", "error_contract", "negative_numbers"),
    ),
    # 10 -----------------------------------------------------------------
    Task(
        id="chunk_even",
        split=SPLIT,
        entry_point="chunk_even",
        signature="def chunk_even(items: list, size: int) -> list[list]:",
        spec=(
            "Split items into consecutive chunks of length size, in order; the last chunk holds the remaining "
            "items and may be shorter, but is never dropped. Each chunk is a new list and items is not modified. "
            "An empty items gives []. Raise ValueError if size < 1."
        ),
        tests=[
            case([1, 2, 3, 4, 5, 6, 7], 3, expected=[[1, 2, 3], [4, 5, 6], [7]]),
            case([1, 2, 3, 4, 5, 6], 2, expected=[[1, 2], [3, 4], [5, 6]]),
            case([], 3, expected=[]),
            case([1, 2], 5, expected=[[1, 2]]),
            case(["a", "b"], 1, expected=[["a"], ["b"]], check_alias=True),
            case([1, 2, 3, 4, 5], 2, expected=[[1, 2], [3, 4], [5]]),
            case([1, 2, 3], 0, expected=raises("ValueError")),
            case([1, 2, 3], -2, expected=raises("ValueError")),
        ],
        reference='''def chunk_even(items, size):
    if size < 1:
        raise ValueError("size must be at least 1")
    n_chunks = -(-len(items) // size)
    return [items[i * size:(i + 1) * size] for i in range(n_chunks)]
''',
        bugs={
            "off_by_one": '''def chunk_even(items, size):
    if size < 1:
        raise ValueError("size must be at least 1")
    n_chunks = len(items) // size
    return [items[i * size:(i + 1) * size] for i in range(n_chunks)]
''',
            "error_contract": '''def chunk_even(items, size):
    n_chunks = -(-len(items) // size)
    return [items[i * size:(i + 1) * size] for i in range(n_chunks)]
''',
        },
        tags=("off_by_one", "error_contract", "empty_input", "shared_reference"),
    ),
    # 11 -----------------------------------------------------------------
    Task(
        id="rotate_right",
        split=SPLIT,
        entry_point="rotate_right",
        signature="def rotate_right(items: list, k: int) -> list:",
        spec=(
            "Return a new list with items rotated right by k positions (the last k items move to the front). "
            "k may be larger than len(items) (it wraps around) or negative (rotate left by -k). The result must "
            "always be a fresh list, even when the rotation is a no-op, and items must not be modified. "
            "An empty items returns [] for any k."
        ),
        tests=[
            case([1, 2, 3, 4, 5], 2, expected=[4, 5, 1, 2, 3]),
            case([1, 2, 3, 4, 5], 7, expected=[4, 5, 1, 2, 3]),
            case([1, 2, 3, 4, 5], -1, expected=[2, 3, 4, 5, 1]),
            case([1, 2, 3, 4, 5], -6, expected=[2, 3, 4, 5, 1]),
            case([], 3, expected=[]),
            case([1, 2, 3], 0, expected=[1, 2, 3], check_alias=True),
            case([1, 2, 3], 3, expected=[1, 2, 3], check_alias=True),
            case(["x"], 5, expected=["x"]),
        ],
        reference='''def rotate_right(items, k):
    n = len(items)
    if n == 0:
        return []
    k %= n
    return items[n - k:] + items[:n - k]
''',
        bugs={
            "empty_input": '''def rotate_right(items, k):
    n = len(items)
    k %= n
    return items[n - k:] + items[:n - k]
''',
            "negative_numbers": '''def rotate_right(items, k):
    n = len(items)
    if n == 0:
        return []
    k = abs(k) % n
    return items[n - k:] + items[:n - k]
''',
            "shared_reference": '''def rotate_right(items, k):
    n = len(items)
    if n == 0:
        return []
    k %= n
    if k == 0:
        return items
    return items[n - k:] + items[:n - k]
''',
        },
        tags=("empty_input", "negative_numbers", "shared_reference", "mutates_input"),
    ),
    # 12 -----------------------------------------------------------------
    Task(
        id="longest_run",
        split=SPLIT,
        entry_point="longest_run",
        signature="def longest_run(s: str) -> list:",
        spec=(
            "Return [char, length] for the longest run of one character repeated consecutively in s. "
            "Comparison is case-sensitive and whitespace counts as a character. If several runs share the "
            "maximum length, return the earliest one. An empty string returns ['', 0]."
        ),
        tests=[
            case("aaabbc", expected=["a", 3]),
            case("abbbcc", expected=["b", 3]),
            case("aabb", expected=["a", 2]),
            case("abc", expected=["a", 1]),
            case("", expected=["", 0]),
            case("z", expected=["z", 1]),
            case("aAAa", expected=["A", 2]),
            case("xyyyzzzz", expected=["z", 4]),
            case("a  b", expected=[" ", 2]),
        ],
        reference='''def longest_run(s):
    if not s:
        return ["", 0]
    best_c, best_n = s[0], 1
    cur_c, cur_n = s[0], 1
    for ch in s[1:]:
        if ch == cur_c:
            cur_n += 1
        else:
            cur_c, cur_n = ch, 1
        if cur_n > best_n:
            best_c, best_n = cur_c, cur_n
    return [best_c, best_n]
''',
        bugs={
            "tie_ordering": '''def longest_run(s):
    if not s:
        return ["", 0]
    best_c, best_n = s[0], 1
    cur_c, cur_n = s[0], 1
    for ch in s[1:]:
        if ch == cur_c:
            cur_n += 1
        else:
            cur_c, cur_n = ch, 1
        if cur_n >= best_n:
            best_c, best_n = cur_c, cur_n
    return [best_c, best_n]
''',
            "empty_input": '''def longest_run(s):
    best_c, best_n = s[0], 1
    cur_c, cur_n = s[0], 1
    for ch in s[1:]:
        if ch == cur_c:
            cur_n += 1
        else:
            cur_c, cur_n = ch, 1
        if cur_n > best_n:
            best_c, best_n = cur_c, cur_n
    return [best_c, best_n]
''',
        },
        tags=("tie_ordering", "empty_input", "whitespace_case"),
    ),
]
