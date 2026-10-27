"""Held-out evaluation tasks. Their pitfalls reuse patterns seen in the train split."""

from __future__ import annotations

from .task import Task, case, expr, raises

# ---------------------------------------------------------------------------
# 1. free_ranges
# ---------------------------------------------------------------------------

_FREE_REF = '''
def free_ranges(lo, hi, busy):
    out = []
    cur = lo
    for a, b in sorted(busy):
        if a > hi:
            break
        if b < cur:
            continue
        if a > cur:
            out.append([cur, a - 1])
        cur = max(cur, b + 1)
    if cur <= hi:
        out.append([cur, hi])
    return out
'''

_FREE_OBO = '''
def free_ranges(lo, hi, busy):
    out = []
    cur = lo
    for a, b in sorted(busy):
        if a > hi:
            break
        if b < cur:
            continue
        if a > cur:
            out.append([cur, a - 1])
        cur = max(cur, b)
    if cur <= hi:
        out.append([cur, hi])
    return out
'''

_FREE_MUT = '''
def free_ranges(lo, hi, busy):
    out = []
    cur = lo
    busy.sort()
    for a, b in busy:
        if a > hi:
            break
        if b < cur:
            continue
        if a > cur:
            out.append([cur, a - 1])
        cur = max(cur, b + 1)
    if cur <= hi:
        out.append([cur, hi])
    return out
'''

FREE_RANGES = Task(
    id="free_ranges",
    split="eval",
    entry_point="free_ranges",
    signature="def free_ranges(lo: int, hi: int, busy: list[list[int]]) -> list[list[int]]:",
    spec=(
        "All ranges are inclusive integer ranges [a, b] with a <= b. Given the window [lo, hi] and a list of busy "
        "ranges (unsorted, possibly overlapping or touching, possibly extending outside the window), return the "
        "maximal free ranges inside [lo, hi] that contain no busy integer, as [start, end] lists sorted by start. "
        "If lo > hi or nothing is free, return []. Do not modify the busy list."
    ),
    tests=[
        case(1, 10, [[3, 4], [7, 8]], expected=[[1, 2], [5, 6], [9, 10]]),
        case(0, 20, [[15, 18], [2, 5], [4, 9]], expected=[[0, 1], [10, 14], [19, 20]]),
        case(1, 10, [], expected=[[1, 10]]),
        case(1, 10, [[0, 3], [9, 15]], expected=[[4, 8]]),
        case(1, 10, [[-5, 20]], expected=[]),
        case(1, 10, [[3, 3], [4, 4], [6, 6]], expected=[[1, 2], [5, 5], [7, 10]]),
        case(5, 5, [[1, 2], [8, 9]], expected=[[5, 5]]),
        case(1, 10, [[20, 30], [-10, -1]], expected=[[1, 10]]),
        case(5, 4, [], expected=[]),
    ],
    reference=_FREE_REF,
    bugs={"off_by_one": _FREE_OBO, "mutates_input": _FREE_MUT},
    tags=("off_by_one", "mutates_input", "empty_input"),
)

# ---------------------------------------------------------------------------
# 2. leaderboard
# ---------------------------------------------------------------------------

_LB_REF = '''
def leaderboard(entries):
    ranked = sorted(entries, key=lambda e: (-e[1], e[0].strip().casefold()))
    return [e[0] for e in ranked]
'''

_LB_TIE = '''
def leaderboard(entries):
    ranked = sorted(entries, key=lambda e: (e[1], e[0].strip().casefold()), reverse=True)
    return [e[0] for e in ranked]
'''

_LB_WS = '''
def leaderboard(entries):
    ranked = sorted(entries, key=lambda e: (-e[1], e[0]))
    return [e[0] for e in ranked]
'''

LEADERBOARD = Task(
    id="leaderboard",
    split="eval",
    entry_point="leaderboard",
    signature="def leaderboard(entries: list[list]) -> list[str]:",
    spec=(
        "Each entry is a [name, score] list with a string name and an integer score. Return the names ordered by "
        "score descending; entries with equal scores are ordered by name.strip().casefold() ascending, and entries "
        "that are still tied keep their input order. Return the original (unmodified) name strings; [] -> []."
    ),
    tests=[
        case([["ann", 5], ["bob", 9], ["cy", 7]], expected=["bob", "cy", "ann"]),
        case([["bob", 3], ["alice", 3]], expected=["alice", "bob"]),
        case([["Bob", 3], ["alice", 3]], expected=["alice", "Bob"]),
        case([["  zed", 1], ["amy ", 1], ["Mia", 1]], expected=["amy ", "Mia", "  zed"]),
        case([["Ann", 2], ["ann", 2], [" ANN", 2]], expected=["Ann", "ann", " ANN"]),
        case([["x", -1], ["y", 0], ["z", -1]], expected=["y", "x", "z"]),
        case([], expected=[]),
        case([["solo", 4]], expected=["solo"]),
    ],
    reference=_LB_REF,
    bugs={"tie_ordering": _LB_TIE, "whitespace_case": _LB_WS},
    tags=("tie_ordering", "whitespace_case"),
)

# ---------------------------------------------------------------------------
# 3. nearest_rank_percentile
# ---------------------------------------------------------------------------

_PCT_REF = '''
def nearest_rank_percentile(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    rank = max(1, (p * n + 99) // 100)
    return s[rank - 1]
'''

_PCT_OBO = '''
def nearest_rank_percentile(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    rank = min(n, max(1, p * n // 100 + 1))
    return s[rank - 1]
'''

_PCT_MUT = '''
def nearest_rank_percentile(vals, p):
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    rank = max(1, (p * n + 99) // 100)
    return vals[rank - 1]
'''

_PCT_EMPTY = '''
def nearest_rank_percentile(vals, p):
    s = sorted(vals)
    n = len(s)
    rank = max(1, (p * n + 99) // 100)
    return s[rank - 1]
'''

NEAREST_RANK_PERCENTILE = Task(
    id="nearest_rank_percentile",
    split="eval",
    entry_point="nearest_rank_percentile",
    signature="def nearest_rank_percentile(vals: list[int], p: int) -> int | None:",
    spec=(
        "Return the p-th percentile (0 <= p <= 100) of vals by the nearest-rank method: sort a copy of vals, let "
        "n = len(vals) and rank = max(1, ceil(p * n / 100)) computed with exact integer arithmetic, and return the "
        "rank-th smallest value (1-based). If vals is empty return None. Do not modify vals."
    ),
    tests=[
        case([15, 20, 35, 40, 50], 30, expected=20),
        case([40, 10, 30, 20], 50, expected=20),
        case([40, 10, 30, 20], 75, expected=30),
        case([3, 1, 2], 100, expected=3),
        case([3, 1, 2], 0, expected=1),
        case([7, 7, -2, 5, 9, 1, 4, 8, 6, 3], 90, expected=8),
        case([5], 50, expected=5),
        case([], 50, expected=None),
    ],
    reference=_PCT_REF,
    bugs={"off_by_one": _PCT_OBO, "mutates_input": _PCT_MUT, "empty_input": _PCT_EMPTY},
    tags=("off_by_one", "mutates_input", "empty_input", "float_rounding"),
)

# ---------------------------------------------------------------------------
# 4. parse_duration
# ---------------------------------------------------------------------------

_DUR_REF = '''
import re

def parse_duration(s):
    if not re.fullmatch(r"\\s*(?:[0-9]+[hms]\\s*)+", s, re.IGNORECASE):
        raise ValueError(f"malformed duration: {s!r}")
    mult = {"h": 3600, "m": 60, "s": 1}
    seen = set()
    total = 0
    for num, unit in re.findall(r"([0-9]+)([hms])", s, re.IGNORECASE):
        unit = unit.lower()
        if unit in seen:
            raise ValueError(f"repeated unit {unit!r}")
        seen.add(unit)
        total += int(num) * mult[unit]
    return total
'''

_DUR_WS = '''
import re

def parse_duration(s):
    if not re.fullmatch(r"\\s*(?:[0-9]+[hms]\\s*)+", s):
        raise ValueError(f"malformed duration: {s!r}")
    mult = {"h": 3600, "m": 60, "s": 1}
    seen = set()
    total = 0
    for num, unit in re.findall(r"([0-9]+)([hms])", s):
        if unit in seen:
            raise ValueError(f"repeated unit {unit!r}")
        seen.add(unit)
        total += int(num) * mult[unit]
    return total
'''

_DUR_ERR = '''
import re

def parse_duration(s):
    if not re.fullmatch(r"\\s*(?:[0-9]+[hms]\\s*)+", s, re.IGNORECASE):
        raise ValueError(f"malformed duration: {s!r}")
    mult = {"h": 3600, "m": 60, "s": 1}
    total = 0
    for num, unit in re.findall(r"([0-9]+)([hms])", s, re.IGNORECASE):
        total += int(num) * mult[unit.lower()]
    return total
'''

PARSE_DURATION = Task(
    id="parse_duration",
    split="eval",
    entry_point="parse_duration",
    signature="def parse_duration(s: str) -> int:",
    spec=(
        "Parse a duration such as \"1h 30m 5s\" into a total number of seconds. The string consists of one or more "
        "tokens, each an ASCII decimal integer immediately followed by a unit letter h (3600 s), m (60 s) or s (1 s); "
        "unit letters are case-insensitive, tokens may appear in any order, and whitespace (spaces, tabs) is optional "
        "before, between and after tokens but not inside a token. Each unit may appear at most once. Raise ValueError "
        "for an empty or whitespace-only string, any malformed text, a number without a unit, or a repeated unit."
    ),
    tests=[
        case("1h 30m 5s", expected=5405),
        case("45s", expected=45),
        case("5s 2m", expected=125),
        case("2H\t5M", expected=7500),
        case("  1h30m  ", expected=5400),
        case("0m", expected=0),
        case("", expected=raises("ValueError")),
        case("90", expected=raises("ValueError")),
        case("1h 2h", expected=raises("ValueError")),
        case("1x", expected=raises("ValueError")),
    ],
    reference=_DUR_REF,
    bugs={"whitespace_case": _DUR_WS, "error_contract": _DUR_ERR},
    tags=("whitespace_case", "error_contract", "empty_input"),
)

# ---------------------------------------------------------------------------
# 5. shortest_cover_span
# ---------------------------------------------------------------------------

_SPAN_REF = '''
def shortest_cover_span(entries, required):
    need = set(required)
    if not need:
        return 0
    counts = {}
    have = 0
    best = 0
    left = 0
    for t, user in entries:
        if user in need:
            counts[user] = counts.get(user, 0) + 1
            if counts[user] == 1:
                have += 1
        while have == len(need):
            span = t - entries[left][0] + 1
            if best == 0 or span < best:
                best = span
            lu = entries[left][1]
            if lu in need:
                counts[lu] -= 1
                if counts[lu] == 0:
                    have -= 1
            left += 1
    return best
'''

_SPAN_SLOW = '''
def shortest_cover_span(entries, required):
    need = set(required)
    if not need:
        return 0
    best = 0
    for i in range(len(entries)):
        seen = set()
        for j in range(i, len(entries)):
            if entries[j][1] in need:
                seen.add(entries[j][1])
            if len(seen) == len(need):
                span = entries[j][0] - entries[i][0] + 1
                if best == 0 or span < best:
                    best = span
                break
    return best
'''

_SPAN_OBO = '''
def shortest_cover_span(entries, required):
    need = set(required)
    if not need:
        return 0
    counts = {}
    have = 0
    best = 0
    left = 0
    for t, user in entries:
        if user in need:
            counts[user] = counts.get(user, 0) + 1
            if counts[user] == 1:
                have += 1
        while have == len(need):
            span = t - entries[left][0]
            if best == 0 or span < best:
                best = span
            lu = entries[left][1]
            if lu in need:
                counts[lu] -= 1
                if counts[lu] == 0:
                    have -= 1
            left += 1
    return best
'''

SHORTEST_COVER_SPAN = Task(
    id="shortest_cover_span",
    split="eval",
    entry_point="shortest_cover_span",
    signature="def shortest_cover_span(entries: list[list], required: list[str]) -> int:",
    spec=(
        "entries is a list of [timestamp, user] pairs (integer timestamp, string user) sorted by timestamp in "
        "non-decreasing order. Find a contiguous run of entries that contains every user in required (duplicates in "
        "required count once) and minimises its inclusive time span, last timestamp - first timestamp + 1; return "
        "that span. Return 0 if required is empty or no run contains all required users. Must handle 200_000 "
        "entries well within a second (aim for O(n))."
    ),
    tests=[
        case([[1, "a"], [2, "b"], [5, "a"], [6, "c"], [7, "b"]], ["a", "b", "c"], expected=3),
        case([[1, "a"], [3, "b"]], ["a", "b"], expected=3),
        case([[4, "a"], [4, "b"]], ["b", "a", "b"], expected=1),
        case([[1, "a"], [2, "x"], [10, "b"], [11, "a"]], ["a", "b"], expected=2),
        case([[1, "a"], [2, "a"]], ["a", "b"], expected=0),
        case([[1, "a"]], [], expected=0),
        case([], ["a"], expected=0),
        case([[3, "z"]], ["z"], expected=1),
        case(
            expr("[[0, 'a']] + [[i, 'x'] for i in range(1, 199999)] + [[199999, 'b']]"),
            ["a", "b"],
            expected=200000,
        ),
    ],
    reference=_SPAN_REF,
    bugs={"inefficient": _SPAN_SLOW, "off_by_one": _SPAN_OBO},
    tags=("inefficient", "off_by_one", "empty_input"),
)

# ---------------------------------------------------------------------------
# 6. pack_batches
# ---------------------------------------------------------------------------

_PACK_REF = '''
def pack_batches(sizes, cap):
    out = []
    cur = []
    total = 0
    for x in sizes:
        if cur and total + x > cap:
            out.append(cur)
            cur = []
            total = 0
        cur.append(x)
        total += x
    if cur:
        out.append(cur)
    return out
'''

_PACK_OBO = '''
def pack_batches(sizes, cap):
    out = []
    cur = []
    total = 0
    for x in sizes:
        if cur and total + x >= cap:
            out.append(cur)
            cur = []
            total = 0
        cur.append(x)
        total += x
    if cur:
        out.append(cur)
    return out
'''

_PACK_EMPTY = '''
def pack_batches(sizes, cap):
    out = []
    cur = []
    total = 0
    for x in sizes:
        if cur and total + x > cap:
            out.append(cur)
            cur = []
            total = 0
        cur.append(x)
        total += x
    out.append(cur)
    return out
'''

PACK_BATCHES = Task(
    id="pack_batches",
    split="eval",
    entry_point="pack_batches",
    signature="def pack_batches(sizes: list[int], cap: int) -> list[list[int]]:",
    spec=(
        "Greedily pack the positive integer sizes, in order, into consecutive batches: keep adding the next item to "
        "the current batch while the batch sum stays <= cap (a sum exactly equal to cap is allowed), otherwise start "
        "a new batch. An item larger than cap goes into a batch of its own. Return the list of batches; an empty "
        "sizes list returns []."
    ),
    tests=[
        case([2, 3, 4, 1], 5, expected=[[2, 3], [4, 1]]),
        case([5, 5, 5], 5, expected=[[5], [5], [5]]),
        case([1, 2, 3, 4], 10, expected=[[1, 2, 3, 4]]),
        case([3, 9, 2], 5, expected=[[3], [9], [2]]),
        case([4, 4, 1, 1], 6, expected=[[4], [4, 1, 1]]),
        case([7], 3, expected=[[7]]),
        case([], 5, expected=[]),
    ],
    reference=_PACK_REF,
    bugs={"off_by_one": _PACK_OBO, "empty_input": _PACK_EMPTY},
    tags=("off_by_one", "empty_input"),
)

# ---------------------------------------------------------------------------
# 7. sanitize_username
# ---------------------------------------------------------------------------

_USER_REF = '''
import re

def sanitize_username(s):
    s = s.strip().casefold()
    s = re.sub(r"\\s+", "_", s)
    s = "".join(c for c in s if c.isalnum() or c == "_")
    return s or "user"
'''

_USER_WS = '''
def sanitize_username(s):
    s = s.strip().casefold()
    s = s.replace(" ", "_")
    s = "".join(c for c in s if c.isalnum() or c == "_")
    return s or "user"
'''

_USER_EMPTY = '''
import re

def sanitize_username(s):
    s = s.strip().casefold()
    s = re.sub(r"\\s+", "_", s)
    return "".join(c for c in s if c.isalnum() or c == "_")
'''

SANITIZE_USERNAME = Task(
    id="sanitize_username",
    split="eval",
    entry_point="sanitize_username",
    signature="def sanitize_username(s: str) -> str:",
    spec=(
        "Normalise a username in this order: strip() surrounding whitespace; casefold(); replace each maximal run of "
        "whitespace characters (spaces, tabs, newlines, any str.isspace() character) with a single \"_\"; then remove "
        "every character that is neither str.isalnum() nor \"_\". If the result is empty, return \"user\"."
    ),
    tests=[
        case("John Smith", expected="john_smith"),
        case("  Ada   Lovelace ", expected="ada_lovelace"),
        case("big\tbird\nfan", expected="big_bird_fan"),
        case("Ünïcode Straße", expected="ünïcode_strasse"),
        case("o'neil-99!", expected="oneil99"),
        case("already_ok", expected="already_ok"),
        case("!!!", expected="user"),
        case("   ", expected="user"),
        case("", expected="user"),
    ],
    reference=_USER_REF,
    bugs={"whitespace_case": _USER_WS, "empty_input": _USER_EMPTY},
    tags=("whitespace_case", "empty_input"),
)

# ---------------------------------------------------------------------------
# 8. apportion
# ---------------------------------------------------------------------------

_APP_REF = '''
def apportion(total, weights):
    if not weights:
        return []
    w_sum = sum(weights)
    base = [total * w // w_sum for w in weights]
    rems = [total * w % w_sum for w in weights]
    left = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (-rems[i], i))
    for i in order[:left]:
        base[i] += 1
    return base
'''

_APP_FLOAT = '''
def apportion(total, weights):
    if not weights:
        return []
    w_sum = sum(weights)
    exact = [total * w / w_sum for w in weights]
    base = [int(x) for x in exact]
    rems = [x - b for x, b in zip(exact, base)]
    left = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (-rems[i], i))
    for i in order[:left]:
        base[i] += 1
    return base
'''

_APP_TIE = '''
def apportion(total, weights):
    if not weights:
        return []
    w_sum = sum(weights)
    base = [total * w // w_sum for w in weights]
    rems = [total * w % w_sum for w in weights]
    left = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (rems[i], i), reverse=True)
    for i in order[:left]:
        base[i] += 1
    return base
'''

APPORTION = Task(
    id="apportion",
    split="eval",
    entry_point="apportion",
    signature="def apportion(total: int, weights: list[int]) -> list[int]:",
    spec=(
        "Split the non-negative integer total into integer shares proportional to weights (non-negative integers "
        "with a positive sum) using the largest remainder method with exact integer arithmetic (totals may be as "
        "large as 10**18): each share starts as floor(total * w / W) where W = sum(weights), then the leftover units "
        "go one each to the indices with the largest fractional remainders, ties going to the lower index. The "
        "shares sum to total. An empty weights list returns []."
    ),
    tests=[
        case(10, [1, 1, 1], expected=[4, 3, 3]),
        case(5, [1, 1, 1, 1], expected=[2, 1, 1, 1]),
        case(7, [2, 3, 5], expected=[1, 2, 4]),
        case(100, [0, 1, 3], expected=[0, 25, 75]),
        case(0, [4, 5], expected=[0, 0]),
        case(10**18, [1, 1, 1], expected=[333333333333333334, 333333333333333333, 333333333333333333]),
        case(10**18 + 1, [2, 7], expected=[222222222222222222, 777777777777777779]),
        case(3, [5], expected=[3]),
        case(9, [], expected=[]),
    ],
    reference=_APP_REF,
    bugs={"float_rounding": _APP_FLOAT, "tie_ordering": _APP_TIE},
    tags=("float_rounding", "tie_ordering", "empty_input"),
)

# ---------------------------------------------------------------------------
# 9. snapshot_diff
# ---------------------------------------------------------------------------

_DIFF_REF = '''
def snapshot_diff(old, new):
    return {
        "added": sorted(k for k in new if k not in old),
        "removed": sorted(k for k in old if k not in new),
        "changed": sorted(k for k in old if k in new and old[k] != new[k]),
    }
'''

_DIFF_MUT = '''
def snapshot_diff(old, new):
    removed = []
    changed = []
    for k, v in old.items():
        if k not in new:
            removed.append(k)
        elif new.pop(k) != v:
            changed.append(k)
    return {"added": sorted(new), "removed": sorted(removed), "changed": sorted(changed)}
'''

_DIFF_TIE = '''
def snapshot_diff(old, new):
    return {
        "added": [k for k in new if k not in old],
        "removed": [k for k in old if k not in new],
        "changed": [k for k in old if k in new and old[k] != new[k]],
    }
'''

SNAPSHOT_DIFF = Task(
    id="snapshot_diff",
    split="eval",
    entry_point="snapshot_diff",
    signature="def snapshot_diff(old: dict, new: dict) -> dict:",
    spec=(
        "Compare two snapshots (dicts with string keys and JSON-like values). Return a dict with exactly the keys "
        "\"added\" (keys only in new), \"removed\" (keys only in old) and \"changed\" (keys in both whose values are "
        "unequal under ==), each an ascending sorted list of keys (empty list when none). Do not modify either input."
    ),
    tests=[
        case({"a": 1, "b": 2}, {"b": 3, "c": 4}, expected={"added": ["c"], "removed": ["a"], "changed": ["b"]}),
        case({"z": 1, "m": 2, "a": 3}, {"y": 0, "b": 0, "m": 2},
             expected={"added": ["b", "y"], "removed": ["a", "z"], "changed": []}),
        case({"k2": [1], "k1": {"x": 1}}, {"k2": [1, 2], "k1": {"x": 1}},
             expected={"added": [], "removed": [], "changed": ["k2"]}),
        case({}, {"q": 1, "p": 2}, expected={"added": ["p", "q"], "removed": [], "changed": []}),
        case({"q": None, "p": 0}, {}, expected={"added": [], "removed": ["p", "q"], "changed": []}),
        case({}, {}, expected={"added": [], "removed": [], "changed": []}),
        case({"x": 1, "y": 2}, {"x": 1, "y": 2}, expected={"added": [], "removed": [], "changed": []}),
        case({"d": 1, "c": 1, "b": 1}, {"d": 2, "c": 2, "b": 1, "e": 0},
             expected={"added": ["e"], "removed": [], "changed": ["c", "d"]}),
    ],
    reference=_DIFF_REF,
    bugs={"mutates_input": _DIFF_MUT, "tie_ordering": _DIFF_TIE},
    tags=("mutates_input", "tie_ordering", "empty_input"),
)

# ---------------------------------------------------------------------------
# 10. max_drawdown
# ---------------------------------------------------------------------------

_DD_REF = '''
def max_drawdown(prices):
    if len(prices) < 2:
        return 0.0
    peak = prices[0]
    best = 0.0
    for p in prices:
        if p > peak:
            peak = p
        else:
            best = max(best, (peak - p) / peak)
    return round(best, 4)
'''

_DD_SLOW = '''
def max_drawdown(prices):
    best = 0.0
    for i in range(len(prices)):
        for j in range(i + 1, len(prices)):
            best = max(best, (prices[i] - prices[j]) / prices[i])
    return round(best, 4)
'''

_DD_EMPTY = '''
def max_drawdown(prices):
    peak = prices[0]
    best = 0.0
    for p in prices:
        if p > peak:
            peak = p
        else:
            best = max(best, (peak - p) / peak)
    return round(best, 4)
'''

MAX_DRAWDOWN = Task(
    id="max_drawdown",
    split="eval",
    entry_point="max_drawdown",
    signature="def max_drawdown(prices: list[float]) -> float:",
    spec=(
        "prices is a list of positive numbers in time order. Return the maximum drawdown: the largest value of "
        "(peak - trough) / peak over all pairs where the peak occurs strictly before the trough, rounded with "
        "round(x, 4). Return 0.0 if prices never decline or has fewer than 2 elements (including empty). Must run in "
        "O(n): inputs may have 200_000 prices."
    ),
    tests=[
        case([100.0, 80.0, 120.0, 60.0, 90.0], expected=0.5),
        case([3.0, 2.0, 1.0], expected=0.6667),
        case([1.0, 2.0, 3.0], expected=0.0),
        case([10.0, 9.0, 12.0, 11.0], expected=0.1),
        case([5.0, 5.0, 5.0], expected=0.0),
        case([7.5], expected=0.0),
        case([], expected=0.0),
        case(expr("[float(i + 1) for i in range(200000)] + [100000.0]"), expected=0.5),
    ],
    reference=_DD_REF,
    bugs={"inefficient": _DD_SLOW, "empty_input": _DD_EMPTY},
    tags=("inefficient", "empty_input", "float_rounding"),
)

# ---------------------------------------------------------------------------
# 11. slice_inclusive
# ---------------------------------------------------------------------------

_SL_REF = '''
def slice_inclusive(items, i, j):
    n = len(items)
    if i < 0:
        i += n
    if j < 0:
        j += n
    i = max(i, 0)
    j = min(j, n - 1)
    if i > j:
        return []
    return items[i:j + 1]
'''

_SL_OBO = '''
def slice_inclusive(items, i, j):
    n = len(items)
    if i < 0:
        i += n
    if j < 0:
        j += n
    i = max(i, 0)
    j = min(j, n - 1)
    if i > j:
        return []
    return items[i:j]
'''

_SL_NEG = '''
def slice_inclusive(items, i, j):
    n = len(items)
    i = max(i, 0)
    j = min(j, n - 1)
    if i > j:
        return []
    return items[i:j + 1]
'''

SLICE_INCLUSIVE = Task(
    id="slice_inclusive",
    split="eval",
    entry_point="slice_inclusive",
    signature="def slice_inclusive(items: list, i: int, j: int) -> list:",
    spec=(
        "Return a new list with the elements of items from index i through index j, both inclusive. A negative index "
        "counts from the end (add len(items) to it, once). After resolving negatives, clamp i to at least 0 and j to "
        "at most len(items) - 1; if then i > j (including when items is empty), return []."
    ),
    tests=[
        case([10, 20, 30, 40, 50], 1, 3, expected=[20, 30, 40]),
        case([10, 20, 30, 40, 50], 2, 2, expected=[30]),
        case([10, 20, 30, 40, 50], -3, -1, expected=[30, 40, 50]),
        case([10, 20, 30, 40, 50], 0, -2, expected=[10, 20, 30, 40]),
        case([10, 20, 30, 40, 50], -10, 1, expected=[10, 20]),
        case([10, 20, 30, 40, 50], 3, 99, expected=[40, 50]),
        case([10, 20, 30, 40, 50], 3, 1, expected=[]),
        case([], 0, 0, expected=[]),
    ],
    reference=_SL_REF,
    bugs={"off_by_one": _SL_OBO, "negative_numbers": _SL_NEG},
    tags=("off_by_one", "negative_numbers", "empty_input"),
)

# ---------------------------------------------------------------------------
# 12. word_freq_ci
# ---------------------------------------------------------------------------

_WF_REF = '''
def word_freq_ci(text):
    counts = {}
    word = []
    for ch in text + " ":
        if ch.isalnum():
            word.append(ch)
        elif word:
            w = "".join(word).casefold()
            counts[w] = counts.get(w, 0) + 1
            word = []
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [[w, c] for w, c in ranked]
'''

_WF_WS = '''
def word_freq_ci(text):
    counts = {}
    for w in text.split():
        w = w.casefold()
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [[w, c] for w, c in ranked]
'''

_WF_TIE = '''
def word_freq_ci(text):
    counts = {}
    word = []
    for ch in text + " ":
        if ch.isalnum():
            word.append(ch)
        elif word:
            w = "".join(word).casefold()
            counts[w] = counts.get(w, 0) + 1
            word = []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [[w, c] for w, c in ranked]
'''

WORD_FREQ_CI = Task(
    id="word_freq_ci",
    split="eval",
    entry_point="word_freq_ci",
    signature="def word_freq_ci(text: str) -> list[list]:",
    spec=(
        "Words are maximal runs of characters for which str.isalnum() is true; every other character separates "
        "words. Count words case-insensitively by their casefold() form. Return a list of [word, count] pairs, where "
        "word is the casefolded form, sorted by count descending, with equal counts ordered by the position of the "
        "word's first appearance in text. Text with no words returns []."
    ),
    tests=[
        case("the cat and the hat", expected=[["the", 2], ["cat", 1], ["and", 1], ["hat", 1]]),
        case("Hello, hello! HELLO?", expected=[["hello", 3]]),
        case("b a b a c", expected=[["b", 2], ["a", 2], ["c", 1]]),
        case("zeta--alpha;beta", expected=[["zeta", 1], ["alpha", 1], ["beta", 1]]),
        case("Run\trun\nRUN fast", expected=[["run", 3], ["fast", 1]]),
        case("x1 X1 y2", expected=[["x1", 2], ["y2", 1]]),
        case("   ...  ", expected=[]),
        case("", expected=[]),
    ],
    reference=_WF_REF,
    bugs={"whitespace_case": _WF_WS, "tie_ordering": _WF_TIE},
    tags=("whitespace_case", "tie_ordering", "empty_input"),
)


TASKS: list[Task] = [
    FREE_RANGES,
    LEADERBOARD,
    NEAREST_RANK_PERCENTILE,
    PARSE_DURATION,
    SHORTEST_COVER_SPAN,
    PACK_BATCHES,
    SANITIZE_USERNAME,
    APPORTION,
    SNAPSHOT_DIFF,
    MAX_DRAWDOWN,
    SLICE_INCLUSIVE,
    WORD_FREQ_CI,
]
