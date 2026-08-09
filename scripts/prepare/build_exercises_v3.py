"""Generate the verified novice-code exercise corpus for the Brittain3 pilot.

This is the 10% "verified novice coding exercises" slice of
`configs/data/brittain3_pilot_corpus.json`. Every emitted exercise is EXECUTED
against its own assertions before it is kept. Nothing that fails is written.

    python3 scripts/prepare/build_exercises_v3.py --target-bytes 234000000

Why generate rather than scrape. Brittain2 saw 14.7B scraped tokens and still
writes code that is shaped right and wrong underneath. Scraped code carries no
guarantee that it does what its name says. A generated exercise whose assertions
have actually run is the one kind of training text where the mapping from
description to behaviour is known to be correct.

Design constraints that come straight from the evaluation work:

- **Exercises are written in the shape the model must learn to continue**: a
  short `#` comment description, a worked example, then the `def`/`class`
  header, then the body. Measured on brittain2_50m_bs, a prompt ending after a
  closing docstring makes this model class emit EOT immediately, so training on
  docstring-terminated functions teaches exactly the wrong stopping behaviour.
- **Nothing here may overlap the evaluation suite.** Templates are checked
  against `benchmarks/novice/tasks.jsonl` entry points and prompts at startup,
  and the check is fatal. If the pilot trains on its own gate, the gate measures
  memorisation and the whole experiment is void.

Verification executes generated code. It is OUR code from OUR templates, not
third-party text, and it runs under `-I` in an empty temporary directory with a
timeout. Values are drawn from a seeded RNG so a run is reproducible.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "generated" / "brittain3-pilot" / "exercises.jsonl"
EVAL_TASKS = PROJECT_ROOT / "benchmarks" / "novice" / "tasks.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-bytes", type=int, default=234_000_000,
                        help="UTF-8 bytes to emit; the pilot config asks for 234MB")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="exercises verified per subprocess. Every exercise "
                             "still runs every assertion; this only amortises "
                             "interpreter startup. A failing batch is retried "
                             "one at a time.")
    parser.add_argument("--max-per-template", type=int, default=None,
                        help="cap documents from any one template. Without it the "
                             "corpus is dominated by whichever templates happen to "
                             "have the largest literal spaces: measured at 234MB, "
                             "the top 10 of 42 templates produced 75%% of documents "
                             "while eight produced under 15 each. Set this to buy "
                             "curriculum balance at the cost of total volume.")
    parser.add_argument("--stall-attempts", type=int, default=200_000,
                        help="give up when this many consecutive candidates are all "
                             "duplicates or capped; prevents an unreachable target "
                             "from spinning forever")
    parser.add_argument("--report", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Templates.
#
# Each template is a callable taking a seeded Random and returning
# (name, description_lines, example_line, header, body, tests).
# Names are parameterised so the corpus does not repeat one identifier 40,000
# times — a model trained on that learns the identifier, not the operation.
# ---------------------------------------------------------------------------

NOUNS = ["values", "items", "numbers", "entries", "records", "readings", "scores",
         "amounts", "totals", "samples", "rows", "points", "counts", "weights"]
THINGS = ["item", "entry", "record", "value", "element", "sample", "row", "point"]
FIELDS = ["name", "label", "key", "title", "kind", "code", "tag", "slug"]
# Pools for templates that would otherwise be constants. A template with no
# randomisation can only ever emit ONE unique document, so it contributes
# nothing to a large corpus while still consuming a share of the random picks.
SWITCH_NAMES = ["Switch", "Toggle", "Flag", "Lamp", "Relay", "Latch"]
SWITCH_ATTRS = ["on", "active", "enabled", "lit", "closed"]
QUEUE_NAMES = ["Queue", "Line", "Pipeline", "Buffer", "Backlog", "Inbox"]
TRACKER_NAMES = ["Tracker", "PeakWatcher", "HighScore", "Maximum", "Recorder"]
TRACKER_ATTRS = ["best", "peak", "highest", "top", "record"]
STORE_NAMES = ["Registry", "Store", "Catalog", "Lookup", "Directory", "Book"]
BLANK_NAMES = ["is_blank", "is_empty_text", "has_no_content", "is_whitespace_only"]
COLLECT_NAMES = ["collect", "gather", "accumulate", "append_to"]


def _int_list(rng, n=None, low=-20, high=40):
    n = n if n is not None else rng.randint(2, 7)
    return [rng.randint(low, high) for _ in range(n)]


def t_sum_of(rng):
    noun = rng.choice(NOUNS)
    fn = f"total_{noun}"
    sample = _int_list(rng)
    return (fn,
            [f"Return the sum of {noun}. An empty list sums to 0."],
            f"{fn}({sample}) -> {sum(sample)}",
            f"def {fn}({noun}):",
            f"    total = 0\n    for value in {noun}:\n        total += value\n    return total",
            [f"assert {fn}({sample}) == {sum(sample)}", f"assert {fn}([]) == 0"])


def t_filter_above(rng):
    noun = rng.choice(NOUNS)
    fn = f"above_threshold"
    limit = rng.randint(0, 20)
    sample = _int_list(rng)
    expected = [v for v in sample if v > limit]
    return (fn,
            [f"Return the {noun} greater than limit, keeping their order."],
            f"{fn}({sample}, {limit}) -> {expected}",
            f"def {fn}({noun}, limit):",
            f"    kept = []\n    for value in {noun}:\n        if value > limit:\n            kept.append(value)\n    return kept",
            [f"assert {fn}({sample}, {limit}) == {expected}", f"assert {fn}([], 0) == []"])


def t_count_matching(rng):
    noun = rng.choice(NOUNS)
    fn = "count_matching"
    sample = _int_list(rng, low=0, high=5)
    target = rng.randint(0, 5)
    return (fn,
            [f"Return how many times target appears in {noun}."],
            f"{fn}({sample}, {target}) -> {sample.count(target)}",
            f"def {fn}({noun}, target):",
            f"    count = 0\n    for value in {noun}:\n        if value == target:\n            count += 1\n    return count",
            [f"assert {fn}({sample}, {target}) == {sample.count(target)}",
             f"assert {fn}([], 1) == 0"])


def t_build_index(rng):
    field = rng.choice(FIELDS)
    thing = rng.choice(THINGS)
    fn = f"index_by_{field}"
    rows = [{field: f"{thing}{i}", "size": rng.randint(1, 40)} for i in range(rng.randint(2, 4))]
    expected = {row[field]: row for row in rows}
    return (fn,
            [f"Return a dict mapping each row's {field!r} to the row itself."],
            f"{fn}(rows) -> {{row[{field!r}]: row}}",
            f"def {fn}(rows):",
            f"    index = {{}}\n    for row in rows:\n        index[row[{field!r}]] = row\n    return index",
            [f"assert {fn}({rows}) == {expected}", f"assert {fn}([]) == {{}}"])


def t_safe_get(rng):
    field = rng.choice(FIELDS)
    fn = "read_setting"
    value = rng.randint(1, 99)
    return (fn,
            ["Return config[key] when present, otherwise fallback."],
            f"{fn}({{{field!r}: {value}}}, {field!r}, 0) -> {value}",
            "def read_setting(config, key, fallback=None):",
            "    if key in config:\n        return config[key]\n    return fallback",
            [f"assert {fn}({{{field!r}: {value}}}, {field!r}, 0) == {value}",
             f"assert {fn}({{}}, {field!r}, 7) == 7"])


def t_parse_numbers(rng):
    fn = "parse_numbers"
    values = _int_list(rng, low=0, high=99)
    text = ",".join(str(v) for v in values)
    return (fn,
            ["Parse a comma-separated string of integers into a list of ints.",
             "An empty string returns an empty list."],
            f"{fn}({text!r}) -> {values}",
            "def parse_numbers(text):",
            "    if not text.strip():\n        return []\n    return [int(part) for part in text.split(',')]",
            [f"assert {fn}({text!r}) == {values}", f"assert {fn}('') == []"])


def t_running_max(rng):
    noun = rng.choice(NOUNS)
    fn = "running_max"
    sample = _int_list(rng)
    expected, best = [], None
    for v in sample:
        best = v if best is None or v > best else best
        expected.append(best)
    return (fn,
            [f"Return a list of the largest value seen so far at each position."],
            f"{fn}({sample}) -> {expected}",
            f"def {fn}({noun}):",
            f"    result = []\n    best = None\n    for value in {noun}:\n"
            f"        if best is None or value > best:\n            best = value\n"
            f"        result.append(best)\n    return result",
            [f"assert {fn}({sample}) == {expected}", f"assert {fn}([]) == []"])


def t_tally(rng):
    fn = "tally"
    words = [rng.choice(["red", "blue", "green", "grey", "amber"]) for _ in range(rng.randint(2, 6))]
    text = " ".join(words)
    expected = {}
    for w in words:
        expected[w] = expected.get(w, 0) + 1
    return (fn,
            ["Return a dict mapping each whitespace-separated word to its count."],
            f"{fn}({text!r}) -> {expected}",
            "def tally(text):",
            "    counts = {}\n    for word in text.split():\n"
            "        counts[word] = counts.get(word, 0) + 1\n    return counts",
            [f"assert {fn}({text!r}) == {expected}", f"assert {fn}('') == {{}}"])


def t_accumulator_class(rng):
    fn = "Accumulator"
    start = rng.randint(0, 10)
    add = rng.randint(1, 9)
    return (fn,
            ["A running total that starts at a given value.",
             "add(amount) increases it. The total attribute holds the current value."],
            f"a = {fn}({start}); a.add({add}); a.total -> {start + add}",
            f"class {fn}:",
            f"    def __init__(self, total=0):\n        self.total = total\n\n"
            f"    def add(self, amount):\n        self.total += amount\n        return self.total",
            [f"_a = {fn}({start}); assert _a.total == {start}",
             f"_a = {fn}({start}); _a.add({add}); assert _a.total == {start + add}"])


def t_bounded_list(rng):
    fn = "add_recent"
    limit = rng.randint(2, 4)
    history = _int_list(rng, n=limit, low=0, high=9)
    entry = rng.randint(10, 19)
    expected = (history + [entry])[-limit:]
    return (fn,
            ["Append entry to history, keeping at most limit newest items.",
             "Modify history in place and return it."],
            f"{fn}({history}, {entry}, {limit}) -> {expected}",
            "def add_recent(history, entry, limit=3):",
            "    history.append(entry)\n    while len(history) > limit:\n"
            "        history.pop(0)\n    return history",
            [f"assert {fn}({history}, {entry}, {limit}) == {expected}",
             f"assert {fn}([], 1, 3) == [1]"])


# --- validation -------------------------------------------------------------

def t_validate_range(rng):
    fn = "check_in_range"
    low, high = rng.randint(0, 5), rng.randint(10, 40)
    good, bad = rng.randint(low, high), high + rng.randint(1, 9)
    return (fn,
            ["Return True when value lies between low and high inclusive."],
            f"{fn}({good}, {low}, {high}) -> True",
            "def check_in_range(value, low, high):",
            "    return low <= value <= high",
            [f"assert {fn}({good}, {low}, {high}) is True",
             f"assert {fn}({bad}, {low}, {high}) is False"])


def t_require_fields(rng):
    fn = "missing_fields"
    fields = rng.sample(FIELDS, 3)
    present = {fields[0]: 1, fields[1]: 2}
    return (fn,
            ["Return the sorted names of required fields absent from record."],
            f"{fn}({present}, {fields}) -> [{fields[2]!r}]",
            "def missing_fields(record, required):",
            "    return sorted(name for name in required if name not in record)",
            [f"assert {fn}({present}, {fields}) == [{fields[2]!r}]",
             f"assert {fn}({{}}, []) == []"])


def t_non_empty(rng):
    fn = rng.choice(BLANK_NAMES)
    pad = " " * rng.randint(1, 5)
    word = rng.choice(["x", "ok", "data", "value"])
    return (fn,
            ["Return True when text is empty or only whitespace."],
            f"{fn}({pad!r}) -> True",
            f"def {fn}(text):",
            "    return text.strip() == ''",
            [f"assert {fn}({pad!r}) is True", f"assert {fn}('') is True",
             f"assert {fn}({' ' + word + ' '!r}) is False"])


def t_clamp_percent(rng):
    fn = "as_percent"
    value = rng.randint(-20, 140)
    expected = max(0, min(100, value))
    return (fn,
            ["Limit value to the range 0..100."],
            f"{fn}({value}) -> {expected}",
            "def as_percent(value):",
            "    if value < 0:\n        return 0\n    if value > 100:\n        return 100\n    return value",
            [f"assert {fn}({value}) == {expected}", f"assert {fn}(50) == 50"])


# --- strings ----------------------------------------------------------------

def t_title_words(rng):
    fn = "capitalise_words"
    words = [rng.choice(["alpha", "beta", "gamma", "delta"]) for _ in range(rng.randint(2, 3))]
    text = " ".join(words)
    expected = " ".join(w.capitalize() for w in words)
    return (fn,
            ["Return text with the first letter of each word made uppercase."],
            f"{fn}({text!r}) -> {expected!r}",
            "def capitalise_words(text):",
            "    return ' '.join(word.capitalize() for word in text.split())",
            [f"assert {fn}({text!r}) == {expected!r}", f"assert {fn}('') == ''"])


def t_truncate(rng):
    fn = "shorten"
    limit = rng.randint(4, 8)
    text = "".join(rng.choice("abcdefghij") for _ in range(limit + rng.randint(2, 6)))
    expected = text[:limit] + "..."
    return (fn,
            ["Cut text to limit characters and append '...' when it was cut."],
            f"{fn}({text!r}, {limit}) -> {expected!r}",
            "def shorten(text, limit):",
            "    if len(text) <= limit:\n        return text\n    return text[:limit] + '...'",
            [f"assert {fn}({text!r}, {limit}) == {expected!r}",
             f"assert {fn}('ab', 5) == 'ab'"])


def t_starts_with_any(rng):
    fn = "has_prefix"
    prefixes = rng.sample(["get_", "set_", "is_", "do_"], 2)
    name = prefixes[0] + "value"
    return (fn,
            ["Return True when name begins with any of the given prefixes."],
            f"{fn}({name!r}, {prefixes}) -> True",
            "def has_prefix(name, prefixes):",
            "    for prefix in prefixes:\n        if name.startswith(prefix):\n            return True\n    return False",
            [f"assert {fn}({name!r}, {prefixes}) is True",
             f"assert {fn}('other', {prefixes}) is False"])


def t_join_nonempty(rng):
    fn = "join_parts"
    parts = [rng.choice(["one", "", "two", "three", ""]) for _ in range(4)]
    expected = ", ".join(p for p in parts if p)
    return (fn,
            ["Join the non-empty parts with ', '."],
            f"{fn}({parts}) -> {expected!r}",
            "def join_parts(parts):",
            "    return ', '.join(part for part in parts if part)",
            [f"assert {fn}({parts}) == {expected!r}", f"assert {fn}([]) == ''"])


def t_reverse_words(rng):
    fn = "reverse_words"
    words = [rng.choice(["red", "blue", "green", "grey"]) for _ in range(rng.randint(2, 4))]
    text = " ".join(words)
    expected = " ".join(reversed(words))
    return (fn,
            ["Return text with its whitespace-separated words in reverse order."],
            f"{fn}({text!r}) -> {expected!r}",
            "def reverse_words(text):",
            "    return ' '.join(reversed(text.split()))",
            [f"assert {fn}({text!r}) == {expected!r}", f"assert {fn}('') == ''"])


# --- collections ------------------------------------------------------------

def t_sort_by_field(rng):
    field = rng.choice(["size", "score", "rank"])
    fn = f"sort_by_{field}"
    rows = [{"name": f"r{i}", field: rng.randint(0, 50)} for i in range(rng.randint(2, 4))]
    expected = sorted(rows, key=lambda r: r[field])
    return (fn,
            [f"Return rows sorted by their {field!r} value, smallest first."],
            f"{fn}(rows) -> rows ordered by {field}",
            f"def {fn}(rows):",
            f"    return sorted(rows, key=lambda row: row[{field!r}])",
            [f"assert {fn}({rows}) == {expected}", f"assert {fn}([]) == []"])


def t_group_by_field(rng):
    field = rng.choice(["kind", "team", "colour"])
    fn = "group_rows"
    values = [rng.choice(["a", "b"]) for _ in range(rng.randint(2, 5))]
    rows = [{field: v, "n": i} for i, v in enumerate(values)]
    expected = {}
    for r in rows:
        expected.setdefault(r[field], []).append(r)
    return (fn,
            [f"Group rows into a dict keyed by their {field!r} value."],
            f"{fn}(rows, {field!r}) -> dict of lists",
            "def group_rows(rows, field):",
            "    groups = {}\n    for row in rows:\n"
            "        groups.setdefault(row[field], []).append(row)\n    return groups",
            [f"assert {fn}({rows}, {field!r}) == {expected}", f"assert {fn}([], 'x') == {{}}"])


def t_flatten(rng):
    fn = "flatten"
    nested = [[rng.randint(0, 9) for _ in range(rng.randint(1, 3))] for _ in range(rng.randint(2, 3))]
    expected = [v for group in nested for v in group]
    return (fn,
            ["Return one flat list containing every item of the nested lists."],
            f"{fn}({nested}) -> {expected}",
            "def flatten(groups):",
            "    out = []\n    for group in groups:\n        for value in group:\n"
            "            out.append(value)\n    return out",
            [f"assert {fn}({nested}) == {expected}", f"assert {fn}([]) == []"])


def t_chunk(rng):
    fn = "chunk"
    size = rng.randint(2, 3)
    values = _int_list(rng, n=rng.randint(3, 7), low=0, high=9)
    expected = [values[i:i + size] for i in range(0, len(values), size)]
    return (fn,
            ["Split values into consecutive lists of at most size items."],
            f"{fn}({values}, {size}) -> {expected}",
            "def chunk(values, size):",
            "    out = []\n    for start in range(0, len(values), size):\n"
            "        out.append(values[start:start + size])\n    return out",
            [f"assert {fn}({values}, {size}) == {expected}", f"assert {fn}([], 2) == []"])


def t_zip_to_dict(rng):
    fn = "pair_up"
    keys = rng.sample(["a", "b", "c", "d"], 3)
    vals = _int_list(rng, n=3, low=0, high=9)
    expected = dict(zip(keys, vals))
    return (fn,
            ["Build a dict pairing each key with the value at the same position."],
            f"{fn}({keys}, {vals}) -> {expected}",
            "def pair_up(keys, values):",
            "    result = {}\n    for index, key in enumerate(keys):\n"
            "        result[key] = values[index]\n    return result",
            [f"assert {fn}({keys}, {vals}) == {expected}", f"assert {fn}([], []) == {{}}"])


def t_dedupe_keep_last(rng):
    fn = "last_seen"
    values = [rng.choice(["x", "y", "z"]) for _ in range(rng.randint(3, 6))]
    seen = {}
    for i, v in enumerate(values):
        seen[v] = i
    expected = seen
    return (fn,
            ["Return a dict mapping each value to the last index it appeared at."],
            f"{fn}({values}) -> {expected}",
            "def last_seen(values):",
            "    positions = {}\n    for index, value in enumerate(values):\n"
            "        positions[value] = index\n    return positions",
            [f"assert {fn}({values}) == {expected}", f"assert {fn}([]) == {{}}"])


def t_min_max_pair(rng):
    fn = "span"
    values = _int_list(rng, n=rng.randint(2, 6))
    expected = (min(values), max(values))
    return (fn,
            ["Return a (smallest, largest) tuple, or None when values is empty."],
            f"{fn}({values}) -> {expected}",
            "def span(values):",
            "    if not values:\n        return None\n    return (min(values), max(values))",
            [f"assert {fn}({values}) == {expected}", f"assert {fn}([]) is None"])


# --- parsing and records ----------------------------------------------------

def t_parse_pairs(rng):
    fn = "parse_settings"
    keys = rng.sample(FIELDS, 2)
    vals = [str(rng.randint(1, 30)) for _ in keys]
    text = ";".join(f"{k}={v}" for k, v in zip(keys, vals))
    expected = dict(zip(keys, vals))
    return (fn,
            ["Parse a ';'-separated list of key=value pairs into a dict.",
             "An empty string returns an empty dict."],
            f"{fn}({text!r}) -> {expected}",
            "def parse_settings(text):",
            "    settings = {}\n    if not text.strip():\n        return settings\n"
            "    for part in text.split(';'):\n        key, _, value = part.partition('=')\n"
            "        settings[key.strip()] = value.strip()\n    return settings",
            [f"assert {fn}({text!r}) == {expected}", f"assert {fn}('') == {{}}"])


def t_read_lines(rng):
    fn = "clean_lines"
    lines = [rng.choice(["alpha", "", "  beta ", "gamma", "   "]) for _ in range(4)]
    text = "\n".join(lines)
    expected = [l.strip() for l in lines if l.strip()]
    return (fn,
            ["Split text into lines, strip each, and drop the blank ones."],
            f"{fn}(text) -> {expected}",
            "def clean_lines(text):",
            "    out = []\n    for line in text.split('\\n'):\n        line = line.strip()\n"
            "        if line:\n            out.append(line)\n    return out",
            [f"assert {fn}({text!r}) == {expected}", f"assert {fn}('') == []"])


def t_status_message(rng):
    fn = "describe_status"
    code = rng.choice([200, 404, 500])
    table = {200: "ok", 404: "not found", 500: "server error"}
    return (fn,
            ["Return a short message for a status code, or 'unknown'."],
            f"{fn}({code}) -> {table[code]!r}",
            "def describe_status(code):",
            "    messages = {200: 'ok', 404: 'not found', 500: 'server error'}\n"
            "    return messages.get(code, 'unknown')",
            [f"assert {fn}({code}) == {table[code]!r}", f"assert {fn}(123) == 'unknown'"])


def t_build_response(rng):
    fn = "make_response"
    code = rng.choice([200, 201, 400])
    body = rng.choice(["done", "created", "bad input"])
    return (fn,
            ["Return a response dict with 'status' and 'body' keys.",
             "'ok' is True only for codes below 400."],
            f"{fn}({code}, {body!r}) -> dict with ok={code < 400}",
            "def make_response(status, body):",
            "    return {'status': status, 'body': body, 'ok': status < 400}",
            [f"assert {fn}({code}, {body!r}) == {{'status': {code}, 'body': {body!r}, 'ok': {code < 400}}}",
             f"assert {fn}(500, '')['ok'] is False"])


def t_query_string(rng):
    fn = "build_query"
    keys = rng.sample(FIELDS, 2)
    vals = [str(rng.randint(1, 9)) for _ in keys]
    params = dict(zip(keys, vals))
    expected = "&".join(f"{k}={v}" for k, v in params.items())
    return (fn,
            ["Turn a params dict into a 'key=value&key=value' string."],
            f"{fn}({params}) -> {expected!r}",
            "def build_query(params):",
            "    return '&'.join(key + '=' + str(value) for key, value in params.items())",
            [f"assert {fn}({params}) == {expected!r}", f"assert {fn}({{}}) == ''"])


# --- numbers ----------------------------------------------------------------

def t_round_to(rng):
    fn = "round_down_to"
    step = rng.choice([5, 10, 25])
    value = rng.randint(1, 200)
    expected = (value // step) * step
    return (fn,
            ["Round value down to the nearest multiple of step."],
            f"{fn}({value}, {step}) -> {expected}",
            "def round_down_to(value, step):",
            "    return (value // step) * step",
            [f"assert {fn}({value}, {step}) == {expected}", f"assert {fn}(0, 5) == 0"])


def t_percentage(rng):
    fn = "percentage"
    part, whole = rng.randint(1, 40), rng.randint(50, 200)
    expected = part / whole * 100
    return (fn,
            ["Return part as a percentage of whole. A whole of 0 returns 0.0."],
            f"{fn}({part}, {whole}) -> {expected:.4f}",
            "def percentage(part, whole):",
            "    if whole == 0:\n        return 0.0\n    return part / whole * 100",
            [f"assert abs({fn}({part}, {whole}) - {expected!r}) < 1e-9",
             f"assert {fn}(1, 0) == 0.0"])


def t_count_digits(rng):
    fn = "digit_count"
    value = rng.randint(0, 99999)
    expected = len(str(abs(value)))
    return (fn,
            ["Return how many digits the absolute value of a number has."],
            f"{fn}({value}) -> {expected}",
            "def digit_count(value):",
            "    return len(str(abs(value)))",
            [f"assert {fn}({value}) == {expected}", f"assert {fn}(0) == 1"])


def t_is_multiple(rng):
    fn = "divides_evenly"
    factor = rng.randint(2, 9)
    value = factor * rng.randint(1, 12)
    return (fn,
            ["Return True when value divides by factor with no remainder."],
            f"{fn}({value}, {factor}) -> True",
            "def divides_evenly(value, factor):",
            "    if factor == 0:\n        return False\n    return value % factor == 0",
            [f"assert {fn}({value}, {factor}) is True",
             f"assert {fn}({value + 1}, {factor}) is False",
             f"assert {fn}(7, 0) is False"])


# --- classes and state ------------------------------------------------------

def t_toggle_class(rng):
    fn = rng.choice(SWITCH_NAMES)
    attr = rng.choice(SWITCH_ATTRS)
    return (fn,
            [f"A {fn.lower()} that starts off.",
             f"toggle() flips it and returns the new state. The {attr} attribute holds it."],
            f"s = {fn}(); s.toggle() -> True",
            f"class {fn}:",
            f"    def __init__(self):\n        self.{attr} = False\n\n"
            f"    def toggle(self):\n        self.{attr} = not self.{attr}\n        return self.{attr}",
            [f"_s = {fn}(); assert _s.{attr} is False",
             f"_s = {fn}(); assert _s.toggle() is True and _s.{attr} is True",
             f"_s = {fn}(); _s.toggle(); assert _s.toggle() is False"])


def t_queue_class(rng):
    fn = rng.choice(QUEUE_NAMES)
    return (fn,
            ["A first-in first-out queue.",
             "push(item) adds. pop() removes and returns the oldest, or None when empty."],
            f"q = {fn}(); q.push(1); q.push(2); q.pop() -> 1",
            f"class {fn}:",
            "    def __init__(self):\n        self.items = []\n\n"
            "    def push(self, item):\n        self.items.append(item)\n\n"
            "    def pop(self):\n        if not self.items:\n            return None\n"
            "        return self.items.pop(0)",
            [f"_q = {fn}(); assert _q.pop() is None",
             f"_q = {fn}(); _q.push(1); _q.push(2); assert _q.pop() == 1"])


def t_registry_class(rng):
    fn = rng.choice(STORE_NAMES)
    key = rng.choice(FIELDS)
    value = rng.randint(1, 99)
    return (fn,
            ["A name to value store.",
             "set(name, value) stores. get(name) returns the value or None."],
            f"r = {fn}(); r.set({key!r}, {value}); r.get({key!r}) -> {value}",
            f"class {fn}:",
            "    def __init__(self):\n        self.items = {}\n\n"
            "    def set(self, name, value):\n        self.items[name] = value\n\n"
            "    def get(self, name):\n        return self.items.get(name)",
            [f"_r = {fn}(); _r.set({key!r}, {value}); assert _r.get({key!r}) == {value}",
             f"_r = {fn}(); assert _r.get('nope') is None"])


def t_minmax_tracker(rng):
    fn = rng.choice(TRACKER_NAMES)
    attr = rng.choice(TRACKER_ATTRS)
    values = _int_list(rng, n=rng.randint(2, 5), low=0, high=50)
    return (fn,
            ["Tracks the largest value it has been shown.",
             "record(value) stores it. The best attribute holds the largest, or None."],
            f"t = {fn}(); t.record(3); t.best -> 3",
            f"class {fn}:",
            "    def __init__(self):\n        self.best = None\n\n"
            "    def record(self, value):\n"
            "        if self.best is None or value > self.best:\n            self.best = value\n"
            "        return self.best",
            [f"_t = {fn}(); assert _t.best is None",
             f"_t = {fn}()\nfor _v in {values}:\n    _t.record(_v)\nassert _t.best == {max(values)}"])


# --- bug fixes --------------------------------------------------------------

def t_fix_off_by_one(rng):
    fn = "last_item"
    values = _int_list(rng, n=rng.randint(2, 5))
    return (fn,
            ["Return the final item of values, or None when values is empty."],
            f"{fn}({values}) -> {values[-1]}",
            "def last_item(values):",
            "    if not values:\n        return None\n    return values[len(values) - 1]",
            [f"assert {fn}({values}) == {values[-1]}", f"assert {fn}([]) is None"])


def t_fix_mutable_default(rng):
    fn = rng.choice(COLLECT_NAMES)
    first, second = rng.randint(1, 9), rng.randint(10, 99)
    return (fn,
            ["Append item to target and return it.",
             "A missing target starts as a fresh empty list each call."],
            f"{fn}({first}) -> [{first}]",
            f"def {fn}(item, target=None):",
            "    if target is None:\n        target = []\n    target.append(item)\n    return target",
            [f"assert {fn}({first}) == [{first}]", f"assert {fn}({second}) == [{second}]",
             f"_t = []; {fn}({first}, _t); assert _t == [{first}]"])


def t_fix_division_guard(rng):
    fn = "mean_or_zero"
    values = _int_list(rng, n=rng.randint(2, 5), low=0, high=20)
    expected = sum(values) / len(values)
    return (fn,
            ["Return the mean of values without failing on an empty list."],
            f"{fn}({values}) -> {expected:.4f}",
            "def mean_or_zero(values):",
            "    if not values:\n        return 0.0\n    return sum(values) / len(values)",
            [f"assert abs({fn}({values}) - {expected!r}) < 1e-9",
             f"assert {fn}([]) == 0.0"])


TEMPLATES = [
    # core
    t_sum_of, t_filter_above, t_count_matching, t_build_index, t_safe_get,
    t_parse_numbers, t_running_max, t_tally, t_accumulator_class, t_bounded_list,
    # validation
    t_validate_range, t_require_fields, t_non_empty, t_clamp_percent,
    # strings
    t_title_words, t_truncate, t_starts_with_any, t_join_nonempty, t_reverse_words,
    # collections
    t_sort_by_field, t_group_by_field, t_flatten, t_chunk, t_zip_to_dict,
    t_dedupe_keep_last, t_min_max_pair,
    # parsing and request/response shapes
    t_parse_pairs, t_read_lines, t_status_message, t_build_response, t_query_string,
    # numbers
    t_round_to, t_percentage, t_count_digits, t_is_multiple,
    # classes and state
    t_toggle_class, t_queue_class, t_registry_class, t_minmax_tracker,
    # bug-fix shapes
    t_fix_off_by_one, t_fix_mutable_default, t_fix_division_guard,
]


def render(name, description, example, header, body, tests):
    """Assemble one exercise in the prompt shape the evaluation uses."""
    comments = "\n".join(f"# {line}" for line in description)
    return f"{comments}\n# {example}\n{header}\n{body}\n"


BATCH_RUNNER = '''
import json, sys
failed = []
with open(sys.argv[1], encoding="utf-8") as handle:
    items = json.load(handle)
for index, (source, tests) in enumerate(items):
    namespace = {}
    try:
        exec(compile(source, "<exercise>", "exec"), namespace)
        exec(compile(tests, "<tests>", "exec"), namespace)
    except BaseException:
        failed.append(index)
print(json.dumps(failed))
'''


def verify_batch(items: list[tuple[str, str]], timeout: float) -> set[int]:
    """Execute a batch of exercises in one subprocess. Returns failing indices.

    Every exercise still runs every one of its own assertions — batching changes
    only how many interpreters we pay for, not what gets checked. Each exercise
    executes in its own namespace dict, so two exercises defining the same
    function name cannot shadow one another and have their tests cross over.

    A whole batch that dies (timeout, crash, segfault) is retried one at a time
    so a single bad exercise cannot discard thousands of good ones.
    """
    with tempfile.TemporaryDirectory() as workdir:
        payload = Path(workdir) / "batch.json"
        runner = Path(workdir) / "runner.py"
        payload.write_text(json.dumps(items), encoding="utf-8")
        runner.write_text(BATCH_RUNNER, encoding="utf-8")
        try:
            finished = subprocess.run(
                [sys.executable, "-I", str(runner), str(payload)],
                cwd=workdir, capture_output=True, text=True,
                timeout=timeout * max(1, len(items) / 8),
            )
        except (subprocess.TimeoutExpired, OSError):
            return _verify_individually(items, timeout)
        if finished.returncode != 0:
            return _verify_individually(items, timeout)
        try:
            return set(json.loads(finished.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            return _verify_individually(items, timeout)


def _verify_individually(items: list[tuple[str, str]], timeout: float) -> set[int]:
    return {index for index, item in enumerate(items) if _run_one(item, timeout)}


def _run_one(item: tuple[str, str], timeout: float) -> bool:
    """True when this single exercise FAILS."""
    source, tests = item
    with tempfile.TemporaryDirectory() as workdir:
        path = Path(workdir) / "exercise.py"
        path.write_text(source + "\n\n" + tests + "\n", encoding="utf-8")
        try:
            finished = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=workdir, capture_output=True, text=True, timeout=timeout,
            )
        except (subprocess.TimeoutExpired, OSError):
            return True
    return finished.returncode != 0


def evaluation_identifiers() -> tuple[set[str], set[str]]:
    """Entry points and full prompts from the held-out suite."""
    if not EVAL_TASKS.exists():
        raise SystemExit(f"cannot decontaminate: {EVAL_TASKS} is missing")
    entry_points, prompts = set(), set()
    for line in EVAL_TASKS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        entry_points.add(row["entry_point"])
        prompts.add(row["prompt"].strip())
    return entry_points, prompts


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    banned_names, banned_prompts = evaluation_identifiers()
    rng = random.Random(args.seed)

    # Fatal pre-flight: no template may share an entry point with the gate.
    probe = random.Random(args.seed)
    clashes = sorted({t(probe)[0] for t in TEMPLATES} & banned_names)
    if clashes:
        raise SystemExit(
            f"templates collide with the evaluation suite entry points: {clashes}. "
            f"Training on the gate makes the gate meaningless."
        )

    written = kept_bytes = failed = duplicate = contaminated = 0
    seen: set[str] = set()
    counts = {t.__name__: 0 for t in TEMPLATES}
    pending: list[dict] = []

    def flush(batch: list[dict]) -> None:
        """Verify a batch, then write only what passed."""
        nonlocal written, kept_bytes, failed
        if not batch:
            return
        bad = verify_batch([(item["text"], item["tests"]) for item in batch], args.timeout)
        failed += len(bad)
        for index, item in enumerate(batch):
            if index in bad:
                continue
            counts[item["template"]] += 1
            handle.write(json.dumps({
                "text": item["text"],
                "source": "novice-exercises",
                "category": "exercises",
                "language": "python",
                "repository": f"generated/{item['template']}",
                "path": f"exercises/{item['name']}.py",
                "license": "GENERATED",
            }) + "\n")
            written += 1
            kept_bytes += len(item["text"].encode("utf-8"))

    stalled = 0
    capped = 0
    queued = Counter()

    with output.open("w", encoding="utf-8") as handle:
        while kept_bytes < args.target_bytes and stalled < args.stall_attempts:
            template = rng.choice(TEMPLATES)
            if args.max_per_template and queued[template.__name__] >= args.max_per_template:
                capped += 1
                stalled += 1
                continue
            name, description, example, header, body, tests = template(rng)
            text = render(name, description, example, header, body, tests)

            if text.strip() in banned_prompts or name in banned_names:
                contaminated += 1
                continue
            if text in seen:
                duplicate += 1
                stalled += 1
                continue
            seen.add(text)
            stalled = 0
            queued[template.__name__] += 1
            pending.append({"text": text, "tests": "\n".join(tests),
                            "template": template.__name__, "name": name})
            if len(pending) >= args.batch_size:
                flush(pending)
                pending = []
        flush(pending)

    stopped_early = stalled >= args.stall_attempts

    report = {
        "format": "brittain3-exercises-report-v1",
        "seed": args.seed,
        "output": str(output),
        "target_bytes": args.target_bytes,
        "accepted_bytes": kept_bytes,
        "accepted_documents": written,
        "verification_failures": failed,
        "duplicates_skipped": duplicate,
        "capped_skipped": capped,
        "max_per_template": args.max_per_template,
        "stopped_early": stopped_early,
        "contaminated_skipped": contaminated,
        "per_template": counts,
        "batch_size": args.batch_size,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if stopped_early:
        print(f"\nWARNING: stopped {args.target_bytes - kept_bytes:,} bytes short of "
              f"the target. The template set cannot produce more unique exercises "
              f"under --max-per-template={args.max_per_template}. Lower the target, "
              f"raise the cap, or add templates.", file=sys.stderr)
    if failed:
        print(f"\nWARNING: {failed} generated exercises failed their own assertions "
              f"and were discarded. A nonzero count means a template is wrong.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
