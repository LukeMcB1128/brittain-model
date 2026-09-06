"""Guards for the novice-code evaluation suite and its shared metrics.

The novice suite is the Brittain3 49M pilot's primary go/no-go gate. A silently
broken task — one whose tests can never pass, or one that duplicates an id and
gets counted twice — would corrupt that decision without failing loudly. These
tests keep the suite honest. They do not load a model.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.metrics import distinct_ngram_ratio, pass_at_k, repetition_collapse

TASKS_PATH = PROJECT_ROOT / "benchmarks" / "novice" / "tasks.jsonl"
REFERENCE_PATH = PROJECT_ROOT / "benchmarks" / "novice" / "reference.jsonl"
EXPECTED_CATEGORIES = {"functions", "arrays", "objects", "loops", "parsing", "state"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def tasks() -> list[dict]:
    return read_jsonl(TASKS_PATH)


@pytest.fixture(scope="module")
def references() -> dict[str, str]:
    return {row["id"]: row["body"] for row in read_jsonl(REFERENCE_PATH)}


def test_task_ids_are_unique(tasks):
    duplicates = [name for name, count in Counter(t["id"] for t in tasks).items() if count > 1]
    assert not duplicates, f"duplicate task ids would be scored twice: {duplicates}"


def test_every_task_has_required_fields(tasks):
    for task in tasks:
        for field in ("id", "category", "language", "entry_point", "prompt", "tests"):
            assert field in task, f"{task.get('id', '?')} is missing {field}"
        assert task["tests"], f"{task['id']} has no assertions, so it can never fail"


def test_categories_are_the_six_named_skills(tasks):
    assert {task["category"] for task in tasks} == EXPECTED_CATEGORIES


def test_prompts_parse_once_the_body_is_supplied(tasks, references):
    """A prompt that is not valid Python even with its solution is malformed."""
    for task in tasks:
        source = task["prompt"] + references[task["id"]]
        ast.parse(source)          # raises SyntaxError with the offending task


def test_prompts_end_at_a_definition_header(tasks):
    """Every prompt must end at a `def` or `class` header line.

    Measured on brittain2_50m_bs 2026-08-09: a prompt ending after a completed
    docstring or `#` comment made the model emit EOT immediately in 12 of 12
    samples, producing an empty body. Ending at the header instead gave 0 of 60
    empty. A docstring-terminated prompt therefore does not measure the model's
    coding ability, it measures where it thinks the document ended.
    """
    for task in tasks:
        last = task["prompt"].rstrip("\n").splitlines()[-1]
        assert last.startswith(("def ", "class ")) and last.rstrip().endswith(":"), (
            f"{task['id']} ends with {last!r}; it must end at a def/class header "
            f"or the model will stop before writing a body"
        )
        assert task["prompt"].endswith("\n"), f"{task['id']} must end with a newline"


def test_prompt_descriptions_are_comments_above_the_header(tasks):
    """The description must precede the header, since it cannot follow it."""
    for task in tasks:
        lines = task["prompt"].rstrip("\n").splitlines()
        assert len(lines) >= 2, f"{task['id']} has no description"
        for line in lines[:-1]:
            assert line.startswith("#"), (
                f"{task['id']} has a non-comment line {line!r} before the header"
            )


def test_every_task_has_a_reference_solution(tasks, references):
    missing = sorted({task["id"] for task in tasks} - set(references))
    assert not missing, f"no reference solution for {missing}"


def test_reference_solutions_pass_their_own_tests(tasks, references, tmp_path):
    """The suite must be solvable, or it measures nothing but its own bugs."""
    failures = []
    for task in tasks:
        program = task["prompt"] + references[task["id"]] + "\n\n" + "\n".join(task["tests"]) + "\n"
        source = tmp_path / "candidate.py"
        source.write_text(program, encoding="utf-8")
        finished = subprocess.run(
            [sys.executable, "-I", str(source)],
            cwd=tmp_path, capture_output=True, text=True, timeout=30,
        )
        if finished.returncode != 0:
            failures.append((task["id"], finished.stderr.strip().splitlines()[-1:]))
    assert not failures, f"reference solutions failing: {failures}"


def test_tasks_are_easier_than_humaneval(tasks, references):
    """Novice means novice. A reference solution needing many lines is too hard."""
    too_long = [task["id"] for task in tasks
                if len([line for line in references[task["id"]].splitlines() if line.strip()]) > 12]
    assert not too_long, f"these reference solutions are too long for a novice suite: {too_long}"


def test_pass_at_k_matches_known_values():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 1, 1) == pytest.approx(0.1)
    # 1 correct of 10, drawing 10, always contains it
    assert pass_at_k(10, 1, 10) == pytest.approx(1.0)
    # 5 of 10 correct, drawing 2: 1 - C(5,2)/C(10,2) = 1 - 10/45
    assert pass_at_k(10, 5, 2) == pytest.approx(1 - 10 / 45)


def test_pass_at_k_is_nan_when_k_exceeds_samples():
    import math
    assert math.isnan(pass_at_k(4, 1, 10))


def test_repetition_collapse_detects_a_repeating_loop():
    assert repetition_collapse([1, 2, 3, 4] * 40) is True
    assert repetition_collapse([7] * 200) is True


def test_repetition_collapse_accepts_varied_text():
    assert repetition_collapse(list(range(200))) is False


def test_repetition_collapse_ignores_short_generations():
    """Too few n-grams for the ratio to mean anything — must not report collapse."""
    assert repetition_collapse([1, 2] * 4) is False


def test_distinct_ngram_ratio_bounds():
    assert distinct_ngram_ratio(list(range(50))) == pytest.approx(1.0)
    assert distinct_ngram_ratio([1] * 50) == pytest.approx(1 / 47)
    assert distinct_ngram_ratio([1, 2]) == 1.0          # shorter than the order
