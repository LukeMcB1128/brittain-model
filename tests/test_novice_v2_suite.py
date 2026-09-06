"""Integrity checks for the frozen multi-language novice-v2 suite."""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.verification_v3 import verify_program

TASKS_PATH = PROJECT_ROOT / "benchmarks" / "novice_v2" / "tasks.jsonl"
REFERENCE_PATH = PROJECT_ROOT / "benchmarks" / "novice_v2" / "reference.jsonl"
OLD_TASKS_PATH = PROJECT_ROOT / "benchmarks" / "novice" / "tasks.jsonl"
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate" / "novice.py"
SPEC = importlib.util.spec_from_file_location("novice_evaluator", EVALUATOR)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)

EXPECTED_LANGUAGES = {"python", "typescript", "javascript"}
EXPECTED_CATEGORIES = {
    "functions", "arrays", "objects", "loops", "parsing", "state",
    "error_handling", "small_multi_step",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_v2_has_balanced_languages_and_all_core_categories():
    tasks = read_jsonl(TASKS_PATH)
    assert len(tasks) == 30
    assert Counter(task["language"] for task in tasks) == {
        "python": 10, "typescript": 10, "javascript": 10,
    }
    categories = Counter(task["category"] for task in tasks)
    assert set(categories) == EXPECTED_CATEGORIES
    assert min(categories.values()) >= 3


def test_v2_ids_and_entry_points_do_not_overlap_v1():
    current = read_jsonl(TASKS_PATH)
    old = read_jsonl(OLD_TASKS_PATH)
    ids = [task["id"] for task in current]
    assert len(ids) == len(set(ids))
    assert not ({task["entry_point"] for task in current} & {task["entry_point"] for task in old})


def test_v2_prompts_end_at_language_definition_headers():
    for task in read_jsonl(TASKS_PATH):
        lines = task["prompt"].rstrip("\n").splitlines()
        assert task["prompt"].endswith("\n")
        prefix = "#" if task["language"] == "python" else "//"
        assert all(line.startswith(prefix) for line in lines[:-1])
        if task["language"] == "python":
            assert lines[-1].startswith(("def ", "class ")) and lines[-1].endswith(":")
        else:
            assert lines[-1].startswith(("function ", "class ")) and lines[-1].endswith("{")


def test_v2_references_compile_and_pass_hidden_tests():
    tasks = read_jsonl(TASKS_PATH)
    references = {row["id"]: row["body"] for row in read_jsonl(REFERENCE_PATH)}
    assert set(references) == {task["id"] for task in tasks}
    failures = []
    for task in tasks:
        checked = verify_program(
            task["language"], task["prompt"] + references[task["id"]],
            "\n".join(task["tests"]), timeout=15,
        )
        if not checked.ok:
            failures.append((task["id"], checked.phase, checked.detail))
    assert not failures


def test_braced_completion_truncation_keeps_one_complete_definition():
    completion = (
        "  const value = {name: '}'};\n"
        "  if (value) { return 1; }\n"
        "}\n"
        "function unrelated() { return 2; }\n"
    )
    kept = module.truncate_completion(completion, "typescript")
    assert kept.endswith("}\n".rstrip("\n"))
    assert "unrelated" not in kept


def test_python_completion_truncation_is_unchanged():
    completion = "    return 1\n\ndef unrelated():\n    return 2\n"
    assert module.truncate_completion(completion, "python") == "    return 1\n\n"
