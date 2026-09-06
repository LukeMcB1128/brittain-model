"""Guards for evaluation-suite decontamination.

Decontamination has two ways to fail and both are silent. Too loose and the
pilot trains on its own gate, so a good novice score means nothing. Too tight
and it deletes a large share of ordinary Python — every file defining `def add`
— which costs real training signal for no benefit. These tests pin both ends.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare" / "decontaminate_v3.py"
TASKS = PROJECT_ROOT / "benchmarks" / "novice" / "tasks.jsonl"
REFERENCE = PROJECT_ROOT / "benchmarks" / "novice" / "reference.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row(text: str, path: str) -> dict:
    return {"text": text, "path": path, "source": "test", "category": "code",
            "language": "python", "repository": "owner/repo", "license": "MIT"}


def run_filter(tmp_path: Path, rows: list[dict], extra: list[str] | None = None):
    source = tmp_path / "in.jsonl"
    target = tmp_path / "out.jsonl"
    report = tmp_path / "report.json"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    finished = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(target),
         "--report", str(report), "--overwrite", *(extra or [])],
        capture_output=True, text=True, timeout=120,
    )
    kept = read_jsonl(target) if target.exists() else []
    data = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    return finished, kept, data


@pytest.fixture(scope="module")
def tasks() -> list[dict]:
    return read_jsonl(TASKS)


@pytest.fixture(scope="module")
def references() -> dict[str, str]:
    return {r["id"]: r["body"] for r in read_jsonl(REFERENCE)}


def test_exact_task_document_is_removed(tmp_path, tasks, references):
    task = next(t for t in tasks if t["id"] == "arrays/unique_preserve_order")
    rows = [row(task["prompt"] + references[task["id"]], "contam.py"),
            row("import os\n\n\ndef unrelated():\n    return os.getcwd()\n", "clean.py")]
    _, kept, report = run_filter(tmp_path, rows, ["--max-drop-fraction", "1.0"])
    assert [k["path"] for k in kept] == ["clean.py"]
    assert report["removed_by_rule"]["document_hash"] == 1


def test_verbatim_assertion_is_removed(tmp_path, tasks):
    task = next(t for t in tasks if t["id"] == "arrays/unique_preserve_order")
    embedded = f"def helper():\n    return 1\n\n{task['tests'][0]}\n"
    _, kept, report = run_filter(tmp_path, [row(embedded, "contam.py")],
                                 ["--max-drop-fraction", "1.0"])
    assert kept == []
    assert report["removed_by_rule"]["verbatim_assertion"] == 1


def test_verbatim_description_is_removed(tmp_path, tasks):
    task = next(t for t in tasks if t["id"] == "parsing/split_key_value")
    description = task["prompt"].splitlines()[0].lstrip("#").strip()
    _, kept, report = run_filter(tmp_path, [row(f"# {description}\ndef whatever():\n    pass\n", "c.py")],
                                 ["--max-drop-fraction", "1.0"])
    assert kept == []
    assert report["removed_by_rule"]["verbatim_description"] == 1


def test_distinctive_entry_point_with_test_data_is_removed(tmp_path):
    """Rule 3: a distinctive name plus two literals lifted from its tests."""
    text = (
        "def split_key_value(line):\n"
        "    key, _, value = line.partition('=')\n"
        "    return (key.strip(), value.strip())\n"
        "\n"
        "sample = 'name = ada'\n"
        "expected = ('name', 'ada')\n"
    )
    _, kept, report = run_filter(tmp_path, [row(text, "c.py")], ["--max-drop-fraction", "1.0"])
    assert kept == []
    assert any(rule.startswith("entry_point") for rule in report["removed_by_rule"])


def test_generic_names_alone_are_kept(tmp_path):
    """`def add` and `class Stack` are everywhere. Removing them costs signal."""
    rows = [
        row("def add(a, b):\n    return a + b\n", "add.py"),
        row("class Stack:\n    def __init__(self):\n        self.items = []\n", "stack.py"),
        row("def average(values):\n    return sum(values) / len(values)\n", "avg.py"),
        row("def factorial(n):\n    return 1 if n < 2 else n * factorial(n - 1)\n", "fact.py"),
        row("class Counter:\n    def __init__(self):\n        self.value = 0\n", "counter.py"),
    ]
    _, kept, report = run_filter(tmp_path, rows)
    assert len(kept) == len(rows), f"generic code was removed: {report['removed_by_rule']}"
    assert report["documents_removed"] == 0


def test_a_distinctive_implementation_without_test_data_is_kept(tmp_path):
    """An independent implementation is not the benchmark's answer key.

    One weak signal must not be enough, or ordinary utility code disappears.
    """
    text = ("def running_total(values):\n"
            "    out = []\n"
            "    acc = 0\n"
            "    for v in values:\n"
            "        acc += v\n"
            "        out.append(acc)\n"
            "    return out\n")
    _, kept, _ = run_filter(tmp_path, [row(text, "impl.py")])
    assert len(kept) == 1


def test_excessive_removal_fails_loudly(tmp_path, tasks, references):
    """A misfiring rule must fail the run, not quietly shrink the corpus."""
    task = next(t for t in tasks if t["id"] == "arrays/unique_preserve_order")
    rows = [row(task["prompt"] + references[task["id"]], f"contam{i}.py") for i in range(4)]
    rows.append(row("def unrelated():\n    return 1\n", "clean.py"))
    finished, _, report = run_filter(tmp_path, rows, ["--max-drop-fraction", "0.05"])
    assert finished.returncode == 1
    assert report["removed_fraction"] > 0.05
    assert "misfiring" in finished.stderr


def test_report_records_what_it_watched(tmp_path):
    _, _, report = run_filter(tmp_path, [row("def unrelated():\n    return 1\n", "a.py")])
    assert report["format"] == "brittain3-decontamination-report-v1"
    assert "split_key_value" in report["entry_points_watched"]
    # Generic names must be explicitly excluded from name-based matching.
    assert "add" in report["generic_names_ignored"]
    assert "add" not in report["entry_points_watched"]
    assert report["tasks_checked"] == 66
    assert any("novice_v2" in value for value in report["task_suites"])


def test_v2_verbatim_assertion_is_removed(tmp_path):
    tasks = read_jsonl(PROJECT_ROOT / "benchmarks" / "novice_v2" / "tasks.jsonl")
    task = next(row for row in tasks if row["id"] == "javascript/functions/days_in_month")
    _, kept, report = run_filter(
        tmp_path, [row("function helper() {}\n" + task["tests"][0], "v2.js")],
        ["--max-drop-fraction", "1.0"],
    )
    assert kept == []
    assert report["removed_by_rule"]["verbatim_assertion"] == 1


def test_javascript_definition_with_v2_evidence_is_removed(tmp_path):
    text = (
        "function daysInMonth(year, month) { return 30; }\n"
        "if (daysInMonth(2024,2) !== 29) fail();\n"
        "if (daysInMonth(2023,11) !== 30) fail();\n"
    )
    _, kept, _ = run_filter(
        tmp_path, [row(text, "calendar.js")], ["--max-drop-fraction", "1.0"],
    )
    assert kept == []


def test_long_typescript_declaration_is_not_a_description(tmp_path):
    text = (
        "function changedSettings(before: Record<string, unknown>, "
        "after: Record<string, unknown>): string[] {\n  return [];\n}\n"
    )
    _, kept, report = run_filter(tmp_path, [row(text, "settings.ts")])
    assert len(kept) == 1
    assert report["documents_removed"] == 0
