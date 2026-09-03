"""Unit tests for the universal generation benchmark."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.universal_bench import (
    TRACKS,
    UNIVERSAL_DIR,
    JSONValidationResult,
    evaluate_prose_completion,
    load_universal_references,
    load_universal_tasks,
    normalize_indentation,
    truncate_braced_completion,
    truncate_json_completion,
    truncate_prose_completion,
    truncate_python_completion,
    validate_universal_suite,
    verify_json_completion,
)


def test_tracks_and_counts():
    tasks = load_universal_tasks()
    assert len(tasks) == 75
    counts = Counter(t.track for t in tasks)
    assert set(counts.keys()) == set(TRACKS)
    for track in TRACKS:
        assert counts[track] == 15, f"Track {track} has {counts[track]} tasks instead of 15"


def test_unique_task_ids():
    tasks = load_universal_tasks()
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "Task IDs must be unique"


def test_references_match_tasks():
    tasks = load_universal_tasks()
    references = load_universal_references()
    assert len(references) == 75
    task_ids = {t.id for t in tasks}
    assert set(references.keys()) == task_ids


def test_validate_universal_suite_passes():
    ok, failures = validate_universal_suite()
    assert ok is True, f"Suite validation failed: {failures}"
    assert len(failures) == 0


def test_truncate_python_completion():
    raw = "    return x + 1\n\ndef unrelated():\n    return 0\n"
    truncated = truncate_python_completion(raw)
    assert truncated == "    return x + 1\n\n"
    assert "unrelated" not in truncated


def test_truncate_braced_completion():
    raw = "    const x = {a: 1};\n    return x;\n}\nfunction extra() {}\n"
    truncated = truncate_braced_completion(raw)
    assert truncated.endswith("}\n".rstrip("\n"))
    assert "extra" not in truncated


def test_truncate_json_completion():
    prompt = '{\n  "config": {\n    "port": 8080,\n'
    body = '    "host": "localhost"\n  }\n}\nextra content after json'
    truncated = truncate_json_completion(prompt, body)
    full = prompt + truncated
    parsed = json.loads(full)
    assert parsed == {"config": {"port": 8080, "host": "localhost"}}


def test_verify_json_completion_success():
    prompt = '{\n  "title": "Report",\n'
    body = '  "status": "done"\n}'
    res = verify_json_completion(prompt, body, expected_keys=["title", "status"], schema_type="object")
    assert res.is_valid_json is True
    assert res.schema_valid is True
    assert res.parsed_object == {"title": "Report", "status": "done"}


def test_verify_json_completion_missing_keys():
    prompt = '{\n  "title": "Report",\n'
    body = '  "other": 123\n}'
    res = verify_json_completion(prompt, body, expected_keys=["title", "status"], schema_type="object")
    assert res.is_valid_json is True
    assert res.schema_valid is False
    assert "Missing keys" in res.error


def test_verify_json_completion_invalid_json():
    prompt = '{\n  "title": "Report",\n'
    body = '  status: not valid json'
    res = verify_json_completion(prompt, body)
    assert res.is_valid_json is False
    assert res.schema_valid is False


def test_evaluate_prose_completion_metrics():
    # Normal diverse prose
    text = "The quick brown fox jumps over the lazy dog in the sunny morning."
    metrics = evaluate_prose_completion(text)
    assert metrics.distinct_1 > 0.8
    assert metrics.distinct_2 > 0.8
    assert metrics.is_repetition_collapse is False
    assert metrics.sentence_closed is True

    # Repetition collapse text
    loop_text = "and the and the and the and the and the and the and the and the and the"
    loop_metrics = evaluate_prose_completion(loop_text)
    assert loop_metrics.is_repetition_collapse is True
    assert loop_metrics.distinct_1 < 0.3


def test_normalize_indentation():
    mixed = "\tdef foo():\n\t\treturn 1\n"
    normalized = normalize_indentation(mixed)
    assert "\t" not in normalized
    assert normalized == "    def foo():\n        return 1\n"
