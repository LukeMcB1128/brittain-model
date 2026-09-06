import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare" / "build_teacher_seeds_v3.py"
SPEC = importlib.util.spec_from_file_location("build_teacher_seeds_v3", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_largest_remainder_allocation_is_exact():
    shares = {"python": 0.30, "typescript": 0.20, "javascript": 0.15,
              "rust": 0.12, "cpp": 0.10, "c": 0.08, "go": 0.05}
    counts = module.allocate_counts(70, shares)
    assert counts == {"python": 21, "typescript": 14, "javascript": 11,
                      "rust": 8, "cpp": 7, "c": 6, "go": 3}
    assert sum(counts.values()) == 70


def test_small_allocation_does_not_drop_the_target():
    counts = module.allocate_counts(1, {"python": 0.6, "javascript": 0.4})
    assert counts == {"python": 1, "javascript": 0}


def test_balanced_smoke_covers_every_language():
    weighted = {"python": 2, "typescript": 1, "javascript": 1, "rust": 1,
                "cpp": 1, "c": 1, "go": 0}
    assert module.balance_smoke_counts(weighted) == {
        "python": 1, "typescript": 1, "javascript": 1, "rust": 1,
        "cpp": 1, "c": 1, "go": 1,
    }


def test_slug_normalization_is_portable():
    assert module.normalize_slug("Parse HTTP Header!") == "parse_http_header"


def test_task_digest_is_stable_and_language_specific():
    brief = {"goal": "Count rows", "input_contract": "a list", "output_contract": "an int"}
    assert module.task_digest("python", brief) == module.task_digest("python", brief)
    assert module.task_digest("python", brief) != module.task_digest("javascript", brief)


def test_semantic_text_includes_slug_for_behavior_guards():
    brief = {
        "slug": "clamp_range", "goal": "Limit a number", "input_contract": "three numbers",
        "output_contract": "one number", "edge_cases": ["equal limits"],
    }
    assert module.semantic_text(brief).startswith("clamp_range ")


def test_atomic_json_and_jsonl_writes_do_not_leave_temporary_files(tmp_path):
    json_path = tmp_path / "state.json"
    jsonl_path = tmp_path / "rows.jsonl"
    module.atomic_write_json(json_path, {"stage": "author"})
    module.atomic_write_jsonl(jsonl_path, [{"id": "one"}, {"id": "two"}])
    assert json.loads(json_path.read_text()) == {"stage": "author"}
    assert [json.loads(line) for line in jsonl_path.read_text().splitlines()] == [
        {"id": "one"}, {"id": "two"},
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_recovery_signature_changes_when_generation_settings_change(tmp_path):
    base = {
        "verify_timeout": 15.0,
        "timeout": 600.0,
        "review_timeout": 240.0,
        "tsc": str(tmp_path / "tsc"),
        "author_repairs": 1,
        "reviewer_repairs": 1,
        "json_retries": 1,
        "review_num_predict": 2560,
        "endpoint": "http://127.0.0.1:11434",
        "output": str(tmp_path / "out.jsonl"),
        "report": str(tmp_path / "report.json"),
    }
    first = module.recovery_signature(
        {"author_model": "coder"}, {"python": 2}, SimpleNamespace(**base)
    )
    changed = dict(base, review_num_predict=3000)
    second = module.recovery_signature(
        {"author_model": "coder"}, {"python": 2}, SimpleNamespace(**changed)
    )
    assert first != second


def test_interrupted_author_record_is_not_marked_complete(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "format": "brittain3-teacher-curriculum-v1",
        "seed": 7,
        "planner_model": "planner",
        "author_model": "author",
        "reviewer_model": "reviewer",
        "language_shares": {"python": 1.0},
        "categories": ["functions"],
    }))
    output = tmp_path / "seeds.jsonl"
    args = SimpleNamespace(
        config=str(config_path), count=1, languages=["python"],
        endpoint="http://127.0.0.1:11434", output=str(output),
        report=str(tmp_path / "report.json"), state=None, resume=False,
        timeout=600.0, review_timeout=240.0, review_num_predict=2560,
        verify_timeout=15.0, tsc=str(tmp_path / "tsc"), balanced_smoke=False,
        author_repairs=1, reviewer_repairs=1, json_retries=0, overwrite=True,
    )
    brief = {
        "slug": "measure_delay", "category": "functions",
        "goal": "Measure a packet delay", "input_contract": "numeric samples",
        "output_contract": "one number", "edge_cases": ["empty", "one sample"],
        "language": "python",
    }
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "backend_status", lambda _tsc: {"python": "python3"})
    monkeypatch.setattr(module, "plan_tasks", lambda *_args: [brief])
    monkeypatch.setattr(
        module, "author_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        module.main()
    state = json.loads(Path(str(output) + ".state.json").read_text())
    assert state["stage"] == "author"
    assert state["author_next"] == 0
    assert state["authored_candidates"] == []
