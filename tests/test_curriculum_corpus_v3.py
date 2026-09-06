import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare" / "build_curriculum_corpus_v3.py"
SPEC = importlib.util.spec_from_file_location("build_curriculum_corpus_v3", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def row(text, repository="owner/repo", path="src/file.py", language="python"):
    return {
        "text": text, "repository": repository, "path": path,
        "language": language, "source": "upstream", "category": "code",
    }


def test_test_document_detection_uses_paths_and_language_markers():
    assert module.is_test_document(row("plain", path="tests/test_math.py"))
    assert module.is_test_document(row("#[test]\nfn works() {}", path="src/lib.rs"))
    assert module.is_test_document(row("func TestSum(t *testing.T) {}", path="sum.go"))
    assert not module.is_test_document(row("def add(a, b): return a + b", path="src/math.py"))


def test_c_cpp_language_is_split_by_extension():
    assert module.normalize_language(row("x", path="src/a.c", language="c_cpp")) == "c"
    assert module.normalize_language(row("x", path="src/a.hpp", language="c_cpp")) == "cpp"


def test_config_targets_add_to_ten_million_tokens():
    config = json.loads(
        (PROJECT_ROOT / "configs" / "data" / "brittain3_49m_curriculum_corpus.json").read_text()
    )
    targets = module.target_keys(config, 1.0)
    assert sum(targets.values()) == 10_000_000
    assert targets["test_code:python"] == 1_200_000
    assert targets["companion_code:go"] == 150_000


def test_pool_selection_deduplicates_and_caps_repositories():
    candidates = [
        module.Candidate(
            "test_code:python", row(f"def f{i}(): return {i}"), 10, f"{i:064x}"
        )
        for i in range(3)
    ]
    duplicate = module.Candidate(
        "test_code:python", row("def f0(): return 0", repository="other/repo"),
        10, "f" * 64,
    )
    selected, totals, duplicates = module.select_pools(
        {"test_code:python": candidates + [duplicate]},
        {"test_code:python": 30}, repository_cap=2, already_selected=[],
    )
    assert len(selected) == 2
    assert totals["test_code:python"] == 20
    assert duplicates == 1


def test_exercise_collection_accepts_equal_priorities(tmp_path, monkeypatch):
    source = tmp_path / "exercises.jsonl"
    value = row("def identity(value):\n    return value", repository="family")
    source.write_text(
        "\n".join(json.dumps(value) for _ in range(2)) + "\n", encoding="utf-8"
    )

    class Tokenizer:
        @staticmethod
        def encode(text):
            return text.split()

    class Guard:
        @staticmethod
        def reason(_text):
            return None

    monkeypatch.setattr(module, "priority", lambda *_args: "same")
    candidates = module.collect_exercises(
        source, Tokenizer(), 100, 2, 1337, Guard(), module.Counter()
    )
    assert len(candidates) == 2


def test_pool_selection_can_exempt_owned_tool_repository_from_cap():
    candidates = [
        module.Candidate(
            "tool", row(f"tool call example {index}", repository="brittain-code"),
            10, f"{index:064x}",
        )
        for index in range(3)
    ]
    selected, totals, _ = module.select_pools(
        {"tool": candidates}, {"tool": 30}, repository_cap=1,
        already_selected=[], repository_cap_exempt_sources={"tool"},
    )
    assert len(selected) == 3
    assert totals["tool"] == 30


def test_fill_gate_allows_small_shift_between_test_and_companion_code():
    targets = {
        "test_code:c": 320,
        "companion_code:c": 240,
        "test_code:cpp": 400,
        "companion_code:cpp": 300,
        "english": 100,
    }
    accepted = {
        "test_code:c": 298,
        "companion_code:c": 241,
        "test_code:cpp": 402,
        "companion_code:cpp": 301,
        "english": 100,
    }
    failures, by_source, by_language = module.fill_failures(targets, accepted, 0.05)
    assert failures == []
    assert by_language["c"] == {"target": 560, "accepted": 539}
    assert by_source["test_code"] == {"target": 720, "accepted": 700}


def test_fill_gate_rejects_missing_total_language_capacity():
    failures, _, _ = module.fill_failures(
        {"test_code:c": 320, "companion_code:c": 240},
        {"test_code:c": 100, "companion_code:c": 241},
        0.05,
    )
    assert failures == ["code source test_code shortfall is 68.8%",
                        "code language c shortfall is 39.1%"]


def test_teacher_replays_keep_one_repository_for_split_grouping(tmp_path):
    source = tmp_path / "teacher.jsonl"
    source.write_text(json.dumps(row("def value(): return 1", repository="teacher/task")) + "\n")

    class Tokenizer:
        @staticmethod
        def encode(text):
            return text.split()

    candidates = module.collect_teacher(source, Tokenizer(), 6, 3, 1337)
    assert len({candidate.row["repository"] for candidate in candidates}) == 1
    assert len({candidate.row["path"] for candidate in candidates}) == 2
