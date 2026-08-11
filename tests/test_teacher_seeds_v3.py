import importlib.util
from pathlib import Path

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
