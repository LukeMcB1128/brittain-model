import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.curriculum_quality_v3 import (
    DuplicateGuard,
    EvaluationGuard,
    structural_fingerprint,
)


def test_evaluation_guard_rejects_entry_point_and_verbatim_text():
    guard = EvaluationGuard.novice_v1()
    assert guard.reason("sum_list", ["unrelated"], "unrelated task") == "evaluation_entry_point"
    assertion = "assert unique_preserve_order([3, 1, 3, 2, 1]) == [3, 1, 2]"
    assert guard.reason("fresh_name", [assertion], "new task") == "evaluation_assertion"


def test_evaluation_guard_accepts_an_unrelated_task():
    guard = EvaluationGuard.novice_v1()
    assert guard.reason(
        "measure_packet_delay", ["def measure_packet_delay(samples):\n    return max(samples)"],
        "Measure the largest packet delay from numeric samples",
    ) is None


def test_structural_fingerprint_ignores_names_and_literals():
    first = "def add_tax(price):\n    return price * 1.2\n"
    second = "def scale_score(value):\n    return value * 9.5\n"
    assert structural_fingerprint("python", first) == structural_fingerprint("python", second)


def test_duplicate_guard_rejects_exact_structure_and_near_semantics():
    guard = DuplicateGuard()
    guard.add("python", "Clamp a score to the allowed numeric range", "def clamp_score(x):\n return max(0, x)")
    assert guard.reason(
        "python", "Scale an unrelated value", "def floor_value(y):\n return max(3, y)"
    ) == "duplicate_structure"
    assert guard.reason(
        "python", "The allowed numeric range must clamp a score", "def different(x):\n return x"
    ) == "duplicate_semantics"


def test_duplicate_guard_compares_semantics_within_one_language():
    guard = DuplicateGuard()
    guard.add("python", "Parse status records into a count mapping", "def one(x):\n return x")
    assert guard.reason(
        "javascript", "Parse status records into a count mapping", "function two(x) { return x; }"
    ) is None
