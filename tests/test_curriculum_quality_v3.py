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
    assert guard.reason(
        "validate_iso_date_format", ["unrelated source"],
        "Validate an input string that contains digits in ISO date format",
    ) is None
    assert guard.reason(
        "validate_password_strength", ["unrelated source"],
        "Validate password strength. Reject a password that contains only digits",
    ) is None


def test_evaluation_guard_rejects_renamed_or_translated_held_out_behaviors():
    guard = EvaluationGuard.novice_v1()
    cases = [
        "Remove duplicate elements from a list while preserving the original order",
        "Clamp an i32 value between minimum and maximum bounds",
        "Count the frequency of each word and return a dictionary mapping words to counts",
        "Filter a C string and return only digit characters from the input",
        "clamp_range Write a function that limits a number to min_val and max_val",
    ]
    for semantic_text in cases:
        assert guard.reason("new_name", ["unrelated source"], semantic_text).startswith(
            "evaluation_behavior_"
        )


def test_evaluation_guard_exposes_behavior_only_planner_exclusions():
    guard = EvaluationGuard.novice_v1()
    assert "Return the sum of a and b." in guard.behavior_summaries
    assert all("assert " not in value and "->" not in value for value in guard.behavior_summaries)


def test_all_frozen_guard_includes_multilanguage_v2():
    guard = EvaluationGuard.all_frozen()
    assert "tiered_shipping" in guard.entry_points
    assert "windowSums" in guard.entry_points
    assert "daysInMonth" in guard.entry_points
    assert any("Return shipping cost" in value for value in guard.behavior_summaries)


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
