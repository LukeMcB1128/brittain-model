import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare" / "audit_teacher_seeds_v3.py"
SPEC = importlib.util.spec_from_file_location("audit_teacher_seeds_v3", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def make_row(identifier, slug, entry_point, solution, tests):
    return {
        "id": identifier,
        "text": "# Measure the largest packet delay\n" + solution,
        "language": "python",
        "entry_point": entry_point,
        "brief": {
            "slug": slug, "goal": "Measure the largest packet delay",
            "input_contract": "numeric samples", "output_contract": "one number",
            "edge_cases": ["one sample", "negative samples"],
        },
        "solution": solution,
        "author_tests": tests,
        "reviewer_tests": tests,
        "semantic_category": "functions",
    }


def test_audit_reverifies_clean_rows_and_rejects_held_out_behavior():
    clean = make_row(
        "python/measure_delay/one", "measure_delay", "measure_delay",
        "def measure_delay(values):\n    return max(values)\n",
        "assert measure_delay([1, 3, 2]) == 3\n",
    )
    contaminated = make_row(
        "python/fresh_name/two", "clamp_range", "fresh_name",
        "def fresh_name(value, low, high):\n    return max(low, min(value, high))\n",
        "assert fresh_name(12, 0, 10) == 10\n",
    )
    contaminated["brief"]["goal"] = "Clamp a value to lower and upper bounds"
    accepted, report = module.audit_rows([clean, contaminated], 5.0, "tsc")
    assert [row["id"] for row in accepted] == [clean["id"]]
    assert report["accepted"] == 1
    assert report["rejected"] == {"evaluation_behavior_clamp_value": 1}
