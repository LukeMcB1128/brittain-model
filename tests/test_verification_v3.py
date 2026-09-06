import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.verification_v3 import backend_status, verify_program, verify_syntax


def test_python_verifier_accepts_correct_code():
    result = verify_program(
        "python", "def add(a, b):\n    return a + b", "assert add(2, 3) == 5"
    )
    assert result.ok


def test_python_verifier_rejects_wrong_code():
    result = verify_program(
        "python", "def add(a, b):\n    return a - b", "assert add(2, 3) == 5"
    )
    assert not result.ok
    assert result.phase == "run"


def test_javascript_verifier_when_node_is_available():
    if backend_status()["javascript"] is None:
        return
    result = verify_program(
        "javascript", "function twice(value) { return value * 2; }",
        "if (twice(4) !== 8) throw new Error('bad');",
    )
    assert result.ok


def test_javascript_syntax_check_does_not_run_top_level_code():
    if backend_status()["javascript"] is None:
        return
    result = verify_syntax("javascript", "throw new Error('must not run');")
    assert result.ok


def test_typescript_syntax_check_is_strict_when_tsc_is_available():
    if backend_status()["typescript"] is None:
        return
    result = verify_syntax(
        "typescript", "function twice(value: number): number { return value * 2; }"
    )
    assert result.ok


def test_c_verifier_when_compiler_is_available():
    if backend_status()["c"] is None:
        return
    result = verify_program(
        "c", "int twice(int value) { return value * 2; }",
        "#include <assert.h>\nint main(void) { assert(twice(4) == 8); return 0; }",
    )
    assert result.ok


def test_c_warnings_do_not_reject_correct_code():
    if backend_status()["c"] is None:
        return
    result = verify_program(
        "c", "typedef struct { int first; int second; } pair_t;",
        "#include <assert.h>\nint main(void) { pair_t value = {4}; assert(value.first == 4); return 0; }",
    )
    assert result.ok


def test_rust_test_module_when_compiler_is_available():
    if backend_status()["rust"] is None:
        return
    result = verify_program(
        "rust", "fn twice(value: i32) -> i32 { value * 2 }",
        "#[cfg(test)]\nmod tests {\nuse super::*;\n#[test]\nfn works() { assert_eq!(twice(4), 8); }\n}",
    )
    assert result.ok


def test_go_compile_failure_is_separate_from_test_failure():
    if backend_status()["go"] is None:
        return
    result = verify_program(
        "go", 'package main\nimport "math"\nfunc twice(value int) int { return value * 2 }',
        'func main() { if twice(4) != 8 { panic("bad") } }',
    )
    assert not result.ok
    assert result.phase == "compile"


def test_missing_toolchain_is_reported():
    result = verify_program(
        "typescript", "function twice(value: number): number { return value * 2; }",
        "if (twice(4) !== 8) throw new Error('bad');", tsc="/does/not/exist/tsc",
    )
    assert not result.ok
    assert result.phase == "unavailable"


def test_teacher_bakeoff_references_pass_when_backends_are_available():
    config = json.loads(
        (PROJECT_ROOT / "configs" / "data" / "brittain3_teacher_bakeoff.json").read_text()
    )
    status = backend_status()
    for case in config["cases"]:
        if status[case["language"]] is None:
            continue
        result = verify_program(case["language"], case["reference"], case["tests"])
        assert result.ok, (case["id"], result)
