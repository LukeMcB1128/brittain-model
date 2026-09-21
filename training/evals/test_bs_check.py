"""Does the checker agree with reality on the probe cases?

A checker for generated code is only worth having if it separates the good from
the broken. Same sixteen programs as the error probe, now scored.
"""
import sys

sys.path.insert(0, "/tmp")
from bs_check import check

CASES = {
    "ok_push":            ('push("hello")\n', True),
    "ok_arith":           ("x = 2\ny = 3\npush(x + y)\n", True),
    "ok_cond":            ('x = 5\ncond x > 3\n    push("big")\nend\n', True),
    "ok_func":            ("func add(a, b)\n    return a + b\nend\npush(add(1, 2))\n", True),
    "ok_while":           ("i = 0\nwhile i < 3\n    push(i)\n    i = i + 1\nend\n", True),
    "ok_pyimport":        'math = pyimport("math")\npush(math.sqrt(16))\n',
    "syntax_trailing_op": ("x = 2 +\n", False),
    "syntax_unclosed":    ('push("unterminated\n', False),
    "unknown_func":       ("print(1)\n", False),
    "undefined_var":      ("push(nope)\n", False),
    "bad_keyword":        ("if x > 1:\n    push(1)\n", False),
    "python_def":         ("def f(x):\n    return x\n", False),
    "python_class":       ("class A:\n    pass\n", False),
    "div_zero":           ("push(1 / 0)\n", False),
}
CASES["ok_pyimport"] = ('math = pyimport("math")\npush(math.sqrt(16))\n', True)

wrong = 0
for name, (src, expected) in CASES.items():
    r = check(src)
    ok = r["valid"] == expected
    wrong += not ok
    print("%-20s expected=%-5s got=%-5s %s  %s"
          % (name, expected, r["valid"], "OK " if ok else "MISMATCH",
             (r["reason"] or "")[:60]))

print()
print("mismatches: %d of %d" % (wrong, len(CASES)))
sys.exit(1 if wrong else 0)
