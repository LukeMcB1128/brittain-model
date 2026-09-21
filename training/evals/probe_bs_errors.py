"""Learn what failure looks like from the BrittainScript interpreter.

`bs` exits 0 whether a program runs or dies, so a checker cannot use the exit
code -- it has to recognise the error text. Rather than guess the vocabulary,
feed it a spread of broken programs and record exactly what comes back.

Also confirms the working surface: push() is output (print is undefined), and
the constructs the corpus filter looks for are func/cond/loop/while/for/repeat.
"""
import json
import subprocess
import tempfile
import os

CASES = {
    # Should succeed.
    "ok_push":            'push("hello")\n',
    "ok_arith":           'x = 2\ny = 3\npush(x + y)\n',
    "ok_cond":            'x = 5\ncond x > 3\n    push("big")\nend\n',
    "ok_func":            'func add(a, b)\n    return a + b\nend\npush(add(1, 2))\n',
    "ok_while":           'i = 0\nwhile i < 3\n    push(i)\n    i = i + 1\nend\n',
    "ok_pyimport":        'math = pyimport("math")\npush(math.sqrt(16))\n',

    # Should fail, in different ways.
    "syntax_trailing_op": "x = 2 +\n",
    "syntax_unclosed":    'push("unterminated\n',
    "unknown_func":       "print(1)\n",
    "undefined_var":      "push(nope)\n",
    "bad_keyword":        "if x > 1:\n    push(1)\n",
    "python_def":         "def f(x):\n    return x\n",
    "python_class":       "class A:\n    pass\n",
    "kwargs":             'push(sep="x")\n',
    "empty":              "",
    "div_zero":           "push(1 / 0)\n",
}

results = {}
for name, src in CASES.items():
    with tempfile.NamedTemporaryFile("w", suffix=".bs", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run(["bs", path], capture_output=True, text=True, timeout=30)
        results[name] = {
            "exit": p.returncode,
            "stdout": p.stdout.strip()[:200],
            "stderr": p.stderr.strip()[:200],
        }
    except subprocess.TimeoutExpired:
        results[name] = {"exit": None, "stdout": "", "stderr": "<timeout>"}
    finally:
        os.unlink(path)

for name, r in results.items():
    expect = "PASS" if name.startswith("ok_") else "FAIL"
    print("%-20s expect=%s exit=%s" % (name, expect, r["exit"]))
    if r["stdout"]:
        print("    stdout: %s" % r["stdout"].replace("\n", " | "))
    if r["stderr"]:
        print("    stderr: %s" % r["stderr"].replace("\n", " | "))

print()
print(json.dumps({k: v["exit"] for k, v in results.items()}))
