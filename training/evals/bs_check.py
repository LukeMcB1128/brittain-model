"""Decide whether a BrittainScript program is valid, for scoring generated code.

    python3 bs_check.py file.bs
    python3 bs_check.py --stdin < program.bs
    from bs_check import check ; check(source) -> dict

WHY NOT THE EXIT CODE
Measured across sixteen probe programs, `bs` returns 0 for every one of them --
a clean run, a syntax error, an undefined function, division by zero. It also
writes errors to STDOUT mixed in with real output, so the stream does not
separate them either, and it does not stop on an error: a Python-style `def`
produced five diagnostics and kept going.

So validity has to be read out of the text. Any checker built on `bs file.bs;
echo $?` marks every broken program as passing, which is worth knowing about
any corpus that was filtered that way.

The patterns below were collected by feeding the interpreter deliberately
broken programs, not taken from documentation.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

# The interpreter runs as a subprocess, so it does not have to live in the same
# venv as the caller. That matters: the eval runs under venv-vllm, while
# brittainscript is installed in venv -- and installing it into both is worse
# than resolving it, because the wheel claims the top-level module name "core"
# and would sit in the import path of everything else in that environment.
BS_BIN = (os.environ.get("BS_BIN")
          or shutil.which("bs")
          or os.path.expanduser("~/venv/bin/bs"))

ERROR_PATTERNS = [
    re.compile(r"^Syntax error\b", re.M),          # "at end of input", "at 'x'"
    re.compile(r"^Illegal character:", re.M),
    re.compile(r"^Undefined function:", re.M),
    re.compile(r"^Undefined variable:", re.M),
    re.compile(r"^Error: ", re.M),                 # division by zero, return outside func
]

# A program that only assigns literals runs cleanly but is not a program. Same
# constructs filter_bs.py looks for, so the two definitions agree.
LOGIC = re.compile(r"(?m)^\s*(func|cond|loop|while|for|repeat)\b|\bpush\s*\(")


def check(source, timeout=30, require_logic=False):
    """Run one program and report why it failed, if it did."""
    with tempfile.NamedTemporaryFile("w", suffix=".bs", delete=False,
                                     encoding="utf-8") as f:
        f.write(source)
        path = f.name
    try:
        p = subprocess.run([BS_BIN, path], capture_output=True, text=True, timeout=timeout)
        output = (p.stdout or "") + "\n" + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return {"valid": False, "reason": "timeout", "errors": [], "output": ""}
    except FileNotFoundError:
        raise SystemExit("BrittainScript interpreter not found at %r. "
                         "pip install brittainscript, or set BS_BIN." % BS_BIN)
    finally:
        os.unlink(path)

    errors = []
    for pat in ERROR_PATTERNS:
        errors.extend(m.group(0) for m in pat.finditer(output))

    if errors:
        # Report the first whole diagnostic line, which names the token.
        first = next((l for l in output.splitlines()
                      if any(p.match(l) for p in ERROR_PATTERNS)), errors[0])
        return {"valid": False, "reason": first.strip()[:160],
                "errors": errors, "output": output[:400]}
    if require_logic and not LOGIC.search(source):
        return {"valid": False, "reason": "no logic constructs",
                "errors": [], "output": output[:400]}
    return {"valid": True, "reason": None, "errors": [], "output": output[:400]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--require-logic", action="store_true")
    args = ap.parse_args()

    src = sys.stdin.read() if args.stdin else open(args.path, encoding="utf-8").read()
    r = check(src, require_logic=args.require_logic)
    print("valid" if r["valid"] else "INVALID: %s" % r["reason"])
    sys.exit(0 if r["valid"] else 1)


if __name__ == "__main__":
    main()
