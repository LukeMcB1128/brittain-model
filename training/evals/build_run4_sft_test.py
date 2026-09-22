# -*- coding: utf-8 -*-
"""A guard that has never rejected anything is not known to work."""
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "b", "/mnt/c/Coding/brittain-model/training/evals/build_run4_sft.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

good = b.known_syntax() + b.sourced_figures() + b.tool_account() + b.no_op_turns()
assert not b.check(good), "the real set should pass: %s" % b.check(good)
print("real set passes")

CASES = [
 ("known_syntax names a tool",
  [b.turn("known_syntax", "reverse a string",
          "I won't search for this. Use s[::-1].")]),
 ("a greeting answered with a speech",
  [b.turn("no_op_turns", "hey", "Welcome! " + "I can help with lots. " * 20)]),
 ("withheld turn states a figure anyway",
  [b.turn("sourced_figures", "how hard is the ap test",
          "Not in the file, but about 11 percent get a 5.",
          note="figure absent, none supplied"),
   b.turn("sourced_figures", "code?", "3070.P000.Y",
          note="figure present, quoted exact")]),
 ("quoted turn cites something not in its source",
  [b.turn("sourced_figures", "whats the course number", "It is 9999.X000.Z.",
          note="figure present, quoted exact"),
   b.turn("sourced_figures", "price?", "I don't have a current one.",
          note="figure absent, none supplied")]),
 ("unpaired sourced_figures",
  [b.turn("sourced_figures", "price?", "I don't have a current one.",
          note="figure absent, none supplied")]),
 ("row not ending on the assistant",
  [{"kind": "known_syntax", "source": "x", "note": "none",
    "messages": [{"role": "assistant", "content": "hi"},
                 {"role": "user", "content": "hi"}]}]),
]
failures = 0
for name, rows in CASES:
    problems = b.check(rows)
    print("  %-46s %s" % (name, "caught" if problems else "MISSED"))
    if not problems:
        failures += 1
sys.exit(1 if failures else 0)
