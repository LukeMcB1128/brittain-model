"""Explain the 40-character gap between my TOOL_DEFS and the spec's figure.

My extraction serialised to 32,330 chars; the spec, using JSON.stringify in
node, reported 32,290. Either that is a serialisation artefact or it is a
content difference in the array we are about to bake permanently into weights.

Hypothesis: Python's json.dumps defaults to ensure_ascii=True, escaping every
non-ASCII character as \\uXXXX -- six chars where JSON.stringify emits one.
The descriptions are full of em-dashes. If the count of non-ASCII characters
times five accounts for the gap, the arrays are identical and the difference
is mine.
"""
import json
import sys

defs = json.load(open(sys.argv[1], encoding="utf-8"))
names = [d["function"]["name"] for d in defs]

ascii_escaped = json.dumps(defs, separators=(",", ":"))
utf8_direct = json.dumps(defs, separators=(",", ":"), ensure_ascii=False)

print("tools                       : %d" % len(defs))
print("serialised, ensure_ascii=True : %d chars" % len(ascii_escaped))
print("serialised, ensure_ascii=False: %d chars  <- what JSON.stringify emits"
      % len(utf8_direct))
print("spec reported                 : 32290 chars")
print()

non_ascii = [c for c in utf8_direct if ord(c) > 127]
from collections import Counter
print("non-ASCII characters: %d" % len(non_ascii))
for ch, n in Counter(non_ascii).most_common():
    print("   %r U+%04X  x%d" % (ch, ord(ch), n))
print()
print("predicted inflation: %d chars (each becomes \\uXXXX, 6 for 1)"
      % (len(non_ascii) * 5))
print("actual difference  : %d chars" % (len(ascii_escaped) - len(utf8_direct)))
print()
match = len(utf8_direct) == 32290
print("MATCHES THE SPEC EXACTLY" if match
      else "still differs by %d chars from the spec figure"
           % (len(utf8_direct) - 32290))

print("\napp's own estimator (len/4, main.js:113): %d tokens"
      % round(len(utf8_direct) / 4))
print("spec reported: 8073 tokens")

print("\n--- the 55 names, in array order ---")
for i, n in enumerate(names, 1):
    print("%3d %s" % (i, n))
