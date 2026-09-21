# -*- coding: utf-8 -*-
"""Does the training corpus ever show recovering from a failed tool?

The web chat asked for BrittainScript code, searched, got told the provider was
unavailable and not to retry, and searched again -- four times, never writing a
line. The model announced what it was about to do ten times over instead.

Over-calling explains reaching for the tool. It does not explain what happened
after the tool failed. If the corpus contains failed tool results that are
followed by another tool call, and almost never by a prose answer, then "a tool
failed, so try a different tool" is exactly what was trained, and no amount of
restraint data fixes it.

Counts, over the mix that produced run 1:
  * turns whose context contains a failed tool result
  * of those, how many have a target that is prose vs another tool call
"""
import json
import re

MIX = "/home/lukeb/brittain4/data/train_mix.jsonl"
FAILED = re.compile(
    r"^\s*(error|Error:|failed|Failed|Traceback|command not found"
    r"|no such file|ENOENT|permission denied|HTTP [45]\d\d"
    r"|temporarily unavailable|not available)", re.I)


def looks_failed(text):
    if not isinstance(text, str) or not text.strip():
        return False
    head = text.strip()[:200]
    if FAILED.search(head):
        return True
    return bool(re.search(r"\b(exit code [1-9]|stderr)\b", head, re.I))


rows = [json.loads(l) for l in open(MIX, encoding="utf-8") if l.strip()]
traj = [r for r in rows if r.get("kind") == "trajectory" and r.get("messages")]
print("trajectory rows: %d" % len(traj))

after_failure = 0
recovered_prose = 0
called_again = 0
same_tool_again = 0
examples = []

for row in traj:
    messages = row["messages"]
    # The most recent tool result in the context is what the target responds to.
    last_tool = None
    last_tool_name = None
    for index, message in enumerate(messages):
        if message.get("role") == "tool":
            last_tool = message.get("content")
            # The assistant turn just before it names the tool.
            for back in range(index - 1, -1, -1):
                calls = messages[back].get("tool_calls")
                if calls:
                    fn = (calls[0].get("function") or calls[0])
                    last_tool_name = fn.get("name")
                    break
    if last_tool is None or messages[-1].get("role") == "assistant":
        pass
    if last_tool is None or not looks_failed(last_tool):
        continue

    after_failure += 1
    target = row["target"]
    calls = target.get("tool_calls") or []
    if calls:
        called_again += 1
        name = ((calls[0].get("function") or calls[0]).get("name"))
        if name and name == last_tool_name:
            same_tool_again += 1
    elif (target.get("content") or "").strip():
        recovered_prose += 1
        if len(examples) < 3:
            examples.append((last_tool[:90].replace("\n", " "),
                             (target["content"] or "")[:150].replace("\n", " ")))

print("\nturns whose context ends in a FAILED tool result: %d" % after_failure)
if after_failure:
    print("  target is another tool call : %d (%.0f%%)"
          % (called_again, 100.0 * called_again / after_failure))
    print("    of which the SAME tool    : %d" % same_tool_again)
    print("  target is prose             : %d (%.0f%%)"
          % (recovered_prose, 100.0 * recovered_prose / after_failure))

for failure, reply in examples:
    print("\n  failed: %s" % failure)
    print("  reply : %s" % reply)
