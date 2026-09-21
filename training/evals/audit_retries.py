# -*- coding: utf-8 -*-
"""Are the post-failure retries in the corpus corrective, or blind?

86% of turns that follow a failed tool result call another tool, and 25 of them
call the SAME tool. That looked like the cause of the web chat's loop. But the
two cases are opposite in quality:

  corrective  read_file fails with ENOENT -> read_file with a fixed path
  blind       search fails -> the identical search again

Training the first one out would make the agent worse. So this compares the
failed call's arguments with the retry's arguments and splits them.

Also reports what KIND of failure each one follows, because "file not found"
and "the provider is refusing you" deserve different responses and the corpus
may only contain the first.
"""
import collections
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
    return bool(FAILED.search(head)
                or re.search(r"\b(exit code [1-9]|stderr)\b", head, re.I))


def failure_kind(text):
    head = text.strip()[:220].lower()
    if "no such file" in head or "enoent" in head or "not found" in head and "http" not in head:
        return "missing file/path"
    if re.search(r"http [45]\d\d", head):
        return "http error"
    if "permission" in head or "not allowed" in head or "rejected" in head:
        return "refused by policy"
    if "unavailable" in head or "timed out" in head:
        return "provider unavailable"
    if "traceback" in head or "exit code" in head or "stderr" in head:
        return "command/program error"
    return "other"


def call_of(message):
    calls = message.get("tool_calls") or []
    if not calls:
        return None, None
    fn = calls[0].get("function") or calls[0]
    args = fn.get("arguments")
    if not isinstance(args, str):
        args = json.dumps(args or {}, sort_keys=True)
    return fn.get("name"), args


rows = [json.loads(l) for l in open(MIX, encoding="utf-8") if l.strip()]
traj = [r for r in rows if r.get("kind") == "trajectory" and r.get("messages")]

kinds = collections.Counter()
verdicts = collections.Counter()
blind_examples = []

for row in traj:
    messages = row["messages"]
    failed_at = None
    for index, message in enumerate(messages):
        if message.get("role") == "tool" and looks_failed(message.get("content")):
            failed_at = index
    if failed_at is None:
        continue
    # Only count it when the failure is the LAST tool result in the context --
    # otherwise the target is responding to something later.
    if any(m.get("role") == "tool" for m in messages[failed_at + 1:]):
        continue

    failure_text = messages[failed_at].get("content") or ""
    kinds[failure_kind(failure_text)] += 1

    prior_name = prior_args = None
    for back in range(failed_at - 1, -1, -1):
        name, args = call_of(messages[back])
        if name:
            prior_name, prior_args = name, args
            break

    target = row["target"]
    calls = target.get("tool_calls") or []
    if not calls:
        verdicts["recovered to prose" if (target.get("content") or "").strip()
                 else "empty target"] += 1
        continue
    fn = calls[0].get("function") or calls[0]
    name = fn.get("name")
    args = fn.get("arguments")
    if not isinstance(args, str):
        args = json.dumps(args or {}, sort_keys=True)

    if name != prior_name:
        verdicts["different tool"] += 1
    elif args != prior_args:
        verdicts["same tool, CHANGED arguments (corrective)"] += 1
    else:
        verdicts["same tool, IDENTICAL arguments (blind)"] += 1
        if len(blind_examples) < 4:
            blind_examples.append((failure_text[:80].replace("\n", " "),
                                   name, args[:90]))

print("failures the target responds to: %d\n" % sum(kinds.values()))
print("what failed:")
for kind, count in kinds.most_common():
    print("  %-24s %d" % (kind, count))

print("\nwhat the target did:")
for verdict, count in verdicts.most_common():
    print("  %-44s %d" % (verdict, count))

if blind_examples:
    print("\nblind repeats:")
    for failure, name, args in blind_examples:
        print("  after %-42s -> %s %s" % (failure, name, args))
