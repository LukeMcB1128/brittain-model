# -*- coding: utf-8 -*-
"""Measure the mix by TARGET tokens, which is what the loss is actually over.

build_mix.py reports share by context+target characters, and by that measure the
new restraint set is 0% -- a trajectory context runs to the 4,096-token window
while a restraint prompt is a couple of hundred tokens. That comparison makes
the set look pointless.

But train_lora.py masks the labels to the target turn: everything before the
final assistant header is -100. Context length decides how much the example
costs to train, not how much it teaches. What competes for the adapter's
capacity is target tokens.

So this counts both, using the trainer's own tokenizer and template, and prints
them side by side.
"""
import collections
import json

from transformers import AutoTokenizer

MIX = "/home/lukeb/brittain4/data/train_mix.jsonl"
WINDOW = 4096

tok = AutoTokenizer.from_pretrained(
    "/home/lukeb/brittain4/models/brittain4-base-w4a16", trust_remote_code=True)
tok.chat_template = open("/home/lukeb/brittain4/chat_template_brittain4.jinja",
                         encoding="utf-8").read()


def normalize_calls(calls):
    fixed = []
    for call in calls or []:
        fn = dict(call.get("function") or {})
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError:
                return None
        if not isinstance(arguments, dict):
            return None
        fixed.append({"id": call.get("id", ""), "type": "function",
                      "function": {"name": fn.get("name"), "arguments": arguments}})
    return fixed


rows = [json.loads(l) for l in open(MIX, encoding="utf-8") if l.strip()]

target_tokens = collections.Counter()
total_tokens = collections.Counter()
examples = collections.Counter()
skipped = 0

for row in rows:
    kind = row["kind"]
    messages = []
    ok = True
    for message in row["messages"]:
        clean = dict(message)
        if clean.get("tool_calls"):
            fixed = normalize_calls(clean["tool_calls"])
            if fixed is None:
                ok = False
                break
            clean["tool_calls"] = fixed
        messages.append(clean)
    if not ok or not messages:
        skipped += 1
        continue

    target = row["target"]
    assistant = {"role": "assistant", "content": target.get("content") or ""}
    if target.get("tool_calls"):
        fixed = normalize_calls(target["tool_calls"])
        if fixed is None:
            skipped += 1
            continue
        assistant["tool_calls"] = fixed

    kw = dict(tokenize=False, chat_template_kwargs={"enable_thinking": False})
    try:
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, **kw)
        full = tok.apply_chat_template(messages + [assistant], **kw)
    except Exception:
        skipped += 1
        continue
    if not full.startswith(prompt):
        skipped += 1
        continue

    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    full_ids = tok(full, add_special_tokens=False).input_ids
    if len(full_ids) <= len(prompt_ids):
        skipped += 1
        continue

    # The trainer keeps the END of an over-long example, so the target always
    # survives and only the context is cut.
    kept = min(len(full_ids), WINDOW)
    target_len = len(full_ids) - len(prompt_ids)
    examples[kind] += 1
    target_tokens[kind] += min(target_len, kept)
    total_tokens[kind] += kept

print("skipped (unrenderable): %d\n" % skipped)
grand_target = sum(target_tokens.values())
grand_total = sum(total_tokens.values())

print("=" * 76)
print("%-12s %9s %14s %8s %14s %8s"
      % ("kind", "examples", "target tok", "share", "window tok", "share"))
print("=" * 76)
for kind in ("trajectory", "general", "identity", "restraint"):
    if not examples[kind]:
        continue
    print("%-12s %9d %14d %7.1f%% %14d %7.1f%%"
          % (kind, examples[kind],
             target_tokens[kind], 100.0 * target_tokens[kind] / grand_target,
             total_tokens[kind], 100.0 * total_tokens[kind] / grand_total))
print("=" * 76)
print("%-12s %9d %14d %7s %14d" % ("total", sum(examples.values()),
                                   grand_target, "", grand_total))
print("\ntarget tokens are what the loss is over; window tokens are what the"
      "\nrun costs. The first column is the one to tune.")

for kind in ("trajectory", "restraint"):
    if examples[kind]:
        print("\n%s: %.0f target tokens per example"
              % (kind, target_tokens[kind] / examples[kind]))
