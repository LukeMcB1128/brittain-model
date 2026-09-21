# -*- coding: utf-8 -*-
"""Score the tools-stripped shape, which is the shape the adapter is trained for.

WHY THIS IS NOT PART OF eval_checkpoint.py's CHAT PATH
eval_checkpoint.py reads `message.tool_calls` from /v1/chat/completions. vLLM
only runs --tool-call-parser when the request declares `tools`, and the whole
point of this shape is that it declares none. A correctly formed

    </think>

    <tool_call>
    <function=read_file>
    <parameter=path>...

therefore never reaches `tool_calls`, and because --reasoning-parser qwen3 is
also active it does not survive into `content` either. Both models scored a
clean 0/108 that way, and that zero was read as "the adapter bakes in nothing"
-- the opposite of the truth. Measured here instead, the adapter calls a tool
on 96 of 108 against the base's 61.

So this talks to /v1/completions: the prompt goes in as text, rendered with the
same template the trainer used, and the completion comes back as text. Nothing
sits between the model and the score.

ask_user IS REPORTED SEPARATELY
Some held-out references are `ask_user`: the model was supposed to stop and ask
rather than act. That measures a behavioural preference, not the ability to pick
a tool, so it is counted on its own rather than folded into tool selection.

It is a small slice -- 5 of 108 -- so it moves the headline barely at all. It is
split out because it was briefly mistaken for the opposite: `ask_user -> ...`
lines dominated a sample of wrong picks and were read as a large suppressed
share, when they were only the first rows in file order. Reporting the two
separately makes the size of the slice visible instead of arguable.

Usage:
    python3 score_stripped.py                          # base + every -mm checkpoint
    python3 score_stripped.py --models brittain4 step-0205-mm
"""
import argparse
import json
import os
import re
import urllib.request

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:11435/v1")
ap.add_argument("--eval", default="/home/lukeb/brittain4/data/eval_driver.jsonl")
ap.add_argument("--template",
                default="/home/lukeb/brittain4/chat_template_brittain4.jinja")
ap.add_argument("--tokenizer",
                default="/home/lukeb/brittain4/models/brittain4-base-w4a16")
ap.add_argument("--models", nargs="*", default=None,
                help="default: brittain4 plus every registered *-mm checkpoint")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--out", default="")
args = ap.parse_args()

KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()
CALL = re.compile(r"<tool_call>\s*<function=([A-Za-z0-9_.\-]+)>")

tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
tok.chat_template = open(args.template, encoding="utf-8").read()


def get(path):
    req = urllib.request.Request(
        args.base_url + path, headers={"Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def raw(model, prompt):
    body = {"model": model, "prompt": prompt, "temperature": 0,
            "max_tokens": args.max_tokens}
    req = urllib.request.Request(
        args.base_url + "/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        payload = json.loads(r.read().decode("utf-8", "replace"))
    return payload["choices"][0]["text"]


def normalize_calls(message):
    """Arguments must be a MAPPING before the template sees them.

    The template does `arguments | items`. The corpus stores both shapes. This
    is the trainer's direction, deliberately -- scoring through a different
    normalisation than training used would measure a different model.
    """
    calls = message.get("tool_calls")
    if not calls:
        return message
    fixed = []
    for call in calls:
        fn = dict(call.get("function") or {})
        name = fn.get("name") or call.get("name")
        arguments = fn.get("arguments", call.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError:
                return None
        if not isinstance(arguments, dict):
            return None
        fixed.append({"id": call.get("id", ""), "type": "function",
                      "function": {"name": name, "arguments": arguments}})
    out = dict(message)
    out["tool_calls"] = fixed
    return out


# --- the held-out items ----------------------------------------------------
items = []
for line in open(args.eval, encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    wanted = (row.get("reference") or {}).get("tool_names") or []
    if not wanted:
        continue                          # a text-only turn, nothing to score
    messages, ok = [], True
    for message in row.get("context") or []:
        fixed = normalize_calls(dict(message))
        if fixed is None:
            ok = False                    # malformed arguments in the context
            break
        messages.append(fixed)
    if not ok or not messages:
        continue
    prompt = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        chat_template_kwargs={"enable_thinking": False})
    items.append({"want": wanted[0], "prompt": prompt})
if args.limit:
    items = items[:args.limit]

tool_items = [i for i in items if i["want"] != "ask_user"]
ask_items = [i for i in items if i["want"] == "ask_user"]
print("held-out items: %d  (tool selection %d, ask_user %d)"
      % (len(items), len(tool_items), len(ask_items)))

models = args.models
if not models:
    registered = [m["id"] for m in get("/models")["data"]]
    models = ["brittain4"] + sorted(m for m in registered if m.endswith("-mm"))
print("models: %s\n" % ", ".join(models))


def score(model, subset):
    called = correct = failed = 0
    confusion = {}
    for item in subset:
        try:
            text = raw(model, item["prompt"])
        except Exception as error:
            failed += 1
            if failed <= 2:
                print("    request failed: %s" % str(error)[:110])
            continue
        found = CALL.search(text)
        if not found:
            continue
        called += 1
        got = found.group(1)
        if got == item["want"]:
            correct += 1
        else:
            key = "%s -> %s" % (item["want"], got)
            confusion[key] = confusion.get(key, 0) + 1
    return {"items": len(subset), "called": called, "correct": correct,
            "failed": failed, "confusion": confusion}


results = {}
for model in models:
    print("scoring %s ..." % model)
    results[model] = {"tool_selection": score(model, tool_items),
                      "ask_user": score(model, ask_items)}
    t = results[model]["tool_selection"]
    a = results[model]["ask_user"]
    print("  tool selection  called %3d/%-3d correct %3d   failed %d"
          % (t["called"], t["items"], t["correct"], t["failed"]))
    print("  ask_user        called %3d/%-3d correct %3d   failed %d\n"
          % (a["called"], a["items"], a["correct"], a["failed"]))

W = 78
print("=" * W)
print("%-16s %-27s %s" % ("", "tool selection (n=%d)" % len(tool_items),
                          "ask_user (n=%d)" % len(ask_items)))
print("%-16s %9s %8s %8s %9s %8s"
      % ("model", "called", "correct", "%", "called", "correct"))
print("=" * W)
for model in models:
    t = results[model]["tool_selection"]
    a = results[model]["ask_user"]
    pct = (100.0 * t["correct"] / t["items"]) if t["items"] else 0.0
    print("%-16s %9d %8d %7.1f%% %9d %8d"
          % (model, t["called"], t["correct"], pct, a["called"], a["correct"]))
print("=" * W)

best = max(models, key=lambda m: results[m]["tool_selection"]["correct"])
print("\nbest on tool selection: %s (%d correct)"
      % (best, results[best]["tool_selection"]["correct"]))
worst = sorted(results[best]["tool_selection"]["confusion"].items(),
               key=lambda kv: -kv[1])[:8]
print("its most common wrong picks:")
for key, count in worst:
    print("  %-46s %d" % (key, count))

if args.out:
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote %s" % args.out)
