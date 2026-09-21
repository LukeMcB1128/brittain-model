# -*- coding: utf-8 -*-
"""Score tool calling across the prompt shapes the app can actually send.

Brittain Code's toolIndex setting decides what reaches the model:

    off     full schemas for every enabled tool          8,475 tokens
    on      full schemas for common/called tools, short definitions for the rest
    auto    short definitions when the context is <= 49,152 tokens

The served model is 32k, so `auto` takes the short-definition path. That is a
shape nobody has measured. Everything known so far covers the two extremes:

    full schemas   base 52/108, adapter 49/108   -- training buys nothing
    no tools       base 25/103, adapter 39/103   -- training is the whole gain

The middle is what ships, and how much agent data the next run needs depends
on it. If the base copes with short definitions, the trajectory slice can
shrink; if it collapses the way it does with nothing, it cannot.

ARGUMENTS ARE SCORED, NOT JUST NAMES
Under short definitions the model sees what a tool is FOR and never sees its
parameters, so the arguments have to come from the weights. A run that picks
the right tool and invents its parameters is not a working run, and a
name-only score would call it one.

Scoring goes through /v1/completions with the prompt rendered here, because
vLLM only runs its tool-call parser when a request declares `tools` -- reading
message.tool_calls reported a clean 0/108 for every model once, which was read
as the adapter having learned nothing.
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
ap.add_argument("--tools", default="/home/lukeb/brittain4/data/tool_defs.json")
ap.add_argument("--template",
                default="/home/lukeb/brittain4/chat_template_brittain4.jinja")
ap.add_argument("--tokenizer",
                default="/home/lukeb/brittain4/models/brittain4-base-w4a16")
ap.add_argument("--models", nargs="*", default=["brittain4", "step-0100-mm"])
ap.add_argument("--modes", nargs="*",
                default=["none", "names", "names_desc", "full"])
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--out", default="")
args = ap.parse_args()

KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()
CALL = re.compile(r"<tool_call>\s*<function=([A-Za-z0-9_.\-]+)>(.*?)</function>",
                  re.S)
PARAM = re.compile(r"<parameter=([A-Za-z0-9_.\-]+)>\n?(.*?)\n?</parameter>", re.S)

tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
tok.chat_template = open(args.template, encoding="utf-8").read()

BAKED = json.load(open(args.tools, encoding="utf-8"))
if isinstance(BAKED, dict):
    BAKED = BAKED.get("tools") or []
SCHEMA = {}
for tool in BAKED:
    function = tool.get("function") or tool
    parameters = function.get("parameters") or {}
    SCHEMA[function.get("name")] = {
        "required": set(parameters.get("required") or []),
        "known": set((parameters.get("properties") or {}).keys()),
    }


def declare(mode, extra_names):
    if mode == "none":
        return None
    out = []
    for tool in BAKED:
        function = tool.get("function") or tool
        if mode == "full":
            out.append(tool)
            continue
        entry = {"name": function.get("name")}
        if mode == "names_desc":
            description = (function.get("description") or "").strip()
            entry["description"] = (description.split(". ")[0].rstrip(".") + "."
                                    if description else "")
        out.append({"type": "function", "function": entry})
    for name in extra_names or []:
        out.append({"type": "function", "function": {"name": name}})
    return out


def normalize_calls(message):
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


def raw(model, prompt):
    body = {"model": model, "prompt": prompt, "temperature": 0,
            "max_tokens": args.max_tokens}
    req = urllib.request.Request(
        args.base_url + "/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8", "replace"))["choices"][0]["text"]


# --- items -----------------------------------------------------------------
items = []
for line in open(args.eval, encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    wanted = (row.get("reference") or {}).get("tool_names") or []
    if not wanted or wanted[0] == "ask_user":
        continue                       # ask_user is a preference, not selection
    messages, ok = [], True
    for message in row.get("context") or []:
        fixed = normalize_calls(dict(message))
        if fixed is None:
            ok = False
            break
        messages.append(fixed)
    if ok and messages:
        items.append({"want": wanted[0], "messages": messages,
                      "extra": row.get("extra_tools_names")
                      or [t.get("function", {}).get("name")
                          for t in (row.get("extra_tools") or [])]})
if args.limit:
    items = items[:args.limit]
print("held-out tool-selection items: %d" % len(items))


def score(model, mode):
    called = correct = args_ok = failed = 0
    missing_required = unknown_param = 0
    for item in items:
        tools = declare(mode, item["extra"])
        kw = dict(tokenize=False, add_generation_prompt=True,
                  chat_template_kwargs={"enable_thinking": False})
        if tools:
            kw["tools"] = tools
        try:
            prompt = tok.apply_chat_template(item["messages"], **kw)
            text = raw(model, prompt)
        except Exception as error:
            failed += 1
            if failed <= 2:
                print("    failed: %s" % str(error)[:100])
            continue
        found = CALL.search(text)
        if not found:
            continue
        called += 1
        name, body = found.group(1), found.group(2)
        if name != item["want"]:
            continue
        correct += 1
        schema = SCHEMA.get(name)
        if not schema:
            continue                   # an MCP tool; no schema to check against
        supplied = {key for key, _ in PARAM.findall(body)}
        if not schema["required"] <= supplied:
            missing_required += 1
            continue
        if schema["known"] and not supplied <= schema["known"]:
            unknown_param += 1
            continue
        args_ok += 1
    return {"items": len(items), "called": called, "correct": correct,
            "args_ok": args_ok, "missing_required": missing_required,
            "unknown_param": unknown_param, "failed": failed}


results = {}
for model in args.models:
    for mode in args.modes:
        print("\nscoring %s / %s ..." % (model, mode))
        row = score(model, mode)
        results["%s|%s" % (model, mode)] = row
        print("  called %3d  correct %3d  args ok %3d  "
              "(missing req %d, unknown param %d, failed %d)"
              % (row["called"], row["correct"], row["args_ok"],
                 row["missing_required"], row["unknown_param"], row["failed"]))

W = 82
print("\n" + "=" * W)
print("%-16s %-12s %8s %9s %9s %9s"
      % ("model", "tool-mode", "called", "correct", "args ok", "% args ok"))
print("=" * W)
for key, row in results.items():
    model, mode = key.split("|")
    pct = 100.0 * row["args_ok"] / row["items"] if row["items"] else 0.0
    print("%-16s %-12s %8d %9d %9d %8.1f%%"
          % (model, mode, row["called"], row["correct"], row["args_ok"], pct))
print("=" * W)
print("\ncorrect = right tool name. args ok = right name AND every required\n"
      "parameter present with no invented ones. Under short definitions the\n"
      "parameters are not in the prompt, so the second column is the one that\n"
      "says whether the schemas are really in the weights.")

if args.out:
    json.dump(results, open(args.out, "w", encoding="utf-8"), indent=2)
    print("\nwrote %s" % args.out)
