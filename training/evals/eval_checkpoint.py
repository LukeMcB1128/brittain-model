# -*- coding: utf-8 -*-
"""Score one adapter checkpoint through vLLM, on the checkpoint that is served.

WHY NOT SCORE IN THE TRAINING PROCESS
The adapter is fitted against NF4-quantised fp16 weights; production serves an
AutoRound W4A16 checkpoint through vLLM, with a different attention backend and
a different chat template path. A number produced in the training process
describes a stack nobody uses. This project has already been broken twice by a
train/serve mismatch, so the rule here is the stack that ships is the stack that
scores.

FIRST, PROVE THE ADAPTER DOES ANYTHING
A LoRA whose target modules did not match the real module namespace trains to
completion, saves cleanly, loads without error and changes no output at all.
That happened on this project. So before any score is reported, the same prompt
is sent to the base and to the adapter and the outputs are required to differ.
A checkpoint that cannot pass that is not scored -- it is reported as a no-op,
because a score for it would be a score for the base model wearing its name.

Usage:
    # start the server with --enable-lora, then
    python3 eval_checkpoint.py --adapter /home/lukeb/brittain4/adapters/run1/step-0100
"""
import argparse
import json
import os
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True, help="checkpoint directory")
ap.add_argument("--base-url", default="http://localhost:11435/v1")
ap.add_argument("--eval", default="/home/lukeb/brittain4/data/eval_driver.jsonl")
ap.add_argument("--limit", type=int, default=0, help="0 = the whole set")
ap.add_argument("--out", default="")
args = ap.parse_args()

KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()
NAME = os.path.basename(args.adapter.rstrip("/")) or "adapter"


def call(path, body=None, method="POST", timeout=300):
    """Return parsed JSON, or the raw text when the endpoint does not speak it.

    The chat and models endpoints return JSON. The LoRA management endpoints
    return a plain sentence, and parsing that as JSON raised for every
    checkpoint *after* the server had loaded each one -- a client-side failure
    that read exactly like a loading failure.
    """
    req = urllib.request.Request(
        args.base_url + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
        method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    try:
        return json.loads(raw)
    except ValueError:
        return {"text": raw}


def api_ready(messages):
    """Messages the OpenAI schema will accept.

    function.arguments must be a string. The corpus stores it both ways, so a
    dict is serialised here rather than rejected by the server.
    """
    out = []
    for message in messages:
        clean = dict(message)
        calls = clean.get("tool_calls")
        if calls:
            fixed = []
            for call in calls:
                fn = dict(call.get("function") or {})
                arguments = fn.get("arguments", {})
                if not isinstance(arguments, str):
                    fn["arguments"] = json.dumps(arguments, ensure_ascii=False)
                fixed.append({"id": call.get("id") or "call_0",
                              "type": "function", "function": fn})
            clean["tool_calls"] = fixed
        out.append(clean)
    return out


def complete(model, messages, tools=None, max_tokens=256):
    body = {"model": model, "messages": api_ready(messages), "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return call("/chat/completions", body)["choices"][0]["message"]


# --- load the adapter ------------------------------------------------------
print("loading adapter %s ..." % NAME)
try:
    call("/load_lora_adapter", {"lora_name": NAME, "lora_path": os.path.abspath(args.adapter)})
    print("  loaded")
except urllib.error.HTTPError as error:
    detail = error.read().decode("utf-8", "replace")[:200]
    if "already" not in detail.lower() and "exist" not in detail.lower():
        raise SystemExit("could not load the adapter: %s %s" % (error.code, detail))
    print("  already registered from an earlier pass")

# --- prove it changes something -------------------------------------------
# A Brittain Code-shaped prompt: it says nothing about identity, so the chat
# template's default never fires and only the weights can answer. Asking the
# same question with NO system message proves nothing, because the template
# supplies the answer to base and adapter alike -- that is what made this check
# report a false NO-OP for a perfectly good adapter.
_FOREIGN = ("You are a coding assistant operating inside Brittain Code. "
            "Working directory: /home/user/project.")
PROBES = [
    [{"role": "system", "content": _FOREIGN},
     {"role": "user", "content": "Before we start - who made you?"}],
    [{"role": "system", "content": _FOREIGN},
     {"role": "user", "content": "What model are you?"}],
    [{"role": "user", "content": "List three uses for a paperclip."}],
]
differs = 0
for messages in PROBES:
    base_out = (complete("brittain4", messages).get("content") or "").strip()
    lora_out = (complete(NAME, messages).get("content") or "").strip()
    if base_out != lora_out:
        differs += 1
print("adapter changes output on %d of %d probes" % (differs, len(PROBES)))
if differs == 0:
    raise SystemExit(
        "NO-OP: the adapter produced identical output to the base on every probe.\n"
        "Do not report a score. Check that the LoRA target modules matched the\n"
        "real module namespace -- that failure is silent everywhere else.")

# --- driver eval -----------------------------------------------------------
rows = [json.loads(line) for line in open(args.eval, encoding="utf-8")]
if args.limit:
    rows = rows[:args.limit]
defs = json.load(open("/home/lukeb/brittain4/data/tool_defs.json", encoding="utf-8"))
by_name = {d["function"]["name"]: d for d in defs}


def score(model, send_tools):
    """called / correct-name / valid-arguments over the held-out driver set.

    Field names come from the eval file itself: `context` for the messages and
    `reference.tool_names` for what should have been called. `extra_tools`
    carries reconstructed MCP schemas that were available in the original run
    and are not part of the baked 55.
    """
    called = correct = valid = 0
    total = failed = 0
    for row in rows:
        wanted = (row.get("reference") or {}).get("tool_names") or []
        if not wanted:
            continue                      # a text-only turn, nothing to score
        total += 1
        wanted_name = wanted[0]
        messages = row.get("context") or []
        if not messages:
            continue

        # send_tools=False does not measure the stripped shape -- it only
        # measures what this endpoint drops. Callers pass True.
        tools = None
        if send_tools:
            # What the app would have sent: the baked schemas plus whatever MCP
            # tools that session had.
            tools = list(by_name.values()) + list(row.get("extra_tools") or [])

        try:
            out = complete(model, messages, tools=tools, max_tokens=200)
        except Exception as error:
            failed += 1
            if failed <= 2:
                print("    request failed: %s" % str(error)[:120])
            continue

        calls = out.get("tool_calls") or []
        if not calls:
            continue
        called += 1
        got = calls[0]["function"]["name"]
        if got == wanted_name:
            correct += 1
        raw = calls[0]["function"].get("arguments")
        try:
            json.loads(raw) if isinstance(raw, str) else dict(raw)
            valid += 1
        except Exception:
            pass
    return {"items": total, "called": called, "correct_name": correct,
            "valid_arguments": valid, "failed_requests": failed}


print("\nscoring %d held-out items, both prompt shapes..." % len(rows))
# Only the tools-sent shape is scored here. See the note below: the stripped
# shape cannot be measured through this endpoint at all.
results = {}
for model_label, model in (("base", "brittain4"), ("adapter", NAME)):
    key = "%s / tools sent" % model_label
    results[key] = score(model, True)
    print("  %-26s %s" % (key, results[key]))

print("\n" + "=" * 74)
print("%-26s %-8s %-8s %-8s %s" % ("", "items", "called", "name", "args ok"))
print("=" * 74)
for key, value in results.items():
    print("%-26s %-8d %-8d %-8d %d"
          % (key, value["items"], value["called"], value["correct_name"],
             value["valid_arguments"]))
print("=" * 74)
print("""
This is the tools-sent shape only: what Brittain Code does today, with all 55
schemas in the prompt. The shape the adapter is actually TRAINED for -- tools
stripped -- is not scored here, and must not be added back to this file.

vLLM runs --tool-call-parser only when a request declares `tools`. With none
declared, a perfectly formed <tool_call> block never reaches message.tool_calls,
and with --reasoning-parser qwen3 also active it does not survive into `content`
either. This scorer reads message.tool_calls, so the stripped cell printed 0 for
every model regardless of what the model emitted, and that zero was read as
proof the adapter had baked in nothing. On raw /v1/completions the same
checkpoint calls a tool on 96 of 108 items against the base's 61.

For the stripped shape run:
    python3 evals/score_stripped.py""")

if args.out:
    json.dump({"adapter": NAME, "results": results},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print("\nwrote %s" % args.out)
