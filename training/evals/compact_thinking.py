# -*- coding: utf-8 -*-
"""Does thinking-on explain the /compact failure at Brittain Code's real budgets?

Reading the app's actual code changed the diagnosis. It does NOT send tools for
compaction (streamChat receives toolset=null), it chunks, it keeps a verbatim
tail, and it sizes max_tokens from the context. So the earlier budget analysis
described a problem the app does not have.

What it cannot do is turn thinking off. src/main/inference.js has two
transports: the Ollama one honours `think`, and the OpenAI-compatible one does
not even destructure it -- no `think`, no `chat_template_kwargs`. So every
compaction through the API runs with thinking on, and the trace is charged to
the same max_tokens as the summary.

Reproduced at the app's own numbers:
  retainedBudget(32768) = floor(32768 * 0.28)      = 9175
  tailBudget            = floor(9175 * 0.6)        = 5505
  summaryRoom           = 9175 - (pinned + tail)   ~ 3670
  minimumSummaryTokens  = clamp(source * 0.02, 120, 900)

The app then rejects a summary that is under `required` tokens or missing any
of the five headings, retries once, and accepts whatever the second attempt
gives. So the question is whether thinking-on pushes the summary under those
gates -- not whether the request errors.
"""
import glob
import json
import os
import re
import urllib.request

BASE = "http://localhost:11435/v1/chat/completions"
KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()

SECTIONS = ["GOAL", "CONSTRAINTS", "DECISIONS", "STATE", "NEXT"]
HINTS = {
    "GOAL": "the original objective, in the user's own terms where possible",
    "CONSTRAINTS": "user corrections, rejected approaches, and stated preferences",
    "DECISIONS": "what was chosen and why - the part that costs most to work out twice",
    "STATE": "where the work actually stands right now",
    "NEXT": "the concrete remaining steps",
}


def summary_instruction(tail_turns, minimum_tokens):
    """Port of compaction.js summaryInstruction()."""
    scope = ("Summarize the conversation above so work can continue in a fresh "
             "session. The %d most recent turns are being kept word for word "
             "and are not shown to you, so do not try to cover them - carry "
             "forward everything earlier that they would not reveal on their "
             "own." % tail_turns)
    lines = [scope, "", "Use exactly these five headings, in this order:"]
    lines += ["%s: %s" % (name, HINTS[name]) for name in SECTIONS]
    lines += ["", "Write at least %d tokens. Detail that is expensive to "
                  "rediscover is worth the room." % minimum_tokens, ""]
    lines += ["Do not continue the work, call tools, or ask questions. "
              "Output only the summary."]
    return "\n".join(lines)


def section_present(text, name):
    """Port of compaction.js sectionPresent()."""
    pattern = (r"^\s*(?:#{1,6}\s*)?(?:\*\*)?%s(?:\*\*)?\s*[:\-—]?\s*$"
               r"|^\s*(?:#{1,6}\s*)?(?:\*\*)?%s(?:\*\*)?\s*[:\-—]\s*\S"
               % (name, name))
    return re.search(pattern, text, re.I | re.M) is not None


def estimate_tokens(value):
    """Port of estimateTokensDefault: JSON.stringify(value).length / 4."""
    return round(len(json.dumps(value)) / 4)


pool = []
for path in glob.glob("/home/lukeb/brittain4/data/chats/**/*.json", recursive=True):
    if os.path.basename(path) == "index.json":
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    for m in data.get("conversation") or []:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"].strip():
            pool.append({"role": m["role"], "content": m["content"]})
    if len(pool) > 300:
        break


def build_input(target_tokens):
    msgs, total, want = [], 0, target_tokens * 4
    cap = max(2000, want // 14)
    for m in pool:
        if total >= want:
            break
        chunk = m["content"][:min(cap, want - total)]
        if not chunk.strip():
            continue
        msgs.append({"role": "user" if len(msgs) % 2 == 0 else "assistant",
                     "content": chunk})
        total += len(chunk)
    return msgs


SUMMARY_ROOM = 3670          # what the app computes for a 32k context
TAIL_TURNS = 6

print("=" * 94)
print("%-22s %-9s %-9s %-8s %-11s %s"
      % ("configuration", "content", "reason", "finish", "headings", "app verdict"))
print("=" * 94)

for source_size in (12000, 20000):
    conv = build_input(source_size)
    source_tokens = estimate_tokens(conv)
    required = min(900, max(120, round(source_tokens * 0.02)))
    msgs = conv + [{"role": "user",
                    "content": summary_instruction(TAIL_TURNS, required)}]
    print("\nsource ~%d tokens, app requires >= %d summary tokens"
          % (source_tokens, required))

    for label, thinking in (("thinking ON (today)", True),
                            ("thinking OFF (fix)", False)):
        body = {"model": "brittain4", "messages": msgs, "stream": False,
                "temperature": 0.2, "max_tokens": SUMMARY_ROOM}
        if not thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(
            BASE, data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + KEY,
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.load(r)
        except Exception as error:
            print("  %-22s ERROR %s" % (label, str(error)[:60]))
            continue
        choice = data["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        reasoning = choice["message"].get("reasoning_content") or ""
        tokens = estimate_tokens(content)
        missing = [s for s in SECTIONS if not section_present(content, s)]
        if not content:
            verdict = "REJECT (empty)"
        elif tokens < required:
            verdict = "REJECT (too short: %d < %d)" % (tokens, required)
        elif missing:
            verdict = "accept, unstructured (missing %s)" % ",".join(missing)
        else:
            verdict = "ACCEPT (structured)"
        print("  %-22s %-9d %-9d %-8s %-11s %s"
              % (label, len(content), len(reasoning),
                 choice.get("finish_reason"),
                 "%d/5" % (5 - len(missing)), verdict))
print("=" * 94)
