# -*- coding: utf-8 -*-
"""How many trajectory examples would have their tool block truncated away?

train_lora.py keeps the END of an over-long example: `full_ids = full_ids[drop:]`.
That was right when the prompt was only conversation, because the recent turns
are what the next step depends on. It is wrong now. The chat template emits the
tool block FIRST, before any message, so front-truncation eats the tool
declarations -- and it eats them partially, leaving a fragment of JSON at the
start of the sequence.

Training on a half-eaten <tools> block would teach exactly nothing useful about
tools, while looking like a mix that simply did not help.

This counts, at the trainer's real window, how many trajectory examples exceed
it and how much of the tool block survives.
"""
import json

from transformers import AutoTokenizer

MIX = "/home/lukeb/brittain4/data/train_mix.jsonl"
tok = AutoTokenizer.from_pretrained(
    "/home/lukeb/brittain4/models/brittain4-base-w4a16", trust_remote_code=True)
tok.chat_template = open("/home/lukeb/brittain4/chat_template_brittain4.jinja",
                         encoding="utf-8").read()


def normalize(calls):
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
traj = [r for r in rows if r["kind"] == "trajectory" and r.get("tools")]
print("trajectory rows declaring tools: %d" % len(traj))

for window in (2048, 4096, 8192):
    over = intact = gutted = unusable = 0
    for row in traj[:300]:
        messages = []
        ok = True
        for message in row["messages"]:
            clean = dict(message)
            if clean.get("tool_calls"):
                fixed = normalize(clean["tool_calls"])
                if fixed is None:
                    ok = False
                    break
                clean["tool_calls"] = fixed
            messages.append(clean)
        if not ok or not messages:
            continue
        target = row["target"]
        assistant = {"role": "assistant", "content": target.get("content") or ""}
        if target.get("tool_calls"):
            fixed = normalize(target["tool_calls"])
            if fixed is None:
                continue
            assistant["tool_calls"] = fixed
        kw = dict(tokenize=False, tools=row["tools"],
                  chat_template_kwargs={"enable_thinking": False})
        try:
            prompt = tok.apply_chat_template(messages, add_generation_prompt=True, **kw)
            full = tok.apply_chat_template(messages + [assistant], **kw)
        except Exception:
            continue
        if not full.startswith(prompt):
            continue
        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        full_ids = tok(full, add_special_tokens=False).input_ids
        # Where the tool block ends, in tokens.
        marker = prompt.find("</tools>")
        tools_end = len(tok(prompt[:marker], add_special_tokens=False).input_ids) if marker >= 0 else 0

        if len(full_ids) <= window:
            intact += 1
            continue
        over += 1
        drop = len(full_ids) - window
        if len(prompt_ids) - drop < 16:
            unusable += 1
        elif drop >= tools_end:
            gutted += 1          # the entire tool block is cut away
        else:
            gutted += 1          # partially cut: a JSON fragment survives
    sample = min(len(traj), 300)
    print("\nwindow %d  (sample of %d)" % (window, sample))
    print("  fits whole, tool block intact : %d" % intact)
    print("  over the window               : %d" % over)
    print("    tool block cut or mangled   : %d" % gutted)
    print("    example unusable            : %d" % unusable)
