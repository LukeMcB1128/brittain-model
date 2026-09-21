"""Render the patched template in all four system/tools combinations.

Checking this offline first: a template bug surfaces at serve time as garbled
prompts across every request, and the server takes ~90s to come up. Jinja
renders the same here as it does inside vLLM.
"""
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/home/lukeb/brittain4/models/brittain4-base-w4a16")
tpl = open("/home/lukeb/brittain4/chat_template_brittain4.jinja", encoding="utf-8").read()

defs = json.load(open("/home/lukeb/brittain4/data/tool_defs.json", encoding="utf-8"))
tools = defs[:2]          # two is enough to prove placement
CALLER = "You are Brittain Code, a coding agent. Working directory: /srv/app."

cases = [
    ("no system, no tools", [{"role": "user", "content": "hi"}], None),
    ("no system, WITH tools", [{"role": "user", "content": "hi"}], tools),
    ("caller system, no tools",
     [{"role": "system", "content": CALLER}, {"role": "user", "content": "hi"}], None),
    ("caller system, WITH tools",
     [{"role": "system", "content": CALLER}, {"role": "user", "content": "hi"}], tools),
]

for label, msgs, tl in cases:
    out = tok.apply_chat_template(msgs, tools=tl, tokenize=False,
                                  add_generation_prompt=True, chat_template=tpl)
    n = len(tok(out).input_ids)
    has_id = "BRITTAIN-4" in out
    has_caller = "Brittain Code" in out
    sys_blocks = out.count("<|im_start|>system")
    print("=" * 70)
    print("%s   [%d tokens, %d system block(s)]" % (label, n, sys_blocks))
    print("  identity present: %s     caller present: %s" % (has_id, has_caller))
    # show the system block only, elided in the middle so tool JSON doesn't bury it
    start = out.find("<|im_start|>system")
    end = out.find("<|im_end|>", start) if start >= 0 else -1
    blk = out[start:end] if start >= 0 else "(no system block at all)"
    print("  ---")
    print("  " + (blk if len(blk) < 700 else blk[:340] + "\n  [... %d chars ...]\n  " % (len(blk) - 640) + blk[-300:]).replace("\n", "\n  "))
print("=" * 70)
