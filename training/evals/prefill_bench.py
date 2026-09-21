# -*- coding: utf-8 -*-
"""How long does a long prompt take before the first token comes back?

KV memory sets a hard ceiling on context, but it is not usually the binding
one. A context you can hold and cannot prefill in reasonable time is not a
context you can serve. This times prefill at several lengths with max_tokens=1,
so the measurement is prompt processing and nothing else.
"""
import json
import os
import time
import urllib.request

from transformers import AutoTokenizer

KEY = open(os.path.expanduser("~/.brittain4_key")).read().strip()
BASE = "http://localhost:11435/v1"
MODEL = "step-0100-mm"

tok = AutoTokenizer.from_pretrained(
    "/home/lukeb/brittain4/models/brittain4-base-w4a16", trust_remote_code=True)

# Ordinary prose, so the tokenizer behaves like it would on real input. A
# repeated single token would prefill unrealistically well.
SEED = (
    "The committee reviewed the quarterly figures and found that the northern "
    "division had outperformed expectations, though the margin was narrower "
    "than the previous year and several one-off costs remained unresolved. "
)
unit = tok(SEED, add_special_tokens=False).input_ids
print("seed is %d tokens" % len(unit))


def prompt_of(target):
    text = SEED * (target // len(unit) + 2)
    ids = tok(text, add_special_tokens=False).input_ids[:target]
    return tok.decode(ids)


def timed(prompt):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 1,
            "temperature": 0}
    req = urllib.request.Request(
        BASE + "/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"}, method="POST")
    start = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        payload = json.loads(r.read().decode("utf-8", "replace"))
    return time.time() - start, payload.get("usage", {}).get("prompt_tokens")


print("\n%-10s %10s %10s %12s" % ("target", "actual", "seconds", "tokens/s"))
print("-" * 46)
for target in (2048, 4096, 8192, 16384, 32000):
    prompt = prompt_of(target)
    # A second pass would hit the prefix cache and measure nothing, so each
    # length gets a distinct prefix.
    prompt = ("Report %d. " % target) + prompt
    try:
        seconds, counted = timed(prompt)
    except Exception as error:
        print("%-10d %10s %10s   %s" % (target, "-", "-", str(error)[:60]))
        continue
    rate = (counted or target) / seconds
    print("%-10d %10s %10.1f %12.0f" % (target, counted, seconds, rate))
