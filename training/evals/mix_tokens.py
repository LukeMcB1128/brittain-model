# -*- coding: utf-8 -*-
"""Answer "25% of what?" with the real tokenizer, and audit the one flagged row.

Examples and tokens give very different mixes here: a trajectory example is a
whole agent step with its accumulated context, while a Tulu example is usually
one question and one answer. Counting by example would call 2,000 general rows
a 65% share; counting by tokens may well call the same 2,000 rows a tenth of
the data. The training loss is over tokens, so tokens are what the model
actually experiences.
"""
import json

from transformers import AutoTokenizer

MODEL = "/home/lukeb/brittain4/models/brittain4-base-w4a16"
tok = AutoTokenizer.from_pretrained(MODEL)
D = "/home/lukeb/brittain4/data"


def count(path, render):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    total = 0
    for row in rows:
        total += len(tok(render(row), add_special_tokens=False).input_ids)
    return len(rows), total


def from_messages(row):
    return "\n".join(str(m.get("content") or "") for m in row["messages"])


def from_trajectory(row):
    parts = [str(m.get("content") or "") for m in row.get("context") or []]
    target = row.get("target") or {}
    parts.append(str(target.get("content") or ""))
    for call in target.get("tool_calls") or []:
        parts.append(json.dumps(call))
    return "\n".join(parts)


print("counting with the served tokenizer...\n")
sets = [
    ("trajectories (train)", "trajectories_train.jsonl", from_trajectory),
    ("identity", "identity_sft.jsonl", from_messages),
    ("general (tulu 2k)", "general_sft.jsonl", from_messages),
]
results = []
for label, name, render in sets:
    n, tokens = count("%s/%s" % (D, name), render)
    results.append((label, n, tokens))

ex_total = sum(n for _, n, _ in results)
tok_total = sum(t for _, _, t in results)

print("=" * 78)
print("%-24s %-10s %-8s %-14s %-8s %s"
      % ("set", "examples", "by ex.", "tokens", "by tok.", "mean tokens/ex"))
print("=" * 78)
for label, n, tokens in results:
    print("%-24s %-10d %-8.0f%% %-14s %-8.0f%% %d"
          % (label, n, 100.0 * n / ex_total, "{:,}".format(tokens),
             100.0 * tokens / tok_total, tokens / max(1, n)))
print("=" * 78)
print("totals: %d examples, %s tokens" % (ex_total, "{:,}".format(tok_total)))

print("")
print("NOTE: raw tokens overstate the trajectory side. Training windows at 4,096,")
print("so an example longer than that contributes 4,096 tokens, not its full length.")
print("")
print("=" * 78)
print("%-24s %-10s %-16s %s" % ("set", "examples", "capped tokens", "share of what is trained"))
print("=" * 78)
capped = []
for (label, name, render), (_, n, _) in zip(sets, results):
    rows = [json.loads(line) for line in open("%s/%s" % (D, name), encoding="utf-8")]
    total = 0
    for row in rows:
        total += min(len(tok(render(row), add_special_tokens=False).input_ids), 4096)
    capped.append((label, n, total))
cap_total = sum(t for _, _, t in capped)
for label, n, total in capped:
    print("%-24s %-10d %-16s %.0f%%" % (label, n, "{:,}".format(total), 100.0 * total / cap_total))
print("=" * 78)

# How many general examples to hit a token-share target?
general_mean = capped[2][2] / max(1, capped[2][1])
other_tokens = capped[0][2] + capped[1][2]
print("to reach a target share of TRAINED tokens:")
for target in (0.10, 0.15, 0.25):
    # share = g*mean / (g*mean + other)  ->  g = other*share / (mean*(1-share))
    need = other_tokens * target / (general_mean * (1 - target))
    print("  %2.0f%% of TOKENS needs about %5d general examples" % (target * 100, need))

print("\n--- the row flagged as mentioning tulu/ai2 ---")
for row in (json.loads(l) for l in open(D + "/general_sft.jsonl", encoding="utf-8")):
    blob = json.dumps(row["messages"]).lower()
    if "tulu" in blob or "ai2" in blob:
        print("source: %s" % row["source"])
        for m in row["messages"]:
            text = " ".join(str(m["content"]).split())
            print("  %-9s %s" % (m["role"], text[:300]))
        break
