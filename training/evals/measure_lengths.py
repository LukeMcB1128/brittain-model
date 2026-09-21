"""How many harvested examples actually fit the training ceiling?

Measured earlier: 4096 tokens is the longest sequence that trains on this card
(11.21 GB peak, 283 tok/s). The harvest has a median context of 35 messages and
a p90 of 235, so the question is not academic -- it decides whether the dataset
needs truncation, and how much of it survives untouched.

Uses the real Qwen3.5 tokenizer; a character estimate would be worthless at a
248,320-token vocabulary.
"""
import argparse
import json

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", required=True)
ap.add_argument("--ceiling", type=int, default=4096)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")


def render(msg):
    """Approximate the wire form: role, content, calls and results all cost tokens."""
    parts = [msg.get("role") or "", msg.get("content") or ""]
    if msg.get("thinking"):
        parts.append(msg["thinking"])
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        parts.append(fn.get("name") or "")
        parts.append(json.dumps(fn.get("arguments") or {}))
    if msg.get("tool_name"):
        parts.append(msg["tool_name"])
    return "\n".join(p for p in parts if p)


rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
print("examples: %d" % len(rows))

lengths, target_lengths = [], []
for r in rows:
    ctx = "\n".join(render(m) for m in r["context"])
    tgt = render({"role": "assistant", **r["target"]})
    lengths.append(len(tok(ctx + "\n" + tgt).input_ids))
    target_lengths.append(len(tok(tgt).input_ids))

lengths.sort()
target_lengths.sort()


def pct(xs, p):
    return xs[min(len(xs) - 1, int(len(xs) * p))]


print("\nfull example (context + target), tokens:")
for p in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
    print("   p%-3d %8d" % (p * 100, pct(lengths, p)))
print("   max  %8d" % lengths[-1])

print("\ntarget alone, tokens:")
for p in (0.50, 0.90, 0.99):
    print("   p%-3d %8d" % (p * 100, pct(target_lengths, p)))
print("   max  %8d" % target_lengths[-1])

for ceiling in (2048, 4096, 8192, 16384, 32768):
    fits = sum(1 for n in lengths if n <= ceiling)
    print("\nceiling %5d: %4d/%d examples fit (%.1f%%)"
          % (ceiling, fits, len(lengths), 100 * fits / len(lengths)))
    kept_tokens = sum(n for n in lengths if n <= ceiling)
    print("             %s tokens in the fitting subset" % f"{kept_tokens:,}")

# What a driver actually has to emit is short; it is the history that is long.
print("\ntargets over 1024 tokens: %d" % sum(1 for n in target_lengths if n > 1024))
