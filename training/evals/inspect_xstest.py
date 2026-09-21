"""How does XSTest label safe vs unsafe prompts?

Over-refusal is only meaningful measured against both sides. A model that
refuses nothing scores perfectly on the safe half and is useless as a safety
signal; one that refuses everything scores perfectly on the unsafe half. The
eval needs both, reported separately.
"""
import collections

from datasets import load_dataset

ds = load_dataset("natolambert/xstest-v2-copy", split="gpt4")
print("n =", len(ds))
print("columns:", ds.column_names)
print()

types = collections.Counter(ds["type"])
print("type distribution:")
for t, n in sorted(types.items()):
    print("   %-34s %3d" % (t, n))

print()
labels = collections.Counter(ds["final_label"])
print("final_label:", dict(labels))

print()
print("cross-tab type x label:")
cross = collections.Counter((r["type"], r["final_label"]) for r in ds)
for (t, l), n in sorted(cross.items()):
    print("   %-34s %-22s %3d" % (t, l, n))

print("\nsamples:")
seen = set()
for r in ds:
    if r["type"] in seen:
        continue
    seen.add(r["type"])
    print("   [%s] %s" % (r["type"], r["prompt"][:95]))
