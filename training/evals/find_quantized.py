"""Look for an already-quantized Qwen3.5-9B so we do not spend hours making one.

Ampere (sm_86) has no native fp8, so an -FP8 release is useless here even
though Qwen usually publishes one. AWQ / GPTQ / int4 compressed-tensors are
the ones that matter.
"""
from huggingface_hub import HfApi

api = HfApi()
seen = {}
for query in ("Qwen3.5-9B", "Qwen3.5"):
    for m in api.list_models(search=query, limit=200):
        seen[m.id] = m

interesting = []
for mid, m in seen.items():
    low = mid.lower()
    if "9b" not in low:
        continue
    if any(k in low for k in ("awq", "gptq", "int4", "int8", "fp8", "w4a16", "quant", "gguf", "bnb", "nvfp4", "mxfp4")):
        interesting.append((mid, getattr(m, "downloads", 0) or 0))

print("--- quantized Qwen3.5-9B candidates ---")
for mid, dl in sorted(interesting, key=lambda x: -x[1]):
    print("%9d  %s" % (dl, mid))
if not interesting:
    print("(none found)")

print()
print("--- all 9B repos seen ---")
for mid in sorted(m for m in seen if "9b" in m.lower()):
    print("  ", mid)
