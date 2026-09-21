"""AutoRound 0.15 dispatches through *args/**kwargs; find the real entry point.

Goal: quantize lm_head (2.03 GiB bf16) alongside the transformer layers, which
no published Qwen3.5-9B checkpoint does.
"""
import inspect

import auto_round

print("auto_round version:", getattr(auto_round, "__version__", "?"))
print("top-level exports:", [n for n in dir(auto_round) if not n.startswith("_")][:40])

print()
for cls_name in ("AutoRoundLLM", "AutoRoundMLLM", "AutoRoundConfig", "BaseCompressor"):
    cls = getattr(auto_round, cls_name, None)
    if cls is None:
        continue
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        continue
    interesting = [n for n in sig.parameters
                   if any(k in n for k in ("lm_head", "layer_config", "bits", "group",
                                           "iters", "scheme", "device", "batch", "seqlen",
                                           "nsamples", "low_gpu", "sym"))]
    print("%s:" % cls_name)
    for n in interesting:
        p = sig.parameters[n]
        d = "" if p.default is inspect.Parameter.empty else " = %r" % (p.default,)
        print("    %-24s%s" % (n, d))
    print()

# The CLI is often the documented surface even when the API moves.
try:
    from auto_round.schemes import PRESET_SCHEMES
    print("preset schemes:", list(PRESET_SCHEMES)[:12])
except Exception as e:  # noqa: BLE001
    print("no PRESET_SCHEMES:", type(e).__name__)
