"""What does AutoRound 0.15 actually expose for quantizing lm_head?"""
import inspect

from auto_round import AutoRound

sig = inspect.signature(AutoRound.__init__)
print("AutoRound.__init__ parameters:")
for name, p in sig.parameters.items():
    if name == "self":
        continue
    default = "" if p.default is inspect.Parameter.empty else " = %r" % (p.default,)
    print("   %-28s%s" % (name, default))

print()
print("methods:", [m for m in dir(AutoRound) if not m.startswith("_")])

doc = (AutoRound.__init__.__doc__ or AutoRound.__doc__ or "")
for line in doc.splitlines():
    if "lm_head" in line or "layer_config" in line:
        print("DOC:", line.strip())
