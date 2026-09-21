"""How does vLLM 0.28 let you pick the attention backend, and why did ninja fail?

VLLM_ATTENTION_BACKEND=TRITON_ATTN was ignored -- the engine still logged
"Using FLASHINFER attention backend" -- so either the variable was renamed or
selection moved into config.
"""
import inspect
import re

import vllm.envs as envs

print("=== env vars mentioning attention/backend ===")
for name in sorted(dir(envs)):
    if name.isupper() and re.search(r"ATTENTION|BACKEND", name):
        try:
            print("  %-42s = %r" % (name, getattr(envs, name)))
        except Exception as e:  # noqa: BLE001
            print("  %-42s = <%s>" % (name, type(e).__name__))

print()
print("=== does EngineArgs take an attention backend? ===")
from vllm.engine.arg_utils import EngineArgs

fields = [f for f in EngineArgs.__dataclass_fields__ if "attention" in f or "backend" in f]
print(" ", fields)

print()
print("=== valid backend names ===")
try:
    from vllm.attention.backends.registry import AttentionBackendEnum
    print(" ", [b.name for b in AttentionBackendEnum])
except Exception as e:  # noqa: BLE001
    print("  could not import registry:", type(e).__name__, e)
