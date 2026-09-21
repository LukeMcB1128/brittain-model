"""Can PEFT target the multimodal class directly, so no key remap is needed?

Serving loads Qwen3_5ForConditionalGeneration. Training against
AutoModelForCausalLM produced model.layers.N.* keys that vLLM silently ignored.
Training against the multimodal class instead should produce
model.language_model.layers.N.* natively -- one source of truth rather than a
rename step that can be forgotten.

Two things to confirm:
  1. the regex selects exactly the language-model projections, and
  2. it excludes the vision tower and the mtp head, which must not be adapted.

Meta device: no weights, no VRAM, safe to run beside the quantization.
"""
import collections
import re

import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig

TARGETS = (
    r"^model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj"
    r"|linear_attn\.(in_proj_(a|b|qkv|z)|out_proj)"
    r"|mlp\.(gate|up|down)_proj)$"
)

cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B")

loaded = None
for auto_name in ("AutoModelForImageTextToText", "AutoModelForConditionalGeneration",
                  "AutoModelForVision2Seq"):
    try:
        import transformers
        auto = getattr(transformers, auto_name, None)
        if auto is None:
            continue
        with init_empty_weights():
            model = auto.from_config(cfg)
        loaded = auto_name
        break
    except Exception as e:  # noqa: BLE001
        print("%-38s failed: %s" % (auto_name, type(e).__name__))

if loaded is None:
    raise SystemExit("no Auto class produced the multimodal model")

print("loaded via:", loaded, "->", type(model).__name__)
print()

linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
print("total nn.Linear:", len(linears))

rx = re.compile(TARGETS)
matched = [n for n in linears if rx.match(n)]
print("matched by regex:", len(matched))

groups = collections.Counter(re.sub(r"\.\d+\.", ".N.", n) for n in matched)
for name, n in sorted(groups.items()):
    print("   %4d  %s" % (n, name))

print()
unmatched = collections.Counter(
    re.sub(r"\.\d+\.", ".N.", n) for n in linears if not rx.match(n))
print("NOT matched (must include the vision tower, mtp and lm_head):")
for name, n in sorted(unmatched.items()):
    print("   %4d  %s" % (n, name))

leaked = [n for n in matched if "visual" in n or n.startswith("mtp")]
print()
print("vision/mtp leaked into targets:", leaked or "none")
