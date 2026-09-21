"""Print the Linear module paths of the loaded model, without loading weights."""
import collections
import re

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "Qwen/Qwen3.5-9B"

cfg = AutoConfig.from_pretrained(MODEL)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(cfg)

print("loaded class:", type(model).__name__)
print()

linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
print("nn.Linear modules:", len(linears))
shapes = collections.Counter(re.sub(r"\.\d+\.", ".N.", n) for n in linears)
for name, n in sorted(shapes.items()):
    print("%4d  %s" % (n, name))

print()
print("--- other parameterised leaf types ---")
others = collections.Counter(
    type(m).__name__
    for n, m in model.named_modules()
    if len(list(m.children())) == 0 and any(True for _ in m.parameters(recurse=False))
)
for k, v in sorted(others.items()):
    print("%4d  %s" % (v, k))
