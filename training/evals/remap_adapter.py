# -*- coding: utf-8 -*-
"""Rename adapter modules into the namespace the served checkpoint uses.

Training resolved AutoModelForCausalLM -> Qwen3_5ForCausalLM, whose decoder sits
at model.layers.N. The served checkpoint is Qwen3_5ForConditionalGeneration,
whose decoder sits at model.language_model.layers.N. vLLM loaded the adapter,
matched zero modules, applied nothing, and answered every request from the base
weights -- indistinguishable from a working adapter unless you compare outputs.

The learned tensors are fine. Only the keys are wrong, so this rewrites them
and leaves the originals untouched.

    base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
 -> base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--suffix", default="-mm", help="suffix for the remapped copy")
args = ap.parse_args()

src = args.adapter.rstrip("/")
dst = src + args.suffix
if os.path.exists(dst):
    shutil.rmtree(dst)
os.makedirs(dst)

OLD = "base_model.model.model.layers."
NEW = "base_model.model.model.language_model.layers."

tensors = {}
renamed = untouched = 0
with safe_open(os.path.join(src, "adapter_model.safetensors"), framework="pt") as f:
    metadata = f.metadata() or {}
    for key in f.keys():
        tensor = f.get_tensor(key)
        if key.startswith(OLD):
            tensors[NEW + key[len(OLD):]] = tensor
            renamed += 1
        else:
            # Anything not under the decoder is copied as-is rather than
            # guessed at; a wrong rename is worse than a key vLLM ignores.
            tensors[key] = tensor
            untouched += 1

if renamed == 0:
    raise SystemExit("nothing matched %r -- the namespace is not what was expected" % OLD)

save_file(tensors, os.path.join(dst, "adapter_model.safetensors"), metadata=metadata)

for name in ("adapter_config.json", "README.md"):
    source = os.path.join(src, name)
    if os.path.exists(source):
        shutil.copy2(source, os.path.join(dst, name))

# peft records the module suffixes it targeted (q_proj, gate_proj ...). Those
# are suffix matches and stay valid across the rename, so the config needs no
# change -- but say so rather than leaving it unexplained.
config = json.load(open(os.path.join(dst, "adapter_config.json"), encoding="utf-8"))
print("remapped : %s" % os.path.basename(src))
print("  renamed keys   : %d" % renamed)
print("  copied as-is   : %d" % untouched)
print("  target_modules : %s (suffix matches, unaffected by the rename)"
      % ",".join(sorted(config.get("target_modules") or [])))
print("  written to     : %s" % dst)
