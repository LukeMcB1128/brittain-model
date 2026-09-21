"""Save an untrained LoRA adapter so vLLM can be asked to load it.

The point is the key names, not the weights. Training happens against
Qwen3_5ForCausalLM, whose modules are model.layers.N.*, while the checkpoint
declares Qwen3_5ForConditionalGeneration, whose modules are
model.language_model.layers.N.*. If vLLM cannot reconcile those, the adapter
silently fails to apply at serve time -- the exact class of train/serve
mismatch that has burned this project before, so it gets tested now rather
than after a training run.

CPU/meta only: no GPU, safe to run beside a measurement.
"""
import argparse
import json

import torch
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

TARGETS = (
    r"^model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj"
    r"|linear_attn\.(in_proj_(a|b|qkv|z)|out_proj)"
    r"|mlp\.(gate|up|down)_proj)$"
)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/home/lukeb/brittain4/adapters/dummy-r8")
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--scale", type=float, default=0.02,
                help="stddev of the adapter weights. At 0.02 the perturbation is "
                     "small enough that greedy decoding can be unchanged even when "
                     "the adapter IS applied, which makes a no-op indistinguishable "
                     "from a pass. Use a large value to force a visible effect.")
args = ap.parse_args()

cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(cfg)

peft_model = get_peft_model(
    model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2,
                      task_type="CAUSAL_LM", target_modules=TARGETS)
)

# Materialise the adapter tensors; the frozen base stays on meta and is not saved.
state = {}
for name, param in peft_model.named_parameters():
    if "lora_" in name:
        state[name] = torch.zeros(param.shape, dtype=torch.bfloat16)
        # A zero adapter is a no-op; make it non-zero so a silent failure to
        # apply is distinguishable from a correctly applied adapter. The
        # magnitude has to be big enough to move greedy decoding, or an
        # unchanged output proves nothing.
        state[name].normal_(0, args.scale)

peft_model.save_pretrained(args.out, state_dict=state, safe_serialization=True)
print("saved to", args.out)

keys = sorted(state)
print("adapter tensors:", len(keys))
print("first 3 keys:")
for k in keys[:3]:
    print("   ", k)
print("adapter_config target_modules:")
print("   ", json.load(open(args.out + "/adapter_config.json"))["target_modules"])
