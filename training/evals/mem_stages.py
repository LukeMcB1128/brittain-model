"""Attribute the memory: load -> forward(no loss) -> forward(loss) -> backward.

Peak was 20.5 GB at 512 tokens with gradient checkpointing confirmed active,
which none of the obvious suspects explains. Measure each stage instead of
arguing about it.
"""
import argparse
import gc

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "Qwen/Qwen3.5-9B"
TARGETS = (
    r"^model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj"
    r"|linear_attn\.(in_proj_(a|b|qkv|z)|out_proj)"
    r"|mlp\.(gate|up|down)_proj)$"
)


def g(n=None):
    return round((torch.cuda.memory_allocated() if n is None else n) / 1024**3, 2)


def peak():
    return round(torch.cuda.max_memory_allocated() / 1024**3, 2)


def stage(label):
    print("%-34s cur=%6.2f GB   peak=%6.2f GB" % (label, g(), peak()))
    torch.cuda.reset_peak_memory_stats()


ap = argparse.ArgumentParser()
ap.add_argument("--seq-len", type=int, default=512)
ap.add_argument("--lora", action="store_true")
args = ap.parse_args()

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0}
)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
if args.lora:
    model = get_peft_model(
        model, LoraConfig(r=32, lora_alpha=64, task_type="CAUSAL_LM", target_modules=TARGETS)
    )
# HF guards checkpointing with `if self.gradient_checkpointing and self.training`,
# so without train() the checkpoint branch is skipped and every activation is kept.
model.train()
torch.cuda.reset_peak_memory_stats()
stage("after load" + (" + lora" if args.lora else ""))

ids = torch.randint(0, 248320, (1, args.seq_len), device="cuda")
backbone = model.get_decoder()  # PEFT wraps model.model, so ask for it by role
print("backbone:", type(backbone).__name__)

# 1. Hidden states only -- no lm_head, no logits, no loss.
with torch.no_grad():
    h = backbone(input_ids=ids).last_hidden_state
stage("no_grad backbone only")
del h
gc.collect()
torch.cuda.empty_cache()

# 2. Backbone WITH autograd graph retained.
h = backbone(input_ids=ids).last_hidden_state
stage("backbone (graph retained)")
print("   hidden shape", tuple(h.shape), h.dtype)

# 3. lm_head on top of it -> the 248k-wide logits.
head = model.get_output_embeddings()
logits = head(h)
stage("+ lm_head (logits)")
print("   logits shape", tuple(logits.shape), logits.dtype,
      "= %.2f GB" % (logits.numel() * logits.element_size() / 1024**3))

# 4. Cross entropy.
loss = torch.nn.functional.cross_entropy(
    logits.float().view(-1, logits.shape[-1]), ids.view(-1)
)
stage("+ cross_entropy (fp32)")

del logits
gc.collect()
loss.backward()
stage("+ backward")
