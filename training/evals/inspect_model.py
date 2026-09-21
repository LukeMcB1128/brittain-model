"""Summarise the Qwen3.5-9B checkpoint: parameter budget and module names.

Reads the safetensors index only -- no weights are loaded, so this is instant
and costs no VRAM.
"""
import collections
import glob
import json
import re

snap = glob.glob(
    "/home/lukeb/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*/"
)[0]
idx = json.load(open(snap + "model.safetensors.index.json"))
print("checkpoint bytes: %.2f GB" % (idx["metadata"]["total_size"] / 1e9))

names = list(idx["weight_map"])
print("tensor count:", len(names))

# Collapse layer indices so the repeated block shows up once.
shapes = collections.Counter(re.sub(r"\.\d+\.", ".N.", n) for n in names)
print("\n--- distinct module names (count = how many layers have it) ---")
for name, n in sorted(shapes.items()):
    print("%4d  %s" % (n, name))

cfg = json.load(open(snap + "config.json"))
t = cfg["text_config"]
layer_types = collections.Counter(t["layer_types"])
print("\n--- layer mix ---")
for k, v in layer_types.items():
    print("%4d  %s" % (v, k))

v, h = t["vocab_size"], t["hidden_size"]
print("\n--- parameter budget (bf16) ---")
print("vocab x hidden embed : %.2fB  (%.1f GB)" % (v * h / 1e9, v * h * 2 / 1e9))
print("untied lm_head       : %.2fB  (%.1f GB)" % (v * h / 1e9, v * h * 2 / 1e9))
print("tie_word_embeddings  :", cfg["tie_word_embeddings"])

# KV cache is charged only by the full-attention layers; the linear-attention
# layers hold a fixed-size recurrent state instead, independent of length.
full = layer_types["full_attention"]
per_token = full * t["num_key_value_heads"] * t["head_dim"] * 2 * 2
print("\n--- KV cache, bf16 ---")
print("full-attention layers:", full, "of", t["num_hidden_layers"])
print("bytes/token          : %d" % per_token)
for ctx in (8192, 16384, 32768, 131072):
    print("  %6d ctx -> %.2f GB" % (ctx, per_token * ctx / 1e9))
