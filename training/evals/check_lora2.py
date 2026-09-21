"""LoRA support via the attribute (the Protocol has non-method members, so
issubclass() raises), plus on-disk size of each quantized candidate.

Size is the decisive number: QuantTrio's AWQ leaves visual, linear_attn,
self_attn, layer 0 and mtp in bf16 and quantizes only the MLPs, which may not
leave room for a KV cache on a 12 GB card.
"""
from vllm.model_executor.models.registry import ModelRegistry

print("=== vLLM LoRA support ===")
for arch in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration",
             "Qwen3ForCausalLM", "Qwen3NextForCausalLM"):
    cls = ModelRegistry._try_load_model_cls(arch)
    if cls is None:
        print("%-38s class not loadable" % arch)
        continue
    print("%-38s supports_lora=%-6s supports_multimodal=%s"
          % (arch, getattr(cls, "supports_lora", False),
             getattr(cls, "supports_multimodal", False)))
    for attr in ("packed_modules_mapping", "embedding_modules", "lora_skip_prefixes"):
        val = getattr(cls, attr, None)
        if val:
            print("      %-24s %s" % (attr, str(val)[:220]))

print()
print("=== on-disk size of quantized candidates ===")
from huggingface_hub import HfApi

api = HfApi()
for repo in ("Qwen/Qwen3.5-9B", "QuantTrio/Qwen3.5-9B-AWQ",
             "kaitchup/Qwen3.5-9B-autoround-W4A16",
             "cyankiwi/Qwen3.5-9B-AWQ-4bit",
             "Intel/Qwen3.5-9B-int4-AutoRound"):
    try:
        info = api.model_info(repo, files_metadata=True)
        total = sum(f.size or 0 for f in info.siblings if f.rfilename.endswith(".safetensors"))
        print("%-38s %6.2f GB" % (repo, total / 1e9))
    except Exception as e:  # noqa: BLE001
        print("%-38s ERROR %s" % (repo, type(e).__name__))
