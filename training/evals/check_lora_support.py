"""Does vLLM support per-request LoRA for Qwen3.5, and on a quantized base?

This is the load-bearing check for the whole project: if vLLM cannot apply a
LoRA to this architecture at serve time, "keep it a LoRA so vLLM applies it
per-request" is not achievable and the plan needs rethinking before any
training happens.
"""
from vllm.model_executor.models.registry import ModelRegistry

for arch in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration",
             "Qwen3ForCausalLM", "Qwen3NextForCausalLM"):
    try:
        info = ModelRegistry.inspect_model_cls(arch)
        # vLLM returns a lazily-inspected info object; field names have moved
        # between releases, so report whatever it exposes.
        fields = {
            k: getattr(info, k)
            for k in dir(info)
            if k.startswith("supports_") or k in ("is_multimodal", "is_pooling_model")
        }
        print(arch)
        for k, v in sorted(fields.items()):
            print("    %-28s %s" % (k, v))
    except Exception as e:  # noqa: BLE001
        print("%-38s ERROR %s: %s" % (arch, type(e).__name__, e))
    print()
