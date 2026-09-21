"""What does this vLLM build actually support for Qwen3.5, before we spend an hour on it."""
import json

from vllm.model_executor.models.registry import ModelRegistry

archs = sorted(ModelRegistry.get_supported_archs())
hits = [a for a in archs if "qwen3" in a.lower() or "Qwen3" in a]
print("vLLM-registered Qwen3-family architectures:")
for a in hits:
    print("  ", a)

for want in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"):
    print("%-38s registered: %s" % (want, want in archs))

print()
print("--- quantization methods ---")
try:
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    print(sorted(QUANTIZATION_METHODS))
except Exception as e:  # noqa: BLE001
    print("could not enumerate:", e)

print()
print("--- what the checkpoint declares ---")
import glob
cfg = json.load(open(glob.glob(
    "/home/lukeb/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*/config.json")[0]))
print("architectures:", cfg["architectures"])
