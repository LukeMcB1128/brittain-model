"""Two gates for the serving plan, checked before any 6 GB download.

1. Does the vLLM model class for Qwen3.5 declare LoRA support? If not, the
   "keep it a LoRA so vLLM applies it per-request" requirement is unmet and
   the whole approach needs rethinking.
2. What exactly is in the candidate AWQ repo -- which architecture, which
   quant config, and is the vision tower quantized too?
"""
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsMultiModal
from vllm.model_executor.models.registry import ModelRegistry

print("=== vLLM LoRA support ===")
for arch in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration", "Qwen3ForCausalLM"):
    try:
        cls = ModelRegistry._try_load_model_cls(arch)
        if cls is None:
            print("%-38s could not load class" % arch)
            continue
        print("%-38s LoRA=%-5s  MultiModal=%s"
              % (arch, issubclass(cls, SupportsLoRA), issubclass(cls, SupportsMultiModal)))
        packed = getattr(cls, "packed_modules_mapping", None)
        if packed:
            print("      packed_modules_mapping:", packed)
    except Exception as e:  # noqa: BLE001
        print("%-38s ERROR %s: %s" % (arch, type(e).__name__, e))

print()
print("=== candidate quantized bases ===")
import json

from huggingface_hub import hf_hub_download, list_repo_files

for repo in ("QuantTrio/Qwen3.5-9B-AWQ", "kaitchup/Qwen3.5-9B-autoround-W4A16"):
    print("---", repo)
    try:
        files = list(list_repo_files(repo))
        weights = [f for f in files if f.endswith(".safetensors")]
        print("   safetensors shards:", len(weights))
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        print("   architectures:", cfg.get("architectures"))
        qc = cfg.get("quantization_config", {})
        print("   quant:", {k: v for k, v in qc.items() if k != "modules_to_not_convert"})
        skip = qc.get("modules_to_not_convert") or qc.get("ignore")
        if skip:
            s = str(skip)
            print("   not quantized:", s[:300])
    except Exception as e:  # noqa: BLE001
        print("   ERROR", type(e).__name__, e)
