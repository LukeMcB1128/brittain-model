"""Find a Qwen3.5-9B checkpoint that quantizes lm_head, before building one.

lm_head is 248,320 x 4,096 = 1.02B params = 2.03 GiB in bf16, and every
checkpoint checked so far leaves it -- plus embed_tokens and the vision tower
-- unquantized. That is ~4.9 GiB of the 8.11 GiB the weights occupy.

Quantizing lm_head alone should free ~1.5 GiB, which is more than enough to
reach 32k. Downloading one beats spending two hours quantizing one.
"""
import json

from huggingface_hub import hf_hub_download, list_repo_files

CANDIDATES = [
    "Intel/Qwen3.5-9B-int4-AutoRound",
    "kaitchup/Qwen3.5-9B-autoround-W4A16",
    "cyankiwi/Qwen3.5-9B-AWQ-4bit",
    "cyankiwi/Qwen3.5-9B-AWQ-BF16-INT4",
    "apolo13x/Qwen3.5-9B-quantized.w4a16",
    "Vishva007/Qwen3.5-9B-W4A16-AutoRound-GPTQ",
    "Vishva007/Qwen3.5-9B-W4A16-AutoRound",
    "QuantTrio/Qwen3.5-9B-AWQ",
    "OpenVINO/Qwen3.5-9B-int8-ov",
]


def probe(repo):
    files = set(list_repo_files(repo))
    cfg = json.load(open(hf_hub_download(repo, "config.json")))
    qc = cfg.get("quantization_config", {}) or {}
    method = qc.get("quant_method", "?")

    # Each toolchain names its exclusions differently.
    skip = (qc.get("modules_to_not_convert") or qc.get("ignore")
            or qc.get("modules_in_block_to_quantize") or [])
    extra = {k: qc[k] for k in ("quant_lm_head", "lm_head", "quantize_lm_head") if k in qc}

    lm_head_skipped = any("lm_head" in str(s) for s in skip)
    verdict = "lm_head QUANTIZED" if (not lm_head_skipped and extra.get("quant_lm_head")) else \
              "lm_head skipped" if lm_head_skipped else "unclear"

    print("%-45s %-12s %s" % (repo, method, verdict))
    if skip:
        print("      not converted: %s" % str(skip)[:150])
    if extra:
        print("      flags: %s" % extra)
    # The index tells the truth regardless of what the config claims.
    idx = [f for f in files if f.endswith("index.json")]
    if idx:
        wm = json.load(open(hf_hub_download(repo, idx[0])))["weight_map"]
        head = sorted(k for k in wm if "lm_head" in k)
        print("      lm_head tensors: %s" % (head[:4] or "none"))


for repo in CANDIDATES:
    try:
        probe(repo)
    except Exception as e:  # noqa: BLE001
        print("%-45s ERROR %s" % (repo, type(e).__name__))
