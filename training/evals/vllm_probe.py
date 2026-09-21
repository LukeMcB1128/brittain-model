"""Measure what vLLM actually costs to serve one quantized base at a context length.

"VRAM at 8k/16k/32k" is not directly observable: vLLM preallocates a fixed
fraction of the card (gpu_memory_utilization) regardless of max_model_len, then
divides what is left after weights into KV blocks. So the numbers that mean
something for concurrency are:

  * weights on the GPU (fixed),
  * KV cache bytes actually available,
  * how many tokens that buys -- i.e. how many concurrent requests of a given
    length can be in flight.

Only 8 of the 32 layers keep a KV cache; the other 24 are linear attention with
a fixed-size recurrent state, so context length costs far less here than on a
normal dense transformer.

Prints one JSON line.
"""
import argparse
import json
import subprocess
import sys
import time


def smi():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--enable-lora", action="store_true")
    ap.add_argument("--kv-cache-dtype", default="auto",
                    help="fp8 halves KV bytes/token; head_dim is 256 here so KV is 32 KB/token in bf16")
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--attention-backend", default="",
                    help="TRITON_ATTN avoids FlashInfer, whose fp8 prefill kernel "
                         "will not compile for sm_86. It is an EngineArgs field in "
                         "0.28, not the VLLM_ATTENTION_BACKEND env var, which is ignored.")
    ap.add_argument("--max-batched-tokens", type=int, default=0,
                    help="chunked prefill cap; a small value shrinks the activation peak, "
                         "which is where 2.2 GiB of an 11 GiB budget was going")
    ap.add_argument("--text-only", action="store_true",
                    help="force the text architecture, skipping the vision tower")
    args = ap.parse_args()

    out = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "gpu_util": args.gpu_util,
        "lora": args.enable_lora,
        "text_only": args.text_only,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_num_seqs": args.max_num_seqs,
        "max_batched_tokens": args.max_batched_tokens,
        "attention_backend": args.attention_backend or "auto",
        "vram_used_before_mb": smi(),
    }

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=True,          # CUDA graphs add memory that muddies the reading
        max_num_seqs=args.max_num_seqs,
        kv_cache_dtype=args.kv_cache_dtype,
    )
    if args.max_batched_tokens:
        kwargs["max_num_batched_tokens"] = args.max_batched_tokens
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if args.enable_lora:
        kwargs.update(enable_lora=True, max_lora_rank=32)
    if args.text_only:
        kwargs["hf_overrides"] = {"architectures": ["Qwen3_5ForCausalLM"]}

    t0 = time.time()
    llm = LLM(**kwargs)
    out["startup_s"] = round(time.time() - t0, 1)
    out["vram_used_after_mb"] = smi()

    # Dig the cache numbers out wherever this vLLM version keeps them.
    blocks = block_size = None
    for path in ("llm_engine.cache_config", "llm_engine.vllm_config.cache_config"):
        obj = llm
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            blocks = getattr(obj, "num_gpu_blocks", None) or blocks
            block_size = getattr(obj, "block_size", None) or block_size
        except AttributeError:
            continue
    out["num_gpu_blocks"] = blocks
    out["block_size"] = block_size
    if blocks and block_size:
        out["kv_cache_tokens"] = blocks * block_size
        out["concurrent_at_max_len"] = round(blocks * block_size / args.max_model_len, 2)

    # A real generation, to prove the config serves rather than merely loads.
    t0 = time.time()
    res = llm.generate(
        ["Write a haiku about compilers."] * 4,
        SamplingParams(max_tokens=64, temperature=0.0),
    )
    dt = time.time() - t0
    out["gen_ok"] = all(len(r.outputs[0].token_ids) > 0 for r in res)
    out["gen_tok_s"] = round(sum(len(r.outputs[0].token_ids) for r in res) / dt, 1)
    out["peak_vram_used_mb"] = smi()
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print("RESULT " + json.dumps({"ok": False, "error": type(e).__name__, "detail": str(e)[:400]}))
        sys.exit(0)
