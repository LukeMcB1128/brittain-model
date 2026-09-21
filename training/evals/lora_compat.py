"""Does an adapter trained on Qwen3_5ForCausalLM load onto the served checkpoint?

Training runs against AutoModelForCausalLM, which resolves to
Qwen3_5ForCausalLM: modules are model.layers.N.*, so PEFT writes keys like
  base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight

The serving checkpoint declares Qwen3_5ForConditionalGeneration, whose modules
live at model.language_model.layers.N.*. If vLLM cannot reconcile those two
namings, the adapter either errors or -- far worse -- loads as a no-op and we
ship a finetune that does nothing.

The dummy adapter has non-zero random weights precisely so a no-op is
detectable: applying it MUST change the output. Identical output with and
without the adapter is a silent failure, not a pass.

Everything runs under a __main__ guard: vLLM spawns its engine in a child
process, and module-level work re-executes on import in the child.
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Intel/Qwen3.5-9B-int4-AutoRound")
    parser.add_argument("--adapter", default="/home/lukeb/brittain4/adapters/dummy-r8")
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        attention_backend="TRITON_ATTN",
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
    )

    prompts = ["The capital of France is", "def fibonacci(n):"]
    sp = SamplingParams(max_tokens=24, temperature=0.0)

    base = [o.outputs[0].text for o in llm.generate(prompts, sp)]
    print("BASE:", json.dumps(base))

    result = {"loaded": False, "changed": None, "error": None}
    try:
        req = LoRARequest("dummy", 1, args.adapter)
        withl = [o.outputs[0].text for o in llm.generate(prompts, sp, lora_request=req)]
        result["loaded"] = True
        print("LORA:", json.dumps(withl))
        result["changed"] = withl != base
    except Exception as e:  # noqa: BLE001
        result["error"] = "%s: %s" % (type(e).__name__, str(e)[:400])
        print("LOAD FAILED:", result["error"])

    if not result["loaded"]:
        result["verdict"] = "REJECTED - adapter did not load"
    elif result["changed"]:
        result["verdict"] = "APPLIED - adapter loaded and changed the output"
    else:
        result["verdict"] = "SILENT NO-OP - loaded but output identical"

    print("COMPAT " + json.dumps(result))


if __name__ == "__main__":
    main()
