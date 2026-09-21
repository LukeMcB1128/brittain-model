"""One QLoRA training step at a fixed sequence length; report peak VRAM.

Run one process per sequence length. An OOM fragments the allocator badly
enough that the next measurement in the same process is not trustworthy, and
the whole point of this probe is a number worth designing a dataset around.

Prints a single JSON line on stdout so the runner can collect results.
"""
import argparse
import json
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "Qwen/Qwen3.5-9B"

# AutoModelForCausalLM resolves to Qwen3_5ForCausalLM, the text-only variant:
# the vision tower and the multi-token-prediction head in the checkpoint are
# never instantiated, so paths are model.layers.N.*, NOT the checkpoint's
# model.language_model.layers.N.*. That difference matters again at export
# time if the serving stack loads the ConditionalGeneration class instead.
TARGETS = (
    r"^model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj"
    r"|linear_attn\.(in_proj_(a|b|qkv|z)|out_proj)"
    r"|mlp\.(gate|up|down)_proj)$"
)


def gb(n):
    return round(n / 1024**3, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--no-gc", action="store_true", help="disable gradient checkpointing")
    args = ap.parse_args()

    out = {"seq_len": args.seq_len, "batch_size": args.batch_size, "rank": args.rank}
    torch.cuda.reset_peak_memory_stats()

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=quant, dtype=torch.bfloat16, device_map={"": 0}
    )
    out["load_s"] = round(time.time() - t0, 1)
    out["vram_after_load"] = gb(torch.cuda.memory_allocated())
    out["vram_reserved_after_load"] = gb(torch.cuda.memory_reserved())

    model.config.use_cache = False
    if not args.no_gc:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    out["grad_ckpt_requested"] = not args.no_gc
    out["grad_ckpt_active"] = bool(getattr(model, "is_gradient_checkpointing", False))
    # Layers can silently ignore the flag, so confirm on a decoder layer too.
    layer = model.model.layers[0]
    out["layer_gc_flag"] = bool(getattr(layer, "gradient_checkpointing", False))

    # Which gated-delta-rule implementation actually got bound: the Triton
    # kernel from the Hub, or the pure-torch fallback that retains ~63
    # intermediates per layer.
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq

    fn = mq.torch_chunk_gated_delta_rule
    out["delta_rule_impl"] = getattr(fn, "__module__", "?") + "." + getattr(fn, "__name__", "?")

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGETS,
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    out["lora_params_m"] = round(trainable / 1e6, 2)
    out["total_params_b"] = round(total / 1e9, 2)
    out["lora_modules"] = len(
        [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    )

    # Checkpointing is guarded by `self.training`; without this the flag reads
    # as enabled and does nothing, and a 512-token step costs 20.5 GB.
    model.train()
    out["training_mode"] = model.training

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    cfg = AutoConfig.from_pretrained(MODEL)
    vocab = getattr(cfg, "text_config", cfg).vocab_size
    out["vocab_size"] = vocab

    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, vocab, (args.batch_size, args.seq_len), device="cuda")
    labels = ids.clone()

    step_s = []
    for _ in range(args.steps):
        t = time.time()
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_s.append(time.time() - t)

    out["ok"] = True
    out["loss"] = round(float(loss), 3)
    out["step_s"] = round(min(step_s), 2)
    out["tokens_per_s"] = round(args.batch_size * args.seq_len / min(step_s))
    out["peak_vram"] = gb(torch.cuda.max_memory_allocated())
    out["peak_reserved"] = gb(torch.cuda.max_memory_reserved())

    # On WSL2 the Windows driver oversubscribes into system RAM rather than
    # raising OOM, so a run can "succeed" while thrashing over PCIe at 17
    # tok/s. Peak above physical VRAM is the tell -- treat it as a failure.
    physical = torch.cuda.get_device_properties(0).total_memory / 1024**3
    out["physical_vram"] = round(physical, 2)
    out["fits_in_vram"] = out["peak_reserved"] <= physical * 0.98
    print(json.dumps(out))


if __name__ == "__main__":
    try:
        main()
    except torch.OutOfMemoryError as e:
        print(json.dumps({"ok": False, "error": "OOM", "detail": str(e)[:200]}))
        sys.exit(0)
