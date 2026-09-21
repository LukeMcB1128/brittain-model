"""QLoRA step at a fixed sequence length, optionally with a chunked loss.

Two things learned the hard way and encoded here:

  * HF guards gradient checkpointing with `self.training`, so model.train() is
    mandatory -- without it the flag reports enabled, does nothing, and a
    512-token step costs 20.5 GB instead of 9.7.
  * On WSL2 the Windows driver oversubscribes into system RAM instead of
    raising OOM, so a run can report success while crawling over PCIe. The
    reliable tell is throughput collapse (~20 tok/s vs ~177), not the peak
    number, because the caching allocator also over-reserves near the limit.

With a 248,320-token vocabulary the logits dominate the per-token cost, so
--chunk-ce recomputes them per chunk in backward instead of holding the whole
sequence's worth at once.
"""
import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "Qwen/Qwen3.5-9B"
TARGETS = (
    r"^model\.layers\.\d+\."
    r"(self_attn\.(q|k|v|o)_proj"
    r"|linear_attn\.(in_proj_(a|b|qkv|z)|out_proj)"
    r"|mlp\.(gate|up|down)_proj)$"
)


def gb(n):
    return round(n / 1024**3, 2)


def chunked_ce(backbone, head, ids, labels, chunk):
    """Cross entropy over sequence chunks, logits recomputed in backward.

    Peak becomes one chunk's logits rather than the whole sequence's, which at
    248k vocab is the difference between 0.24 GB and several GB.
    """
    hidden = backbone(input_ids=ids).last_hidden_state[:, :-1]
    target = labels[:, 1:]
    total = torch.zeros((), device=hidden.device, dtype=torch.float32)

    def one(h, t):
        logits = head(h)
        return F.cross_entropy(
            logits.float().view(-1, logits.shape[-1]), t.reshape(-1), reduction="sum"
        )

    for i in range(0, hidden.shape[1], chunk):
        total = total + torch.utils.checkpoint.checkpoint(
            one, hidden[:, i : i + chunk], target[:, i : i + chunk], use_reentrant=False
        )
    return total / target.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--chunk-ce", type=int, default=0,
                    help="chunk size for the loss; 0 uses the model's own loss")
    args = ap.parse_args()

    out = {"seq_len": args.seq_len, "batch_size": args.batch_size,
           "rank": args.rank, "chunk_ce": args.chunk_ce}

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
    model = get_peft_model(
        model,
        LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                   bias="none", task_type="CAUSAL_LM", target_modules=TARGETS),
    )
    model.train()

    out["vram_weights"] = gb(torch.cuda.memory_allocated())
    out["lora_params_m"] = round(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    vocab = getattr(AutoConfig.from_pretrained(MODEL), "text_config").vocab_size

    backbone, head = model.get_decoder(), model.get_output_embeddings()
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, vocab, (args.batch_size, args.seq_len), device="cuda")

    step_s = []
    for _ in range(args.steps):
        t = time.time()
        if args.chunk_ce:
            loss = chunked_ce(backbone, head, ids, ids, args.chunk_ce)
        else:
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_s.append(time.time() - t)

    fastest = min(step_s)
    out.update(
        ok=True,
        loss=round(float(loss), 3),
        step_s=round(fastest, 2),
        tokens_per_s=round(args.batch_size * args.seq_len / fastest),
        peak_vram=gb(torch.cuda.max_memory_allocated()),
        peak_reserved=gb(torch.cuda.max_memory_reserved()),
    )
    # Spilling into system RAM shows up as a throughput collapse, not an error.
    out["spilled"] = out["tokens_per_s"] < 60
    print(json.dumps(out))


if __name__ == "__main__":
    try:
        main()
    except torch.OutOfMemoryError:
        print(json.dumps({"ok": False, "error": "OOM"}))
        sys.exit(0)
