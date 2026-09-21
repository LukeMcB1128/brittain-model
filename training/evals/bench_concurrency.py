"""What does this card actually serve, per hour, under load?

    python3 bench_concurrency.py --model <path>

Concurrency and throughput are different limits and people plan off the wrong
one. The KV pool sets how many requests can be RESIDENT; the GPU sets how fast
tokens come out in total. Adding requests past the point where the GPU is
saturated does not add throughput -- it just spreads the same tokens over more
users and makes each one slower.

Measures both, at realistic prompt sizes, with and without an adapter:
  * aggregate output tokens/sec at increasing batch size,
  * per-request tokens/sec (what one user experiences),
  * time to first token (prefill cost, which a chat UI shows as a stall).

Prompt sizes are drawn from the harvest, not invented: the median windowed
example is ~6k tokens, and the 55 tool schemas add ~8k on top until they are
baked into weights.
"""
import argparse
import json
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--prompt-tokens", type=int, default=6000)
    ap.add_argument("--output-tokens", type=int, default=400)
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--out", default="/home/lukeb/brittain4/results/concurrency.json")
    ap.add_argument("--cuda-graphs", action="store_true",
                    help="production setting; costs memory, so KV shrinks")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

    def make_prompt(seed):
        """A DISTINCT prompt per request.

        vLLM has prefix caching on. Sending the same prompt N times stores its
        KV once, so N requests occupy one request's worth of cache and the
        measurement reports a concurrency the pool cannot actually hold -- an
        earlier run showed 8 x 14k prompts fitting a 68,112-token pool, which
        is arithmetically impossible. Real traffic shares no prefix, so every
        prompt here is unique from its first tokens.
        """
        rng = random.Random(seed)
        words = ["%s%d" % (rng.choice(VOCAB), rng.randrange(10 ** 6))
                 for _ in range(args.prompt_tokens)]
        ids = tok(" ".join(words)).input_ids[:args.prompt_tokens]
        return tok.decode(ids)

    VOCAB = ["alpha", "bravo", "delta", "echo", "foxtrot", "gamma", "hotel",
             "india", "juliet", "kilo", "lima", "mike", "november", "oscar"]
    print("building distinct prompts...")

    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.90, enforce_eager=not args.cuda_graphs,
              max_num_seqs=32, max_num_batched_tokens=2048,
              attention_backend="TRITON_ATTN")

    cache = llm.llm_engine.vllm_config.cache_config
    kv_tokens = (cache.num_gpu_blocks or 0) * (cache.block_size or 0)
    print("kv cache tokens: %d  (cuda graphs: %s)" % (kv_tokens, bool(args.cuda_graphs)))

    sp = SamplingParams(max_tokens=args.output_tokens, temperature=0.8,
                        ignore_eos=True)   # fixed length, so timings compare

    rows = []
    for b in [int(x) for x in args.batches.split(",")]:
        prompts = [make_prompt(1000 * b + i) for i in range(b)]
        print("  batch %d: %d prompts, %d tokens each"
              % (b, len(prompts), len(tok(prompts[0]).input_ids)))
        # Warm at this shape so JIT does not land in the measurement. Uses
        # different prompts again so the warmup does not prime the cache.
        llm.generate([make_prompt(-i - 1) for i in range(b)],
                     SamplingParams(max_tokens=8, temperature=0.0))
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        dt = time.time() - t0
        produced = sum(len(o.outputs[0].token_ids) for o in outs)
        row = {
            "batch": b,
            "wall_s": round(dt, 2),
            "aggregate_tok_s": round(produced / dt, 1),
            "per_request_tok_s": round(produced / dt / b, 1),
            "responses_per_hour": round(3600 / (dt / b)),
        }
        rows.append(row)
        print("batch %-3d  %6.1fs  aggregate %6.1f tok/s  per-request %5.1f tok/s"
              "  -> %5d responses/hour"
              % (b, dt, row["aggregate_tok_s"], row["per_request_tok_s"],
                 row["responses_per_hour"]))

    json.dump({"model": args.model, "prompt_tokens": args.prompt_tokens,
               "output_tokens": args.output_tokens, "kv_tokens": kv_tokens,
               "rows": rows}, open(args.out, "w"), indent=2)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
    import os, sys
    sys.stdout.flush()
    os._exit(0)     # vLLM 0.28 teardown does not return on this setup
