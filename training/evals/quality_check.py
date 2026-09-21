"""Compare the new lm_head-quantized checkpoint against the published baseline.

The new checkpoint was made with --iters 0, i.e. round-to-nearest, and it
quantizes lm_head -- the layer everyone else deliberately leaves in bf16. Both
choices trade accuracy for memory, and neither has been measured. Serving it
without a number here would repeat exactly the mistake of trusting a
configuration because it fits.

Metric: mean negative log-likelihood over fixed passages (lower is better),
read from vLLM's prompt_logprobs so it is measured on the serving stack rather
than in a notebook. Plus greedy samples, because a perplexity that looks fine
can still hide broken generation.
"""
import argparse
import json
import math


PASSAGES = [
    "The mitochondrion is a double-membrane-bound organelle found in most "
    "eukaryotic organisms. Mitochondria generate most of the cell's supply of "
    "adenosine triphosphate, which is used as a source of chemical energy.",
    "def binary_search(arr, target):\n"
    "    low, high = 0, len(arr) - 1\n"
    "    while low <= high:\n"
    "        mid = (low + high) // 2\n"
    "        if arr[mid] == target:\n"
    "            return mid\n"
    "        elif arr[mid] < target:\n"
    "            low = mid + 1\n"
    "        else:\n"
    "            high = mid - 1\n"
    "    return -1\n",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon. "
    "Neil Armstrong and Buzz Aldrin spent about two and a quarter hours "
    "outside the spacecraft collecting lunar material to bring back to Earth.",
]

PROMPTS = [
    "Explain in one sentence why the sky is blue.",
    "Write a Python function that reverses a string.",
    "The three largest planets in the solar system are",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        attention_backend="TRITON_ATTN",
    )

    # prompt_logprobs=0 returns the logprob of each actual prompt token.
    scored = llm.generate(
        PASSAGES, SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    )
    nlls, ntok = [], 0
    for out in scored:
        for pos in out.prompt_logprobs or []:
            if not pos:
                continue          # the first token has no prediction
            lp = next(iter(pos.values())).logprob
            nlls.append(-lp)
            ntok += 1
    mean_nll = sum(nlls) / max(len(nlls), 1)

    gens = [o.outputs[0].text.strip()
            for o in llm.generate(PROMPTS, SamplingParams(max_tokens=40, temperature=0.0))]

    print("QUALITY " + json.dumps({
        "label": args.label,
        "mean_nll": round(mean_nll, 4),
        "perplexity": round(math.exp(mean_nll), 3),
        "scored_tokens": ntok,
        "samples": gens,
    }))


if __name__ == "__main__":
    main()
