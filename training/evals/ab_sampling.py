# -*- coding: utf-8 -*-
"""Is the frequency penalty wrecking long math, and does a thinking budget fix stalls?

frequency_penalty 0.3 went in to stop a short reply repeating a sentence, and
it did: 3 in 48 to 0 in 48. It was only measured on short replies. The penalty
accumulates over the whole generation, reasoning included, and long math
repeats x, =, brackets and digits hundreds of times -- past some point every
sensible next token is penalised and the model is pushed into rare words.
That fits what the calculus probes showed: "world peace achieved universal
peace", "attics attics roof roof", "ready ready ready".

So: the same eight hand-checked problems, twice each, under the production
sampling, Qwen's published thinking-mode sampling (temperature 0.6, top_p
0.95, top_k 20, no frequency penalty), that plus a 2048-token thinking budget
-- vLLM 0.28's thinking_token_budget, which ends the reasoning and moves on to
the answer -- and thinking off with and without the penalty.

    python3 ab_sampling.py
"""
import concurrent.futures
import sys

import jev
from eval_calculus import CALIBRATION, PROBLEMS, RIGHT, ask
from eval_defects import vllm_key


QWEN_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "frequency_penalty": 0.0}
QWEN_PLAIN = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "frequency_penalty": 0.0}
ARMS = [
    ("production: think, freq 0.3", True, {}),
    ("think, Qwen sampling, no penalty", True, QWEN_THINK),
    ("think, Qwen sampling, budget 2048", True, dict(QWEN_THINK, thinking_token_budget=2048)),
    ("no think, freq 0.3", False, {}),
    ("no think, Qwen sampling, no penalty", False, QWEN_PLAIN),
]
REPEATS = 2


def collapsed(text):
    """Prose tail cycling a small vocabulary; same test as site/server/collapse.js."""
    import re
    masks = [r"```[\s\S]*?(?:```|$)", r"`[^`\n]*`", r"\$\$[\s\S]*?(?:\$\$|$)", r"\$[^$\n]*\$",
             r"\\\[[\s\S]*?(?:\\\]|$)", r"\\\([\s\S]*?(?:\\\)|$)", r"\\[A-Za-z]+"]
    for pattern in masks:
        text = re.sub(pattern, lambda m: " " * len(m.group()), text)
    words = re.findall(r"[A-Za-z']+", text.lower())
    tail = words[-200:]
    return len(tail) >= 150 and len(set(tail)) / float(len(tail)) < 0.3


def main():
    jev_key = jev.api_key()
    for reference, reply, want in CALIBRATION:
        p = jev.grade({"reference": reference, "reply": reply}, RIGHT, key=jev_key)["answers"]["matches"]["noul"]
        if (p >= 0.5) != want:
            raise SystemExit("grader failed calibration")
    key = vllm_key()
    jobs = [pr for pr in PROBLEMS for _ in range(REPEATS)]
    total = len(jobs)
    print("%-36s %9s %8s %9s %10s %9s %8s" % ("", "finished", "right", "unsure", "collapsed", "median t", "median s"))
    for name, think, sampling in ARMS:
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            runs = list(pool.map(lambda pr: ask("run5b-step-0124", pr[0], think, 4096, key, sampling), jobs))
        finished = right = unsure = broken = 0
        for (problem, reference), run in zip(jobs, runs):
            text = run["content"].strip()
            broken += collapsed(text)
            if run["finish"] == "length" or not text:
                continue
            finished += 1
            p = jev.grade({"reference": reference, "reply": text[-1500:]}, RIGHT,
                          key=jev_key)["answers"]["matches"]["noul"]
            right += p >= jev.DECIDED_HIGH
            unsure += jev.DECIDED_LOW < p < jev.DECIDED_HIGH
        tokens = sorted(r["tokens"] for r in runs)
        seconds = sorted(r["seconds"] for r in runs)
        print("%-36s %6d/%d %5d/%d %6d/%d %7d/%d %9d %8.0f"
              % (name, finished, total, right, total, unsure, total, broken, total,
                 tokens[total // 2], seconds[total // 2]))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
