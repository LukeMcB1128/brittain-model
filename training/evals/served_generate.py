# -*- coding: utf-8 -*-
"""Generate through the running vLLM server, for benching adapters.

run_general_eval.py and run_bs_eval.py load the model offline with
LLM(model=...). That was fine for the base, and it cannot score an adapter at
all: there is no LoRA path, and it wants 90% of a GPU the server is already
holding. So both gain --server NAME, which sends the SAME raw prompt string
they would have generated from to /v1/completions, at the same temperature
and token limit, under the served model or adapter NAME.

Completions, not chat completions: the prompt is already templated, and the
chat endpoint would template it a second time with a different template.
Sending the identical string is what keeps these numbers comparable with the
offline baselines in BASELINE.md -- and the base is re-run through this path
first, so if it does not land on its recorded score, the path is wrong and no
adapter number from it should be believed.

Returns objects shaped like vLLM's RequestOutput (o.outputs[0].text), so the
scoring code in both scripts is untouched.
"""
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request


class _Completion:
    def __init__(self, text):
        self.text = text


class _Output:
    def __init__(self, text):
        self.outputs = [_Completion(text)]


def served_generate(prompts, max_tokens, model,
                    base="http://localhost:11435/v1", workers=4):
    """Greedy completions for every prompt, in order.

    `max_tokens` is one int per prompt. Four workers matches the server's
    --max-num-seqs; more would only queue.
    """
    key = open(os.path.expanduser("~/.brittain4_key"), encoding="utf-8").read().strip()
    done = [0]

    def one(index):
        body = json.dumps({
            "model": model,
            "prompt": prompts[index],
            "max_tokens": max_tokens[index],
            "temperature": 0.0,
        }).encode("utf-8")
        delay = 2.0
        for attempt in range(5):
            request = urllib.request.Request(base + "/completions", body, {
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            })
            try:
                with urllib.request.urlopen(request, timeout=900) as response:
                    text = json.load(response)["choices"][0]["text"]
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 4:
                    raise
                time.sleep(delay)
                delay *= 2
        done[0] += 1
        if done[0] % 50 == 0:
            sys.stderr.write("  %d/%d generated\n" % (done[0], len(prompts)))
        return text

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        texts = list(pool.map(one, range(len(prompts))))
    return [_Output(text) for text in texts]
