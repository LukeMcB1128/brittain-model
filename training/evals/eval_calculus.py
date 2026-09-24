# -*- coding: utf-8 -*-
"""Does thinking finish, and is it right, on hard calculus?

A web chat user gave run5b a hard calculus problem twice. Both times it
thought its whole 4096-token budget away, returned nothing until told to
finish, and then gave a wrong answer. This separates the two failures:

  finishes   did any answer arrive before the token limit
  correct    is the final answer equal to the reference

across the settings that could be responsible: the production budget, a
larger one, the base with the same budget (is the adapter hurting a
reasoning path it was never trained on?), and thinking off.

Every reference was worked by hand. Jev grades the final answer AGAINST THE
SUPPLIED REFERENCE -- it is never asked whether an answer is true -- and is
calibrated on right, wrong and equivalent-form answers before it grades.

    python3 eval_calculus.py
"""
import concurrent.futures
import json
import os
import sys
import time
import urllib.request

import jev
from eval_defects import system_prompt, vllm_key

URL = "http://localhost:11435/v1/chat/completions"

# (problem, reference). Each reference checked by hand.
PROBLEMS = [
    ("Evaluate the integral from 0 to 1 of x^2 e^x dx. Give an exact answer.",
     "e - 2 (about 0.7183)"),
    ("Find the limit as x approaches 0 of (sin x - x) / x^3.",
     "-1/6"),
    ("Find the derivative of x^x and evaluate it at x = 2.",
     "4(ln 2 + 1), about 6.7726"),
    ("A 10 ft ladder leans against a wall. Its base slides away from the wall at "
     "2 ft/s. How fast is the top sliding down when the base is 6 ft from the wall?",
     "1.5 ft/s (the top moves down at 3/2 ft/s)"),
    ("Find the area of the region between y = x^2 and y = 2x - x^2.",
     "1/3"),
    ("Evaluate the integral from 0 to pi/4 of sec^3(x) dx. Give an exact answer.",
     "(1/2)(sqrt(2) + ln(1 + sqrt(2))), about 1.1478"),
    ("What is the coefficient of x^5 in the Maclaurin series of sin(x)cos(x)?",
     "2/15"),
    ("Find the volume when the region under y = sqrt(x) from x = 0 to 4 is rotated "
     "about the x-axis.",
     "8 pi, about 25.13"),
]

RIGHT = {
    "matches": {
        "type": "noul",
        "instructions": "Compare the reply's FINAL answer with the reference. Is it "
                        "mathematically equal to the reference? Equivalent forms "
                        "count as equal (e.g. 3/2 and 1.5, or e - 2 and 0.718). If "
                        "the reply gives no final answer, it is not equal.",
        "criteria": {"true": "The final answer equals the reference.",
                     "false": "The final answer differs, or there is none."},
    },
}
CALIBRATION = [
    ("-1/6", "Using the Taylor series, sin x - x = -x^3/6 + ..., so the limit is -1/6.", True),
    ("-1/6", "Applying L'Hopital three times gives -1/3.", False),
    ("1.5 ft/s (the top moves down at 3/2 ft/s)", "dy/dt = -3/2, so the top slides down at three halves of a foot per second.", True),
    ("e - 2 (about 0.7183)", "Integrating by parts twice gives e - 1.", False),
    ("2/15", "I would need to expand both series first.", False),
]


def ask(model, problem, think, max_tokens, key, sampling=None):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0.7,
            "frequency_penalty": 0.3, "chat_template_kwargs": {"enable_thinking": think},
            "messages": [{"role": "system", "content": system_prompt()},
                         {"role": "user", "content": problem}]}
    body.update(sampling or {})
    started = time.time()
    request = urllib.request.Request(URL, json.dumps(body).encode(),
                                     {"Authorization": "Bearer " + key,
                                      "Content-Type": "application/json"})
    data = json.load(urllib.request.urlopen(request, timeout=1800))
    choice = data["choices"][0]
    return {"content": choice["message"].get("content") or "",
            "finish": choice.get("finish_reason"),
            "tokens": data["usage"]["completion_tokens"],
            "seconds": time.time() - started}


def main():
    jev_key = jev.api_key()
    for reference, reply, want in CALIBRATION:
        p = jev.grade({"reference": reference, "reply": reply}, RIGHT, key=jev_key)["answers"]["matches"]["noul"]
        if (p >= 0.5) != want:
            raise SystemExit("grader failed calibration on %r (p=%.2f)" % (reply[:50], p))
    print("grader calibrated: %d of %d" % (len(CALIBRATION), len(CALIBRATION)))

    key = vllm_key()
    arms = [
        ("run5b, think on, 4096 (production)", "run5b-step-0124", True, 4096),
        ("run5b, think on, 12288", "run5b-step-0124", True, 12288),
        ("base, think on, 12288", "brittain4", True, 12288),
        ("run5b, think off, 4096", "run5b-step-0124", False, 4096),
    ]
    for name, model, think, budget in arms:
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            runs = list(pool.map(lambda pr: ask(model, pr[0], think, budget, key), PROBLEMS))
        finished = right = 0
        for (problem, reference), run in zip(PROBLEMS, runs):
            done = run["finish"] != "length" and run["content"].strip()
            finished += bool(done)
            if done:
                p = jev.grade({"reference": reference, "reply": run["content"][-1500:]},
                              RIGHT, key=jev_key)["answers"]["matches"]["noul"]
                right += p >= 0.5
        tokens = sorted(r["tokens"] for r in runs)
        print("%-36s finished %d/8 | correct %d/8 | median %5d tokens, max %5d | median %4.0f s"
              % (name, finished, right, tokens[4], tokens[-1],
                 sorted(r["seconds"] for r in runs)[4]))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
