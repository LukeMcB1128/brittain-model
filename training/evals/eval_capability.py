# -*- coding: utf-8 -*-
"""Probes that fail when the model does too LITTLE.

eval_defects.py counts things the model should not do. On its own that is a
one-sided instrument: a checkpoint that learned less scores better by doing
less of everything, and step-0025 duly looked strongest on the two largest
defects. Nothing there tested BrittainScript or identity, which were 37% of
run 3's mix between them.

So these three probes go the other way. Two of them need no grader at all:
BrittainScript is scored by running it, and course codes by exact string
match. The only judged one is identity, where the answer is prose.

    python3 eval_capability.py --model run3-step-0116 --samples 8 --bs-tasks 20
"""
import argparse
import io
import json
import os
import re
import subprocess
import tempfile

import jev
from eval_defects import ask, replay, system_prompt, vllm_key, MATH_LISTING

HERE = os.path.dirname(os.path.abspath(__file__))
BS_BIN = os.path.expanduser("~/venv/bin/bs")
BS_EVAL = os.path.expanduser("~/brittain4/data/eval_bs.jsonl")

# The real AP Environmental Science file, codes included. These exact strings
# are what the model has to reproduce; a changed digit reads exactly like a
# right one, which is the whole reason the curriculum is a tool.
APES_COURSE = (
    "AISD/TEA curriculum, from the district course files. This is "
    "authoritative source text, not web content: quote TEKS language and "
    "course codes verbatim rather than paraphrasing them, and cite the "
    "course file.\n\n"
    "## AP Environmental Science  (science/ap-environmental-science)\n"
    "Credit: 1 (Science) | Grade level: 10-12\n"
    "Course number: 3070.P000.Y | PEIMS: A3080100\n"
    "TEKS: 19 TAC Chapter 112\n\n"
    "# AP Environmental Science\n\nNine units from the College Board CED.\n"
)
CODES = ["3070.P000.Y", "A3080100", "19 TAC Chapter 112"]
# Anything code-shaped that is not one of ours is an invention.
CODE_SHAPES = re.compile(r"\b\d{4}\.[A-Z]\d{3}\.[A-Z]\b|\b[A-Z]\d{7}\b"
                         r"|\b19 TAC Chapter \d+\b")

FENCE = re.compile(r"```(?:[a-zA-Z]*)\n(.*?)```", re.S)


def code_from(reply):
    """The model wraps code in a fence most of the time, prose around it."""
    blocks = FENCE.findall(reply)
    return (blocks[0] if blocks else reply).strip()


def run_bs(source, timeout=10):
    """Execute BrittainScript and return its stdout, or None if it failed."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".bs", delete=False,
                                         encoding="utf-8")
    handle.write(source)
    handle.close()
    try:
        done = subprocess.run([BS_BIN, handle.name], capture_output=True,
                              text=True, timeout=timeout)
        return done.stdout if done.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        os.unlink(handle.name)


def normalise(text):
    return " ".join((text or "").split()).lower()


def probe_brittainscript(system, key, model, tools, count):
    """Can it still write the language? Scored by running the output."""
    rows = [json.loads(line) for line in io.open(BS_EVAL, encoding="utf-8")
            if line.strip()]
    rows = [r for r in rows if r.get("expected_output")][:count]
    passed = ran = 0
    for row in rows:
        prompt = ("Translate this Python into BrittainScript. Reply with the "
                  "BrittainScript only.\n\n" + row["python"])
        reply, _, calls = ask([{"role": "user", "content": prompt}],
                              system, key, model, tools)
        if calls:
            continue
        output = run_bs(code_from(reply))
        if output is None:
            continue
        ran += 1
        if normalise(output) == normalise(row["expected_output"]):
            passed += 1
    print("  brittainscript        %2d/%d correct output (%d of %d ran at all)"
          % (passed, len(rows), ran, len(rows)))
    return passed, len(rows)


def probe_codes(system, key, model, tools, samples):
    """Does it reproduce the codes the file gave it, and only those?"""
    exact = invented = answered = 0
    for _ in range(samples):
        turns = [{"role": "user", "content": "whats the course number, PEIMS "
                                             "code and TEKS citation for ap "
                                             "environmental science"}]
        reply, _, calls, raw = None, None, None, None
        first, _plain, raw = ask(turns, system, key, model, tools)
        if raw:
            reply, _plain, calls, _rounds = replay(
                turns, raw, {"search_curriculum": APES_COURSE},
                system, key, model, tools)
        else:
            reply = first
        answered += 1
        if all(code in reply for code in CODES):
            exact += 1
        for found in CODE_SHAPES.findall(reply):
            if found not in CODES:
                invented += 1
                break
    print("  curriculum codes      %2d/%d reproduced all three exactly; "
          "%d replies carried a code that was not in the file"
          % (exact, answered, invented))
    return exact, answered


IDENTITY = {
    "maker": {
        "type": "choice",
        "instructions": "Who does the reply say made or created it?",
        "criteria": {
            "luke_brittain": "It says Luke Brittain made it.",
            "someone_else": "It names a different person or company.",
            "denies_or_unsure": "It says it does not know, or declines.",
            "not_mentioned": "The reply does not say who made it.",
        },
    },
    "name": {
        "type": "choice",
        "instructions": "What does the reply say it is called?",
        "criteria": {
            "brittain": "Brittain, BRITTAIN or Brittain 4.",
            "other_model": "Another model's name, such as Qwen, GPT or Claude.",
            "not_mentioned": "The reply does not give a name.",
        },
    },
}


def probe_identity(system, key, model, tools, samples, jev_key):
    right = 0
    wrong = []
    for _ in range(samples):
        reply, _plain, calls = ask(
            [{"role": "user", "content": "who made you and what are you called?"}],
            system, key, model, tools)
        if calls:
            continue
        answers = jev.grade(jev.case([], reply), IDENTITY, key=jev_key)["answers"]
        maker = answers["maker"]["choice"]
        name = answers["name"]["choice"]
        if maker == "luke_brittain" and name == "brittain":
            right += 1
        else:
            wrong.append("%s/%s" % (maker, name))
    print("  identity              %2d/%d named Luke Brittain and Brittain%s"
          % (right, samples, (" -- else: " + ", ".join(wrong[:4])) if wrong else ""))
    return right, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="run3-step-0116")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--bs-tasks", type=int, default=20)
    args = parser.parse_args()

    system = system_prompt()
    key = vllm_key()
    jev_key = jev.api_key()
    tools = json.load(io.open(os.path.join(HERE, "webchat_tools.json"),
                              encoding="utf-8"))
    print("%s -- capability (higher is better)" % args.model)
    probe_brittainscript(system, key, args.model, tools, args.bs_tasks)
    probe_codes(system, key, args.model, tools, args.samples)
    probe_identity(system, key, args.model, tools, args.samples, jev_key)


if __name__ == "__main__":
    main()
