# -*- coding: utf-8 -*-
"""A code eval with enough problems to attribute a regression.

WHY
Every adapter scores 62.5-75 on the 40 HumanEval problems in eval_general
against the base's 80.0, and two mixes thirty rows apart landed ten points
apart. At 40 problems one item is 2.5 points, so the set can say that code got
worse but not which part of the training mix did it.

This builds the full 164-problem HumanEval plus MBPP's 500-problem test split,
about 660 problems, in exactly the shape run_general_eval.py already scores:

  humaneval  identical prompt wording to build_general_eval.py, and the same
             `program + test + check(entry_point)` execution.
  mbpp       the task text plus its asserts in the prompt, as MBPP is normally
             posed. The asserts become the test and the function name they
             call becomes the entry point, so run_humaneval scores it
             unchanged -- a model that names the function differently fails,
             which the asserts would fail anyway.

    python3 build_code_eval.py --out ~/brittain4/data/eval_code.jsonl
"""
import argparse
import json
import os
import re

HUMANEVAL_ASK = ("Complete this Python function. Reply with the full function "
                 "in a single ```python code block, no explanation.\n\n"
                 "```python\n%s```")
MBPP_ASK = ("Write a Python function for this task. Reply with the code in a "
            "single ```python code block, no explanation.\n\nTask: %s\n\n"
            "Your code should pass these tests:\n%s")
CALLED = re.compile(r"assert\s+(?:not\s+)?\(?\s*(?:set\()?\s*([A-Za-z_]\w*)\s*\(")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=640)
    args = parser.parse_args()
    from datasets import load_dataset

    rows = []
    for item in load_dataset("openai/openai_humaneval", split="test"):
        rows.append({
            "task": "humaneval", "task_id": item["task_id"],
            "prompt": HUMANEVAL_ASK % item["prompt"],
            "test": item["test"], "entry_point": item["entry_point"],
            "max_tokens": args.max_tokens,
        })

    skipped = 0
    for item in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        tests = list(item["test_list"])
        found = CALLED.search(tests[0]) if tests else None
        if not found:
            skipped += 1
            continue
        entry = found.group(1)
        test = "%s\n%s\n\n\ndef check(candidate):\n    return None\n" % (
            item.get("test_setup_code") or "", "\n".join(tests))
        rows.append({
            "task": "mbpp", "task_id": "mbpp/%d" % item["task_id"],
            "prompt": MBPP_ASK % (item["text"], "\n".join(tests)),
            "test": test, "entry_point": entry,
            "max_tokens": args.max_tokens,
        })

    path = os.path.expanduser(args.out)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    counts = {}
    for row in rows:
        counts[row["task"]] = counts.get(row["task"], 0) + 1
    print("wrote %s: %s (%d mbpp items skipped, no function name in the tests)"
          % (path, counts, skipped))


if __name__ == "__main__":
    main()
