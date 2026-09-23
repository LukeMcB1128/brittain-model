# -*- coding: utf-8 -*-
"""Is any eval probe sitting in the training data?

Run 4's behaviour data was written from the same failing transcripts the
defect probes were built from. One overlap was found by accident -- the
`settled syntax` probe asks how to reverse a string in Python, and the first
known_syntax row was "reverse a string in python" -- and a model that learns
that one row scores perfectly on that probe while learning nothing. So every
probe is checked against every training row, by word overlap rather than by
exact match, because a near-paraphrase contaminates just as well.

    python3 audit_contamination.py --train ~/brittain4/data/run4_sft.jsonl

Exits non-zero if any probe is contaminated, so it can gate a scoring run.
"""
import argparse
import io
import json
import os
import re
import sys

WORD = re.compile(r"[a-z0-9]+")
# Words that carry no identity between two prompts.
STOP = set("a an the is are was were be do does did i you it to of in on for "
           "and or that this what whats who whos how can me my your with at "
           "right now just".split())


def words(text):
    return {w for w in WORD.findall(text.lower()) if w not in STOP}


def overlap(a, b):
    """Share of the smaller prompt's content words found in the other."""
    if not a or not b:
        return 0.0
    return len(a & b) / float(min(len(a), len(b)))


def probe_prompts(heldout=False):
    """(probe name, the question it asks) for every probe and every variant."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_defects import HELDOUT, PROBES
    out = []
    for probe in (HELDOUT if heldout else PROBES):
        for variant in probe.get("variants") or [probe]:
            users = [t["content"] for t in variant["turns"] if t["role"] == "user"
                     and not t["content"].startswith("The following tool evidence")]
            out.append((probe["name"], users[-1]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--heldout", action="store_true",
                        help="audit the held-out probes instead of the originals")
    args = parser.parse_args()

    rows = [json.loads(l) for l in io.open(os.path.expanduser(args.train), encoding="utf-8")
            if l.strip()]
    trained = []
    for row in rows:
        for message in row["messages"]:
            if message["role"] == "user":
                # Tool results are pasted into the user turn in some groups;
                # compare only the question the user actually asked.
                text = message["content"].split("\n")[-1]
                trained.append((row["kind"], text, words(text)))

    dirty = 0
    prompts = probe_prompts(args.heldout)
    for name, prompt in prompts:
        mine = words(prompt)
        best = max(trained, key=lambda t: overlap(mine, t[2]))
        score = overlap(mine, best[2])
        flag = "CONTAMINATED" if score >= args.threshold else "clean"
        dirty += score >= args.threshold
        print("  %-12s %-38s %.2f  <- %s: %.50s"
              % (flag, name[:38], score, best[0], best[1]))
    print("\n%d of %d probes contaminated at overlap >= %.2f"
          % (dirty, len(prompts), args.threshold))
    return 1 if dirty else 0


if __name__ == "__main__":
    raise SystemExit(main())
