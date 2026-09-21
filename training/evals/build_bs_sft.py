# -*- coding: utf-8 -*-
"""BrittainScript training data, sampled from the parallel corpus.

WHY IT IS IN THE MIX AT ALL
Asked to write BrittainScript, the served model searched the web four times and
never wrote a line. Asked what it knew about the language, it said there is "a
Python library called BrittainScript" that adds "B4-specific syntax and editor
integration" -- invented. The name is on PyPI, so the base has seen the package
metadata and nothing else: recognition with nothing behind it.

The corpus has been sitting ready since 8 September and was simply never wired
into build_mix.py.

HELD-OUT ITEMS ARE EXCLUDED BY HASH
eval_bs.jsonl scores by RUNNING generated code and comparing output, and it
carries the same `hash` field as the corpus. Training on those rows would turn
the eval into a memorisation check. This drops them before sampling.

TARGETS ARE VERIFIED BY THE INTERPRETER
A sample is executed with ~/venv/bin/bs and compared against Python. Training
on snippets that do not run would teach a language that does not exist, and the
whole point is that the model currently invents one.

python3 build_bs_sft.py --keep 600
"""
import argparse
import json
import os
import random
import subprocess
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="/home/lukeb/brittain4/data/bs_corpus.jsonl")
ap.add_argument("--eval", default="/home/lukeb/brittain4/data/eval_bs.jsonl")
ap.add_argument("--out", default="/home/lukeb/brittain4/data/bs_sft.jsonl")
ap.add_argument("--keep", type=int, default=600)
ap.add_argument("--verify", type=int, default=40,
                help="how many sampled targets to actually execute")
ap.add_argument("--bs", default=os.path.expanduser("~/venv/bin/bs"))
ap.add_argument("--seed", type=int, default=4)
args = ap.parse_args()

held_out = set()
for line in open(args.eval, encoding="utf-8"):
    if line.strip():
        held_out.add(json.loads(line).get("hash"))
print("held-out hashes: %d" % len(held_out))

rows = []
for line in open(args.corpus, encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("hash") in held_out:
        continue
    py = (row.get("py") or "").strip()
    bs = (row.get("bs") or "").strip()
    # Very short pairs teach the tokenizer more than the language, and very
    # long ones crowd out everything else in a 2,048-token window.
    if not py or not bs or len(bs) < 30 or len(bs) > 1200:
        continue
    rows.append(row)
print("usable pairs after exclusions: %d of the corpus" % len(rows))

random.Random(args.seed).shuffle(rows)
rows = rows[: args.keep]

# --- verify a sample actually runs -----------------------------------------
def run(path_text, binary):
    with tempfile.NamedTemporaryFile("w", suffix=".bs", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(path_text)
        name = fh.name
    try:
        done = subprocess.run([binary, name], capture_output=True, text=True,
                              timeout=10)
        return done.returncode, (done.stdout or "").strip()
    except Exception as error:
        return -1, str(error)[:80]
    finally:
        os.unlink(name)


checked = ok = 0
if args.verify and os.path.exists(args.bs):
    for row in rows[: args.verify]:
        checked += 1
        code, _out = run(row["bs"], args.bs)
        if code == 0:
            ok += 1
    print("interpreter check: %d of %d sampled targets run cleanly"
          % (ok, checked))
    if checked and ok < checked * 0.6:
        raise SystemExit("most sampled BrittainScript does not run; the corpus "
                         "or the interpreter is not what this assumes")
else:
    print("interpreter not found at %s -- targets unverified" % args.bs)

# --- three directions -------------------------------------------------------
# Translation matches how the eval scores. Reading and explaining are included
# because the live failure was partly comprehension: the model could describe
# a program put in front of it but could not produce one.
TRANSLATE = [
    "Write this in BrittainScript:\n\n```python\n%s\n```",
    "Translate this Python to BrittainScript:\n\n```python\n%s\n```",
    "What's the BrittainScript equivalent of this?\n\n```python\n%s\n```",
]
READ = [
    "What does this BrittainScript do?\n\n```\n%s\n```",
    "Translate this BrittainScript to Python:\n\n```\n%s\n```",
]

out = []
random.seed(args.seed)
for index, row in enumerate(rows):
    py, bs = row["py"].strip(), row["bs"].strip()
    if index % 4 == 3:
        prompt = random.choice(READ) % bs
        reply = "```python\n%s\n```" % py
        kind = "bs_read"
    else:
        prompt = random.choice(TRANSLATE) % py
        reply = "```\n%s\n```" % bs
        kind = "bs_write"
    out.append({
        "kind": kind,
        "source": "bs_corpus",
        "hash": row.get("hash"),
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ],
    })

random.Random(args.seed).shuffle(out)
with open(args.out, "w", encoding="utf-8") as fh:
    for row in out:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

counts = {}
for row in out:
    counts[row["kind"]] = counts.get(row["kind"], 0) + 1
print("\nwrote %d examples -> %s" % (len(out), args.out))
for kind in sorted(counts):
    print("  %-10s %d" % (kind, counts[kind]))

leaked = {r["hash"] for r in out} & held_out
if leaked:
    raise SystemExit("%d held-out hashes leaked into training" % len(leaked))
print("no held-out hash appears in the training set")
