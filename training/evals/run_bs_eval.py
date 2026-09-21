"""Score a model at writing BrittainScript, by running what it writes.

    python3 run_bs_eval.py --model <path> --eval eval_bs.jsonl --out r.json
    python3 run_bs_eval.py ... --spec        # include a language reference

TWO FRAMINGS, AND THEY ANSWER DIFFERENT QUESTIONS
  without --spec  what the model knows about BrittainScript. A stock model has
                  never seen the language, so this floors near zero; after
                  training it is the number that means something.
  with --spec     whether it can follow a language reference it is handed. A
                  stock model can score here, so this measures in-context
                  learning and sets a ceiling worth beating.

Run the baseline both ways. If training only moves the with-spec number, the
model learned to read a spec, not the language.

SCORING
  parses    the generated program runs without a diagnostic
  matches   it runs AND prints what the reference program prints

The interpreter exits 0 on broken programs and writes errors to stdout, so
bs_check reads the diagnostics out of the text rather than trusting a status.
"""
import argparse
import collections
import json
import re
import sys
import os

sys.path.insert(0, "/tmp")
from bs_check import check                                     # noqa: E402

# Distilled from the constraints documented in the brittain-model .bs headers
# and confirmed against the interpreter.
SPEC = """BrittainScript reference:
- Output is push(x). There is no print().
- Blocks open with func/cond/while/for/loop/repeat and close with `end`.
- func name(a, b) ... end, with `return x` inside.
- cond <expr> ... end is the conditional. There is no `if`/`elif`/`else:` syntax.
- No classes, no decorators, no comprehensions, no `def`.
- Calls are positional only; there are no keyword arguments.
- No multi-dimensional indexing; use .chunk() style calls instead.
- No scientific notation: write 0.0006, never 6e-4.
- No attribute assignment (obj.x = y is not allowed).
- Reach Python with pyimport: math = pyimport("math") then math.sqrt(16).
- Comments start with #.
"""


def extract(text):
    """Pull the program out of whatever fence the model used.

    Match ANY language tag. An earlier version accepted only
    brittainscript/bs/none, so a reply fenced as ```code fell through to
    returning the whole message -- backticks included -- and all 120 items
    failed with "Illegal character". The prompt had asked for "one ```code
    block", which is where the model got that tag from; the prompt is fixed
    too, but the extractor should not depend on the model's choice of tag.
    """
    m = re.search(r"```[^\n]*\n(.*?)```", text or "", re.S)
    if m:
        return m.group(1).strip()
    # No fence at all: strip stray backticks rather than feed them in.
    return (text or "").replace("```", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--spec", action="store_true")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    kw = {} if args.think else {"enable_thinking": False}
    prompts = []
    for r in rows:
        head = (SPEC + "\n") if args.spec else ""
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content":
              "%sRewrite this Python program in BrittainScript. It must print "
              "exactly what the Python prints. Reply with only the "
              "BrittainScript in a single fenced code block.\n\n```python\n%s\n```"
              % (head, r["python"].strip())}],
            add_generation_prompt=True, tokenize=False, **kw))

    llm = LLM(model=args.model, max_model_len=8192, gpu_memory_utilization=0.90,
              max_num_seqs=16, max_num_batched_tokens=2048,
              attention_backend="TRITON_ATTN")
    outs = llm.generate(prompts, SamplingParams(max_tokens=768, temperature=0.0))

    tally = collections.Counter()
    reasons = collections.Counter()
    records = []
    for r, o in zip(rows, outs):
        code = extract(o.outputs[0].text)
        res = check(code, timeout=20)
        got = (res["output"] or "").strip()
        parses = res["valid"]
        matches = parses and got == r["expected_output"].strip()
        tally["total"] += 1
        tally["parses"] += parses
        tally["matches"] += matches
        if not parses:
            reasons[res["reason"][:70]] += 1
        records.append({"hash": r["hash"], "parses": bool(parses),
                        "matches": bool(matches), "reason": res["reason"],
                        "expected": r["expected_output"][:200], "got": got[:200],
                        "code": code[:1200]})

    n = max(1, tally["total"])
    print("n         : %d" % tally["total"])
    print("parses    : %5.1f%%  (%d)" % (100 * tally["parses"] / n, tally["parses"]))
    print("matches   : %5.1f%%  (%d)" % (100 * tally["matches"] / n, tally["matches"]))
    print("\nwhy programs failed to run:")
    for why, c in reasons.most_common(10):
        print("   [%d] %s" % (c, why))

    json.dump({"summary": {"label": args.label or args.model,
                           "spec_given": bool(args.spec),
                           "n": tally["total"],
                           "parses": round(100 * tally["parses"] / n, 1),
                           "matches": round(100 * tally["matches"] / n, 1)},
               "records": records},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)          # vLLM 0.28 teardown does not return on this setup
