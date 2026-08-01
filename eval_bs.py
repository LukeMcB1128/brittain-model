"""
Score a BrittainScript specialist by RUNNING what it writes.

    python3 eval_bs.py weights/brittain_xs_bs.pt --n 200
    python3 eval_bs.py a.pt --out samples_a.jsonl      # keep the programs

Generates unconditionally (no prompt — native emission, the thing the specialist
is for), executes each program in the real interpreter, and reports how many run.

WHY NOT VALIDATION LOSS
Loss says how surprised the model is by held-out text. It cannot say whether the
model's own output is a working program, which is the entire claim being made.
This project is unusual in being able to answer that directly: there is an
interpreter, so correctness is executable rather than inferred.

WHY NOVELTY IS REPORTED NEXT TO IT, AND WHY IT IS NOT OPTIONAL
The corpus is ~19,700 programs and a specialization run passes over them several
times. A model that simply memorised them scores BRILLIANTLY on execution rate:
every program it recites parses and runs, because a human wrote it. Execution
rate alone cannot tell that apart from having learned the language, and the two
call for opposite responses — one means ship it, the other means fewer epochs.

So the headline number here is RUNS AND IS NOVEL. A model at 80% runs / 90% novel
has learned BrittainScript. One at 85% runs / 15% novel has learned the corpus.
Judge two training recipes on the joint figure or the comparison is worthless.

Novelty is 8-gram containment against the TRAINING split: what share of a
sample's 8-grams already appear somewhere in the training data. Whole-program
equality is far too weak a test — changing one variable name defeats it.

EXECUTING GENERATED CODE
The interpreter runs in a subprocess with a timeout, in a temp directory (that is
py2bs.verify's own runner, reused). Programs that pull in the gui or io libraries
are NOT executed: this repo has already been through one round of verification
opening windows on the screen, and a sampler that hits `add gui` a hundred times
would do it again. They are counted separately rather than silently dropped.
"""
import os
import re
import sys
import json
import time
import random
import argparse
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.normpath(
    os.path.join(HERE, "..", "bs-corpus", "bs_corpus.clean.train.jsonl"))

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--n", type=int, default=200, help="programs to generate")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--max_tokens", type=int, default=400)
p.add_argument("--temperature", type=float, default=0.4)
p.add_argument("--top_p", type=float, default=0.95)
p.add_argument("--top_k", type=int, default=None)
p.add_argument("--repetition_penalty", type=float, default=1.12)
p.add_argument("--timeout", type=int, default=10, help="seconds per program")
p.add_argument("--corpus", default=DEFAULT_CORPUS,
               help="training split, for the novelty check")
p.add_argument("--ngram", type=int, default=8)
p.add_argument("--novel_below", type=float, default=0.8,
               help="8-gram containment at or above this counts as a copy")
p.add_argument("--py2bs_path", default="../BrittainScript")
p.add_argument("--out", default=None, help="write every sample and verdict here")
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

PY2BS = os.path.abspath(os.path.expanduser(args.py2bs_path))

# gui opens windows; io touches the filesystem. Neither belongs in an automated
# eval loop running hundreds of generated programs.
UNSAFE_LIBS = {"gui", "io"}
ADD_LINE = re.compile(r"(?m)^\s*add\s+(\w+)")
# The interpreter reports problems on stdout and keeps going, so a clean exit
# does not mean a clean run — verify.py learned this the same way.
ERROR_PREFIXES = ("Error:", "Syntax error", "Undefined ")


def normalise(text):
    return re.sub(r"\s+", " ", text).strip()


def ngrams(text, n):
    words = normalise(text).split()
    return {hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)}


def load_corpus():
    if not os.path.exists(args.corpus):
        sys.exit(f"no corpus at {args.corpus} — run filter_bs.py first")
    exact, grams = set(), set()
    n = 0
    for line in open(args.corpus):
        try:
            bs = json.loads(line)["bs"]
        except (ValueError, KeyError):
            continue
        n += 1
        exact.add(normalise(bs))
        grams |= ngrams(bs, args.ngram)
    print(f"novelty reference: {n:,} training programs, {len(grams):,} distinct "
          f"{args.ngram}-grams")
    return exact, grams


def generate(n):
    """Unconditional samples: start from end-of-text and let it open a document."""
    import torch
    from model import Brittain, GPTConfig
    from tok_util import load_tokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    ck = torch.load(args.checkpoint, map_location=device)
    cfg = GPTConfig(**ck["cfg"])
    model = Brittain(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    enc = load_tokenizer(ck)
    print(f"loaded {args.checkpoint} ({model.num_params():,} params, "
          f"iter {ck.get('iter', '?')}) on {device.type}")

    out = []
    t0 = time.time()
    while len(out) < n:
        b = min(args.batch, n - len(out))
        idx = torch.full((b, 1), enc.eot, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type,
                                             dtype=torch.bfloat16):
            done = model.generate(idx, args.max_tokens,
                                  temperature=args.temperature,
                                  top_k=args.top_k, top_p=args.top_p,
                                  repetition_penalty=args.repetition_penalty)
        for row in done.tolist():
            body = row[1:]                       # drop the priming end-of-text
            if enc.eot in body:                  # stop at the document boundary
                body = body[:body.index(enc.eot)]
            out.append(enc.decode(body))
        print(f"  generated {len(out)}/{n} "
              f"({len(out)/max(1e-9, time.time()-t0):.1f}/s)", flush=True)
    return out


def execute(source):
    """Run one program. Returns a verdict string."""
    from py2bs.verify import run_brittainscript

    used = set(ADD_LINE.findall(source))
    if used & UNSAFE_LIBS:
        return "skipped (gui/io)"
    result = run_brittainscript(source, timeout=args.timeout)
    if result.timed_out:
        return "timeout"
    if not result.ok:
        return "crash"
    for line in result.stdout.splitlines():
        if line.startswith(ERROR_PREFIXES):
            return "interpreter error"
    return "runs"


def main():
    sys.path.insert(0, PY2BS)
    try:
        import py2bs.verify                                    # noqa: F401
    except ImportError as exc:
        sys.exit(f"cannot import py2bs from {PY2BS}: {exc}")

    exact, grams = load_corpus()
    samples = generate(args.n)

    verdicts = collections.Counter()
    rows = []
    n_runs = n_novel = n_both = 0
    for text in samples:
        if not text.strip():
            verdicts["empty"] += 1
            rows.append({"bs": text, "verdict": "empty", "containment": None})
            continue
        verdict = execute(text)
        verdicts[verdict] += 1

        mine = ngrams(text, args.ngram)
        if normalise(text) in exact:
            containment = 1.0
        elif not mine:                    # too short for an n-gram comparison
            containment = 1.0 if normalise(text) in exact else 0.0
        else:
            containment = len(mine & grams) / len(mine)
        novel = containment < args.novel_below

        n_runs += verdict == "runs"
        n_novel += novel
        n_both += (verdict == "runs") and novel
        rows.append({"bs": text, "verdict": verdict,
                     "containment": round(containment, 3), "novel": novel})

    n = len(samples)
    print(f"\n{'='*60}\n{n} unconditional samples\n{'='*60}")
    for verdict, count in verdicts.most_common():
        print(f"  {verdict:<22}{count:>6,}  {100*count/n:5.1f}%")
    print(f"\n  runs                  {n_runs:>6,}  {100*n_runs/n:5.1f}%")
    print(f"  novel                 {n_novel:>6,}  {100*n_novel/n:5.1f}%")
    print(f"  RUNS AND IS NOVEL     {n_both:>6,}  {100*n_both/n:5.1f}%   <- compare this")
    if n_runs and n_both / max(1, n_runs) < 0.5:
        print("\n  WARNING: most working programs are near-copies of training data.")
        print("  That is memorisation, not competence — cut epochs or add data.")

    if args.out:
        with open(args.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"\nsamples -> {args.out}")


if __name__ == "__main__":
    main()
