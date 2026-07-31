"""
Build a BrittainScript corpus by translating The Stack through py2bs.

    python3 prepare_bs.py --tokens 10e6
    python3 prepare_bs.py --tokens 10e6 --workers 10 --no_verify   # ~20x faster

Writes data/bs_corpus.jsonl, one {"py": ..., "bs": ...} per accepted file. Keeping
the Python alongside costs nothing and leaves the door open to a translation-pair
SFT later ("write this in BrittainScript"), which the .bs alone would not support.

WHY THIS IS SLOW, AND WHY IT IS PARALLEL
Verification runs BOTH programs and compares stdout — two interpreter startups per
candidate file. That, not translation, is the entire cost, and it is what makes a
translation trustworthy rather than merely plausible. It is also embarrassingly
parallel, so it runs in a process pool. On a 12-core machine expect roughly 20
files/sec verified.

    --no_verify is ~20x faster and produces UNVERIFIED data. The whole argument for
    this corpus is that every line is known-correct, so only use it to estimate
    yield before committing to a real run.

SAFETY
Verification EXECUTES code from The Stack, which is scraped from public GitHub.
The safety boundary is py2bs's frontend, which rejects unsafe imports before
anything runs — no os, subprocess, socket, or file I/O survives validation, and
each run has a timeout. That is a meaningful boundary, not a sandbox. Run this on
a machine you would not mind reinstalling, or in a container.

RESUMING
The output is appended to, and already-seen files are skipped by content hash, so
re-running continues rather than restarting. Ctrl-C is safe.
"""
import os
import sys
import json
import time
import hashlib
import argparse
import collections
import multiprocessing as mp

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet backend has thrown SIGBUS here

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=float, default=10e6, help="stop at this many BS tokens")
p.add_argument("--out", default="data/bs_corpus.jsonl")
p.add_argument("--py2bs_path", default="../BrittainScript",
               help="checkout containing the py2bs package (it is not on PyPI)")
p.add_argument("--dataset", default="bigcode/the-stack-dedup")
p.add_argument("--lang", default="python")
p.add_argument("--tokenizer", default="data/code_bpe.json")
p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
p.add_argument("--timeout", type=int, default=10, help="seconds per verified program")
p.add_argument("--max_chars", type=int, default=20_000,
               help="skip larger files; they rarely translate and cost the most")
p.add_argument("--no_verify", action="store_true",
               help="skip differential execution — MUCH faster, UNVERIFIED output")
args = p.parse_args()

PY2BS = os.path.abspath(os.path.expanduser(args.py2bs_path))


def init_worker(py2bs_path):
    """Each worker imports py2bs once, from the checkout rather than PyPI."""
    if py2bs_path not in sys.path:
        sys.path.insert(0, py2bs_path)
    global translate
    from py2bs import translate


def attempt(source):
    """Translate one file. Returns (bs_or_None, rejection_reason_or_None)."""
    try:
        r = translate(source, verify=not args.no_verify, timeout=args.timeout)
    except Exception as exc:                       # a crash is a rejection, not a stop
        return None, f"crash: {type(exc).__name__}"
    if not r.ok:
        return None, (r.rejected_features[0] if r.rejected_features
                      else (r.error or "unknown")[:40])
    if not r.brittainscript or not r.brittainscript.strip():
        return None, "empty output"
    return r.brittainscript, None


def main():
    from tokenizers import Tokenizer
    from datasets import load_dataset

    tok = Tokenizer.from_file(args.tokenizer)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # resume: remember what we already translated, and how far we got
    seen = set()
    n_tokens = n_written = 0
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                seen.add(row["hash"])
                n_tokens += row.get("tokens", 0)
                n_written += 1
        print(f"resuming: {n_written:,} files, {n_tokens:,} tokens already in {args.out}")

    ds = load_dataset(args.dataset, data_dir=f"data/{args.lang}",
                      split="train", streaming=True)

    def candidates():
        for ex in ds:
            text = ex.get("content") or ex.get("text") or ""
            if not text.strip() or len(text) > args.max_chars:
                continue
            h = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]
            if h in seen:
                continue
            seen.add(h)
            yield h, text

    rejections = collections.Counter()
    n_scanned = 0
    t0 = t_log = time.time()
    out = open(args.out, "a")
    pool = mp.Pool(args.workers, initializer=init_worker, initargs=(PY2BS,))
    try:
        pending = {}
        for h, text in candidates():
            pending[pool.apply_async(attempt, (text,))] = (h, text)
            if len(pending) < args.workers * 4:
                continue
            done = [r for r in pending if r.ready()]
            if not done:                            # let the pool catch up
                done = [next(iter(pending))]
            for res in done:
                h, text = pending.pop(res)
                n_scanned += 1
                bs, why = res.get()
                if why is not None:
                    rejections[why] += 1
                    continue
                n = len(tok.encode(bs, add_special_tokens=False).ids)
                out.write(json.dumps({"hash": h, "tokens": n,
                                      "py": text, "bs": bs}) + "\n")
                n_written += 1
                n_tokens += n

            if time.time() - t_log > 30:
                out.flush()
                rate = n_scanned / max(1e-9, time.time() - t0)
                pct = 100 * n_written / max(1, n_scanned)
                print(f"  {n_tokens/1e6:6.2f}M / {args.tokens/1e6:.1f}M tokens | "
                      f"{n_written:,} written / {n_scanned:,} scanned ({pct:.1f}%) | "
                      f"{rate:.1f} files/s", flush=True)
                t_log = time.time()

            if n_tokens >= args.tokens:
                break
    except KeyboardInterrupt:
        print("\ninterrupted — output is valid, re-run to continue")
    finally:
        pool.terminate()
        pool.join()
        out.close()

    print(f"\nDone. {n_written:,} files, {n_tokens:,} tokens -> {args.out}")
    print(f"acceptance {100*n_written/max(1,n_scanned):.1f}% "
          f"({'UNVERIFIED' if args.no_verify else 'every line verified by execution'})")
    if rejections:
        print("\nWhy files were rejected — the top row is the BrittainScript feature")
        print("that would unlock the most real Python:\n")
        total = sum(rejections.values())
        for feature, count in rejections.most_common(20):
            print(f"  {feature:<32} {100*count/total:5.1f}%  {count:,}")


if __name__ == "__main__":
    main()
