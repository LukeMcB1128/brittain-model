"""
Build a BrittainScript corpus by translating The Stack through py2bs.

    python3 prepare_bs.py --tokens 10e6
    python3 prepare_bs.py --tokens 10e6 --workers 10

Writes ../bs-corpus/bs_corpus.jsonl by default — alongside py2bs's own output, and
deliberately OUTSIDE both git repos: at 10M tokens the file is ~130MB, which git
will happily commit and then GitHub will reject at push. One
{"py": ..., "bs": ...} row per accepted file. Keeping
the Python alongside costs nothing and leaves the door open to a translation-pair
SFT later ("write this in BrittainScript"), which the .bs alone would not support.

DO NOT USE --no_verify FOR A REAL CORPUS
Verification runs both programs and compares stdout. Measured on The Stack:

    verified     1.6% acceptance
    unverified   2.6% acceptance

So roughly 38% of everything the emitter produces computes something DIFFERENT
from the Python it came from. Skipping verification does not add 1.6x more data;
it adds 1.6x more data of which a third is wrong.

That matters more here than a bad-data fraction normally would, because transpiler
errors are SYSTEMATIC. A human corpus full of bugs is noisy in random directions
and largely averages out. A mistranslated construct is wrong the same way every
single time, so the model sees it consistently and learns it as correct
BrittainScript — and this model's whole purpose is to emit BrittainScript.

It is not even a speed argument. Only the ~2% of files that pass validation are
ever executed; the other 98% are rejected by parsing alone. Throughput is bound by
streaming The Stack over the network, not by verification. Keep it on.

Verification is embarrassingly parallel, so it runs in a process pool.

SAFETY
Verification EXECUTES code from The Stack, which is scraped from public GitHub.
Two boundaries, and the second was learned the hard way:

  1. py2bs's frontend rejects unsafe imports before anything runs — no os,
     subprocess, socket or file I/O survives validation — and each run has a
     timeout.
  2. Workers chdir into an empty temp directory. The working directory is on
     sys.path, so without this a candidate file containing `import sample`
     imported and RAN this repo's sample.py. `import train` would have started a
     training run.

WHAT IS STILL NOT BOUNDED: MEMORY. `[0] * 10**10` is pure computation, passes the
allowlist, and will drive a laptop into swap. There is no fix available here on
macOS — Darwin does not implement RLIMIT_AS or RLIMIT_DATA (setrlimit raises even
with an infinite hard limit), and RLIMIT_CPU is cumulative per process, so setting
it in a worker kills the worker rather than the candidate. The only real bound is
--timeout, plus running fewer workers so concurrent blowups stay survivable.

If you want actual limits, run this in Docker, where memory is a cgroup setting.
Mount the PARENT of this repo — py2bs lives in ../BrittainScript and the corpus is
written to ../bs-corpus, so mounting only this directory fails immediately. Match
the image to your own Python: py2bs rejects candidates with ast.parse, and a
different version parses a different language, so the container would not
reproduce the local yield.

    cd ~/Downloads/Coding
    docker run --rm -m 4g \
        -e HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
        -v "$PWD":/w -w "/w/BRITTAIN MODEL" python:3.13 \
        sh -c "pip install -q ply tokenizers datasets && \
               python prepare_bs.py --tokens 10e6"

ply IS REQUIRED and is easy to miss: it is BrittainScript's only runtime
dependency, nothing in this repo needs it, and without it the interpreter cannot
start — so every verification fails and every file is rejected. A container that
installed only tokenizers and datasets rejected 990,000 files at 0% acceptance.
self_test() now catches that before the scan begins.

The Stack is gated, and the container has no login of its own, so HF_TOKEN has to
be passed in. Prefer the env var over mounting ~/.cache/huggingface: that
directory also holds stored_tokens, and there is no reason to expose more
credentials than the one download needs. Streaming means the cache buys nothing.

That is a meaningful boundary, not a sandbox.

RESUMING
The output is appended to, and already-seen files are skipped by content hash, so
re-running continues rather than restarting. Ctrl-C is safe.
"""
import os
import sys
import json
import time
import hashlib
import shutil
import tempfile
import argparse
import itertools
import collections
import multiprocessing as mp

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet backend has thrown SIGBUS here

HERE = os.path.dirname(os.path.abspath(__file__))
# Default into the sibling bs-corpus/ — that folder is the corpus, it already holds
# py2bs's own output, and it is NOT a git repo, so a 100MB+ jsonl cannot be
# committed by accident. Resolved from this file's location, not the cwd.
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "bs-corpus", "bs_corpus.jsonl"))

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=float, default=10e6, help="stop at this many BS tokens")
p.add_argument("--out", default=DEFAULT_OUT)
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
               help="skip differential execution. Barely faster, and ~38%% of the "
                    "extra rows compute the wrong thing. See the module docstring.")
args, _ = p.parse_known_args()   # children re-import under spawn; be forgiving

PY2BS = os.path.abspath(os.path.expanduser(args.py2bs_path))
STUCK_AFTER = 300      # no single candidate can legitimately take this long


# Modules a candidate file may import. Pure computation only — nothing that can
# touch the filesystem, the network, the process table, or the display.
#
# This is an ALLOWLIST because py2bs's own UNSAFE_IMPORTS is a denylist of 34
# names, and a denylist cannot bound arbitrary code from GitHub. `turtle` was not
# on it, so verification executed Stack files that opened Tk windows on the
# screen; `pty`, `runpy`, `code`, `pdb`, `curses` and `site` are all still absent
# from it. Anything not named here is rejected as an unresolvable import.
SAFE_MODULES = {
    "math", "cmath", "random", "string", "re", "json", "datetime", "time",
    "itertools", "functools", "operator", "collections", "heapq", "bisect",
    "array", "statistics", "decimal", "fractions", "numbers", "copy", "enum",
    "dataclasses", "typing", "abc", "textwrap", "unicodedata", "struct",
    "binascii", "base64", "hashlib", "hmac", "uuid", "calendar", "zlib",
}


def init_worker(py2bs_path, sandbox, verify, timeout):
    """Each worker imports py2bs once, then moves somewhere harmless.

    THE CHDIR IS A SAFETY FIX, NOT TIDINESS. Verification executes the candidate
    Python, and the working directory is on sys.path — so a file from The Stack
    containing `import sample` imported THIS repo's sample.py and ran it. Any of
    `import train`, `import serve`, `import prepare_code` would have started a
    training run, bound a port, or begun downloading a corpus.

    It also corrupted the measurement: the validator's module_is_available() uses
    find_spec, so local .py files counted as resolvable imports and files were
    accepted that should have been rejected.

    Workers run in an empty directory instead. py2bs_path is absolute, so the
    import above still works from anywhere.
    """
    if py2bs_path not in sys.path:
        sys.path.insert(0, py2bs_path)
    global translate
    from py2bs import translate
    import py2bs.frontend as frontend

    # Replace the denylist with the allowlist above. find_spec is also why local
    # .py files counted as importable, so this closes that too.
    frontend.module_is_available = lambda name: name.split(".")[0] in SAFE_MODULES

    global VERIFY, TIMEOUT
    VERIFY, TIMEOUT = verify, timeout

    sys.path[:] = [p for p in sys.path if p not in ("", ".", os.getcwd())]
    os.chdir(sandbox)


# A program that MUST translate and verify. If this fails, the environment is
# broken and every real candidate would be rejected too — silently, since a broken
# interpreter is indistinguishable from an untranslatable file. That is not
# hypothetical: a Docker run installed tokenizers and datasets but not ply,
# BrittainScript's only runtime dependency, and rejected 990,000 files at 0%
# acceptance before anyone noticed.
CANARY = """def double(n):
    return n * 2

print(double(21))
"""


def self_test():
    """Prove the toolchain works before scanning a million files.

    Runs in the MAIN process, before the pool exists. It cannot run inside the
    pool: if a Pool initializer raises — which is exactly what a missing ply or a
    wrong --py2bs_path does — multiprocessing respawns the worker forever and any
    apply() hangs instead of failing. The failure modes worth catching here are
    process-independent anyway.
    """
    def die(reason, detail=""):
        sys.exit(
            f"\nSelf-test failed: {reason}\n{detail}\n"
            f"Every candidate would fail the same way, so this stops now rather\n"
            f"than reporting a million rejections.\n\n"
            f"Most likely causes:\n"
            f"  * ply is not installed — BrittainScript's interpreter needs it,\n"
            f"    nothing in this repo does, so it is easy to miss:  pip install ply\n"
            f"  * --py2bs_path is not a BrittainScript checkout\n"
            f"      currently: {PY2BS}\n")

    if not os.path.isdir(os.path.join(PY2BS, "py2bs")):
        die("no py2bs package found.", f"  looked in: {PY2BS}")
    sys.path.insert(0, PY2BS)
    try:
        from py2bs import translate as _translate
    except Exception as exc:
        die("py2bs could not be imported.", f"  {type(exc).__name__}: {exc}")
    try:
        r = _translate(CANARY, verify=not args.no_verify, timeout=args.timeout)
    except Exception as exc:
        die("translating a trivial program raised.", f"  {type(exc).__name__}: {exc}")
    if not r.ok:
        die("a trivial program did not translate.",
            f"  {r.error or (r.rejected_features or ['unknown'])[0]}")


def attempt(source):
    """Translate one file. Returns (bs_or_None, rejection_reason_or_None)."""
    try:
        r = translate(source, verify=VERIFY, timeout=TIMEOUT)
    except Exception as exc:                       # a crash is a rejection, not a stop
        return None, f"crash: {type(exc).__name__}"
    if not r.ok:
        return None, (r.rejected_features[0] if r.rejected_features
                      else (r.error or "unknown")[:40])
    if not r.brittainscript or not r.brittainscript.strip():
        return None, "empty output"
    return r.brittainscript, None


def main():
    self_test()          # before the network, the pool, or anything expensive
    from tokenizers import Tokenizer
    from datasets import load_dataset

    tok = Tokenizer.from_file(args.tokenizer)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Resume needs the STREAM POSITION, not just the accepted hashes. `seen` only
    # records files that were accepted, so without a position a resume re-streams
    # The Stack from the start and re-translates every previously REJECTED file
    # purely to reject it again — at 1.6% acceptance that is 98% of the work,
    # observed as ~400,000 files scanned for 1 new row.
    progress_path = os.path.abspath(args.out) + ".progress.json"
    n_raw_done = 0
    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                n_raw_done = json.load(f).get("raw_examples", 0)
        except (ValueError, OSError):
            n_raw_done = 0

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

    if n_raw_done:
        print(f"skipping {n_raw_done:,} dataset examples already scanned")

    n_raw = n_raw_done

    def candidates():
        nonlocal n_raw
        # islice consumes without translating — vastly cheaper than re-rejecting
        for ex in itertools.islice(ds, n_raw_done, None):
            n_raw += 1
            text = ex.get("content") or ex.get("text") or ""
            if not text.strip() or len(text) > args.max_chars:
                continue
            h = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]
            if h in seen:
                continue
            seen.add(h)
            yield h, text

    rejections = collections.Counter()
    # n_written/n_tokens carry the resumed totals so the target is right; acceptance
    # must be measured on THIS session only or a resume reports over 100%.
    n_scanned = n_written_session = 0
    t0 = t_log = time.time()
    out = open(os.path.abspath(args.out), "a")
    sandbox = tempfile.mkdtemp(prefix="prepare_bs_")
    # SPAWN, NOT FORK. A pool worker that dies — the cgroup OOM killer takes one
    # whenever a candidate allocates past the container limit — makes Pool fork a
    # replacement. Forking a process whose helper threads hold internal locks
    # leaves the child with a mutex nobody will ever release, and the whole pool
    # wedges in futex_wait with every process at 0% CPU. Observed after 50 minutes
    # and 2.4M tokens. spawn starts children clean and cannot inherit a held lock.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    pool = mp.Pool(args.workers, initializer=init_worker,
                   initargs=(PY2BS, sandbox, not args.no_verify, args.timeout))
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
                # Backstop: never block forever. If the pool wedges anyway, this
                # surfaces it as an error instead of a silent 0%-CPU hang.
                try:
                    bs, why = res.get(timeout=STUCK_AFTER)
                except mp.TimeoutError:
                    sys.exit(f"\nA task made no progress for {STUCK_AFTER}s — the "
                             f"worker pool is wedged.\nOutput so far is valid; "
                             f"re-run to resume from it.")
                if why is not None:
                    rejections[why] += 1
                    continue
                n = len(tok.encode(bs, add_special_tokens=False).ids)
                out.write(json.dumps({"hash": h, "tokens": n,
                                      "py": text, "bs": bs}) + "\n")
                n_written += 1
                n_written_session += 1
                n_tokens += n

            if time.time() - t_log > 30:
                out.flush()
                with open(progress_path, "w") as pf:
                    json.dump({"raw_examples": n_raw, "written": n_written,
                               "tokens": n_tokens}, pf)
                rate = n_scanned / max(1e-9, time.time() - t0)
                pct = 100 * n_written_session / max(1, n_scanned)
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
        with open(progress_path, "w") as pf:
            json.dump({"raw_examples": n_raw, "written": n_written,
                       "tokens": n_tokens}, pf)
        shutil.rmtree(sandbox, ignore_errors=True)

    print(f"\nDone. {n_written:,} files, {n_tokens:,} tokens -> {args.out}")
    print(f"acceptance {100*n_written_session/max(1,n_scanned):.1f}% "
          f"({'UNVERIFIED' if args.no_verify else 'every line verified by execution'})")
    if rejections:
        print("\nWhy files were rejected — the top row is the BrittainScript feature")
        print("that would unlock the most real Python:\n")
        total = sum(rejections.values())
        for feature, count in rejections.most_common(20):
            print(f"  {feature:<32} {100*count/total:5.1f}%  {count:,}")


if __name__ == "__main__":
    main()
