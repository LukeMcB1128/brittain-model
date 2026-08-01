"""
Measure what a py2bs feature would ACTUALLY unlock, before building it.

    python3 survey_bs.py --files 200e3
    python3 survey_bs.py --report            # re-analyse an existing survey

THE PROBLEM THIS SOLVES
prepare_bs.py's rejection histogram records only rejected_features[0], because
validation raises on the first unsupported construct it meets. A file using
classes AND dicts AND comprehensions is counted once, under whichever the
validator checked first. So the counts cannot be added up, the expensive features
hide the cheap ones, and "classes is the biggest bar" might mean nothing: every
one of those files may also use four other things you have not built.

This walks the whole file (py2bs.frontend.survey_features) and records EVERY
unsupported construct, then answers the question that actually matters:

    how many more files translate if I build this SET of features?

It reports single features by true unlock count, then runs a greedy set cover, so
the output is a build order rather than a histogram.

IMPORT-BLOCKED FILES ARE EXCLUDED FROM THE ROADMAP
Measured over 2.1M rejections, 81.9% were imports. Those are not a feature gap: a
file importing numpy cannot become a BrittainScript program because there is no
numpy to translate to. Counting them drowns everything actionable, so they are
reported separately and left out of the cover.

CHEAP AND SAFE
Surveying is a pure AST walk — it never executes a candidate, so none of
prepare_bs.py's sandbox machinery is needed and there is no pool, no timeout and
no memory hazard. It is bound entirely by streaming The Stack.
"""
import os
import sys
import json
import time
import signal
import argparse
import itertools
import collections

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "bs-corpus", "survey.json"))

p = argparse.ArgumentParser()
p.add_argument("--files", type=float, default=200e3, help="how many to survey")
p.add_argument("--out", default=DEFAULT_OUT)
p.add_argument("--py2bs_path", default="../BrittainScript")
p.add_argument("--dataset", default="bigcode/the-stack-dedup")
p.add_argument("--lang", default="python")
p.add_argument("--max_chars", type=int, default=20_000)
p.add_argument("--report", action="store_true",
               help="skip scanning, re-analyse --out")
p.add_argument("--bundle", type=int, default=6,
               help="how many features deep to run the greedy cover")
args = p.parse_args()

PY2BS = os.path.abspath(os.path.expanduser(args.py2bs_path))

# Not a feature gap — no language work reaches these. See the docstring.
IMPORT_BLOCKED = {"unresolvable import", "unsafe import", "relative imports",
                  "star imports", "dotted import"}
UNFIXABLE = IMPORT_BLOCKED | {"invalid python", "unparseable python",
                              "survey error", "I/O or dynamic execution"}


def scan():
    sys.path.insert(0, PY2BS)
    try:
        from py2bs.frontend import survey_features
    except ImportError as exc:
        sys.exit(f"cannot import py2bs from {PY2BS}: {exc}\n"
                 f"survey_features() is required — it is the collecting variant "
                 f"of the validator.")
    from datasets import load_dataset

    ds = load_dataset(args.dataset, data_dir=f"data/{args.lang}",
                      split="train", streaming=True)

    # One row per surveyed file: the sorted feature set. Kept whole rather than
    # pre-aggregated because set cover needs co-occurrence, which a histogram of
    # counts has already destroyed.
    sets = []
    n = n_clean = 0
    t0 = t_log = time.time()
    target = int(args.files)

    def save():
        with open(args.out, "w") as f:
            json.dump({"surveyed": n, "clean": n_clean, "sets": sets}, f)

    try:
        for ex in itertools.islice(ds, target):
            text = ex.get("content") or ex.get("text") or ""
            if not text.strip() or len(text) > args.max_chars:
                continue
            n += 1
            feats = survey_features(text)
            if not feats:
                n_clean += 1
            else:
                sets.append(sorted(feats))
            if time.time() - t_log > 30:
                rate = n / max(1e-9, time.time() - t0)
                print(f"  {n:,} / {target:,} surveyed | {n_clean:,} already "
                      f"translate | {rate:.0f} files/s", flush=True)
                t_log = time.time()
                save()
    except KeyboardInterrupt:
        print("\ninterrupted — analysing what was surveyed so far")
    save()
    print(f"\nsurveyed {n:,} files -> {args.out}")
    return n, n_clean, sets


def report(n, n_clean, sets):
    if not n:
        sys.exit("nothing surveyed")

    blocked = [s for s in sets if UNFIXABLE & set(s)]
    fixable = [set(s) for s in sets if not (UNFIXABLE & set(s))]

    print(f"\n{'='*66}\n{n:,} files surveyed\n{'='*66}")
    print(f"  {n_clean:>8,}  ({100*n_clean/n:4.1f}%) translate today")
    print(f"  {len(blocked):>8,}  ({100*len(blocked)/n:4.1f}%) blocked by imports "
          f"or unparseable source — unreachable")
    print(f"  {len(fixable):>8,}  ({100*len(fixable)/n:4.1f}%) blocked ONLY by "
          f"missing language features — the whole addressable pool")

    if not fixable:
        print("\nNothing addressable in this sample.")
        return

    # True single-feature unlock: files whose ENTIRE blocking set is that one
    # feature. This is the number the old histogram could not produce.
    solo = collections.Counter()
    appears = collections.Counter()
    for s in fixable:
        for f in s:
            appears[f] += 1
        if len(s) == 1:
            solo[next(iter(s))] += 1

    print(f"\n{'feature':<32}{'appears in':>12}{'unlocks alone':>15}")
    print("-" * 59)
    for feat, c in appears.most_common(15):
        print(f"{feat:<32}{c:>12,}{solo.get(feat, 0):>15,}")
    print("\n'appears in' is what the old histogram approximated. 'unlocks alone'")
    print("is what building only that feature would actually buy you.")

    # Greedy set cover: at each step take the feature that completes the most
    # files given everything already chosen.
    print(f"\n{'='*66}\nBuild order (greedy — each line assumes the ones above it)\n{'='*66}")
    remaining = [set(s) for s in fixable]
    chosen = []
    cumulative = 0
    for _ in range(args.bundle):
        gain = collections.Counter()
        for s in remaining:
            missing = s - set(chosen)
            if len(missing) == 1:
                gain[next(iter(missing))] += 1
        if gain:
            feat, got = gain.most_common(1)[0]
        else:
            # Nothing finishes a file on its own — what is left needs features in
            # combination. Without this the build order stops dead while a large
            # cluster (classes + dict literals, say) sits there needing both, and
            # the report silently understates how much is reachable. Take the
            # feature blocking the most files and let the next pass complete them.
            pool = collections.Counter()
            for s in remaining:
                for f in s - set(chosen):
                    pool[f] += 1
            if not pool:
                break
            feat, _ = pool.most_common(1)[0]
            got = 0
        chosen.append(feat)
        cumulative += got
        remaining = [s for s in remaining if not s <= set(chosen)]
        note = (f"unlocks {got:>7,} more   " if got else
                "unlocks       0 now   ")
        print(f"  + {feat:<28} {note}"
              f"(total {cumulative:,} = {100*cumulative/len(fixable):.1f}% of "
              f"addressable, {100*cumulative/n:.2f}% of all files)")

    if cumulative:
        print(f"\nAcceptance would go from {100*n_clean/n:.2f}% to about "
              f"{100*(n_clean+cumulative)/n:.2f}% with those {len(chosen)} features.")


def main():
    def on_sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_sigterm)

    if args.report:
        if not os.path.exists(args.out):
            sys.exit(f"no survey at {args.out} — run without --report first")
        with open(args.out) as f:
            d = json.load(f)
        report(d["surveyed"], d["clean"], d["sets"])
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        report(*scan())


if __name__ == "__main__":
    main()
