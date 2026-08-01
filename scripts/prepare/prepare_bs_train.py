"""
Tokenize the clean BrittainScript corpus into the .bin files train.py reads.

    python3 scripts/prepare/prepare_bs_train.py \
        --out data/processed/bs_native
    python3 scripts/prepare/prepare_bs_train.py \
        --out data/processed/bs_mixed --mix 0.25

Then pass the selected directory to scripts/train/specialist.py.

TWO VARIANTS, ONE BUDGET
These exist to answer a single question: does adding py->bs translation examples
make the specialist's NATIVE BrittainScript better or worse? That only means
anything if both variants present the same number of tokens per epoch, so the
mixed build is capped at the native build's token count rather than being the
native data plus extra. Otherwise the mixed run simply trains longer and you
credit the objective for what was really more compute.

The mixed variant therefore contains STRICTLY LESS BrittainScript than the native
one — a quarter of its budget is spent on Python prompts. That asymmetry is the
honest form of the question ("at equal compute, is the trade worth it?"), but it
does colour the result: if mixed WINS despite seeing less BrittainScript, the
grounding effect is real and large. If it loses narrowly, the cause is ambiguous
— fewer BrittainScript tokens would explain it just as well as interference. The
token accounting printed at the end is what you need to tell those apart.

VAL IS ALWAYS NATIVE-ONLY, IN BOTH VARIANTS
Validation loss has to measure the same thing in both runs or the two numbers
cannot be compared at all. Held-out native BrittainScript is that thing. (Loss is
still the weaker signal here — eval_bs.py runs the model's output through the
interpreter, which is the number to actually judge on.)

WHY THE PROMPT IS NOT LOSS-MASKED
train.py's loop is plain next-token prediction over packed tokens; it has no
label masking, so in the mixed build the model also learns to predict the Python
side. That is a real cost, counted in the accounting below. train_sft.py is the
harness that masks, and it is the right one if you later want translation as the
primary skill rather than as a seasoning for native emission.
"""
import os
import sys
import json
import pickle
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from brittain.paths import BASE_TOKENIZER, BS_CORPUS_DIR
from brittain.prompts import format_prompt

CORPUS = str(BS_CORPUS_DIR)

p = argparse.ArgumentParser()
p.add_argument("--train_jsonl", default=os.path.join(CORPUS, "bs_corpus.clean.train.jsonl"))
p.add_argument("--val_jsonl", default=os.path.join(CORPUS, "bs_corpus.clean.val.jsonl"))
p.add_argument("--out", required=True, help="directory for train.bin/val.bin/meta.pkl")
p.add_argument("--mix", type=float, default=0.0,
               help="share of the TOKEN BUDGET given to py->bs translation examples")
p.add_argument("--tokenizer", default=str(BASE_TOKENIZER))
p.add_argument("--budget", type=int, default=None,
               help="total train tokens; default = every native token available, "
                    "which is the figure the mixed build must also be given")
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

INSTRUCTION = "Write this Python program in BrittainScript."


def rows(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path} — run filter_bs.py first")
    for line in open(path):
        try:
            yield json.loads(line)
        except ValueError:
            continue


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    vocab = tok.get_vocab_size()
    assert vocab < 65536, "vocab must fit in uint16"
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        sys.exit("tokenizer has no <|endoftext|> token")

    os.makedirs(args.out, exist_ok=True)
    train_rows = list(rows(args.train_jsonl))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_rows)

    encode = lambda text: tok.encode(text, add_special_tokens=False).ids

    # Budget: default to everything native, so `--mix` variants are capped at the
    # same figure and the comparison stays honest.
    native_total = sum(r.get("tokens", 0) for r in train_rows)
    budget = args.budget or native_total
    native_budget = int(budget * (1 - args.mix))

    n_bs = n_py = n_native_rows = n_pair_rows = 0
    with open(os.path.join(args.out, "train.bin"), "wb") as f:
        used = 0
        cut = 0                       # where the native half stopped
        for cut, r in enumerate(train_rows):
            if used >= native_budget:
                break
            ids = encode(r["bs"]) + [eot]
            f.write(np.array(ids, dtype=np.uint16).tobytes())
            used += len(ids)
            n_bs += len(ids)
            n_native_rows += 1

        if args.mix > 0:
            # Translation examples come from the programs the native half did NOT
            # reach. Reusing the same programs in both formats would be
            # augmentation — the model seeing one corpus twice — and this
            # experiment is asking about grounding, which needs new programs.
            for r in train_rows[cut:]:
                if used >= budget:
                    break
                pids = encode(format_prompt(INSTRUCTION, r["py"]))
                bids = encode(r["bs"]) + [eot]
                f.write(np.array(pids + bids, dtype=np.uint16).tobytes())
                used += len(pids) + len(bids)
                n_py += len(pids)
                n_bs += len(bids)
                n_pair_rows += 1

    n_val = 0
    with open(os.path.join(args.out, "val.bin"), "wb") as f:
        for r in rows(args.val_jsonl):        # native only, identical in every variant
            ids = encode(r["bs"]) + [eot]
            f.write(np.array(ids, dtype=np.uint16).tobytes())
            n_val += len(ids)

    with open(os.path.join(args.out, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab, "tokenizer": "code_bpe"}, f)

    total = n_bs + n_py
    print(f"wrote {args.out}/  (train.bin, val.bin, meta.pkl)")
    print(f"  budget          {budget:,} tokens  (mix {args.mix:.0%})")
    print(f"  train total     {total:,}")
    print(f"    BrittainScript{n_bs:>12,}  ({100*n_bs/max(1,total):.1f}%)")
    print(f"    Python prompts{n_py:>12,}  ({100*n_py/max(1,total):.1f}%)")
    print(f"  programs        {n_native_rows:,} native + {n_pair_rows:,} translation "
          f"(disjoint)")
    print(f"  val (native)    {n_val:,}")
    print("\nCompare variants on BrittainScript tokens, not total: the mixed build")
    print("deliberately trades some away, and that is the cost side of the trade.")


if __name__ == "__main__":
    main()
