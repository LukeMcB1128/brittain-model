"""
Build the CODE instruction-tuning set for BRITTAIN-2, with LOSS MASKING.

This is the BRITTAIN-2 counterpart to prepare_sft.py, which stays as-is because
it is what `brittain1:124m-instruct` was built from. Four things differ:

  * the BRITTAIN-2 code BPE (32003 with FIM sentinels), not gpt2 tiktoken
  * a MIX of code instruction sets, not Alpaca alone
  * a PARSE FILTER: every fenced Python block in a response must ast.parse, so
    the model is not taught to emit plausible-looking broken syntax. Measured on
    the real run this rejects only ~0.6% — far less than expected, because the
    filter only sees FENCED blocks and much of CodeAlpaca answers with bare
    unfenced code that is never checked. Extending it to unfenced responses is
    the obvious next improvement; the multi-block and length filters below are
    doing more work than this one right now.
  * a RESPONSE LENGTH CAP. 235M degrades badly over long generations, and 35% of
    its HumanEval completions never emit EOT at all. Training only on short,
    single-code-block answers that terminate is how that behaviour gets fixed —
    SFT teaches format, and format is what this model is missing.

Loss masking works exactly as in prepare_sft.py: labels are the token ids with
PROMPT positions set to -100, so cross-entropy ignores them and the model is
graded only on the response.

    python3 scripts/prepare/prepare_code_sft.py --out_dir data/processed/code_sft

Writes input_ids.npy + labels.npy + meta.json.
"""
import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from datasets import load_dataset

from brittain.paths import FIM_TOKENIZER, PROCESSED_DATA_DIR
from brittain.prompts import format_prompt
from brittain.tokenizer import CodeTok

p = argparse.ArgumentParser()
p.add_argument("--out_dir", default=str(PROCESSED_DATA_DIR / "code_sft"))
p.add_argument("--tokenizer", default=str(FIM_TOKENIZER))
# 1024, not the model's full 2048: these examples are short (mean ~350 tokens),
# and every example is padded to this length, so a larger cap is paid for in
# wasted compute on padding rather than in longer training examples.
p.add_argument("--max_len", type=int, default=1024)
p.add_argument("--max_response_tokens", type=int, default=300,
               help="drop longer responses. Short canonical answers are the "
                    "behaviour being taught; long ones dilute it.")
p.add_argument("--code_alpaca", type=int, default=20000)
p.add_argument("--magicoder", type=int, default=10000)
p.add_argument("--alpaca", type=int, default=3000,
               help="general English instructions, so code SFT does not cost "
                    "plain instruction following")
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
enc = CodeTok(args.tokenizer)
EOT = enc.eot
print(f"tokenizer {args.tokenizer} | vocab {enc.vocab_size} | eot {EOT}")

FENCE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
stats = Counter()


def python_blocks_parse(text):
    """True unless a fenced block that claims to be Python fails to parse.

    Unlabelled fences are checked too: most unlabelled blocks in these datasets
    are Python, and one that does not parse is exactly what we want to drop. A
    block that is genuinely another language will usually fail to parse as
    Python, so this is deliberately conservative — it costs some recall to keep
    the training set clean.
    """
    blocks = FENCE.findall(text)
    if not blocks:
        return True                      # prose-only answer; nothing to check
    for lang, body in blocks:
        if lang and lang.lower() not in ("python", "py", ""):
            continue                     # explicitly another language, skip
        try:
            ast.parse(body)
        except SyntaxError:
            return False
        except Exception:
            return False
    return True


def accept(instruction, inp, response):
    """Shared filter. Returns token arrays, or None with a reason counted."""
    if not instruction or not instruction.strip() or not response.strip():
        stats["empty"] += 1
        return None
    if len(FENCE.findall(response)) > 1:
        # Multiple code blocks means a multi-part answer. At 235M those come out
        # as rambling; one block per answer is the format being taught.
        stats["multi_block"] += 1
        return None
    if not python_blocks_parse(response):
        stats["unparseable"] += 1
        return None

    prompt = format_prompt(instruction, inp)
    pids = enc.encode(prompt)
    rids = enc.encode(response.strip()) + [EOT]
    if len(rids) > args.max_response_tokens:
        stats["response_too_long"] += 1
        return None
    if len(pids) + len(rids) > args.max_len:
        stats["too_long"] += 1
        return None

    ids = pids + rids
    lab = [-100] * len(pids) + rids          # grade the response only
    pad = args.max_len - len(ids)
    ids += [EOT] * pad
    lab += [-100] * pad                      # padding is not graded
    stats["kept"] += 1
    return ids, lab


rows = []


def take(name, ds, n, get):
    """Pull up to n accepted examples from a dataset via a field extractor."""
    before = stats["kept"]
    for ex in ds:
        if stats["kept"] - before >= n:
            break
        got = accept(*get(ex))
        if got:
            rows.append(got)
    print(f"  {name}: kept {stats['kept'] - before}")


print("CodeAlpaca-20k ...")
take("code_alpaca", load_dataset("sahil2801/CodeAlpaca-20k", split="train"),
     args.code_alpaca, lambda e: (e["instruction"], e.get("input", ""), e["output"]))

print("Magicoder-OSS-Instruct-75K ...")
take("magicoder", load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train"),
     args.magicoder, lambda e: (e["problem"], "", e["solution"]))

print("alpaca-cleaned ...")
take("alpaca", load_dataset("yahma/alpaca-cleaned", split="train"),
     args.alpaca, lambda e: (e["instruction"], e.get("input", ""), e["output"]))

rng = np.random.default_rng(args.seed)
rng.shuffle(rows)
input_ids = np.array([r[0] for r in rows], dtype=np.uint16)
labels = np.array([r[1] for r in rows], dtype=np.int32)   # holds ids AND -100

np.save(os.path.join(args.out_dir, "input_ids.npy"), input_ids)
np.save(os.path.join(args.out_dir, "labels.npy"), labels)
meta = {"n": len(rows), "max_len": args.max_len, "vocab_size": enc.vocab_size,
        "tokenizer": "code_bpe_fim", "eot": EOT, "stats": dict(stats)}
with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

graded = int((labels != -100).sum())
print(f"\nWrote {len(rows)} examples, shape {input_ids.shape} -> {args.out_dir}")
print(f"  graded tokens {graded:,} ({graded / labels.size:.1%} of positions)")
print("  filtered:", dict(stats))
