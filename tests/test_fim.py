"""
Does the FIM model actually READ the suffix, or has it just learned that three new
tokens exist?

    python3 test_fim.py brittain_235m_fim_best.pt

Validation loss cannot answer this. A model that learned only "sentinels appear in
this pattern" scores better on FIM-formatted data while still completing purely
left-to-right, which is exactly the capability the run was meant to buy.

THE TEST: give the model an identical prefix twice, with DIFFERENT suffixes, and
see whether the middle changes to fit. A model using the suffix will reach for the
variable the suffix returns. A model ignoring it produces the same text both times.

    <fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle> -> model writes the MIDDLE

That is the PSM ordering prepare_fim.py trained on (it also trained SPM; PSM is the
one an editor uses).
"""
import sys
import codecs
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from brittain.model import Brittain, GPTConfig
from brittain.paths import CHECKPOINT_DIR
from brittain.tokenizer import load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", nargs="?",
                default=str(CHECKPOINT_DIR / "brittain_235m_fim_best.pt"))
ap.add_argument("-n", "--max_tokens", type=int, default=40)
ap.add_argument("--trials", type=int, default=8,
                help="generations per prompt. One sample per prompt cannot tell a "
                     "real difference between checkpoints from a lucky draw — six "
                     "trials gave 6/6 and 3/6 for two checkpoints whose val differed "
                     "by 3%%, which is well inside sampling noise.")
ap.add_argument("--quiet", action="store_true",
                help="summary only — the completions flood a truncating console")
ap.add_argument("-t", "--temperature", type=float, default=0.2)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("-r", "--repetition_penalty", type=float, default=1.12)
args = ap.parse_args()

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

ck = torch.load(args.checkpoint, map_location=device)
cfg = GPTConfig(**ck["cfg"])
model = Brittain(cfg).to(device).eval()
model.load_state_dict(ck["model"])
enc = load_tokenizer(ck)

fim = ck.get("fim")
if fim is None:
    if not getattr(enc, "has_fim", False):
        sys.exit(f"{args.checkpoint} has no FIM sentinels — this is not a FIM model.")
    fim = {"prefix": enc.fim_prefix, "suffix": enc.fim_suffix, "middle": enc.fim_middle}
PRE, SUF, MID = fim["prefix"], fim["suffix"], fim["middle"]

print(f"{args.checkpoint} | iter {ck.get('iter','?')} | val {ck.get('val', float('nan')):.4f}")
print(f"{enc.name} vocab {enc.vocab_size} | sentinels pre={PRE} suf={SUF} mid={MID}")
print("=" * 72)


@torch.no_grad()
def infill(prefix, suffix):
    """Generate the MIDDLE for a prefix/suffix pair, PSM ordering."""
    ids = ([PRE] + enc.encode(prefix) + [SUF] + enc.encode(suffix) + [MID])
    x = torch.tensor([ids[-cfg.block_size:]], dtype=torch.long, device=device)
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    out = []
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for tok in model.stream(x, args.max_tokens, temperature=args.temperature,
                                top_p=args.top_p,
                                repetition_penalty=args.repetition_penalty):
            t = tok[0, -1].item()
            if t in (enc.eot, PRE, SUF, MID):
                break
            out.append(utf8.decode(enc.token_bytes(t)))
    return "".join(out)


# Each case: one prefix, two suffixes that demand DIFFERENT middles. The suffix
# names a variable the middle has to have produced.
CASES = [
    ("def f(items):\n", [("    return total\n", "total"),
                         ("    return count\n", "count")]),
    ("def stats(xs):\n    n = len(xs)\n", [("    return mean\n", "mean"),
                                           ("    return largest\n", "largest")]),
    ("import math\n\ndef area(r):\n", [("    return circle\n", "circle"),
                                       ("    return square\n", "square")]),
]

used = trials_run = ignored = pairs = runaway = 0
for prefix, variants in CASES:
    if not args.quiet:
        print(f"\nPREFIX:\n{prefix.rstrip()}")
    for trial in range(args.trials):
        middles = []
        for suffix, want in variants:
            mid = infill(prefix, suffix)
            middles.append(mid)
            hit = want in mid
            used += hit
            trials_run += 1
            # Termination: the middle should stop where the suffix begins. Starting
            # a fresh def means it ran past the hole it was asked to fill.
            if "\ndef " in mid or mid.strip().startswith("def "):
                runaway += 1
            if not args.quiet and trial == 0:
                print(f"\n  suffix wants `{want}`:")
                print("    " + "\n    ".join(mid.strip().splitlines() or ["(nothing)"]))
                print(f"    -> mentions '{want}': {'YES' if hit else 'no'}")
        pairs += 1
        if middles[0].strip() == middles[1].strip():
            ignored += 1
    if not args.quiet:
        print("-" * 72)

print(f"""
{'=' * 72}
{args.checkpoint}   iter {ck.get('iter', '?')}   val {ck.get('val', float('nan')):.4f}
{args.trials} trials x {len(CASES)} prefixes x 2 suffixes = {trials_run} generations

  names the variable the suffix returns : {used:3d}/{trials_run}  ({100*used/max(1,trials_run):.0f}%)
  identical text for both suffixes      : {ignored:3d}/{pairs}  ({100*ignored/max(1,pairs):.0f}%)
  ran past the hole into a new def      : {runaway:3d}/{trials_run}  ({100*runaway/max(1,trials_run):.0f}%)

Reading this:
  * High naming rate, low identical rate -> the model IS conditioning on what
    comes after the cursor. FIM works.
  * High identical rate -> it learned the token pattern, not the capability.
  * "ran past the hole" is the termination defect: knowing WHERE the middle ends
    is a separate skill from knowing what goes in it. Lower is better, and this is
    the number extra training was meant to improve.
""")
