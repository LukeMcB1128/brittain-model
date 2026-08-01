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

used = ignored = 0
for prefix, variants in CASES:
    print(f"\nPREFIX:\n{prefix.rstrip()}")
    middles = []
    for suffix, want in variants:
        mid = infill(prefix, suffix)
        middles.append(mid)
        hit = want in mid
        used += hit
        print(f"\n  suffix wants `{want}`:")
        print("    " + "\n    ".join(mid.strip().splitlines() or ["(nothing)"]))
        print(f"    -> mentions '{want}': {'YES' if hit else 'no'}")
    if middles[0].strip() == middles[1].strip():
        ignored += 1
        print("\n  ** IDENTICAL output for different suffixes — suffix was ignored **")
    print("-" * 72)

print(f"""
{'=' * 72}
Middles naming the variable the suffix returns : {used} / {2 * len(CASES)}
Cases where both suffixes gave identical text  : {ignored} / {len(CASES)}

Reading this:
  * Different text per suffix, often naming the right variable -> the model IS
    conditioning on what comes after the cursor. FIM works.
  * Identical text regardless of suffix -> it learned the token pattern and not
    the capability. More of the same training is unlikely to fix that.
  * Different but unrelated text -> partial. It sees the suffix but cannot use it
    yet; more annealing may help, and this is the case worth spending on.
""")
