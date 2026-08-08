"""Small local capability report for a Brittain3 checkpoint.

The script does not execute generated code. Use the separate HumanEval pipeline
for execution-based results.
"""
from __future__ import annotations

import argparse
import ast
import codecs
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from brittain.checkpoint_v3 import load_brittain3_checkpoint
from brittain.tokenizer_v3 import Brittain3Tokenizer


CODE_PROMPTS = (
    "def fibonacci(n: int) -> int:\n",
    "def binary_search(values, target):\n",
    "class Stack:\n    def __init__(self):\n",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--code-text", default=str(PROJECT_ROOT / "benchmarks" / "prompts" / "code.py"))
    parser.add_argument("--prose-text", default=None)
    parser.add_argument("--max-bytes", type=int, default=20_000)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--long-context", action="store_true", help="run a real 16K early-token influence check")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def select_device(requested):
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps" or (requested == "auto" and torch.backends.mps.is_available()):
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available")
        return torch.device("mps")
    return torch.device("cpu")


def stop_ids(tokenizer):
    return set(tokenizer.special_ids.values())


@torch.no_grad()
def complete(model, tokenizer, prompt_ids, max_tokens, device):
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []
    ended = False
    hit_eot = False
    for token in model.stream(ids, max_tokens, top_k=1, repetition_penalty=1.0):
        token_id = int(token.item())
        if token_id in stop_ids(tokenizer):
            ended = True
            hit_eot = token_id == tokenizer.eot
            break
        generated.append(token_id)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    text = "".join(decoder.decode(tokenizer.token_bytes(token_id)) for token_id in generated)
    text += decoder.decode(b"", final=True)
    return text, ended, hit_eot, len(generated)


@torch.no_grad()
def bits_per_byte(model, tokenizer, text, device):
    ids = tokenizer.encode(text)
    total_nll = 0.0
    total_bytes = len(text.encode("utf-8"))
    window = min(1024, model.cfg.max_seq_len)
    for start in range(0, len(ids) - 1, window):
        chunk = ids[start:start + window + 1]
        if len(chunk) < 2:
            continue
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        logits, _ = model(x, y)
        total_nll += float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ))
    return total_nll / (math.log(2) * max(1, total_bytes))


def syntax_and_termination(model, tokenizer, args, device):
    valid = ended = eot = total = 0
    for prompt in CODE_PROMPTS:
        for _ in range(args.samples):
            text, did_end, did_eot, _ = complete(
                model, tokenizer, tokenizer.encode(prompt), args.max_tokens, device
            )
            candidate = prompt + text
            if "\n" in candidate:
                candidate = candidate[:candidate.rfind("\n") + 1]
            total += 1
            ended += did_end
            eot += did_eot
            try:
                ast.parse(candidate)
                valid += 1
            except (SyntaxError, ValueError):
                pass
    return {
        "samples": total,
        "syntax_fraction": valid / total,
        "structural_stop_fraction": ended / total,
        "eot_fraction": eot / total,
    }


def fim_probe(model, tokenizer, args, device):
    prefix = "def summarize(values):\n"
    rows = []
    next_logits = []
    for suffix, wanted in (("    return total\n", "total"), ("    return count\n", "count")):
        prompt = [
            tokenizer.fim_prefix, *tokenizer.encode(prefix),
            tokenizer.fim_suffix, *tokenizer.encode(suffix), tokenizer.fim_middle,
        ]
        prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)
        logits, _ = model(prompt_tensor)
        next_logits.append(logits.float().cpu())
        text, ended, hit_eot, length = complete(
            model, tokenizer, prompt, args.max_tokens, device
        )
        rows.append({
            "wanted": wanted,
            "mentioned": wanted in text,
            "ended": ended,
            "hit_eot": hit_eot,
            "tokens": length,
        })
    difference = float((next_logits[0] - next_logits[1]).abs().max())
    return {
        "cases": rows,
        "suffix_use_fraction": sum(row["mentioned"] for row in rows) / len(rows),
        "suffix_changes_next_token_logits": difference > 0,
        "max_next_logit_difference": difference,
    }


@torch.no_grad()
def long_context_probe(model, tokenizer, device):
    context = model.cfg.max_seq_len
    if context < 16384:
        raise ValueError("checkpoint does not declare a 16K context")
    first = tokenizer.encode("EARLY_MARKER_ALPHA") or [tokenizer.eot]
    second = tokenizer.encode("EARLY_MARKER_BETA") or [tokenizer.eot]
    filler_id = tokenizer.encode(" x")[0]
    tail = tokenizer.encode("\nUse the early marker:")
    def sequence(marker):
        filler = context - len(marker) - len(tail)
        if filler < 1:
            raise ValueError("long-context probe metadata exceeds the context")
        return marker + [filler_id] * filler + tail
    a = torch.tensor([sequence(first)], dtype=torch.long, device=device)
    b = torch.tensor([sequence(second)], dtype=torch.long, device=device)
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    difference = float((logits_a.float() - logits_b.float()).abs().max())
    return {
        "context": context,
        "early_token_changes_final_logits": difference > 0,
        "max_logit_difference": difference,
        "note": "This checks the 16K computation path. It is not a repository-quality score.",
    }


def main():
    args = parse_args()
    device = select_device(args.device)
    model, checkpoint = load_brittain3_checkpoint(args.checkpoint, device)
    model.eval()
    tokenizer_path = checkpoint.get("tokenizer_path")
    tokenizer = Brittain3Tokenizer(tokenizer_path)
    code_text = Path(args.code_text).read_text(encoding="utf-8", errors="ignore")[:args.max_bytes]
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "architecture": checkpoint["architecture"],
        "parameter_count": model.num_params(),
        "context": model.cfg.max_seq_len,
        "code_bpb": bits_per_byte(model, tokenizer, code_text, device),
        "generation": syntax_and_termination(model, tokenizer, args, device),
        "fim": fim_probe(model, tokenizer, args, device),
    }
    if args.prose_text:
        prose = Path(args.prose_text).read_text(encoding="utf-8", errors="ignore")[:args.max_bytes]
        report["prose_bpb"] = bits_per_byte(model, tokenizer, prose, device)
    if args.long_context:
        report["long_context"] = long_context_probe(model, tokenizer, device)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
