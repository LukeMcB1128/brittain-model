"""Measure whether the conditioning tags actually steer generation.

The extractors in ``story_tagger`` are deterministic, so the same code that
labelled the training data can score generated samples. That turns "do the tags
work" into a number instead of an opinion.

Adherence alone is not the number that matters. If 88% of the corpus is past
tense, then ``[Tense: Past]`` scoring 88% means the tag did nothing. Every tag is
therefore reported against an unconditional baseline generated with no tag block
at all, and the gap between them ("lift") is the real signal.

    python3 scripts/evaluate/tag_adherence.py checkpoints/shakespeare_18m_pilot/weights.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain import story_tagger
from brittain.checkpoint_v3 import load_brittain3_checkpoint
from brittain.tags import OBJECTIVE_TAGS, TAG_VALUES, render
from brittain.tokenizer_story import StoryTokenizer

# Tags a deterministic extractor can score on generated text.
SCORABLE = OBJECTIVE_TAGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--samples", type=int, default=24, help="samples per tag value")
    parser.add_argument("--baseline-samples", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="runs/shakespeare-tag-adherence.json")
    parser.add_argument("--tags", default=None, help="comma-separated subset to score")
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def generate(model, tokenizer, device, prompt_ids, args) -> str:
    """Sample one story and return its text, stopping at a story boundary."""
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    stop = {tokenizer.eot, tokenizer.story_end, tokenizer.pad}
    pieces: list[int] = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for token in model.stream(
            ids, args.max_tokens,
            temperature=args.temperature, top_k=None, top_p=args.top_p,
            repetition_penalty=1.0,
        ):
            value = int(token[0, -1].item())
            if value in stop:
                break
            pieces.append(value)
    return tokenizer.decode(pieces, skip_special_tokens=True)


def score(text: str, token_count: int) -> dict[str, str]:
    """Run every deterministic extractor over generated text."""
    found = {}
    for name, value in (
        ("Voice", story_tagger.voice_from_text(text)),
        ("POV", story_tagger.point_of_view(text)),
        ("Tense", story_tagger.tense(text)),
        ("Setting", story_tagger.setting(text)),
        ("Cast", story_tagger.cast(text)),
        ("Length", story_tagger.length_from_tokens(token_count)),
    ):
        if value is not None:
            found[name] = value
    # voice_from_text only ever detects the archaic register. Absence of archaic
    # morphology is genuine evidence of a non-Shakespearean voice, but it cannot
    # tell Victorian from Modern, so those two are unscorable here.
    return found


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    checkpoint_path = project_path(args.checkpoint)
    model, checkpoint = load_brittain3_checkpoint(checkpoint_path, device=device)
    model.to(device).eval()
    tokenizer_path = args.tokenizer or checkpoint.get("tokenizer_path")
    tokenizer = StoryTokenizer(project_path(tokenizer_path))
    print(f"loaded {checkpoint_path.name}  {model.num_params():,} params  "
          f"vocab {tokenizer.vocab_size}", flush=True)

    selected = (
        tuple(name.strip() for name in args.tags.split(",")) if args.tags else SCORABLE
    )
    for name in selected:
        if name not in TAG_VALUES:
            raise SystemExit(f"unknown tag: {name}")

    story_start = tokenizer.special_ids["<|story_start|>"]

    # Baseline: no tag block at all. Without this the adherence numbers are
    # meaningless, because a tag cannot take credit for the corpus's own bias.
    print(f"baseline: {args.baseline_samples} unconditional samples", flush=True)
    baseline: dict[str, Counter] = defaultdict(Counter)
    baseline_seen: Counter[str] = Counter()
    for index in range(args.baseline_samples):
        text = generate(model, tokenizer, device, [story_start], args)
        found = score(text, len(tokenizer.encode(text)))
        for name, value in found.items():
            baseline[name][value] += 1
            baseline_seen[name] += 1
        if (index + 1) % 24 == 0:
            print(f"  {index + 1}/{args.baseline_samples}", flush=True)

    results = {}
    samples_kept = []
    for name in selected:
        per_value = {}
        for value in TAG_VALUES[name]:
            hits = misses = undetermined = 0
            prompt = [story_start, tokenizer.tags_start,
                      *tokenizer.encode(render({name: value})),
                      tokenizer.tags_end]
            for index in range(args.samples):
                text = generate(model, tokenizer, device, prompt, args)
                found = score(text, len(tokenizer.encode(text)))
                got = found.get(name)
                if got is None:
                    undetermined += 1
                elif got == value:
                    hits += 1
                else:
                    misses += 1
                if index == 0:
                    samples_kept.append(
                        {"tag": name, "requested": value, "got": got,
                         "text": text[:400]}
                    )
            decided = hits + misses
            base_total = baseline_seen[name]
            base_rate = (baseline[name][value] / base_total) if base_total else None
            adherence = (hits / decided) if decided else None
            per_value[value] = {
                "hits": hits, "misses": misses, "undetermined": undetermined,
                "adherence": adherence,
                "baseline_rate": base_rate,
                "lift": (adherence - base_rate)
                        if adherence is not None and base_rate is not None else None,
            }
            flag = "" if adherence is None else f"{adherence:6.1%}"
            base = "  n/a" if base_rate is None else f"{base_rate:6.1%}"
            print(f"  [{name}: {value:18}] adherence {flag}  baseline {base} "
                  f"  undetermined {undetermined}/{args.samples}", flush=True)
        decided_all = sum(v["hits"] + v["misses"] for v in per_value.values())
        hits_all = sum(v["hits"] for v in per_value.values())
        results[name] = {
            "values": per_value,
            "overall_adherence": (hits_all / decided_all) if decided_all else None,
        }
        overall = results[name]["overall_adherence"]
        print(f"{name}: overall {overall:.1%}" if overall is not None
              else f"{name}: overall n/a", flush=True)

    report = {
        "format": "brittain-shakespeare-adherence-v1",
        "checkpoint": str(checkpoint_path),
        "samples_per_value": args.samples,
        "baseline_samples": args.baseline_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "baseline_distribution": {
            name: dict(counts) for name, counts in baseline.items()
        },
        "tags": results,
        "examples": samples_kept,
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nreport: {output}")


if __name__ == "__main__":
    main()
