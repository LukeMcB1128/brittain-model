"""Build the brittain-shakespeare SFT set from the tagged synthetic stories.

    python3 scripts/prepare/build_story_sft.py

Every synthetic story was generated from a known tag combination, so the tags are
a specification the story already satisfies. Inverting that gives an
instruction/response pair for free: phrase the tags as a request, and the story
is the answer.

The output is the same directory-of-.npy shape the pretraining stages use, so
the trainer reads it with no changes -- SFT is one more stage at a low learning
rate, starting from the pretrained weights.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from brittain.data_v3 import repository_in_validation
from brittain.sft_story import encode_example, example_from_story
from brittain.story_tagger import pronoun_consistency
from brittain.tokenizer_story import STORY_TOKENIZER, StoryTokenizer

SYNTHETIC = "data/raw/brittain-shakespeare-synthetic/stories_renamed.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", default=SYNTHETIC)
    parser.add_argument("--tokenizer", default=str(STORY_TOKENIZER))
    parser.add_argument("--output-dir", default="data/processed/brittain-shakespeare-sft")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--min-pronoun-consistency", type=float, default=1.0,
        help="drop stories whose named characters take contradictory pronouns. "
             "The synthetic stories are only 72-80%% consistent themselves, and "
             "the pretrained model measured 75%%, so the data is the ceiling and "
             "training longer cannot raise it. Stories with no evidence either "
             "way are kept: absence of evidence is not a contradiction. Pass 0 "
             "to keep everything.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_split(name, rows, block_size, pad, output_dir, report):
    """One example per row, padded. No packing.

    Pretraining packs several documents into a row to waste no tokens. Here that
    would put one story's tokens in another story's context, and an instruction
    followed by the wrong story is precisely the thing being trained away.
    """
    if not rows:
        raise SystemExit(f"{name} split is empty")
    directory = output_dir / f"{name}_{block_size}"
    directory.mkdir(parents=True, exist_ok=True)
    inputs = np.lib.format.open_memmap(
        directory / "input_ids.npy", mode="w+",
        dtype=np.uint16, shape=(len(rows), block_size),
    )
    labels = np.lib.format.open_memmap(
        directory / "labels.npy", mode="w+",
        dtype=np.int16, shape=(len(rows), block_size),
    )
    graded = 0
    try:
        for index, (ids, row_labels) in enumerate(rows):
            padding = block_size + 1 - len(ids)
            padded_ids = ids + [pad] * padding
            padded_labels = row_labels + [-100] * padding
            inputs[index] = padded_ids[:-1]
            labels[index] = padded_labels[1:]
            graded += sum(1 for value in padded_labels[1:] if value != -100)
        inputs.flush()
        labels.flush()
    finally:
        del inputs, labels
    report[f"{name}_rows"] = len(rows)
    report[f"{name}_graded_tokens"] = graded
    print(f"wrote {directory}  rows={len(rows):,}  graded={graded:,}", flush=True)


def main():
    args = parse_args()
    tokenizer = StoryTokenizer(project_path(args.tokenizer))
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / f"train_{args.block_size}").exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite to replace it")

    rng = random.Random(args.seed)
    train: list[tuple[list[int], list[int]]] = []
    validation: list[tuple[list[int], list[int]]] = []
    skipped: Counter[str] = Counter()
    seen = 0

    with project_path(args.stories).open(encoding="utf-8") as handle:
        for line in handle:
            if args.limit is not None and seen >= args.limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped["malformed json"] += 1
                continue
            story = row.get("text") or ""
            if args.min_pronoun_consistency > 0 and story:
                consistency = pronoun_consistency(story)
                if consistency is not None and consistency < args.min_pronoun_consistency:
                    skipped["contradictory pronouns"] += 1
                    continue
            example = example_from_story(story, row.get("book_tags") or {}, rng)
            if example is None:
                skipped["no story or no tags"] += 1
                continue
            ids, labels = encode_example(example, tokenizer)
            if len(ids) > args.block_size + 1:
                # Truncating would cut the story's ending off, and the ending is
                # the thing this model most recently learned to produce.
                skipped["longer than the block"] += 1
                continue
            seen += 1
            target = (
                validation
                if repository_in_validation(
                    row["repository"], args.validation_fraction, args.seed
                )
                else train
            )
            target.append((ids, labels))
            if seen % 10000 == 0:
                print(f"  {seen:,} examples", flush=True)

    report = {
        "format": "brittain-shakespeare-sft-report-v1",
        "stories": str(project_path(args.stories)),
        "block_size": args.block_size,
        "examples": seen,
        "min_pronoun_consistency": args.min_pronoun_consistency,
        "skipped": dict(skipped),
    }
    write_split("train", train, args.block_size, tokenizer.pad, output_dir, report)
    write_split("validation", validation, args.block_size, tokenizer.pad,
                output_dir, report)
    path = output_dir / f"sft_{args.block_size}.report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
