"""Turn the brittain-shakespeare corpus JSONL into packed training stages.

Windows each book, tags each window, applies the tag randomization policy, and
packs whole stories into rows. Whole books go to train or validation so no book
spans both splits.

    python3 scripts/prepare/prepare_shakespeare.py --block-size 1024 --max-tokens 220000000

``--max-tokens`` bounds the output, which is what the pilot wants: the 20M pilot
needs about 200M tokens, not the whole corpus.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.data_story import (
    StorySettings, encode_story, pack_story_segments, window_text,
)
from brittain.data_v3 import repository_in_validation
from brittain.story_tagger import extract
from brittain.tags import TagPolicy
from brittain.tokenizer_story import STORY_TOKENIZER, StoryTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/raw/brittain-shakespeare/corpus.jsonl")
    parser.add_argument("--config", default="configs/data/shakespeare_corpus.json")
    parser.add_argument("--tokenizer", default=str(STORY_TOKENIZER))
    parser.add_argument("--output-dir", default="data/processed/brittain-shakespeare")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit-books", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main():
    args = parse_args()
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    windowing = config.get("windowing", {})
    policy_cfg = config.get("tag_policy", {})

    tokenizer = StoryTokenizer(project_path(args.tokenizer))
    # A window plus its tag block and sentinels must fit one row, so the block
    # size caps the window. At small block sizes that cap falls below the
    # configured target, and target and minimum have to come down with it.
    maximum_tokens = min(
        int(windowing.get("maximum_tokens", 3800)), args.block_size - 64
    )
    target_tokens = min(int(windowing.get("target_tokens", 1400)), maximum_tokens)
    minimum_tokens = min(int(windowing.get("minimum_tokens", 200)), target_tokens)
    settings = StorySettings(
        block_size=args.block_size,
        target_tokens=target_tokens,
        minimum_tokens=minimum_tokens,
        maximum_tokens=maximum_tokens,
        prefer_chapter_boundaries=bool(windowing.get("prefer_chapter_boundaries", True)),
        policy=TagPolicy(
            tag_dropout=float(policy_cfg.get("tag_dropout", 0.30)),
            block_dropout=float(policy_cfg.get("block_dropout", 0.10)),
            shuffle_rate=float(policy_cfg.get("shuffle_rate", 0.15)),
            mask_rate=float(policy_cfg.get("mask_rate", 0.80)),
            reverse_rate=float(policy_cfg.get("reverse_rate", 0.05)),
        ),
    )

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"train_{args.block_size}.npz"
    if train_path.exists() and not args.overwrite:
        raise SystemExit(f"{train_path} exists; pass --overwrite to replace it")

    rng = random.Random(args.seed)
    segments = {"train": [], "validation": []}
    tokens = Counter()
    windows = Counter()
    books = 0
    tag_counts: Counter[str] = Counter()
    started = time.time()

    corpus = project_path(args.corpus)
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            if args.max_tokens is not None and tokens["train"] >= args.max_tokens:
                break
            if args.limit_books is not None and books >= args.limit_books:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") or ""
            if not text:
                continue
            repository = row["repository"]
            split = (
                "validation"
                if repository_in_validation(
                    repository, args.validation_fraction, args.seed
                )
                else "train"
            )
            books += 1

            for window in window_text(text, tokenizer, settings):
                token_count = len(tokenizer.encode(window))
                tags = extract(
                    window,
                    token_count=token_count,
                    birth_year=row.get("birth_year"),
                    death_year=row.get("death_year"),
                    subjects=row.get("subjects"),
                    bookshelves=row.get("bookshelves"),
                )
                story = encode_story(
                    window, tags, tokenizer, settings, rng,
                    repository=repository,
                    path=row.get("path", ""),
                    source=row.get("source", ""),
                )
                segments[split].append(story)
                tokens[split] += len(story.ids)
                windows[split] += 1
                for name in story.tags:
                    tag_counts[name] += 1

            if books % 250 == 0:
                rate = tokens["train"] / max(1e-9, time.time() - started)
                print(
                    f"  {books:,} books  {tokens['train']:,} train tokens "
                    f"({rate:,.0f} tok/s)",
                    flush=True,
                )

    report = {
        "format": "brittain-shakespeare-prepare-report-v1",
        "block_size": args.block_size,
        "books": books,
        "seconds": round(time.time() - started, 1),
        "tokens": dict(tokens),
        "windows": dict(windows),
        "tag_coverage": {
            name: round(count / max(1, sum(windows.values())), 4)
            for name, count in tag_counts.most_common()
        },
    }

    for split in ("train", "validation"):
        if not segments[split]:
            raise SystemExit(f"{split} split is empty; lower --validation-fraction")
        inputs, labels, spans = pack_story_segments(
            segments[split], args.block_size, tokenizer.pad
        )
        destination = output_dir / f"{split}_{args.block_size}.npz"
        np.savez(destination, input_ids=inputs, labels=labels)
        report[f"{split}_rows"] = int(inputs.shape[0])
        report[f"{split}_supervised_tokens"] = int((labels != -100).sum())
        print(f"wrote {destination}  rows={inputs.shape[0]:,}", flush=True)
        del inputs, labels, spans

    report_path = output_dir / f"prepare_{args.block_size}.report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
