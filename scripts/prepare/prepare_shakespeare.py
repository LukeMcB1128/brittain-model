"""Turn the brittain-shakespeare corpora into packed training stages.

Windows each document, tags each window, applies the tag randomization policy,
and packs whole stories into rows. Whole books go to train or validation so no
book spans both splits.

    python3 scripts/prepare/prepare_shakespeare.py --block-size 1024 \\
        --corpus data/raw/brittain-shakespeare/corpus.jsonl \\
        --corpus data/raw/brittain-shakespeare-synthetic/stories.jsonl

``--corpus`` is repeatable. The synthetic stories are held to a small share of
the main set and then concentrated into a separate anneal set, because they are
the only documents in the project that are whole stories: everything else is a
window from the middle of a novel. A model that finishes training on
story-shaped data produces story-shaped output; one that finishes on chapter
fourteen produces chapter fourteen.
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
from brittain.keep_awake import keep_awake
from brittain.story_tagger import extract
from brittain.tags import CARRIED_TAGS, TagPolicy
from brittain.tokenizer_story import STORY_TOKENIZER, StoryTokenizer

SYNTHETIC_SOURCE = "synthetic"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", action="append", default=None,
        help="a corpus JSONL; repeat the flag to combine several",
    )
    parser.add_argument("--config", default="configs/data/shakespeare_corpus.json")
    parser.add_argument("--tokenizer", default=str(STORY_TOKENIZER))
    parser.add_argument("--output-dir", default="data/processed/brittain-shakespeare")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--synthetic-share", type=float, default=0.05,
        help="share of the main set's tokens that may come from synthetic stories",
    )
    parser.add_argument(
        "--anneal-tokens", type=int, default=0,
        help="size of a separate high-synthetic set for the end of training; "
             "0 writes no anneal set",
    )
    parser.add_argument(
        "--anneal-share", type=float, default=0.35,
        help="share of the anneal set's tokens that come from synthetic stories",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--limit-books", type=int, default=None,
        help="documents to read FROM EACH corpus, for smoke tests. A global "
             "limit would spend itself on the first corpus and never reach the "
             "second, which is exactly the mix a smoke test needs to check.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_settings(config, block_size):
    windowing = config.get("windowing", {})
    policy_cfg = config.get("tag_policy", {})
    # A window plus its tag block and sentinels must fit one row, so the block
    # size caps the window. At small block sizes that cap falls below the
    # configured target, and target and minimum have to come down with it.
    maximum_tokens = min(int(windowing.get("maximum_tokens", 3800)), block_size - 64)
    target_tokens = min(int(windowing.get("target_tokens", 1400)), maximum_tokens)
    minimum_tokens = min(int(windowing.get("minimum_tokens", 200)), target_tokens)
    return StorySettings(
        block_size=block_size,
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


def encode_document(row, tokenizer, settings, rng):
    """Window and encode one document into an ordered group of segments.

    The group is kept together and in order because consecutive windows of a
    book are packed contiguously and marked as continuations. Shuffling them
    would make that marker a lie.
    """
    text = row.get("text") or ""
    if not text:
        return []
    repository = row["repository"]
    # Twist has no extractor, and Genre and Voice come from metadata, so those
    # three are carried from the document. Every other tag is re-derived from
    # the window, so a tag the source claimed but the text does not support
    # never reaches training.
    carried = {
        name: value
        for name, value in (row.get("book_tags") or {}).items()
        if name in CARRIED_TAGS
    }
    group = []
    for position, window in enumerate(window_text(text, tokenizer, settings)):
        tags = {
            **carried,
            **extract(
                window,
                token_count=len(tokenizer.encode(window)),
                birth_year=row.get("birth_year"),
                death_year=row.get("death_year"),
                subjects=row.get("subjects"),
                bookshelves=row.get("bookshelves"),
            ),
        }
        group.append(encode_story(
            window, tags, tokenizer, settings, rng,
            repository=repository, path=row.get("path", ""),
            source=row.get("source", ""), continues=position > 0,
        ))
    return group


def group_tokens(group):
    return sum(len(segment.ids) for segment in group)


def mix(real_groups, synthetic_groups, share, budget, rng):
    """Interleave document groups so synthetic holds roughly ``share`` of tokens.

    Groups stay whole and internally ordered. Synthetic documents are single
    stories, so each is a group of one.
    """
    real = list(real_groups)
    synthetic = list(synthetic_groups)
    rng.shuffle(real)
    rng.shuffle(synthetic)
    chosen, tokens, synthetic_tokens = [], 0, 0
    real_index = synthetic_index = 0
    while (real_index < len(real) or synthetic_index < len(synthetic)):
        if budget is not None and tokens >= budget:
            break
        want_synthetic = (
            synthetic_index < len(synthetic)
            and (tokens == 0 or synthetic_tokens / max(1, tokens) < share)
        )
        if want_synthetic:
            group = synthetic[synthetic_index]
            synthetic_index += 1
            synthetic_tokens += group_tokens(group)
        elif real_index < len(real):
            group = real[real_index]
            real_index += 1
        else:
            break
        chosen.append(group)
        tokens += group_tokens(group)
    return chosen, tokens, synthetic_tokens


def write_split(name, groups, block_size, pad, output_dir, report):
    segments = [segment for group in groups for segment in group]
    if not segments:
        raise SystemExit(f"{name} split is empty")
    inputs, labels, _ = pack_story_segments(segments, block_size, pad)
    destination = output_dir / f"{name}_{block_size}.npz"
    np.savez(destination, input_ids=inputs, labels=labels)
    report[f"{name}_rows"] = int(inputs.shape[0])
    report[f"{name}_supervised_tokens"] = int((labels != -100).sum())
    print(f"wrote {destination}  rows={inputs.shape[0]:,}", flush=True)
    del inputs, labels


def main():
    args = parse_args()
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    tokenizer = StoryTokenizer(project_path(args.tokenizer))
    settings = build_settings(config, args.block_size)

    corpora = args.corpus or [config.get("output", "data/raw/brittain-shakespeare/corpus.jsonl")]
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / f"train_{args.block_size}.npz").exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite to replace it")

    rng = random.Random(args.seed)
    real_groups: list[list] = []
    synthetic_groups: list[list] = []
    validation_groups: list[list] = []
    documents = 0
    windows = 0
    tag_counts: Counter[str] = Counter()
    started = time.time()

    for corpus_name in corpora:
        corpus = project_path(corpus_name)
        if not corpus.exists():
            raise SystemExit(f"corpus not found: {corpus}")
        print(f"reading {corpus}", flush=True)
        from_this_corpus = 0
        with corpus.open(encoding="utf-8") as handle:
            for line in handle:
                if args.limit_books is not None and from_this_corpus >= args.limit_books:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # The generator appends while this reads, so the final line
                    # can be half written. Skipping it loses one story.
                    continue
                group = encode_document(row, tokenizer, settings, rng)
                if not group:
                    continue
                documents += 1
                from_this_corpus += 1
                windows += len(group)
                for segment in group:
                    for tag in segment.tags:
                        tag_counts[tag] += 1
                if repository_in_validation(
                    row["repository"], args.validation_fraction, args.seed
                ):
                    validation_groups.append(group)
                elif row.get("source") == SYNTHETIC_SOURCE:
                    synthetic_groups.append(group)
                else:
                    real_groups.append(group)
                if documents % 500 == 0:
                    print(f"  {documents:,} documents  {time.time() - started:.0f}s",
                          flush=True)

    print(f"real documents {len(real_groups):,}  synthetic {len(synthetic_groups):,}",
          flush=True)

    train_groups, tokens, synthetic_tokens = mix(
        real_groups, synthetic_groups, args.synthetic_share, args.max_tokens, rng
    )
    report = {
        "format": "brittain-shakespeare-prepare-report-v1",
        "block_size": args.block_size,
        "corpora": [str(project_path(c)) for c in corpora],
        "documents": documents,
        "seconds": round(time.time() - started, 1),
        "train_tokens": tokens,
        "train_synthetic_tokens": synthetic_tokens,
        "train_synthetic_share": round(synthetic_tokens / max(1, tokens), 4),
        "windows": windows,
        "tag_coverage": {
            name: round(count / max(1, windows), 4)
            for name, count in tag_counts.most_common()
        },
    }
    write_split("train", train_groups, args.block_size, tokenizer.pad, output_dir, report)
    write_split("validation", validation_groups, args.block_size, tokenizer.pad,
                output_dir, report)

    if args.anneal_tokens > 0 and synthetic_groups:
        # The anneal is what shapes output format. A model that finishes on
        # story-shaped data produces story-shaped output.
        anneal_groups, anneal_tokens, anneal_synthetic = mix(
            real_groups, synthetic_groups, args.anneal_share, args.anneal_tokens, rng
        )
        report["anneal_tokens"] = anneal_tokens
        report["anneal_synthetic_share"] = round(
            anneal_synthetic / max(1, anneal_tokens), 4
        )
        write_split("anneal", anneal_groups, args.block_size, tokenizer.pad,
                    output_dir, report)

    report_path = output_dir / f"prepare_{args.block_size}.report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with keep_awake("brittain-shakespeare corpus preparation"):
        main()
