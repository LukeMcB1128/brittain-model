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
import os
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool, freeze_support
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.data_story import (
    StorySettings, TokenLengths, encode_story, window_text, write_packed_stories,
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
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="processes encoding documents; the parent needs one core to "
             "collect their results",
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


def encode_document(row, tokenizer, settings, seed):
    """Window and encode one document into an ordered group of segments.

    The group is kept together and in order because consecutive windows of a
    book are packed contiguously and marked as continuations. Shuffling them
    would make that marker a lie.

    The tag policy is drawn from a generator seeded per document rather than
    from one walked across the corpus, so a document's tag presentation depends
    only on the document and the run seed. That is what lets documents be
    encoded in parallel and still produce the same output every time.
    """
    text = row.get("text") or ""
    if not text:
        return []
    repository = row["repository"]
    rng = random.Random(f"{seed}:{repository}")
    # Twist has no extractor, and Genre and Voice come from metadata, so those
    # three are carried from the document. Every other tag is re-derived from
    # the window, so a tag the source claimed but the text does not support
    # never reaches training.
    carried = {
        name: value
        for name, value in (row.get("book_tags") or {}).items()
        if name in CARRIED_TAGS
    }
    # Shared with the windower so the token count below is already cached rather
    # than a second pass over the same window.
    lengths = TokenLengths(tokenizer)
    group = []
    for position, window in enumerate(
        window_text(text, tokenizer, settings, lengths=lengths)
    ):
        tags = {
            **carried,
            **extract(
                window,
                token_count=lengths(window),
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


_WORKER: dict = {}


def _start_worker(tokenizer_path, config_path, block_size, seed):
    """Build the per-process tokenizer once, not once per document."""
    # The tokenizer runs its own thread pool per call, which on a 4-core machine
    # fights the process pool for the same cores.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _WORKER["tokenizer"] = StoryTokenizer(tokenizer_path)
    _WORKER["settings"] = build_settings(config, block_size)
    _WORKER["seed"] = seed


def _encode_line(line):
    """Parse and encode one corpus line in a worker.

    Returns the routing fields alongside the segments so the parent never has to
    parse the JSON itself; at 85,000 documents that parse is not free either.
    """
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        # The generator appends while this reads, so the final line can be half
        # written. Skipping it loses one story.
        return None
    group = encode_document(row, _WORKER["tokenizer"], _WORKER["settings"],
                            _WORKER["seed"])
    if not group:
        return None
    return row["repository"], row.get("source", ""), group


def iter_corpus_lines(corpora, limit_books):
    """Yield raw lines from every corpus, honouring the per-corpus limit."""
    for corpus_name in corpora:
        corpus = project_path(corpus_name)
        if not corpus.exists():
            raise SystemExit(f"corpus not found: {corpus}")
        print(f"reading {corpus}", flush=True)
        taken = 0
        with corpus.open(encoding="utf-8") as handle:
            for line in handle:
                if limit_books is not None and taken >= limit_books:
                    break
                if not line.strip():
                    continue
                taken += 1
                yield line


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
    destination = output_dir / f"{name}_{block_size}"
    stats = write_packed_stories(segments, block_size, pad, destination)
    del segments
    report[f"{name}_rows"] = stats["rows"]
    report[f"{name}_supervised_tokens"] = stats["supervised_tokens"]
    print(f"wrote {destination}  rows={stats['rows']:,}", flush=True)


def main():
    args = parse_args()
    config = json.loads(project_path(args.config).read_text(encoding="utf-8"))
    # The parent only needs the pad id; the workers do the encoding, each with
    # its own tokenizer and settings.
    tokenizer = StoryTokenizer(project_path(args.tokenizer))

    corpora = args.corpus or [config.get("output", "data/raw/brittain-shakespeare/corpus.jsonl")]
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / f"train_{args.block_size}").exists() and not args.overwrite:
        raise SystemExit("output exists; pass --overwrite to replace it")

    rng = random.Random(args.seed)
    real_groups: list[list] = []
    synthetic_groups: list[list] = []
    validation_groups: list[list] = []
    documents = 0
    windows = 0
    tag_counts: Counter[str] = Counter()
    started = time.time()

    lines = iter_corpus_lines(corpora, args.limit_books)
    # Documents are independent, and windowing is pure tokenizer work, so this
    # is the one place in the pipeline where more cores buy anything. Results
    # come back in order, so the output does not depend on the worker count.
    with Pool(
        args.workers,
        initializer=_start_worker,
        initargs=(str(project_path(args.tokenizer)), str(project_path(args.config)),
                  args.block_size, args.seed),
    ) as pool:
        for result in pool.imap(_encode_line, lines, chunksize=8):
            if result is None:
                continue
            repository, source, group = result
            documents += 1
            windows += len(group)
            for segment in group:
                for tag in segment.tags:
                    tag_counts[tag] += 1
            if repository_in_validation(
                repository, args.validation_fraction, args.seed
            ):
                validation_groups.append(group)
            elif source == SYNTHETIC_SOURCE:
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
    freeze_support()
    with keep_awake("brittain-shakespeare corpus preparation"):
        main()
