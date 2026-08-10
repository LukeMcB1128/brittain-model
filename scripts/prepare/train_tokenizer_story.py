"""Train the brittain-shakespeare 8K prose byte-level BPE.

Reads the corpus JSONL produced by ``build_story_corpus.py``. BPE converges on a
few hundred megabytes, so the full 1.27B-word corpus is sampled rather than read
whole: a deterministic per-book stride keeps the sample spread across the whole
catalog instead of the alphabetically first books.

Early Modern and drama text is oversampled on purpose. It is a small share of the
corpus but carries the archaic spellings and elisions the Shakespearean register
depends on, and merges for `'tis`, `o'er`, and `hast` only appear if the trainer
sees enough of them.

    python3 scripts/prepare/train_tokenizer_story.py \\
        --corpus data/raw/brittain-shakespeare/corpus.jsonl \\
        --output tokenizers/brittain-shakespeare-prose-8k/tokenizer.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.tokenizer_story import (
    STORY_SPECIAL_TOKENS,
    STORY_VOCAB_SIZE,
    StoryTokenizer,
    validate_tokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/raw/brittain-shakespeare/corpus.jsonl")
    parser.add_argument(
        "--output", default="tokenizers/brittain-shakespeare-prose-8k/tokenizer.json"
    )
    parser.add_argument("--vocab-size", type=int, default=STORY_VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-token-length", type=int, default=24)
    parser.add_argument(
        "--sample-bytes", type=int, default=900_000_000,
        help="safety cap on sampled UTF-8 bytes",
    )
    parser.add_argument(
        "--per-document-bytes", type=int, default=30_000,
        help="slice taken from each book; a fixed slice per book covers the whole "
             "corpus instead of exhausting the budget on the first few thousand",
    )
    parser.add_argument(
        "--archaic-boost", type=int, default=3,
        help="how many times to repeat Shakespearean-voice documents",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--report", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def iter_training_text(corpus: Path, args):
    """Yield text chunks up to the sample budget, spread across the corpus."""
    rng = random.Random(args.seed)
    budget = args.sample_bytes
    used = 0
    archaic_used = 0
    documents = 0
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            if used >= budget:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or ""
            if not text:
                continue
            documents += 1
            archaic = (row.get("book_tags") or {}).get("Voice") == "Shakespearean"

            # A fixed slice per book, taken from a random offset. Reading books
            # whole until the budget filled would have sampled only the first few
            # thousand and missed most of the corpus, including nearly all the
            # Shakespearean-voice titles.
            span = args.per_document_bytes * (4 if archaic else 1)
            if len(text) > span:
                start = rng.randrange(0, len(text) - span)
                text = text[start: start + span]

            repeats = args.archaic_boost if archaic else 1
            for _ in range(repeats):
                yield text
                size = len(text.encode("utf-8", errors="ignore"))
                used += size
                if archaic:
                    archaic_used += size
                if used >= budget:
                    break
    print(f"sampled {used:,} bytes from {documents:,} documents", flush=True)
    print(f"  of which Shakespearean voice: {archaic_used:,} bytes", flush=True)


def main():
    args = parse_args()
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    corpus = project_path(args.corpus)
    if not corpus.exists():
        raise SystemExit(f"corpus not found: {corpus}; run build_story_corpus.py first")
    output = project_path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.vocab_size != STORY_VOCAB_SIZE:
        raise SystemExit(
            f"brittain-shakespeare requires vocab {STORY_VOCAB_SIZE}, "
            f"got {args.vocab_size}"
        )

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=list(STORY_SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=args.max_token_length,
        show_progress=True,
    )

    print(f"training {args.vocab_size} vocab from {corpus}", flush=True)
    tokenizer.train_from_iterator(iter_training_text(corpus, args), trainer=trainer)
    tokenizer.save(str(output))
    print(f"wrote {output} with vocab {tokenizer.get_vocab_size()}", flush=True)

    loaded = StoryTokenizer(output)
    metrics = validate_tokenizer(loaded)
    print("\nvalidation (bytes per token):", flush=True)
    for name, row in metrics.items():
        print(f"  {name:14} {row['bytes_per_token']:.2f}", flush=True)

    report = {
        "format": "brittain-shakespeare-tokenizer-report-v1",
        "output": str(output),
        "vocab_size": loaded.vocab_size,
        "sample_bytes": args.sample_bytes,
        "archaic_boost": args.archaic_boost,
        "min_frequency": args.min_frequency,
        "max_token_length": args.max_token_length,
        "seed": args.seed,
        "metrics": metrics,
    }
    report_path = project_path(
        args.report or output.parent / "tokenizer.report.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
