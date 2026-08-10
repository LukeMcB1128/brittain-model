"""Build the brittain-shakespeare narrative-prose corpus.

Two stages, deliberately separable:

1. ``--catalog-only`` reads a local Project Gutenberg RDF catalog, applies the
   metadata half of the fiction filter, and writes a funnel report saying how
   many books survive each rule. It reads no book text and downloads nothing, so
   it is cheap enough to run repeatedly while calibrating the filter.
2. The full run additionally reads book text from a local mirror directory,
   applies the dialogue-density and narrative-verb floors, tags each book, and
   writes the corpus JSONL.

Neither stage downloads anything. Fetch the catalog and the text mirror first:

    https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2

Run the funnel on a small slice before trusting the thresholds:

    python3 scripts/prepare/build_story_corpus.py \\
        --catalog data/raw/gutenberg/rdf-files.tar.bz2 \\
        --catalog-only --limit 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.corpus_story import (
    REJECTION_REASONS,
    TEXTURE_CATEGORY,
    BookRecord,
    FilterConfig,
    TextureConfig,
    book_tags,
    classify,
    iter_catalog,
    read_book_text,
    read_catalog_directory,
    text_rejection_reason,
)
from brittain.story_tagger import words

CONFIG_FORMAT = "brittain-shakespeare-corpus-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/shakespeare_corpus.json")
    parser.add_argument(
        "--catalog", required=True,
        help="local rdf-files.tar.bz2, or a directory of loose .rdf files",
    )
    parser.add_argument(
        "--text-dir", default=None,
        help="local Gutenberg text mirror; required unless --catalog-only",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--catalog-only", action="store_true",
        help="apply metadata rules and write the funnel report; read no text",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="stop after this many catalog records; use for a dry run",
    )
    parser.add_argument(
        "--emit-ids", default=None,
        help="write the surviving ebook ids to this file, one per line, so only "
             "those books need fetching instead of the whole 10GB text archive",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path):
    with project_path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("format") != CONFIG_FORMAT:
        raise SystemExit(f"unsupported corpus configuration: {config.get('format')!r}")
    if "fiction_filter" not in config:
        raise SystemExit("configuration is missing fiction_filter")
    return config


def catalog_records(catalog: Path, limit: int | None):
    if catalog.is_dir():
        records = read_catalog_directory(catalog)
        for index, record in enumerate(records):
            if limit is not None and index >= limit:
                return
            yield record
        return
    yield from iter_catalog(catalog, limit=limit)


def book_text_path(text_dir: Path, book_id: int) -> Path | None:
    """Find one book's text in a local mirror, trying the usual layouts."""
    candidates = (
        text_dir / f"{book_id}.txt",
        text_dir / f"pg{book_id}.txt",
        text_dir / f"{book_id}.txt.bz2",
        text_dir / f"pg{book_id}.txt.bz2",
        text_dir / str(book_id) / f"pg{book_id}.txt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def corpus_row(record: BookRecord, text: str, tags: dict[str, str], category: str) -> dict:
    """One JSONL row.

    ``repository`` carries the book identity so the existing
    ``data_v3.split_by_repository`` assigns whole books to train or validation.
    """
    return {
        "repository": f"gutenberg/{record.book_id}",
        "path": record.title or f"ebook-{record.book_id}",
        "text": text,
        "source": category,
        "is_code": False,
        "author": record.author,
        "birth_year": record.birth_year,
        "death_year": record.death_year,
        "subjects": record.subjects,
        "bookshelves": record.bookshelves,
        "lcc": record.lcc,
        "rights": record.rights,
        "book_tags": tags,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    fiction = config["fiction_filter"]
    filter_config = FilterConfig.from_config(config)
    texture_config = TextureConfig.from_config(config)

    catalog = project_path(args.catalog)
    if not catalog.exists():
        raise SystemExit(
            f"catalog not found: {catalog}\n"
            "Fetch https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2 first."
        )

    text_dir = None
    if not args.catalog_only:
        if args.text_dir is None:
            raise SystemExit("--text-dir is required unless --catalog-only is set")
        text_dir = project_path(args.text_dir)
        if not text_dir.is_dir():
            raise SystemExit(f"text mirror not found: {text_dir}")

    output_path = project_path(args.output or config["output"])
    report_path = project_path(args.report or config["report"])
    if not args.catalog_only and output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")

    minimum_speech = float(fiction.get("minimum_speech_fraction", 0.0))
    minimum_verb_ratio = float(fiction.get("minimum_narrative_verb_ratio", 0.0))
    minimum_tokens = int(fiction.get("minimum_document_tokens", 0))
    maximum_tokens = int(fiction.get("maximum_document_tokens", 10**9))

    texture_share = texture_config.maximum_corpus_share if texture_config else 0.0

    seen = 0
    rejected: Counter[str] = Counter()
    metadata_passed = 0
    accepted = 0
    accepted_words = 0
    by_category: Counter[str] = Counter()
    words_by_category: Counter[str] = Counter()
    metadata_by_category: Counter[str] = Counter()
    tag_coverage: Counter[str] = Counter()
    voice_counts: Counter[str] = Counter()
    genre_counts: Counter[str] = Counter()

    handle = None
    if not args.catalog_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", encoding="utf-8")

    ids_handle = None
    if args.emit_ids:
        ids_path = project_path(args.emit_ids)
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_handle = ids_path.open("w", encoding="utf-8")

    try:
        for record in catalog_records(catalog, args.limit):
            seen += 1
            category, reason = classify(record, filter_config, texture_config)
            if category is None:
                rejected[reason] += 1
                continue
            metadata_passed += 1
            metadata_by_category[category] += 1
            if ids_handle is not None:
                ids_handle.write(f"{record.book_id}\t{category}\n")

            tags = book_tags(record)
            for name in tags:
                tag_coverage[name] += 1
            if "Voice" in tags:
                voice_counts[tags["Voice"]] += 1
            if "Genre" in tags:
                genre_counts[tags["Genre"]] += 1

            if args.catalog_only:
                continue

            path = book_text_path(text_dir, record.book_id)
            if path is None:
                rejected["text_missing"] += 1
                continue
            text = read_book_text(path)
            word_count = len(words(text))
            # Token counts run about a third above word counts, but the exact
            # figure needs the tokenizer, which does not exist yet. Words are the
            # honest proxy at this stage.
            if word_count < minimum_tokens * 0.75:
                rejected["too_short"] += 1
                continue
            if word_count > maximum_tokens:
                rejected["too_long"] += 1
                continue
            # The texture path carries more expository prose, so it faces
            # stricter floors. They are what separate a book of legends from a
            # regnal chronology.
            if category == TEXTURE_CATEGORY:
                floor_speech = texture_config.minimum_speech_fraction
                floor_verbs = texture_config.minimum_narrative_verb_ratio
            else:
                floor_speech, floor_verbs = minimum_speech, minimum_verb_ratio
            text_reason = text_rejection_reason(
                text,
                minimum_speech=floor_speech,
                minimum_verb_ratio=floor_verbs,
            )
            if text_reason is not None:
                rejected[f"{category}:{text_reason}"] += 1
                continue

            # Texture is capped so it can never dilute the story corpus.
            if category == TEXTURE_CATEGORY and accepted_words:
                projected = words_by_category[TEXTURE_CATEGORY] + word_count
                if projected > texture_share * (accepted_words + word_count):
                    rejected["texture_cap_reached"] += 1
                    continue

            handle.write(json.dumps(corpus_row(record, text, tags, category)) + "\n")
            accepted += 1
            accepted_words += word_count
            by_category[category] += 1
            words_by_category[category] += word_count
    finally:
        if handle is not None:
            handle.close()
        if ids_handle is not None:
            ids_handle.close()

    report = {
        "format": "brittain-shakespeare-corpus-report-v1",
        "catalog": str(catalog),
        "catalog_only": args.catalog_only,
        "limit": args.limit,
        "seen": seen,
        "metadata_passed": metadata_passed,
        "accepted": accepted if not args.catalog_only else None,
        "accepted_words": accepted_words if not args.catalog_only else None,
        "rejected": dict(rejected.most_common()),
        "metadata_by_category": dict(metadata_by_category),
        "accepted_by_category": dict(by_category),
        "accepted_words_by_category": dict(words_by_category),
        "texture_share_of_accepted_words": (
            words_by_category[TEXTURE_CATEGORY] / accepted_words
            if accepted_words else None
        ),
        "metadata_tag_coverage": dict(tag_coverage),
        "voice_distribution": dict(voice_counts),
        "genre_distribution": dict(genre_counts),
        "thresholds": {
            "minimum_speech_fraction": minimum_speech,
            "minimum_narrative_verb_ratio": minimum_verb_ratio,
            "minimum_document_tokens": minimum_tokens,
            "maximum_document_tokens": maximum_tokens,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"catalog records read: {seen}")
    print(f"passed metadata rules: {metadata_passed}")
    for name, count in metadata_by_category.most_common():
        print(f"  {name}: {count}")
    for reason, count in sorted(rejected.items(), key=lambda item: -item[1]):
        print(f"  rejected {reason}: {count}")
    if tag_coverage:
        print("metadata tag coverage (of books passing metadata rules):")
        for name, count in tag_coverage.most_common():
            share = count / metadata_passed if metadata_passed else 0.0
            print(f"  {name}: {count} ({share:.1%})")
    if not args.catalog_only:
        print(f"accepted books: {accepted}")
        print(f"accepted words: {accepted_words:,}")
        for name, count in by_category.most_common():
            share = words_by_category[name] / accepted_words if accepted_words else 0.0
            print(f"  {name}: {count} books, {words_by_category[name]:,} words ({share:.1%})")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
