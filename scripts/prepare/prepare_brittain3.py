"""Prepare a local JSONL corpus for Brittain3 training.

This script does not download data. Each JSONL row must contain ``repository``,
``path``, ``text``, ``source``, and ``is_code``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.data_v3 import (
    Document,
    FIMSettings,
    encode_document,
    pack_segments,
    split_by_repository,
    token_controlled_mix,
    write_dataset,
)
from brittain.paths import BRITTAIN3_TOKENIZER, PROCESSED_DATA_DIR
from brittain.tokenizer_v3 import Brittain3Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="local JSONL corpus")
    parser.add_argument("--output-dir", default=str(PROCESSED_DATA_DIR / "brittain3"))
    parser.add_argument("--tokenizer", default=str(BRITTAIN3_TOKENIZER))
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fim-rate", type=float, default=0.40)
    parser.add_argument("--psm-rate", type=float, default=0.50)
    parser.add_argument("--line-rate", type=float, default=0.50)
    parser.add_argument("--block-rate", type=float, default=0.25)
    parser.add_argument("--weights", default=None, help="JSON object of source token weights")
    return parser.parse_args()


def read_documents(path: Path) -> list[Document]:
    documents = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                documents.append(Document(
                    repository=str(row["repository"]),
                    path=str(row["path"]),
                    text=str(row["text"]),
                    source=str(row["source"]),
                    is_code=bool(row["is_code"]),
                ))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise SystemExit(f"invalid input at {path}:{line_number}: {exc}") from exc
    return documents


def prepare_split(name, documents, tokenizer, args, settings, weights):
    if not documents:
        raise SystemExit(
            f"{name} split is empty; add repositories or change --validation-fraction"
        )
    groups = defaultdict(list)
    for document in documents:
        groups[document.source].append(document)
    if weights:
        missing = set(groups) - set(weights)
        if missing:
            raise SystemExit(f"{name} has source names with no weight: {sorted(missing)}")
        local_weights = {source: weights[source] for source in groups}
    else:
        local_weights = {source: 1.0 for source in groups}
    ordered = token_controlled_mix(
        groups, local_weights, lambda document: len(tokenizer.encode(document.text)), args.seed
    )
    rng = random.Random(args.seed + (1 if name == "validation" else 0))
    segments = [encode_document(document, tokenizer, args.block_size, settings, rng) for document in ordered]
    inputs, labels, spans = pack_segments(segments, args.block_size, tokenizer.pad)
    output = Path(args.output_dir) / f"{name}_{args.block_size}.npz"
    metadata = {
        "format": "brittain3-packed-v1",
        "split": name,
        "seed": args.seed,
        "block_size": args.block_size,
        "tokenizer": tokenizer.name,
        "tokenizer_path": str(tokenizer.path),
        "vocab_size": tokenizer.vocab_size,
        "source_weights": local_weights,
        "input": str(Path(args.input).resolve()),
        "fim": vars(settings),
        "documents": len(documents),
        "rows": len(inputs),
        "spans": spans,
    }
    write_dataset(output, inputs, labels, metadata)
    print(f"wrote {output}: {len(inputs)} rows")


def main():
    args = parse_args()
    tokenizer = Brittain3Tokenizer(args.tokenizer)
    documents = read_documents(Path(args.input))
    train, validation = split_by_repository(documents, args.validation_fraction, args.seed)
    settings = FIMSettings(args.fim_rate, args.psm_rate, args.line_rate, args.block_rate)
    weights = json.loads(args.weights) if args.weights else None
    prepare_split("train", train, tokenizer, args, settings, weights)
    prepare_split("validation", validation, tokenizer, args, settings, weights)


if __name__ == "__main__":
    main()
