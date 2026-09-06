"""Prepare a local JSONL corpus for Brittain3 training.

This script does not download data. Each JSONL row must contain ``repository``,
``path``, ``text``, ``source``, and ``is_code``.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.data_v3 import (
    Document,
    FIMSettings,
    encode_document,
    pack_segments_streaming,
    split_by_repository,
    token_controlled_mix,
    write_dataset,
)

# Categories that receive fill-in-the-middle. Prose and structured text do not.
FIM_CATEGORIES = {"code", "exercises"}
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
    parser.add_argument("--keep-spans", action="store_true",
                        help="record per-segment spans in the metadata. One dict "
                             "per segment: at corpus scale this makes the metadata "
                             "JSON hundreds of megabytes. Spans are analysis data "
                             "and are not used by training.")
    return parser.parse_args()


def read_documents(path: Path) -> list[Document]:
    documents = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                # The collector writes `category`, not `is_code`. FIM applies to
                # code and to the generated exercises; prose and structured text
                # must not be holed out.
                if "is_code" in row:
                    is_code = bool(row["is_code"])
                else:
                    is_code = row["category"] in FIM_CATEGORIES
                repository = str(row["repository"])
                # Older curriculum builds named each teacher replay as a new
                # repository. Normalize it so exact copies cannot cross the
                # repository-level train and validation split.
                if str(row.get("source", "")) == "verified_teacher":
                    repository = re.sub(r"/replay-\d+$", "", repository)
                documents.append(Document(
                    repository=repository,
                    path=str(row["path"]),
                    text=str(row["text"]),
                    source=str(row["source"]),
                    is_code=is_code,
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
    # A GENERATOR, not a list. Materialising every EncodedSegment for the pilot
    # corpus was projected to peak near 49GB on a 38GB machine, because each
    # token becomes a boxed Python int. Streaming keeps only one segment alive.
    segments = (
        encode_document(document, tokenizer, args.block_size, settings, rng)
        for document in ordered
    )
    inputs, labels, spans = pack_segments_streaming(
        segments, args.block_size, tokenizer.pad, keep_spans=args.keep_spans,
    )
    valid_labels = int((labels != -100).sum())
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
        "tokens": int(inputs.size),
        "valid_labels": valid_labels,
        "valid_label_fraction": valid_labels / max(1, int(inputs.size)),
        "spans": spans if args.keep_spans else [],
        "spans_recorded": bool(args.keep_spans),
    }
    write_dataset(output, inputs, labels, metadata)
    print(f"wrote {output}: {len(inputs):,} rows, {inputs.size:,} tokens", flush=True)


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
