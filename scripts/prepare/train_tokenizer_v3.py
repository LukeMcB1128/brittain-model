"""Train or validate the Brittain3 24K byte-level BPE tokenizer.

This script reads local text or JSONL files. It does not download data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from brittain.paths import BRITTAIN3_TOKENIZER
from brittain.tokenizer import CodeTok
from brittain.tokenizer_v3 import (
    BRITTAIN3_SPECIAL_TOKENS,
    Brittain3Tokenizer,
    save_validation_report,
    validate_tokenizer,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[], help="local text or JSONL input; repeat as needed")
    parser.add_argument("--jsonl-field", default="text")
    parser.add_argument("--output", default=str(BRITTAIN3_TOKENIZER))
    parser.add_argument("--vocab-size", type=int, default=24_576)
    parser.add_argument("--max-document-chars", type=int, default=100_000)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report", default=None)
    parser.add_argument("--compare-brittain2", action="store_true")
    parser.add_argument("--max-code-regression", type=float, default=0.03)
    return parser.parse_args()


def documents(paths, field, limit):
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"input does not exist: {path}")
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        text = row[field]
                    except (json.JSONDecodeError, KeyError) as exc:
                        raise SystemExit(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
                    if isinstance(text, str) and text.strip():
                        yield text[:limit]
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                yield text[:limit]


def main():
    args = arguments()
    if args.vocab_size != 24_576:
        raise SystemExit("Brittain3 requires --vocab-size 24576")
    output = Path(args.output)
    if not args.validate_only:
        if not args.input:
            raise SystemExit("give at least one --input, or use --validate-only")
        output.parent.mkdir(parents=True, exist_ok=True)
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=list(BRITTAIN3_SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tokenizer.train_from_iterator(
            documents(args.input, args.jsonl_field, args.max_document_chars),
            trainer=trainer,
        )
        tokenizer.save(str(output))
        print(f"saved {output} with vocab {tokenizer.get_vocab_size()}")
    reference = CodeTok() if args.compare_brittain2 else None
    report = validate_tokenizer(Brittain3Tokenizer(output), reference=reference)
    if reference is not None:
        regression = report["samples"]["code"]["token_change_fraction"]
        if regression > args.max_code_regression:
            raise SystemExit(
                f"code token count regressed by {regression:.1%}; "
                f"limit is {args.max_code_regression:.1%}"
            )
    report_path = Path(args.report) if args.report else output.with_name("validation.json")
    save_validation_report(report, report_path)
    print(json.dumps(report, indent=2))
    print(f"saved validation report to {report_path}")


if __name__ == "__main__":
    main()
