"""Train or validate the Brittain3 24K byte-level BPE tokenizer.

This script reads local text or JSONL files. It does not download data.
"""
from __future__ import annotations

import argparse
import json
import os
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
    evaluate_tokenizer_corpus,
    save_validation_report,
    validate_tokenizer,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[], help="local text or JSONL input; repeat as needed")
    parser.add_argument("--jsonl-field", default="text")
    parser.add_argument("--output", default=str(BRITTAIN3_TOKENIZER))
    parser.add_argument("--vocab-size", type=int, default=24_576)
    parser.add_argument("--max-document-chars", type=int, default=300_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-token-length", type=int, default=64)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report", default=None)
    parser.add_argument("--compare-brittain2", action="store_true")
    parser.add_argument("--max-code-regression", type=float, default=0.03)
    parser.add_argument(
        "--evaluation", action="append", default=[],
        help="held-out provenance JSONL; repeat as needed",
    )
    parser.add_argument("--max-evaluation-bytes", type=int, default=10_000_000)
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


def portable_report_path(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def main():
    args = arguments()
    if args.vocab_size != 24_576:
        raise SystemExit("Brittain3 requires --vocab-size 24576")
    if args.min_frequency < 1 or args.max_token_length < 1:
        raise SystemExit("--min-frequency and --max-token-length must be positive")
    output = Path(args.output)
    candidate = output.with_name(output.name + ".candidate")
    if not args.validate_only:
        if not args.input:
            raise SystemExit("give at least one --input, or use --validate-only")
        output.parent.mkdir(parents=True, exist_ok=True)
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=list(BRITTAIN3_SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            max_token_length=args.max_token_length,
            show_progress=True,
        )
        tokenizer.train_from_iterator(
            documents(args.input, args.jsonl_field, args.max_document_chars),
            trainer=trainer,
        )
        tokenizer.save(str(candidate))
        print(f"wrote candidate {candidate} with vocab {tokenizer.get_vocab_size()}")
    reference = CodeTok() if args.compare_brittain2 else None
    validation_path = output if args.validate_only else candidate
    validation_tokenizer = Brittain3Tokenizer(validation_path)
    report = validate_tokenizer(validation_tokenizer, reference=reference)
    if args.evaluation:
        report["corpus_evaluation"] = evaluate_tokenizer_corpus(
            validation_tokenizer, args.evaluation,
            reference=reference, maximum_bytes=args.max_evaluation_bytes,
        )
    if reference is not None:
        corpus_evaluation = report.get("corpus_evaluation", {})
        corpus_code = corpus_evaluation.get("categories", {}).get("code")
        regression = (
            corpus_code["token_change_fraction"] if corpus_code
            else report["samples"]["code"]["token_change_fraction"]
        )
        if regression > args.max_code_regression:
            raise SystemExit(
                f"code token count regressed by {regression:.1%}; "
                f"limit is {args.max_code_regression:.1%}"
            )
        for language, metrics in corpus_evaluation.get("code_languages", {}).items():
            regression = metrics["token_change_fraction"]
            if regression > args.max_code_regression:
                raise SystemExit(
                    f"{language} token count regressed by {regression:.1%}; "
                    f"limit is {args.max_code_regression:.1%}"
                )
    if not args.validate_only:
        os.replace(candidate, output)
        print(f"accepted candidate and saved {output}")
    report["path"] = portable_report_path(output)
    report_path = Path(args.report) if args.report else output.with_name("validation.json")
    save_validation_report(report, report_path)
    print(json.dumps(report, indent=2))
    print(f"saved validation report to {report_path}")


if __name__ == "__main__":
    main()
