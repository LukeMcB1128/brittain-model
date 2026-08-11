#!/usr/bin/env python3
"""Reverify and filter a Brittain3 teacher seed bank.

This command is the last gate between local model generation and corpus
packing. It preserves the raw seed bank and writes a separate clean bank.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.corpus_v3 import SECRET_PATTERNS  # noqa: E402
from brittain.curriculum_quality_v3 import DuplicateGuard, EvaluationGuard  # noqa: E402
from brittain.verification_v3 import DEFAULT_TSC, backend_status, verify_program  # noqa: E402


REQUIRED_FIELDS = {
    "id", "text", "language", "entry_point", "brief", "solution",
    "author_tests", "reviewer_tests",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/generated/brittain3-curriculum/teacher-seeds.jsonl"
    )
    parser.add_argument(
        "--output", default="data/generated/brittain3-curriculum/teacher-seeds.clean.jsonl"
    )
    parser.add_argument(
        "--report", default="runs/brittain3-teacher-seeds.clean.report.json"
    )
    parser.add_argument("--verify-timeout", type=float, default=15.0)
    parser.add_argument("--tsc", default=str(DEFAULT_TSC))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def semantic_text(brief: dict) -> str:
    return " ".join([
        str(brief.get("slug", "")), str(brief.get("goal", "")),
        str(brief.get("input_contract", "")), str(brief.get("output_contract", "")),
        *(str(value) for value in brief.get("edge_cases", [])),
    ])


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def audit_rows(rows: list[dict], verify_timeout: float, tsc: str) -> tuple[list[dict], dict]:
    evaluation_guard = EvaluationGuard.novice_v1()
    duplicate_guard = DuplicateGuard()
    accepted = []
    rejected = Counter()
    rejection_details = []

    def reject(row: dict, reason: str, detail: str) -> None:
        rejected[reason] += 1
        rejection_details.append({
            "id": row.get("id"), "language": row.get("language"),
            "reason": reason, "detail": detail[-4000:],
        })

    for row in rows:
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            reject(row, "invalid_record", f"missing fields: {missing}")
            continue
        language = str(row["language"])
        semantic = semantic_text(row["brief"])
        generated_text = "\n".join([
            str(row["text"]), str(row["solution"]),
            str(row["author_tests"]), str(row["reviewer_tests"]),
        ])
        contamination = evaluation_guard.reason(
            str(row["entry_point"]),
            [str(row["text"]), str(row["author_tests"]), str(row["reviewer_tests"])],
            semantic,
        )
        if contamination:
            reject(row, contamination, "matched the frozen novice-v1 suite")
            continue
        if any(pattern.search(generated_text) for pattern in SECRET_PATTERNS):
            reject(row, "secret", "generated text matched a secret pattern")
            continue
        author_check = verify_program(
            language, str(row["solution"]), str(row["author_tests"]),
            timeout=verify_timeout, tsc=tsc,
        )
        if not author_check.ok:
            reject(row, f"author_{author_check.phase}", author_check.detail)
            continue
        reviewer_check = verify_program(
            language, str(row["solution"]), str(row["reviewer_tests"]),
            timeout=verify_timeout, tsc=tsc,
        )
        if not reviewer_check.ok:
            reject(row, f"review_{reviewer_check.phase}", reviewer_check.detail)
            continue
        duplicate = duplicate_guard.reason(language, semantic, str(row["solution"]))
        if duplicate:
            reject(row, duplicate, "matched an earlier clean seed")
            continue
        clean = dict(row)
        clean["fingerprints"] = duplicate_guard.add(language, semantic, str(row["solution"]))
        accepted.append(clean)

    report = {
        "format": "brittain3-teacher-seeds-audit-v1",
        "input_rows": len(rows),
        "accepted": len(accepted),
        "rejected": dict(rejected),
        "language_counts": dict(Counter(row["language"] for row in accepted)),
        "category_counts": dict(Counter(row.get("semantic_category") for row in accepted)),
        "rejection_details": rejection_details,
    }
    return accepted, report


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    report_path = Path(args.report)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite")
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    languages = {str(row.get("language")) for row in rows}
    status = backend_status(args.tsc)
    missing = sorted(language for language in languages if status.get(language) is None)
    if missing:
        raise SystemExit(f"missing verification toolchains: {missing}")
    accepted, report = audit_rows(rows, args.verify_timeout, args.tsc)
    report["input"] = str(source)
    report["output"] = str(output)
    atomic_write_jsonl(output, accepted)
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
