#!/usr/bin/env python3
"""Build a small, test-rich Brittain3 correctness curriculum.

The builder uses the existing decontaminated pilot corpus. It gives priority to
test files and to implementation files from repositories that contain tests.
It also includes bounded replay of execution-verified teacher and template
exercises. The report fails if a source is short or replay is excessive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from brittain.corpus_v3 import SECRET_PATTERNS  # noqa: E402
from brittain.tokenizer_v3 import Brittain3Tokenizer  # noqa: E402
from brittain.training_v3 import load_json  # noqa: E402
from scripts.prepare.decontaminate_v3 import (  # noqa: E402
    DecontaminationGuard,
    GENERIC_NAMES,
    build_signatures,
)


TEST_PATH = re.compile(
    r"(?:^|/)(?:test|tests|spec|specs|__tests__)(?:/|_)"
    r"|(?:_test|\.test|\.spec)\.[^/]+$",
    re.IGNORECASE,
)
TEST_TEXT = re.compile(
    r"(?:\bdescribe\s*\(|\bit\s*\(|\btest\s*\(|\bassert(?:_eq)?[!(]"
    r"|#\[test\]|\bfunc\s+Test[A-Z]|\bTEST(?:_F)?\s*\()"
)
CPP_EXTENSIONS = {".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx"}


@dataclass(frozen=True)
class Candidate:
    key: str
    row: dict
    tokens: int
    priority: str
    replay: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/brittain3_49m_curriculum_corpus.json")
    parser.add_argument("--output", default="data/generated/brittain3-curriculum/corpus.jsonl")
    parser.add_argument("--report", default="runs/brittain3-curriculum-corpus.report.json")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--audit-existing", action="store_true",
                        help="validate the existing output without scanning the raw corpus")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{number}: {exc}") from exc


def normalized_digest(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def priority(seed: int, row: dict, suffix: str = "") -> str:
    value = "\n".join([
        str(seed), str(row.get("repository", "")), str(row.get("path", "")),
        normalized_digest(str(row.get("text", ""))), suffix,
    ])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sample_hit(priority_value: str, rate: float) -> bool:
    return int(priority_value[:16], 16) / 2**64 < rate


def normalize_language(row: dict) -> str | None:
    language = str(row.get("language", "")).lower()
    if language == "c_cpp":
        suffix = Path(str(row.get("path", ""))).suffix.lower()
        return "cpp" if suffix in CPP_EXTENSIONS else "c"
    return language if language in {"python", "typescript", "javascript", "rust", "cpp", "c", "go"} else None


def is_test_document(row: dict) -> bool:
    path = str(row.get("path", ""))
    text = str(row.get("text", ""))
    return bool(TEST_PATH.search(path) or TEST_TEXT.search(text[:20_000]))


def allocate(total: int, shares: dict[str, float]) -> dict[str, int]:
    weight = sum(shares.values())
    exact = {name: total * share / weight for name, share in shares.items()}
    result = {name: math.floor(value) for name, value in exact.items()}
    for name in sorted(exact, key=lambda key: exact[key] - result[key], reverse=True)[:total - sum(result.values())]:
        result[name] += 1
    return result


def scaled_targets(config: dict, scale: float) -> dict[str, int]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return {
        name: max(1, round(int(value) * scale))
        for name, value in config["source_token_targets"].items()
    }


def target_keys(config: dict, scale: float) -> dict[str, int]:
    targets = scaled_targets(config, scale)
    result = {}
    for source, target in targets.items():
        if source in ("test_code", "companion_code"):
            for language, tokens in allocate(target, config["code_language_shares"]).items():
                result[f"{source}:{language}"] = tokens
        else:
            result[source] = target
    return result


def validate_upstream_reports(config: dict, raw: Path, teacher: Path) -> None:
    raw_report = load_json(config["raw_decontamination_report"])
    if Path(raw_report.get("output", "")).resolve() != raw.resolve():
        raise SystemExit("raw decontamination report does not match raw_corpus")
    if int(raw_report.get("tasks_checked", 0)) < 36:
        raise SystemExit("raw corpus was not decontaminated against frozen novice-v1")
    teacher_report = load_json(config["teacher_audit_report"])
    if Path(teacher_report.get("output", "")).resolve() != teacher.resolve():
        raise SystemExit("teacher audit report does not match teacher_corpus")
    if int(teacher_report.get("evaluation_entry_points", 0)) < 66:
        raise SystemExit("teacher corpus was not audited against both frozen suites")


def frozen_guard() -> DecontaminationGuard:
    tasks = []
    references = {}
    for directory in ("novice", "novice_v2"):
        root = PROJECT_ROOT / "benchmarks" / directory
        tasks.extend(list(read_jsonl(root / "tasks.jsonl")))
        references.update({
            row["id"]: row["body"] for row in read_jsonl(root / "reference.jsonl")
        })
    return DecontaminationGuard(build_signatures(tasks, references), set(GENERIC_NAMES))


def clean_row(row: dict, source: str, language: str | None = None) -> dict:
    value = dict(row)
    value["upstream_source"] = row.get("source")
    value["source"] = source
    if language:
        value["language"] = language
    return value


def collect_teacher(
    path: Path, tokenizer, target: int, maximum_replays: int, seed: int
) -> list[Candidate]:
    rows = list(read_jsonl(path))
    result = []
    total = 0
    for replay_index in range(maximum_replays):
        for row in rows:
            text = str(row["text"])
            tokens = len(tokenizer.encode(text))
            value = clean_row(row, "verified_teacher", normalize_language(row))
            value["path"] = f"replay-{replay_index:02d}/{row['path']}"
            result.append(Candidate(
                "verified_teacher", value, tokens,
                priority(seed, value, str(replay_index)), replay=replay_index > 0,
            ))
            total += tokens
            if total >= target:
                return result
    return result


def collect_exercises(
    path: Path, tokenizer, target: int, cap: int, seed: int,
    evaluation_guard: DecontaminationGuard, rejections: Counter,
) -> list[Candidate]:
    families = defaultdict(list)
    for row in read_jsonl(path):
        reason = evaluation_guard.reason(str(row.get("text", "")))
        if reason:
            rejections[reason.split(":", 1)[0]] += 1
            continue
        value_priority = priority(seed, row)
        families[str(row.get("repository", "unknown"))].append((value_priority, row))
    candidates = []
    for values in families.values():
        # Exact duplicate rows have the same priority. Use a key so Python does
        # not try to compare the row dictionaries when priorities are equal.
        for value_priority, row in sorted(values, key=lambda item: item[0])[:cap]:
            text = str(row["text"])
            candidates.append(Candidate(
                "verified_exercises", clean_row(row, "verified_exercises", "python"),
                len(tokenizer.encode(text)), value_priority,
            ))
    selected = []
    total = 0
    for candidate in sorted(candidates, key=lambda value: value.priority):
        selected.append(candidate)
        total += candidate.tokens
        if total >= target:
            break
    return selected


def find_test_repositories(raw: Path) -> set[str]:
    repositories = set()
    for row in read_jsonl(raw):
        if row.get("category") == "code" and normalize_language(row) and is_test_document(row):
            repositories.add(str(row.get("repository", "")))
    return repositories


def raw_key(row: dict, test_repositories: set[str]) -> str | None:
    category = str(row.get("category", ""))
    if category == "code":
        language = normalize_language(row)
        if not language:
            return None
        if is_test_document(row):
            return f"test_code:{language}"
        if str(row.get("repository", "")) in test_repositories:
            return f"companion_code:{language}"
        return None
    return category if category in {"documentation", "english", "tool", "structured"} else None


def collect_raw_candidates(
    raw: Path, tokenizer, config: dict, targets: dict[str, int],
    test_repositories: set[str], evaluation_guard: DecontaminationGuard,
    rejections: Counter,
) -> dict[str, list[Candidate]]:
    pools = defaultdict(list)
    minimum = int(config["minimum_document_tokens"])
    maximum = int(config["maximum_document_tokens"])
    rates = config["candidate_sample_rates"]
    seed = int(config["seed"])
    for row in read_jsonl(raw):
        key = raw_key(row, test_repositories)
        if key not in targets:
            continue
        source = key.split(":", 1)[0]
        value_priority = priority(seed, row)
        if not sample_hit(value_priority, float(rates[source])):
            continue
        text = str(row.get("text", ""))
        reason = evaluation_guard.reason(text)
        if reason:
            rejections[reason.split(":", 1)[0]] += 1
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            rejections["secret"] += 1
            continue
        tokens = len(tokenizer.encode(text))
        if not minimum <= tokens <= maximum:
            continue
        language = key.split(":", 1)[1] if ":" in key else normalize_language(row)
        pools[key].append(Candidate(
            key, clean_row(row, source, language), tokens, value_priority,
        ))
    return pools


def select_pools(
    pools: dict[str, list[Candidate]], targets: dict[str, int], repository_cap: int,
    already_selected: list[Candidate], repository_cap_exempt_sources: set[str] | None = None,
) -> tuple[list[Candidate], dict[str, int], int]:
    repository_cap_exempt_sources = repository_cap_exempt_sources or set()
    selected = list(already_selected)
    accepted_tokens = Counter(candidate.key for candidate in selected)
    token_totals = Counter()
    for candidate in selected:
        token_totals[candidate.key] += candidate.tokens
    seen = {normalized_digest(str(candidate.row["text"])) for candidate in selected}
    repository_counts = Counter()
    duplicates = 0
    for key, target in targets.items():
        if token_totals[key] >= target:
            continue
        for candidate in sorted(pools.get(key, []), key=lambda value: value.priority):
            digest = normalized_digest(str(candidate.row["text"]))
            if digest in seen:
                duplicates += 1
                continue
            repository = str(candidate.row.get("repository", ""))
            source = key.split(":", 1)[0]
            cap_key = (source, repository)
            if (source not in repository_cap_exempt_sources
                    and repository_counts[cap_key] >= repository_cap):
                continue
            selected.append(candidate)
            seen.add(digest)
            repository_counts[cap_key] += 1
            token_totals[key] += candidate.tokens
            accepted_tokens[key] += 1
            if token_totals[key] >= target:
                break
    return selected, dict(token_totals), duplicates


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


def code_group_totals(
    targets: dict[str, int], token_totals: dict[str, int]
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Aggregate code targets by source and language for fill validation."""
    by_source = defaultdict(lambda: {"target": 0, "accepted": 0})
    by_language = defaultdict(lambda: {"target": 0, "accepted": 0})
    for key, target in targets.items():
        if ":" not in key:
            continue
        source, language = key.split(":", 1)
        accepted = token_totals.get(key, 0)
        by_source[source]["target"] += target
        by_source[source]["accepted"] += accepted
        by_language[language]["target"] += target
        by_language[language]["accepted"] += accepted
    return dict(by_source), dict(by_language)


def fill_failures(
    targets: dict[str, int], token_totals: dict[str, int], maximum_shortfall: float
) -> tuple[list[str], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Fail missing prose pools, code sources, or total code languages."""
    failures = []
    for key, target in targets.items():
        if ":" in key:
            continue
        fraction = max(0, target - token_totals.get(key, 0)) / target
        if fraction > maximum_shortfall:
            failures.append(f"{key} shortfall is {fraction:.1%}")
    by_source, by_language = code_group_totals(targets, token_totals)
    for label, groups in (("code source", by_source), ("code language", by_language)):
        for name, values in groups.items():
            fraction = max(0, values["target"] - values["accepted"]) / values["target"]
            if fraction > maximum_shortfall:
                failures.append(f"{label} {name} shortfall is {fraction:.1%}")
    return failures, by_source, by_language


def output_key(row: dict) -> str | None:
    source = str(row.get("source", ""))
    if source in {"test_code", "companion_code"}:
        language = normalize_language(row)
        return f"{source}:{language}" if language else None
    if source in {
        "verified_teacher", "verified_exercises", "documentation", "english",
        "tool", "structured",
    }:
        return source
    return None


def audit_existing_output(
    output: Path, report_path: Path, config: dict, targets: dict[str, int], tokenizer
) -> int:
    """Audit a built curriculum without another multi-gigabyte source scan."""
    totals = Counter()
    documents = Counter()
    seen = set()
    duplicates = 0
    unknown = 0
    replay_tokens = 0
    guard = frozen_guard()
    guarded = Counter()
    secrets = 0
    for row in read_jsonl(output):
        key = output_key(row)
        if key not in targets:
            unknown += 1
            continue
        text = str(row.get("text", ""))
        tokens = len(tokenizer.encode(text))
        totals[key] += tokens
        documents[key] += 1
        is_teacher_replay = (
            key == "verified_teacher"
            and not str(row.get("path", "")).startswith("replay-00/")
        )
        if not is_teacher_replay:
            value_digest = normalized_digest(text)
            if value_digest in seen:
                duplicates += 1
            seen.add(value_digest)
        reason = guard.reason(text)
        if reason:
            guarded[reason.split(":", 1)[0]] += 1
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            secrets += 1
        if is_teacher_replay:
            replay_tokens += tokens

    total_tokens = sum(totals.values())
    maximum_shortfall = float(config["maximum_shortfall_fraction"])
    failures, code_source_totals, code_language_totals = fill_failures(
        targets, totals, maximum_shortfall
    )
    replay_fraction = replay_tokens / max(1, total_tokens)
    if replay_fraction > float(config["maximum_replay_fraction"]):
        failures.append(f"replay fraction is {replay_fraction:.1%}")
    for name, count in (
        ("duplicate documents", duplicates), ("unknown-source documents", unknown),
        ("evaluation-contaminated documents", sum(guarded.values())),
        ("documents with secret patterns", secrets),
    ):
        if count:
            failures.append(f"{name}: {count}")
    shortfalls = {
        key: max(0, target - totals.get(key, 0)) for key, target in targets.items()
    }
    report = {
        "format": "brittain3-curriculum-corpus-audit-v1",
        "config": str(config.get("_path", "")),
        "output": str(output),
        "target_tokens": sum(targets.values()),
        "accepted_tokens": total_tokens,
        "accepted_documents": sum(documents.values()),
        "target_tokens_by_key": targets,
        "accepted_tokens_by_key": dict(totals),
        "accepted_documents_by_key": dict(documents),
        "shortfall_tokens_by_key": shortfalls,
        "code_source_totals": code_source_totals,
        "code_language_totals": code_language_totals,
        "replay_tokens": replay_tokens,
        "replay_fraction": replay_fraction,
        "duplicates": duplicates,
        "unknown_source_documents": unknown,
        "rejected_by_guard": dict(guarded),
        "secret_pattern_documents": secrets,
        "evaluation_tasks_checked": len(guard.signatures["per_name"]),
        "failures": failures,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    config["_path"] = args.config
    if config.get("format") != "brittain3-curriculum-corpus-v1":
        raise SystemExit("unsupported curriculum corpus configuration")
    output = Path(args.output)
    if output.exists() and not args.overwrite and not args.audit_existing:
        raise SystemExit(f"{output} exists; pass --overwrite")
    raw = Path(config["raw_corpus"])
    teacher = Path(config["teacher_corpus"])
    for path in (raw, teacher, Path(config["exercise_corpus"]), Path(config["tokenizer"])):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    validate_upstream_reports(config, raw, teacher)
    tokenizer = Brittain3Tokenizer(config["tokenizer"])
    targets = target_keys(config, args.scale)
    if args.audit_existing:
        if not output.is_file():
            raise SystemExit(f"output does not exist: {output}")
        return audit_existing_output(output, Path(args.report), config, targets, tokenizer)
    evaluation_guard = frozen_guard()
    exercise_rejections = Counter()
    raw_rejections = Counter()

    fixed = collect_teacher(
        teacher, tokenizer, targets["verified_teacher"],
        int(config["teacher_max_replays"]), int(config["seed"]),
    )
    fixed.extend(collect_exercises(
        Path(config["exercise_corpus"]), tokenizer, targets["verified_exercises"],
        int(config["exercise_max_per_family"]), int(config["seed"]),
        evaluation_guard, exercise_rejections,
    ))
    print(f"fixed verified candidates: {len(fixed):,}", flush=True)
    test_repositories = find_test_repositories(raw)
    print(f"repositories with tests: {len(test_repositories):,}", flush=True)
    pools = collect_raw_candidates(
        raw, tokenizer, config, targets, test_repositories,
        evaluation_guard, raw_rejections,
    )
    print("candidate pools: " + ", ".join(
        f"{key}={len(values):,}" for key, values in sorted(pools.items())
    ), flush=True)
    selected, token_totals, duplicate_count = select_pools(
        pools, targets, int(config["max_documents_per_repository_per_source"]), fixed,
        set(config.get("repository_cap_exempt_sources", [])),
    )
    rng = random.Random(int(config["seed"]))
    rng.shuffle(selected)
    rows = [candidate.row for candidate in selected]
    replay_tokens = sum(candidate.tokens for candidate in selected if candidate.replay)
    total_tokens = sum(candidate.tokens for candidate in selected)
    shortfalls = {
        key: max(0, target - token_totals.get(key, 0)) for key, target in targets.items()
    }
    replay_fraction = replay_tokens / max(1, total_tokens)
    maximum_shortfall = float(config["maximum_shortfall_fraction"])
    failures, code_source_totals, code_language_totals = fill_failures(
        targets, token_totals, maximum_shortfall
    )
    if replay_fraction > float(config["maximum_replay_fraction"]):
        failures.append(f"replay fraction is {replay_fraction:.1%}")
    report = {
        "format": "brittain3-curriculum-corpus-report-v1",
        "config": args.config,
        "scale": args.scale,
        "target_tokens": sum(targets.values()),
        "accepted_tokens": total_tokens,
        "accepted_documents": len(rows),
        "target_tokens_by_key": targets,
        "accepted_tokens_by_key": token_totals,
        "shortfall_tokens_by_key": shortfalls,
        "code_source_totals": code_source_totals,
        "code_language_totals": code_language_totals,
        "replay_tokens": replay_tokens,
        "replay_fraction": replay_fraction,
        "duplicates_skipped": duplicate_count,
        "repositories_with_tests": len(test_repositories),
        "evaluation_tasks_checked": len(evaluation_guard.signatures["per_name"]),
        "rejected_by_guard": {
            "exercises": dict(exercise_rejections),
            "raw_candidates": dict(raw_rejections),
        },
        "language_tokens": {
            language: values["accepted"] for language, values in code_language_totals.items()
        },
        "failures": failures,
        "output": str(output),
    }
    atomic_write_jsonl(output, rows)
    atomic_write_json(Path(args.report), report)
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
