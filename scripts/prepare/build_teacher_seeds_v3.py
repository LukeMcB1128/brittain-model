#!/usr/bin/env python3
"""Build execution-verified curriculum seeds with three local Ollama models.

Qwen 3.6 plans task briefs, Qwen Coder writes solutions and initial tests, and
GLM writes an independent edge-case test set. An item is kept only when its
solution passes both test sets. Grouping work by role avoids frequent model
loads on a unified-memory Mac.

Generated code is executed. The verifier uses timeouts and disposable folders,
but it is not a security sandbox. Use this only with trusted local models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.ollama_teacher import chat_json  # noqa: E402
from brittain.training_v3 import load_json  # noqa: E402
from brittain.verification_v3 import DEFAULT_TSC, backend_status, verify_program  # noqa: E402


EXTENSIONS = {
    "python": "py", "typescript": "ts", "javascript": "js", "rust": "rs",
    "cpp": "cpp", "c": "c", "go": "go",
}

PLANNER_SYSTEM = """You design diverse, original novice coding tasks for model
training. Return JSON that follows the schema. Tasks must be small enough for a
new developer, but they must require correct behavior rather than boilerplate.
Do not copy famous benchmark problems. Vary names, data shapes, edge cases, and
solution structure. Do not write code."""

AUTHOR_SYSTEM = """You write execution-verified training examples. Return JSON
that follows the schema. The prompt field must be two to five clear source-code
comment lines. The solution must be a complete source fragment with the named
entry point and all required imports. It must not contain Markdown fences or a
main function. The tests must be source code that can be appended to the
solution and run. Keep the code simple, correct, and portable."""

REVIEWER_SYSTEM = """You are an independent code-test reviewer. Return JSON
that follows the schema. Review the specification and solution for correctness.
Write a new test set that does not copy the supplied tests. It must be source
code that can be appended to the solution and run by itself. Test edge cases
and input mutation when relevant. Reject ambiguous or internally inconsistent
tasks. Do not use Markdown fences."""


def brief_schema(count: int, categories: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array", "minItems": count, "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "slug": {"type": "string"},
                        "category": {"type": "string", "enum": categories},
                        "goal": {"type": "string"},
                        "input_contract": {"type": "string"},
                        "output_contract": {"type": "string"},
                        "edge_cases": {
                            "type": "array", "minItems": 2, "maxItems": 5,
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "slug", "category", "goal", "input_contract",
                        "output_contract", "edge_cases",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["tasks"],
        "additionalProperties": False,
    }


AUTHOR_SCHEMA = {
    "type": "object",
    "properties": {
        "entry_point": {"type": "string"},
        "prompt": {"type": "string"},
        "solution": {"type": "string"},
        "tests": {"type": "string"},
    },
    "required": ["entry_point", "prompt", "solution", "tests"],
    "additionalProperties": False,
}

REVIEWER_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "reason": {"type": "string"},
        "tests": {"type": "string"},
    },
    "required": ["approved", "reason", "tests"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/brittain3_teacher_curriculum.json")
    parser.add_argument("--count", type=int, default=70, help="total task briefs")
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--output", default="data/generated/brittain3-curriculum/teacher-seeds.jsonl")
    parser.add_argument("--report", default="runs/brittain3-teacher-seeds.report.json")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--verify-timeout", type=float, default=15.0)
    parser.add_argument("--tsc", default=str(DEFAULT_TSC))
    parser.add_argument("--balanced-smoke", action="store_true",
                        help="require one task per selected language; for a small smoke run only")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def allocate_counts(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Use largest remainders so integer counts add up to the exact target."""
    if total <= 0:
        raise ValueError("count must be positive")
    weight = sum(shares.values())
    if weight <= 0:
        raise ValueError("language shares must have positive weight")
    exact = {name: total * share / weight for name, share in shares.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(counts.values())
    order = sorted(exact, key=lambda name: (exact[name] - counts[name], shares[name]), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def balance_smoke_counts(counts: dict[str, int]) -> dict[str, int]:
    """Move tasks from large buckets until each selected language has one."""
    adjusted = dict(counts)
    if sum(adjusted.values()) < len(adjusted):
        raise ValueError("balanced smoke count must cover every selected language")
    for empty in [name for name, count in adjusted.items() if count == 0]:
        donors = [name for name, count in adjusted.items() if count > 1]
        if not donors:
            raise ValueError("cannot balance the selected language counts")
        donor = max(donors, key=lambda name: adjusted[name])
        adjusted[donor] -= 1
        adjusted[empty] = 1
    return adjusted


def load_banned_entry_points() -> set[str]:
    path = PROJECT_ROOT / "benchmarks" / "novice" / "tasks.jsonl"
    return {
        json.loads(line)["entry_point"]
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    }


def normalize_slug(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return "_".join(part for part in cleaned.split("_") if part)[:80]


def task_digest(language: str, brief: dict) -> str:
    normalized = " ".join(
        (brief["goal"] + " " + brief["input_contract"] + " " + brief["output_contract"])
        .lower().split()
    )
    return hashlib.sha256(f"{language}\n{normalized}".encode()).hexdigest()


def language_rules(language: str) -> str:
    rules = {
        "python": "Use Python 3. The tests must use assert statements.",
        "typescript": "Use strict TypeScript. Do not import Node-specific packages.",
        "javascript": "Use modern plain JavaScript supported by Node.js.",
        "rust": ("Use Rust 2021, include use declarations, and do not use external crates. "
                 "Tests are appended at crate root. Put use super::* only inside a test "
                 "module; never put use super::* at crate root."),
        "cpp": "Use C++20, include all headers, and do not use third-party libraries.",
        "c": "Use C17, include all headers, and do not use third-party libraries.",
        "go": ("Begin the solution with package main. Put all imports in the solution. "
               "Tests are appended after declarations, so tests must not contain a package "
               "line, import declaration, or Markdown fence. Use only packages already "
               "imported by the solution."),
    }
    return rules[language]


def plan_tasks(config: dict, counts: dict[str, int], args, banned: set[str]) -> list[dict]:
    planned = []
    for language, count in counts.items():
        if count == 0:
            continue
        prompt = (
            f"Create exactly {count} task briefs for {language}.\n"
            f"Allowed categories: {', '.join(config['categories'])}.\n"
            f"Do not use these held-out entry-point names: {', '.join(sorted(banned))}.\n"
            "Use unique snake_case slugs. Cover different categories and edge cases."
        )
        reply = chat_json(
            args.endpoint, config["planner_model"], PLANNER_SYSTEM, prompt,
            schema=brief_schema(count, config["categories"]),
            seed=int(config["seed"]) + len(planned), timeout=args.timeout,
        )
        tasks = reply.content.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != count:
            raise RuntimeError(f"planner returned the wrong task count for {language}")
        for task in tasks:
            task["language"] = language
            task["slug"] = normalize_slug(task["slug"])
            planned.append(task)
        print(f"planned {count} {language} task(s)", flush=True)
    return planned


def author_task(config: dict, brief: dict, args, seed: int) -> dict:
    prompt = (
        f"Language: {brief['language']}\n"
        f"Task brief: {json.dumps(brief, ensure_ascii=False)}\n"
        f"Rules: {language_rules(brief['language'])}"
    )
    return chat_json(
        args.endpoint, config["author_model"], AUTHOR_SYSTEM, prompt,
        schema=AUTHOR_SCHEMA, seed=seed, timeout=args.timeout,
    ).content


def review_task(config: dict, brief: dict, authored: dict, args, seed: int) -> dict:
    prompt = (
        f"Language: {brief['language']}\n"
        f"Task brief: {json.dumps(brief, ensure_ascii=False)}\n"
        f"Entry point: {authored['entry_point']}\n"
        f"Prompt comments:\n{authored['prompt']}\n"
        f"Solution:\n{authored['solution']}\n"
        f"Existing tests:\n{authored['tests']}\n"
        f"Rules: {language_rules(brief['language'])}"
    )
    return chat_json(
        args.endpoint, config["reviewer_model"], REVIEWER_SYSTEM, prompt,
        schema=REVIEWER_SCHEMA, seed=seed, timeout=args.timeout,
    ).content


def make_row(config: dict, brief: dict, authored: dict, reviewed: dict, digest: str) -> dict:
    language = brief["language"]
    entry_point = authored["entry_point"].strip()
    identifier = f"{language}/{normalize_slug(entry_point)}/{digest[:12]}"
    text = authored["prompt"].rstrip() + "\n" + authored["solution"].rstrip() + "\n"
    return {
        "id": identifier,
        "text": text,
        "source": "teacher-verified",
        "category": "exercises",
        "semantic_category": brief["category"],
        "language": language,
        "repository": f"generated/teacher/{language}/{brief['slug']}",
        "path": f"exercises/{brief['slug']}.{EXTENSIONS[language]}",
        "license": "GENERATED",
        "entry_point": entry_point,
        "brief": brief,
        "solution": authored["solution"],
        "author_tests": authored["tests"],
        "reviewer_tests": reviewed["tests"],
        "review": reviewed["reason"],
        "teacher_models": {
            "planner": config["planner_model"],
            "author": config["author_model"],
            "reviewer": config["reviewer_model"],
        },
    }


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    if config.get("format") != "brittain3-teacher-curriculum-v1":
        raise SystemExit("unsupported teacher curriculum configuration")
    selected = set(args.languages or config["language_shares"])
    unknown = selected - set(config["language_shares"])
    if unknown:
        raise SystemExit(f"unknown languages: {sorted(unknown)}")
    shares = {name: share for name, share in config["language_shares"].items() if name in selected}
    counts = allocate_counts(args.count, shares)
    if args.balanced_smoke:
        counts = balance_smoke_counts(counts)
    status = backend_status(args.tsc)
    missing = sorted(name for name, count in counts.items() if count and status.get(name) is None)
    if missing:
        raise SystemExit(f"missing verification toolchains: {missing}")

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    banned = load_banned_entry_points()
    planned = plan_tasks(config, counts, args, banned)

    unique = set()
    candidates = []
    rejected = Counter()
    rejection_details = []

    def reject(
        reason: str, brief: dict, detail: str, authored: dict | None = None,
        reviewed: dict | None = None,
    ) -> None:
        rejected[reason] += 1
        rejection_details.append({
            "reason": reason,
            "language": brief.get("language"),
            "slug": brief.get("slug"),
            "detail": detail[-4000:],
            "entry_point": authored.get("entry_point") if authored else None,
            "solution": authored.get("solution") if authored else None,
            "tests": authored.get("tests") if authored else None,
            "reviewer_tests": reviewed.get("tests") if reviewed else None,
            "review": reviewed.get("reason") if reviewed else None,
        })
        print(f"rejected {brief.get('language')}/{brief.get('slug')}: {reason}", flush=True)

    for index, brief in enumerate(planned):
        digest = task_digest(brief["language"], brief)
        if digest in unique or not brief["slug"]:
            reject("duplicate_or_invalid_brief", brief, "duplicate digest or empty slug")
            continue
        unique.add(digest)
        try:
            authored = author_task(config, brief, args, int(config["seed"]) + 10_000 + index)
        except (RuntimeError, ValueError) as exc:
            reject("author_error", brief, str(exc))
            continue
        if authored.get("entry_point", "").strip() in banned:
            reject("evaluation_entry_point", brief, "entry point is in the held-out suite", authored)
            continue
        checked = verify_program(
            brief["language"], authored.get("solution", ""), authored.get("tests", ""),
            timeout=args.verify_timeout, tsc=args.tsc,
        )
        if not checked.ok:
            reject(f"author_{checked.phase}", brief, checked.detail, authored)
            continue
        candidates.append((brief, authored, digest))
        print(f"authored {brief['language']}/{brief['slug']}: PASS", flush=True)

    accepted = []
    for index, (brief, authored, digest) in enumerate(candidates):
        try:
            reviewed = review_task(config, brief, authored, args, int(config["seed"]) + 20_000 + index)
        except (RuntimeError, ValueError) as exc:
            reject("reviewer_error", brief, str(exc), authored)
            continue
        if not reviewed.get("approved"):
            reject("reviewer_rejected", brief, str(reviewed.get("reason", "")), authored, reviewed)
            continue
        checked = verify_program(
            brief["language"], authored["solution"], reviewed.get("tests", ""),
            timeout=args.verify_timeout, tsc=args.tsc,
        )
        if not checked.ok:
            reject(f"review_{checked.phase}", brief, checked.detail, authored, reviewed)
            continue
        accepted.append(make_row(config, brief, authored, reviewed, digest))
        print(f"reviewed {brief['language']}/{brief['slug']}: PASS", flush=True)

    with output.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "format": "brittain3-teacher-seeds-report-v1",
        "config": args.config,
        "requested": args.count,
        "planned": len(planned),
        "authored_and_verified": len(candidates),
        "accepted": len(accepted),
        "language_counts": dict(Counter(row["language"] for row in accepted)),
        "category_counts": dict(Counter(row["semantic_category"] for row in accepted)),
        "rejected": dict(rejected),
        "rejection_details": rejection_details,
        "output": str(output),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not accepted:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
