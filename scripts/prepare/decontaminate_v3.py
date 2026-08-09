"""Remove evaluation-suite contamination from a Brittain3 corpus.

The collector's `--exclude-corpus` matches WHOLE documents by exact and
whitespace-normalized hash. The novice tasks are small snippets, so a repository
file that merely *contains* one never hash-matches and passes straight through.
This script closes that gap. It runs on the local JSONL after the fetch and
before `prepare_brittain3.py`, so it never requires a re-download.

    python3 scripts/prepare/decontaminate_v3.py \
      --input data/raw/brittain3-pilot/corpus.jsonl \
      --output data/raw/brittain3-pilot/corpus.clean.jsonl \
      --report data/raw/brittain3-pilot/decontamination.report.json

Three rules, ordered from zero false positives to most aggressive:

1. **Document hash** — exact or whitespace-normalized match against a task's
   prompt, its reference solution, or the two concatenated.
2. **Verbatim assertion** — the document contains one of the suite's own test
   assertions character for character. Almost impossible by chance; a document
   carrying `assert unique_preserve_order([3, 1, 3, 2, 1]) == [3, 1, 2]` is the
   benchmark, not training data.
3. **Distinctive entry point plus evidence** — the document defines a suite
   entry point whose name is distinctive, AND contains at least two distinctive
   literals from that task's tests.

Rule 3 deliberately does NOT fire on generic names alone. Banning every document
that defines `def add(` or `class Stack:` would delete an enormous share of real
Python and teach the model less, not more. Learning to write a summing loop from
similar code is the point of pretraining; reproducing a specific benchmark task
is the thing to prevent. Names are classified by `--generic-names`, which
defaults to the obviously common ones in the suite.

The report records how much each rule removed. **Read it.** A rule removing a
large fraction of the corpus is misfiring, not working well.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TASKS = PROJECT_ROOT / "benchmarks" / "novice" / "tasks.jsonl"
DEFAULT_REFERENCE = PROJECT_ROOT / "benchmarks" / "novice" / "reference.jsonl"

# Entry points common enough in ordinary code that the name alone proves nothing.
GENERIC_NAMES = {
    "add", "greet", "clamp", "average", "factorial", "is_even", "merge_dicts",
    "reverse_list", "count_words", "Counter", "Stack", "Account", "get_field",
    "count_vowels", "tally",
}

# A literal is "distinctive" if its presence alongside a matching definition is
# unlikely to be coincidence: quoted strings, and bracket/brace/paren groups
# carrying actual test data. An earlier version matched bare identifiers of five
# or more characters, which mostly recovered the entry-point name itself — the
# very thing that already triggered the check — so rule 3 could never reach the
# two-hit threshold and never fired.
DISTINCTIVE_LITERAL = re.compile(
    r"'[^']{3,}'"
    r'|"[^"]{3,}"'
    r"|\[[^\]]{4,}\]"
    r"|\{[^}]{4,}\}"
    r"|\([^)]{6,}\)"
)

# A description line long enough that a verbatim copy is not coincidence.
MINIMUM_DESCRIPTION_LENGTH = 30


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--generic-names", default=None,
                        help="comma-separated names that must not trigger rule 3 "
                             "on the name alone; defaults to the built-in list")
    parser.add_argument("--max-drop-fraction", type=float, default=0.05,
                        help="fail if more than this fraction is removed; a large "
                             "removal means a rule is misfiring, not working")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{number}: {exc}") from exc
    return rows


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_signatures(tasks: list[dict], references: dict[str, str]) -> dict:
    hashes: set[str] = set()
    assertions: set[str] = set()
    descriptions: set[str] = set()
    per_name: dict[str, set[str]] = {}

    for task in tasks:
        prompt = task["prompt"]
        body = references.get(task["id"], "")
        for variant in (prompt, body, prompt + body):
            if variant.strip():
                hashes.add(digest(variant))
                hashes.add(digest(normalized(variant)))

        for assertion in task["tests"]:
            cleaned = assertion.strip()
            if len(cleaned) >= 20:          # short asserts collide with real code
                assertions.add(cleaned)

        # The `#` description lines are full English sentences written for this
        # suite. A repository file containing one verbatim is the benchmark.
        for line in prompt.splitlines():
            stripped = line.lstrip("#").strip()
            if len(stripped) >= MINIMUM_DESCRIPTION_LENGTH:
                descriptions.add(stripped)

        name = task["entry_point"]
        literals = set()
        for text in list(task["tests"]) + [prompt]:
            literals.update(DISTINCTIVE_LITERAL.findall(text))
        # The entry-point name is what triggers the check, so it cannot also
        # serve as evidence for it.
        literals = {value for value in literals if value.strip("'\"()[]{} ") != name}
        per_name.setdefault(name, set()).update(literals)

    return {
        "hashes": hashes,
        "assertions": assertions,
        "descriptions": descriptions,
        "per_name": per_name,
    }


def definition_pattern(name: str) -> re.Pattern:
    return re.compile(rf"^\s*(?:def|class)\s+{re.escape(name)}\b", re.MULTILINE)


def main() -> int:
    args = parse_args()
    source = Path(args.input)
    target = Path(args.output)
    if not source.is_file():
        raise SystemExit(f"input corpus does not exist: {source}")
    if target.exists() and not args.overwrite:
        raise SystemExit(f"{target} exists; pass --overwrite")

    tasks = read_jsonl(Path(args.tasks))
    references = {row["id"]: row["body"] for row in read_jsonl(Path(args.reference))}
    signatures = build_signatures(tasks, references)

    generic = (set(args.generic_names.split(",")) if args.generic_names
               else set(GENERIC_NAMES))
    watched = {
        name: literals for name, literals in signatures["per_name"].items()
        if name not in generic and literals
    }
    patterns = {name: definition_pattern(name) for name in watched}

    kept = removed = 0
    reasons = Counter()
    examples: dict[str, str] = {}

    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as handle, target.open("w", encoding="utf-8") as out:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {source}:{number}: {exc}") from exc
            text = row.get(args.text_field, "")
            if not isinstance(text, str):
                raise SystemExit(f"row {number} has no string {args.text_field!r}")

            reason = None
            if digest(text) in signatures["hashes"] or digest(normalized(text)) in signatures["hashes"]:
                reason = "document_hash"
            if reason is None:
                for assertion in signatures["assertions"]:
                    if assertion in text:
                        reason = "verbatim_assertion"
                        break
            if reason is None:
                for description in signatures["descriptions"]:
                    if description in text:
                        reason = "verbatim_description"
                        break
            if reason is None:
                for name, literals in watched.items():
                    if not patterns[name].search(text):
                        continue
                    hits = sum(1 for literal in literals if literal in text)
                    if hits >= 2:
                        reason = f"entry_point:{name}"
                        break

            if reason:
                removed += 1
                reasons[reason.split(":")[0]] += 1
                examples.setdefault(reason, row.get("path", "?"))
                continue
            out.write(line if line.endswith("\n") else line + "\n")
            kept += 1

    total = kept + removed
    fraction = removed / total if total else 0.0
    report = {
        "format": "brittain3-decontamination-report-v1",
        "input": str(source),
        "output": str(target),
        "documents_in": total,
        "documents_kept": kept,
        "documents_removed": removed,
        "removed_fraction": fraction,
        "removed_by_rule": dict(reasons),
        "example_paths": examples,
        "tasks_checked": len(tasks),
        "entry_points_watched": sorted(watched),
        "generic_names_ignored": sorted(generic),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if fraction > args.max_drop_fraction:
        print(f"\nFAIL: removed {fraction:.1%} of the corpus, above the "
              f"{args.max_drop_fraction:.1%} limit. A rule is misfiring — inspect "
              f"removed_by_rule and example_paths before using this output.",
              file=sys.stderr)
        return 1
    print(f"\nkept {kept:,} of {total:,} documents ({fraction:.4%} removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
