"""Redraw the character names in the synthetic stories from a wide pool.

The generator worked from a 30-name pool and was told to reuse those names
throughout each story, so every one of the 66,219 synthetic stories draws its
cast from the same thirty people. Each name lands in roughly 7,300 stories and
repeats inside every one of them, which very likely makes "Ivo" the most common
name in the whole corpus: real books spread their names over thousands of
distinct people, so no individual name there comes close.

That matters most at the end. The synthetic stories are 3.4% of the main mix but
35% of the anneal, and the anneal is the last thing the model sees, so every
complete story it has ever read is populated by those thirty.

This rewrites each story's names, drawing from a pool harvested out of the real
corpus. The mapping is seeded per story, so it is deterministic and reruns give
the same result.

    python3 scripts/prepare/rename_synthetic_characters.py

Writes a new file and leaves the original alone.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.synthetic import NAME_POOL
from brittain.tags import CHARACTER_SEPARATOR

SYNTHETIC = "data/raw/brittain-shakespeare-synthetic"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", default=f"{SYNTHETIC}/stories.jsonl")
    parser.add_argument("--pool", default=f"{SYNTHETIC}/name_pool.json")
    parser.add_argument("--output", default=f"{SYNTHETIC}/stories_renamed.jsonl")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def rename_story(row, pool, seed):
    """Return the row with every old-pool name replaced, or None if unchanged."""
    text = row.get("text") or ""
    if not text:
        return None

    # Which of the old names this story actually uses. The tag is the intent;
    # the text is the truth, and the generator sometimes reached for a pool name
    # it was not given.
    tag_names = [
        part.strip()
        for part in (row.get("book_tags", {}).get("Characters") or "").split(
            CHARACTER_SEPARATOR
        )
        if part.strip()
    ]
    present = [name for name in NAME_POOL if re.search(rf"\b{name}\b", text)]
    used = list(dict.fromkeys(tag_names + present))
    used = [name for name in used if name in set(NAME_POOL)]
    if not used:
        return None

    # Seeded on the story id, so the mapping does not depend on file order and
    # a rerun reproduces it exactly.
    rng = random.Random(f"{seed}:{row['repository']}")
    replacements = dict(zip(used, rng.sample(pool, len(used))))

    # One pass over the text, so a name mapped onto another old name cannot be
    # rewritten twice.
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in used) + r")\b")
    row["text"] = pattern.sub(lambda m: replacements[m.group(1)], text)

    if tag_names:
        row["book_tags"]["Characters"] = CHARACTER_SEPARATOR.join(
            f" {replacements[name]}" if index else replacements[name]
            for index, name in enumerate(tag_names)
            if name in replacements
        ).strip()
    return row


def main():
    args = parse_args()
    pool = json.loads(project_path(args.pool).read_text(encoding="utf-8"))
    if len(pool) < 500:
        raise SystemExit(f"name pool is only {len(pool)} names; too narrow to help")
    source = project_path(args.stories)
    destination = project_path(args.output)

    stories = renamed = 0
    names = Counter()
    with source.open(encoding="utf-8") as handle, \
            destination.open("w", encoding="utf-8") as out:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            stories += 1
            updated = rename_story(row, pool, args.seed)
            if updated is not None:
                renamed += 1
                row = updated
            names.update(
                part.strip()
                for part in (row.get("book_tags", {}).get("Characters") or "").split(
                    CHARACTER_SEPARATOR
                )
                if part.strip()
            )
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{stories:,} stories, {renamed:,} renamed")
    print(f"{len(names):,} distinct names now in use")
    top = names.most_common(5)
    print("most common:", ", ".join(
        f"{name} {count / max(1, stories):.2%}" for name, count in top
    ))
    leaked = [name for name in NAME_POOL if name in names]
    if leaked:
        raise SystemExit(f"old pool names survived: {leaked}")
    print("no old pool name survives in the tags")


if __name__ == "__main__":
    main()
