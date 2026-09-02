"""Generate the synthetic complete-story set against an OpenAI-compatible API.

The synthetic stories are the only in-domain data in the corpus: everything else
is a window cut from the middle of a novel, which cannot teach narrative
structure. See ``src/brittain/synthetic.py`` for the exemplars and the design.

    export BRITTAIN_API_KEY=...
    python3 scripts/prepare/generate_stories.py \\
        --base-url https://openrouter.ai/api/v1 \\
        --model some/model \\
        --target-tokens 60000000

The key is read from the environment and never written to the output, the
report, or the repository.

Every story is verified against its own tags with the same extractors that
labelled the real corpus, and discarded on contradiction. The run is resumable:
output is appended, and restarting counts what is already there and continues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.keep_awake import keep_awake
from brittain.synthetic import build_messages, sample_tags, verify
from brittain.tags import validate

API_KEY_VARIABLE = "BRITTAIN_API_KEY"

# Set on Ctrl+C. Workers check it while sleeping through a backoff, because a
# worker parked in a two minute retry wait would otherwise hold the whole
# process open long after the interrupt, making Ctrl+C look broken.
STOP = threading.Event()

# Set when the API returns something no amount of retrying will fix. An
# exhausted balance turned into 2,320,998 wasted attempts at thousands a second,
# because the loop only ever asked whether the token target was met, never why
# an attempt had failed.
FATAL = threading.Event()

# Statuses where the request itself is the problem: bad key, no credit, no
# permission, wrong model id. Retrying any of them is pure spin.
FATAL_STATUS = frozenset({400, 401, 402, 403, 404})

# Stop if this many attempts in a row fail for any reason at all. Catches the
# causes not enumerated above.
CONSECUTIVE_FAILURE_LIMIT = 250


def interruptible_sleep(seconds: float) -> None:
    """Sleep, but wake immediately once the run has been interrupted."""
    STOP.wait(timeout=seconds)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True,
                        help="OpenAI-compatible root, e.g. https://openrouter.ai/api/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="data/raw/brittain-shakespeare-synthetic/stories.jsonl")
    parser.add_argument("--report", default=None)
    parser.add_argument("--target-tokens", type=int, default=60_000_000,
                        help="stop once the accepted stories reach this many tokens")
    parser.add_argument("--max-stories", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--reasoning-off", action="store_true",
                        help="ask the provider to skip chain-of-thought. Reasoning "
                             "models bill thinking as output and are far slower, "
                             "which is the wrong trade for bulk creative writing")
    parser.add_argument("--verbose", action="store_true",
                        help="report every story and every backoff as it happens")
    parser.add_argument("--dry-run", action="store_true",
                        help="print one built prompt and exit without calling the API")
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def approximate_tokens(text: str) -> int:
    """Cheap token estimate. The real tokenizer is not needed for a budget."""
    return max(1, len(text) // 4)


def fingerprint(story: str) -> str:
    """Stable hash of the normalized story, for duplicate detection."""
    normalized = " ".join(story.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def load_existing(path: Path) -> tuple[int, int, set[str]]:
    """Count what a previous run already wrote so this one can continue."""
    if not path.exists():
        return 0, 0, set()
    stories = tokens = 0
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            stories += 1
            tokens += approximate_tokens(row.get("text", ""))
            seen.add(fingerprint(row.get("text", "")))
    return stories, tokens, seen


def request_story(client, args, key, tags, attempt_log, rng):
    """One chat completion, with backoff on rate limits and transient failures."""
    payload = {
        "model": args.model,
        "messages": build_messages(tags, rng),
        "max_tokens": args.max_output_tokens,
        "temperature": args.temperature,
    }
    if args.reasoning_off:
        # OpenRouter's shape. Providers that do not understand it ignore it.
        payload["reasoning"] = {"enabled": False}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    delay = 2.0
    for attempt in range(args.max_retries):
        if STOP.is_set() or FATAL.is_set():
            return None, "interrupted"
        try:
            response = client.post("/chat/completions", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            attempt_log["network"] += 1
            interruptible_sleep(delay)
            delay = min(delay * 2, 60)
            continue
        if response.status_code == 200:
            body = response.json()
            try:
                choice = body["choices"][0]
                content = choice["message"]["content"].strip()
            except (KeyError, IndexError, AttributeError):
                return None, f"malformed response: {str(body)[:160]}"
            # A response the API cut short is a story with no ending. Only a
            # natural stop is a finished story.
            reason = choice.get("finish_reason") or choice.get("native_finish_reason")
            if reason not in (None, "stop", "end_turn", "eos"):
                return None, f"truncated by api ({reason})"
            return content, None
        if response.status_code in (408, 409, 429) or response.status_code >= 500:
            attempt_log[f"http_{response.status_code}"] += 1
            # Honour Retry-After when the server sends one; free tiers do.
            wait = response.headers.get("retry-after")
            try:
                pause = float(wait) if wait else delay
            except ValueError:
                pause = delay
            pause = min(pause, 120)
            if args.verbose:
                print(f"    http {response.status_code}, waiting {pause:.0f}s",
                      flush=True)
            interruptible_sleep(pause)
            delay = min(delay * 2, 60)
            continue
        if response.status_code in FATAL_STATUS:
            FATAL.set()
            return None, (f"fatal http {response.status_code}: "
                          f"{response.text[:200]}")
        return None, f"http {response.status_code}: {response.text[:160]}"
    return None, "gave up after retries"


def corpus_row(index: int, story: str, tags: dict[str, str]) -> dict:
    """A row shaped like the ones build_story_corpus writes.

    The identifier comes from the story's own content rather than a counter.
    Counters do not survive a resume: indices are handed out per attempt, so
    accepted rows carry sparse ids, and restarting from the accepted *count*
    began renumbering partway through the range already used. Six ids collided
    that way. A content hash cannot collide and does not care how often the run
    is restarted.

    ``book_tags`` carries the requested tags. Preparation reuses only the ones no
    extractor can verify from the text; everything else is re-derived from the
    story itself, so a tag the generator claimed but did not deliver cannot leak
    into training.
    """
    mark = fingerprint(story)[:12]
    return {
        "repository": f"synthetic/{mark}",
        "path": f"story-{mark}",
        "text": story,
        "source": "synthetic",
        "is_code": False,
        "author": "",
        "birth_year": None,
        "death_year": None,
        "subjects": [],
        "bookshelves": [],
        "lcc": [],
        "rights": "OWNED",
        "book_tags": tags,
    }


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    if args.dry_run:
        tags = sample_tags(rng)
        validate(tags)
        for message in build_messages(tags):
            print(f"--- {message['role']} ---")
            print(message["content"][:600])
        return

    key = os.environ.get(API_KEY_VARIABLE)
    if not key:
        raise SystemExit(
            f"{API_KEY_VARIABLE} is not set. Export it in this shell; do not put "
            f"it in a file inside the repository."
        )

    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stories, tokens, seen = load_existing(output)
    if stories:
        print(f"resuming: {stories:,} stories, ~{tokens:,} tokens already written",
              flush=True)

    rejected: Counter[str] = Counter()
    attempt_log: Counter[str] = Counter()
    consecutive_failures = 0
    aborted = None
    lock = threading.Lock()
    started = time.time()

    client = httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout)
    handle = output.open("a", encoding="utf-8")

    def one_story(index: int, tags: dict[str, str]):
        # Each worker gets its own generator: random.Random is not thread
        # safe, and a shared one would make the shuffle non-reproducible.
        story, error = request_story(
            client, args, key, tags, attempt_log, random.Random(index)
        )
        if story is None:
            return "error", error
        ok, why = verify(story, tags)
        if not ok:
            return "rejected", why
        mark = fingerprint(story)
        with lock:
            if mark in seen:
                return "rejected", "duplicate"
            seen.add(mark)
            handle.write(json.dumps(corpus_row(index, story, tags)) + "\n")
            handle.flush()
        return "accepted", story

    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        index = stories
        pending = set()
        while tokens < args.target_tokens:
            if args.max_stories is not None and stories >= args.max_stories:
                break
            if FATAL.is_set():
                aborted = "the API returned a status retrying cannot fix"
                break
            if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                aborted = (f"{consecutive_failures} attempts in a row failed; "
                           f"something is wrong upstream")
                break
            while len(pending) < args.concurrency:
                tags = sample_tags(rng)
                pending.add(pool.submit(one_story, index, tags))
                index += 1
            done = next(as_completed(pending))
            pending.discard(done)
            status, payload = done.result()
            if status == "accepted":
                stories += 1
                tokens += approximate_tokens(payload)
                consecutive_failures = 0
            else:
                rejected[payload or status] += 1
                consecutive_failures += 1
            attempts = stories + sum(rejected.values())
            every = 1 if args.verbose else 10
            if attempts % every == 0:
                elapsed = max(1e-9, time.time() - started)
                print(f"  {stories:,} kept / {attempts:,} tried  "
                      f"~{tokens:,} tokens  {stories / (elapsed / 3600):,.0f}/h  "
                      f"{elapsed / attempts:.1f}s per try", flush=True)
                if args.verbose and status != "accepted":
                    print(f"    rejected: {payload}", flush=True)
    except KeyboardInterrupt:
        STOP.set()
        print("\ninterrupted; the output file is complete up to this point",
              flush=True)
    finally:
        STOP.set()
        # Do not wait for workers parked in a backoff. A worker sleeping through
        # a two minute retry wait would otherwise hold the process open long
        # after the interrupt, which makes Ctrl+C look broken.
        pool.shutdown(wait=False, cancel_futures=True)
        handle.close()
        client.close()

    if aborted:
        print(f"\nSTOPPED: {aborted}", flush=True)
        for reason, count in rejected.most_common(3):
            print(f"  most recent failures: {reason[:120]} ({count})", flush=True)

    report = {
        "format": "brittain-shakespeare-synthetic-report-v1",
        "aborted": aborted,
        "model": args.model,
        "base_url": args.base_url,
        "accepted_stories": stories,
        "approximate_tokens": tokens,
        "seconds": round(time.time() - started, 1),
        "rejected": dict(rejected.most_common(20)),
        "transport": dict(attempt_log),
    }
    report_path = project_path(args.report or output.with_name(output.stem + ".report.json"))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\naccepted {stories:,} stories, ~{tokens:,} tokens")
    for reason, count in rejected.most_common(10):
        print(f"  rejected {reason}: {count}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    with keep_awake("synthetic story generation"):
        main()
