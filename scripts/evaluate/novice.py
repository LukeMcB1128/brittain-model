"""Novice-code evaluation for the Brittain3 49M pilot.

HumanEval is too hard and too noisy at 49M — Brittain2 XS scores a flat 0%, which
tells you nothing about whether a change helped. This suite is deliberately easier:
36 tasks over easy functions, arrays, objects, loops, parsing, and state updates,
each with a docstring, a worked example, and executable assertions.

    python3 scripts/evaluate/novice.py checkpoints/brittain2_50m_bs.pt --samples 10

Validate the suite itself (no model, runs the reference solutions):

    python3 scripts/evaluate/novice.py --validate

These are BASE-model prompts, not instructions. The model sees a signature and a
docstring and continues it, exactly as in pretraining. Nothing here needs an
instruction-tuned checkpoint.

Reported metrics:

- **pass@1 / pass@k** — unbiased estimator from Chen et al. 2021. With `n`
  samples and `c` correct, pass@k is 1 - C(n-c, k)/C(n, k).
- **Per-category pass@1** — which of the six skills is missing, not just a total.
- **Repetition collapse** — fraction of completions where the generated token
  4-grams are mostly duplicates. A 49M model that degenerates into repeated words
  can still parse, so syntax validity alone does not catch this.
- **Syntax validity** — the completion parses as Python.

SAFETY. Candidate code is EXECUTED. Every candidate runs in a subprocess under
`-I` (isolated: no cwd on sys.path, no PYTHON* environment variables) with its
working directory set to a fresh empty temporary directory, under a wall-clock
timeout. That is enough for code sampled from our own small model. It is NOT a
sandbox for untrusted third-party code — for that, run this whole script inside
`docker run --rm -m 4g --network none`.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings("ignore", category=SyntaxWarning)

from brittain.metrics import pass_at_k, repetition_collapse  # noqa: E402
from brittain.loading import document_prefix, strip_specials  # noqa: E402

DEFAULT_TASKS = PROJECT_ROOT / "benchmarks" / "novice" / "tasks.jsonl"
DEFAULT_REFERENCE = PROJECT_ROOT / "benchmarks" / "novice" / "reference.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="*", help="checkpoints to score")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--validate", action="store_true",
                        help="run the reference solutions and exit; proves the suite is solvable")
    parser.add_argument("--samples", type=int, default=10, help="completions per task")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--timeout", type=float, default=5.0, help="seconds per candidate")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default=None, help="write the full JSON report here")
    return parser.parse_args()


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid task at {path}:{number}: {exc}") from exc
    return tasks


def truncate_completion(completion: str) -> str:
    """Cut a base-model continuation at the end of the function or class body.

    Every prompt in this suite ends at a `def` or `class` header in column 0, so
    the body is entirely indented and ends at the first non-empty line starting
    in column 0. Without this, a base model runs on into the next unrelated
    definition and the assertions execute whatever it invented.
    """
    kept = []
    for line in completion.splitlines(keepends=True):
        if line.strip() and not line[0].isspace():
            break
        kept.append(line)
    return "".join(kept)


def normalize_indentation(body: str) -> str:
    """Expand LEADING tabs to four spaces.

    These models frequently indent with tabs. That is valid Python on its own,
    but the class tasks put a space-indented header above a tab-indented body and
    Python rejects the mixture with TabError — which would score an otherwise
    correct answer as a failure. Only leading whitespace is touched, so tabs
    inside string literals survive.
    """
    lines = []
    for line in body.splitlines(keepends=True):
        stripped = line.lstrip("\t ")
        leading = line[:len(line) - len(stripped)]
        lines.append(leading.replace("\t", "    ") + stripped)
    return "".join(lines)


def run_candidate(program: str, timeout: float) -> tuple[bool, str]:
    """Execute one candidate program in an isolated subprocess."""
    with tempfile.TemporaryDirectory() as workdir:
        source = Path(workdir) / "candidate.py"
        source.write_text(program, encoding="utf-8")
        try:
            finished = subprocess.run(
                [sys.executable, "-I", str(source)],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except OSError as exc:
            return False, f"spawn failed: {exc}"
    if finished.returncode == 0:
        return True, ""
    return False, (finished.stderr or "").strip().splitlines()[-1:] and \
        (finished.stderr or "").strip().splitlines()[-1] or "nonzero exit"


def build_program(task: dict, body: str) -> str:
    return task["prompt"] + body + "\n\n" + "\n".join(task["tests"]) + "\n"


def validate_suite(tasks: list[dict], reference_path: Path, timeout: float) -> int:
    """Run every reference solution. A task whose own solution fails is broken."""
    references = {row["id"]: row["body"] for row in load_tasks(reference_path)}
    missing = [task["id"] for task in tasks if task["id"] not in references]
    if missing:
        print(f"FAIL: no reference solution for {len(missing)} task(s): {missing}")
        return 1
    failures = []
    for task in tasks:
        ok, detail = run_candidate(build_program(task, references[task["id"]]), timeout)
        if not ok:
            failures.append((task["id"], detail))
    print(f"suite: {len(tasks)} tasks, {len(tasks) - len(failures)} reference solutions pass")
    for task_id, detail in failures:
        print(f"  FAIL {task_id}: {detail}")
    if failures:
        print("\nThe suite is broken. Fix these before scoring any model.")
        return 1
    categories = defaultdict(int)
    for task in tasks:
        categories[task["category"]] += 1
    print("categories: " + ", ".join(f"{name} {count}" for name, count in sorted(categories.items())))
    return 0


def score_checkpoint(path: str, tasks: list[dict], args) -> dict:
    import torch

    from brittain.loading import generate, load_any, resolve_device

    device = resolve_device(args.device)
    model, block_size, enc = load_any(path, device)
    # Framed models must be prompted the way they were trained. This is prepended
    # to the MODEL INPUT only; the program that gets executed stays unframed.
    prefix = document_prefix(enc)
    if prefix:
        print(f"  using repository/file framing ({len(enc.encode(prefix))} tokens)", flush=True)
    torch.manual_seed(args.seed)

    results = []
    collapses = 0
    syntax_ok = 0
    empty = 0
    generations = 0

    for task in tasks:
        prompt_ids = enc.encode(prefix + task["prompt"])
        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        if context.size(1) >= block_size:
            raise SystemExit(f"{task['id']}: prompt exceeds the model context of {block_size}")
        correct = 0
        for _ in range(args.samples):
            with torch.no_grad():
                out = generate(
                    model, context, args.max_new_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
            new_ids = strip_specials(enc, out[0, context.size(1):].tolist())
            generations += 1
            if repetition_collapse(new_ids):
                collapses += 1
            body = normalize_indentation(truncate_completion(enc.decode(new_ids)))
            # An empty body is NOT valid syntax for our purposes. `def f(): <pass>`
            # with only the header parses fine, so counting it as valid reported
            # ~91% syntax for a model that emitted no code at all. Track it
            # separately — a high empty rate means the prompt shape is wrong for
            # the model, not that the model cannot write code.
            if not body.strip():
                empty += 1
                continue
            # Parse the CANDIDATE only. Parsing prompt+body+tests would fold the
            # always-valid test assertions into the score.
            try:
                ast.parse(task["prompt"] + body)
                syntax_ok += 1
            except (SyntaxError, ValueError):
                continue          # unparseable cannot pass; skip the subprocess
            ok, _ = run_candidate(build_program(task, body), args.timeout)
            correct += ok
        results.append({"id": task["id"], "category": task["category"],
                        "n": args.samples, "correct": correct})

    by_category = defaultdict(lambda: [0, 0])
    for row in results:
        totals = by_category[row["category"]]
        totals[0] += pass_at_k(row["n"], row["correct"], 1)
        totals[1] += 1

    return {
        "checkpoint": path,
        "params": model.num_params(),
        "tasks": len(tasks),
        "samples_per_task": args.samples,
        "pass@1": sum(pass_at_k(r["n"], r["correct"], 1) for r in results) / len(results),
        # Only meaningful when the run actually drew 10 samples per task.
        "pass@10": (sum(pass_at_k(r["n"], r["correct"], 10) for r in results) / len(results)
                    if args.samples >= 10 else None),
        "solved_any": sum(1 for r in results if r["correct"] > 0),
        "syntax_validity": syntax_ok / max(1, generations),
        "empty_completions": empty / max(1, generations),
        "repetition_collapse": collapses / max(1, generations),
        "per_category_pass@1": {name: total / count for name, (total, count) in sorted(by_category.items())},
        "per_task": results,
    }


def main() -> int:
    args = parse_args()
    tasks = load_tasks(Path(args.tasks))
    if not tasks:
        raise SystemExit(f"no tasks in {args.tasks}")

    if args.validate:
        return validate_suite(tasks, Path(args.reference), args.timeout)

    if not args.checkpoints:
        raise SystemExit("give at least one checkpoint, or --validate")

    reports = []
    for path in args.checkpoints:
        if not Path(path).exists():
            print(f"[skip] {path} not found")
            continue
        print(f"evaluating {path} ...", flush=True)
        reports.append(score_checkpoint(path, tasks, args))

    header = (f"{'model':<30}{'pass@1':>9}{'pass@10':>9}{'solved':>8}"
              f"{'syntax':>9}{'empty':>8}{'collapse':>10}")
    print()
    print(header)
    print("-" * len(header))
    for report in reports:
        at_ten = f"{100 * report['pass@10']:>8.1f}%" if report["pass@10"] is not None else f"{'n<10':>9}"
        print(f"{Path(report['checkpoint']).name:<30}"
              f"{100 * report['pass@1']:>8.1f}%{at_ten}"
              f"{report['solved_any']:>5}/{report['tasks']:<2}"
              f"{100 * report['syntax_validity']:>8.0f}%"
              f"{100 * report['empty_completions']:>7.0f}%"
              f"{100 * report['repetition_collapse']:>9.1f}%")

    for report in reports:
        print(f"\n{Path(report['checkpoint']).name} by category (pass@1):")
        for name, value in report["per_category_pass@1"].items():
            print(f"  {name:<12}{100 * value:>7.1f}%")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
