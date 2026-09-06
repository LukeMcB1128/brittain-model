#!/usr/bin/env python3
"""Compare local Ollama models as Brittain3 curriculum teachers.

The fixed tests are not sent to the teacher. A model must solve the stated task,
return structured JSON, and produce code that passes the hidden local tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.ollama_teacher import SOLUTION_SCHEMA, chat_json  # noqa: E402
from brittain.training_v3 import load_json  # noqa: E402
from brittain.verification_v3 import DEFAULT_TSC, backend_status, verify_program  # noqa: E402


SYSTEM = """You create correct reference code for a model-training pipeline.
Return one JSON object that follows the given schema. Put only source code in
the solution field. Do not use Markdown fences. Do not include tests unless the
request asks for them. Keep the solution small, direct, and portable."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/brittain3_teacher_bakeoff.json")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--output", default="runs/brittain3-teacher-bakeoff.json")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--verify-timeout", type=float, default=15.0)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--tsc", default=str(DEFAULT_TSC))
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def print_preflight(tsc: str) -> dict[str, str | None]:
    status = backend_status(tsc)
    print("verification backends:")
    for language, command in status.items():
        print(f"  {language:<12}{command or 'UNAVAILABLE'}")
    return status


def validate_cases(cases: list[dict], status: dict[str, str | None], tsc: str, timeout: float) -> None:
    """Prove that each available task passes with its frozen reference code."""
    required = {"id", "language", "request", "reference", "tests"}
    identifiers = [case.get("id") for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit("teacher bake-off case ids must be unique")
    for case in cases:
        missing = sorted(required - set(case))
        if missing:
            raise SystemExit(f"{case.get('id', '?')} is missing {missing}")
        if status.get(case["language"]) is None:
            continue
        result = verify_program(
            case["language"], case["reference"], case["tests"], timeout=timeout, tsc=tsc
        )
        if not result.ok:
            raise SystemExit(
                f"broken bake-off case {case['id']}: {result.phase}: {result.detail}"
            )


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    result = {}
    for model, items in grouped.items():
        eligible = [item for item in items if item["status"] != "toolchain_unavailable"]
        rates = [item["output_tokens_per_second"] for item in items
                 if item.get("output_tokens_per_second") is not None]
        result[model] = {
            "cases": len(items),
            "eligible_cases": len(eligible),
            "passed": sum(item["status"] == "pass" for item in eligible),
            "pass_rate": (sum(item["status"] == "pass" for item in eligible) / len(eligible)
                          if eligible else None),
            "structured_json_rate": (
                sum(item["status"] not in ("request_error", "json_error") for item in items)
                / len(items) if items else None
            ),
            "mean_output_tokens_per_second": sum(rates) / len(rates) if rates else None,
        }
    return result


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    if config.get("format") != "brittain3-teacher-bakeoff-v1":
        raise SystemExit("unsupported teacher bake-off configuration")
    status = print_preflight(args.tsc)
    validate_cases(config["cases"], status, args.tsc, args.verify_timeout)
    print("available reference solutions: PASS")
    if args.preflight:
        return 0

    models = args.models or config["models"]
    languages = set(args.languages) if args.languages else None
    cases = [case for case in config["cases"]
             if languages is None or case["language"] in languages]
    rows = []
    for model_index, model in enumerate(models):
        print(f"\nmodel: {model}", flush=True)
        for case_index, case in enumerate(cases):
            print(f"  {case['id']} ... ", end="", flush=True)
            if status.get(case["language"]) is None:
                row = {
                    "model": model, "case_id": case["id"], "language": case["language"],
                    "status": "toolchain_unavailable", "detail": "compiler or runtime is absent",
                }
                rows.append(row)
                print("SKIP (toolchain unavailable)")
                continue
            seed = int(config["seed"]) + model_index * 1000 + case_index
            try:
                reply = chat_json(
                    args.endpoint, model, SYSTEM,
                    f"Language: {case['language']}\nTask: {case['request']}",
                    schema=SOLUTION_SCHEMA, seed=seed, timeout=args.timeout,
                    keep_alive=args.keep_alive,
                )
                solution = reply.content.get("solution")
                if not isinstance(solution, str) or not solution.strip():
                    raise ValueError("solution is missing or empty")
                checked = verify_program(
                    case["language"], solution, case["tests"],
                    timeout=args.verify_timeout, tsc=args.tsc,
                )
                row = {
                    "model": model, "case_id": case["id"], "language": case["language"],
                    "status": "pass" if checked.ok else "verification_failed",
                    "phase": checked.phase, "detail": checked.detail,
                    "elapsed_seconds": reply.elapsed_seconds,
                    "prompt_tokens": reply.prompt_tokens,
                    "output_tokens": reply.output_tokens,
                    "output_tokens_per_second": reply.output_tokens_per_second,
                    "solution": solution,
                }
            except ValueError as exc:
                row = {
                    "model": model, "case_id": case["id"], "language": case["language"],
                    "status": "json_error", "detail": str(exc),
                }
            except RuntimeError as exc:
                row = {
                    "model": model, "case_id": case["id"], "language": case["language"],
                    "status": "request_error", "detail": str(exc),
                }
            rows.append(row)
            print(row["status"])

    report = {
        "format": "brittain3-teacher-bakeoff-report-v1",
        "config": args.config,
        "models": models,
        "backend_status": status,
        "results": rows,
        "summary": summarize(rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report["summary"], indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
