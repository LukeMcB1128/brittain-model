"""Universal generation benchmark CLI for the BRITTAIN model family.

Evaluates any BRITTAIN checkpoint across five core tracks:
- Python coding generation (AST syntax, execution pass@k)
- JavaScript coding generation (node --check syntax, execution pass@k)
- TypeScript coding generation (tsc --noEmit type/syntax check, execution pass@k)
- JSON structured generation (strict JSON validity, schema/key adherence)
- Prose generation & quality (lexical diversity Distinct-1/2, repetition collapse, sentence closure)

Usage:
    # Run all tracks on multiple checkpoints
    python3 scripts/evaluate/universal_bench.py checkpoints/brittain2_235m_weights.pt \
        checkpoints/brittain2_50m_bs.pt

    # Run quick smoke test
    python3 scripts/evaluate/universal_bench.py checkpoints/brittain2_50m_bs.pt --quick

    # Run specific tracks
    python3 scripts/evaluate/universal_bench.py checkpoints/brittain2_235m_weights.pt \
        --tracks python,json,prose

    # Validate reference solutions (no models needed)
    python3 scripts/evaluate/universal_bench.py --validate

Every run merges its results into benchmarks/results/universal_benchmark_results.json
(keyed by checkpoint name) and regenerates benchmarks/results/leaderboard.html.
Use --no-merge / --no-html / --output / --html to change that.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_RESULTS_JSON = Path("benchmarks/results/universal_benchmark_results.json")

from brittain.universal_bench import (
    TRACKS,
    UniversalBenchmarkRunner,
    load_universal_tasks,
    validate_universal_suite,
)
from brittain.verification_v3 import DEFAULT_TSC, backend_status

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_html_report import write_report  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Universal BRITTAIN Generation Benchmark")
    parser.add_argument("checkpoints", nargs="*", help="Paths to model checkpoints")
    parser.add_argument("--tracks", default="all",
                        help=f"Comma-separated list of tracks to run ({','.join(TRACKS)}) or 'all'")
    parser.add_argument("--samples", type=int, default=5, help="Samples drawn per prompt (default 5)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum generated tokens per sample")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature (default 0.4)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top_p (default 0.95)")
    parser.add_argument("--repetition-penalty", type=float, default=1.12, help="Repetition penalty (default 1.12)")
    parser.add_argument("--device", default=None, help="Device to use: auto, cuda, mps, or cpu")
    parser.add_argument("--timeout", type=float, default=5.0, help="Execution timeout in seconds")
    parser.add_argument("--tsc", default=str(DEFAULT_TSC), help="Path to tsc compiler executable")
    parser.add_argument("--validate", action="store_true",
                        help="Validate ground-truth reference suite without loading any models")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke run (samples=2, max-tokens=64, 3 tasks per track)")
    parser.add_argument("--output", default=str(DEFAULT_RESULTS_JSON),
                        help=f"Save structured results JSON to this path (default {DEFAULT_RESULTS_JSON})")
    parser.add_argument("--html", default=None,
                        help="HTML leaderboard path (default: leaderboard.html beside the results JSON)")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip regenerating the HTML leaderboard")
    parser.add_argument("--no-merge", action="store_true",
                        help="Overwrite the results JSON instead of merging into existing entries")
    return parser.parse_args()


def format_table(headers: List[str], rows: List[List[str]], alignments: List[str]) -> str:
    """Format an ASCII table with customizable column alignments ('<' or '>')."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    header_parts = []
    for i, (h, w, align) in enumerate(zip(headers, widths, alignments)):
        header_parts.append(f"{h:<{w}}" if align == "<" else f"{h:>{w}}")
    header_line = " | ".join(header_parts)
    separator_line = "-+-".join("-" * w for w in widths)

    row_lines = []
    for row in rows:
        parts = []
        for val, w, align in zip(row, widths, alignments):
            parts.append(f"{val:<{w}}" if align == "<" else f"{val:>{w}}")
        row_lines.append(" | ".join(parts))

    return f"{header_line}\n{separator_line}\n" + "\n".join(row_lines)


def print_results(reports: List[dict]):
    """Print clean summary matrix and per-track breakdown tables."""
    print("\n" + "=" * 108)
    print(" " * 36 + "UNIVERSAL BENCHMARK SUMMARY MATRIX")
    print("=" * 108)

    # Matrix Table
    headers = [
        "Model", "Params", "BPB Cod", "BPB Pro", "Py Syn", "Py p@1",
        "JS Syn", "JS p@1", "TS Syn", "TS p@1", "JSON Syn", "JSON Sch", "Prose D1", "Prose Cls"
    ]
    alignments = ["<", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">"]
    rows = []

    for rep in reports:
        name = Path(rep["checkpoint"]).name
        if len(name) > 22:
            name = name[:19] + "..."
        params = f"{rep['params']:,}" if rep["params"] else "N/A"
        bpb_code = f"{rep['bpb_code']:.3f}" if not math.isnan(rep["bpb_code"]) else "N/A"
        bpb_prose = f"{rep['bpb_prose']:.3f}" if not math.isnan(rep["bpb_prose"]) else "N/A"

        tracks = rep["tracks"]
        py_syn = f"{100*tracks['python']['syntax_validity']:.0f}%" if "python" in tracks else "-"
        py_p1 = f"{100*tracks['python']['pass@1']:.1f}%" if "python" in tracks else "-"

        js_syn = f"{100*tracks['javascript']['syntax_validity']:.0f}%" if "javascript" in tracks else "-"
        js_p1 = f"{100*tracks['javascript']['pass@1']:.1f}%" if "javascript" in tracks else "-"

        ts_syn = f"{100*tracks['typescript']['syntax_validity']:.0f}%" if "typescript" in tracks else "-"
        ts_p1 = f"{100*tracks['typescript']['pass@1']:.1f}%" if "typescript" in tracks else "-"

        json_syn = f"{100*tracks['json']['syntax_validity']:.0f}%" if "json" in tracks else "-"
        json_sch = f"{100*tracks['json'].get('schema_validity', 0):.0f}%" if "json" in tracks else "-"

        prose_d1 = f"{tracks['prose'].get('distinct_1', 0):.2f}" if "prose" in tracks else "-"
        prose_cls = f"{100*tracks['prose'].get('sentence_closed_rate', 0):.0f}%" if "prose" in tracks else "-"

        rows.append([
            name, params, bpb_code, bpb_prose, py_syn, py_p1,
            js_syn, js_p1, ts_syn, ts_p1, json_syn, json_sch, prose_d1, prose_cls
        ])

    print(format_table(headers, rows, alignments))
    print()

    # Per-Track Details
    print("=" * 108)
    print(" " * 42 + "TRACK BREAKDOWN DETAILS")
    print("=" * 108)

    for track in TRACKS:
        has_track = any(track in rep["tracks"] for rep in reports)
        if not has_track:
            continue

        print(f"\n[Track: {track.upper()}]")
        if track in ("python", "javascript", "typescript"):
            th = ["Model", "Tasks", "Syntax %", "Pass@1", "Solved", "Collapse %", "Empty %"]
            ta = ["<", ">", ">", ">", ">", ">", ">"]
            tr = []
            for rep in reports:
                if track not in rep["tracks"]:
                    continue
                d = rep["tracks"][track]
                tr.append([
                    Path(rep["checkpoint"]).name,
                    str(d["tasks"]),
                    f"{100*d['syntax_validity']:.1f}%",
                    f"{100*d['pass@1']:.1f}%",
                    f"{d['solved_tasks']}/{d['tasks']}",
                    f"{100*d['repetition_collapse']:.1f}%",
                    f"{100*d['empty_rate']:.1f}%",
                ])
            print(format_table(th, tr, ta))

        elif track == "json":
            th = ["Model", "Tasks", "JSON Syntax %", "Schema Valid %", "Solved", "Collapse %", "Empty %"]
            ta = ["<", ">", ">", ">", ">", ">", ">"]
            tr = []
            for rep in reports:
                if "json" not in rep["tracks"]:
                    continue
                d = rep["tracks"]["json"]
                tr.append([
                    Path(rep["checkpoint"]).name,
                    str(d["tasks"]),
                    f"{100*d['syntax_validity']:.1f}%",
                    f"{100*d.get('schema_validity', 0):.1f}%",
                    f"{d['solved_tasks']}/{d['tasks']}",
                    f"{100*d['repetition_collapse']:.1f}%",
                    f"{100*d['empty_rate']:.1f}%",
                ])
            print(format_table(th, tr, ta))

        elif track == "prose":
            th = ["Model", "Tasks", "Distinct-1", "Distinct-2", "Sentence Closed %", "Collapse %", "Empty %"]
            ta = ["<", ">", ">", ">", ">", ">", ">"]
            tr = []
            for rep in reports:
                if "prose" not in rep["tracks"]:
                    continue
                d = rep["tracks"]["prose"]
                tr.append([
                    Path(rep["checkpoint"]).name,
                    str(d["tasks"]),
                    f"{d.get('distinct_1', 0):.3f}",
                    f"{d.get('distinct_2', 0):.3f}",
                    f"{100*d.get('sentence_closed_rate', 0):.1f}%",
                    f"{100*d['repetition_collapse']:.1f}%",
                    f"{100*d['empty_rate']:.1f}%",
                ])
            print(format_table(th, tr, ta))



def merge_reports(existing: List[dict], new: List[dict]) -> List[dict]:
    """Merge fresh reports into the stored leaderboard, keyed by checkpoint name.

    A re-run of a checkpoint replaces its old entry in place; checkpoints that
    were not re-run are preserved so the leaderboard accumulates over time.
    """
    merged = list(existing)
    index = {rep.get("checkpoint"): i for i, rep in enumerate(merged)}
    for rep in new:
        key = rep.get("checkpoint")
        if key in index:
            merged[index[key]] = rep
        else:
            index[key] = len(merged)
            merged.append(rep)
    return merged


def load_existing_reports(path: Path) -> List[dict]:
    """Read previously stored reports, tolerating a missing or corrupt file."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        print(f"[!] Could not read existing results at {path} ({exc}); starting a fresh file.")
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        print(f"[!] Unexpected results format in {path}; starting a fresh file.")
        return []
    return [rep for rep in data if isinstance(rep, dict) and "checkpoint" in rep]


def export_results(reports: List[dict], args) -> None:
    """Write the results JSON and refresh the HTML leaderboard."""
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = reports if args.no_merge else merge_reports(load_existing_reports(out_path), reports)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[✓] Results exported to {out_path} ({len(payload)} checkpoint(s) on record)")

    if args.no_html:
        return
    html_path = Path(args.html) if args.html else out_path.parent / "leaderboard.html"
    try:
        written = write_report(out_path, html_path)
    except Exception as exc:  # report generation must never lose the JSON results
        print(f"[!] HTML leaderboard not regenerated: {exc}")
        return
    print(f"[✓] HTML leaderboard updated at {written}")


def main() -> int:
    args = parse_args()

    if args.validate:
        print("[*] Validating ground-truth universal benchmark reference suite...")
        ok, failures = validate_universal_suite(timeout=args.timeout, tsc=args.tsc)
        if ok:
            print("[✓] All 75 reference solutions across Python, JavaScript, TypeScript, JSON, and Prose passed!")
            return 0
        print(f"[✗] Reference validation failed with {len(failures)} error(s):")
        for f in failures:
            print(f"    - {f}")
        return 1

    if not args.checkpoints:
        print("Error: No checkpoints provided. Pass at least one checkpoint path or use --validate.")
        return 1

    selected_tracks = list(TRACKS) if args.tracks == "all" else [t.strip() for t in args.tracks.split(",")]
    invalid_tracks = [t for t in selected_tracks if t not in TRACKS]
    if invalid_tracks:
        print(f"Error: Invalid track(s): {invalid_tracks}. Available tracks: {list(TRACKS)}")
        return 1

    tasks = load_universal_tasks(selected_tracks)
    if not tasks:
        print(f"Error: No tasks found for tracks: {selected_tracks}")
        return 1

    if args.quick:
        args.samples = 2
        args.max_tokens = 64
        # Take first 3 tasks per track
        filtered_tasks = []
        counts = {t: 0 for t in selected_tracks}
        for task in tasks:
            if counts[task.track] < 3:
                filtered_tasks.append(task)
                counts[task.track] += 1
        tasks = filtered_tasks
        print(f"[*] Quick mode enabled: {len(tasks)} tasks total (3/track), 2 samples/task, max 64 tokens.")
    else:
        print(f"[*] Loaded {len(tasks)} tasks across tracks: {', '.join(selected_tracks)}.")

    runner = UniversalBenchmarkRunner(tsc_path=Path(args.tsc), timeout=args.timeout, device=args.device)

    reports = []
    for ckpt_path in args.checkpoints:
        if not Path(ckpt_path).exists():
            print(f"[!] Warning: Checkpoint not found, skipping: {ckpt_path}")
            continue
        print(f"\n[*] Evaluating checkpoint: {ckpt_path} ...", flush=True)
        try:
            report = runner.evaluate_checkpoint(
                ckpt_path,
                tasks,
                samples_per_task=args.samples,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            reports.append(report)
        except Exception as exc:
            print(f"[!] Error evaluating {ckpt_path}: {exc}")

    if not reports:
        print("No valid checkpoints evaluated.")
        return 1

    print_results(reports)

    export_results(reports, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
