"""Evaluate BrittainScript completion, execution, correctness, and translation.

Unlike validation loss, this suite asks a model to finish functions and then
runs hidden tests against the completed programs. It also measures Python-to-BS
translation by comparing stdout from both programs, and optionally measures BPB
over external human-written BrittainScript.

Examples:

    python3 scripts/evaluate/bs_capabilities.py \
        checkpoints/xs_bs_native.pt checkpoints/xs_bs_mixed.pt --samples 5

    python3 scripts/evaluate/bs_capabilities.py checkpoints/xs_bs_native.pt \
        --samples 1 --skip-translations --out runs/bs_quick.jsonl

The generated programs execute in py2bs.verify's temporary subprocess with a
timeout. Programs containing imports, file/input helpers, or pyimport are
classified as unsafe and are never executed.
"""

import argparse
import collections
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from brittain.model import Brittain, GPTConfig
from brittain.model_bs import load as load_bs
from brittain.paths import BS_CORPUS_DIR
from brittain.prompts import format_prompt
from brittain.tokenizer import load_tokenizer

BENCHMARK_DIR = PROJECT_ROOT / "benchmarks" / "brittainscript"
DEFAULT_TASKS = BENCHMARK_DIR / "tasks.jsonl"
DEFAULT_TRANSLATIONS = BENCHMARK_DIR / "translations.jsonl"
DEFAULT_HUMAN_DIR = BENCHMARK_DIR / "human"
DEFAULT_CORPUS = BS_CORPUS_DIR / "bs_corpus.clean.train.jsonl"
DEFAULT_PY2BS = PROJECT_ROOT.parent / "BrittainScript"

ERROR_PREFIXES = ("Error:", "Undefined ")
SYNTAX_PREFIXES = ("Syntax error", "Illegal character")
UNSAFE = re.compile(
    r"(?mi)^\s*add\s+\w+|"
    r"\b(?:pyimport|readfile|readlines|createfile|writefile|appendfile|"
    r"deletefile|input|clear)\s*\("
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="checkpoint files to compare")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--translations", default=str(DEFAULT_TRANSLATIONS))
    parser.add_argument("--human-dir", default=str(DEFAULT_HUMAN_DIR))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                        help="specialist training split used only for novelty")
    parser.add_argument("--py2bs-path", default=str(DEFAULT_PY2BS))
    parser.add_argument("--samples", type=int, default=5,
                        help="samples per functional and translation task")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--translation-max-tokens", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--ngram", type=int, default=8)
    parser.add_argument("--novel-below", type=float, default=0.8)
    parser.add_argument("--skip-translations", action="store_true")
    parser.add_argument("--skip-human", action="store_true")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "runs" / "bs_capabilities.jsonl"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    return args


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise SystemExit(f"invalid JSON at {path}:{number}: {exc}") from exc
    return rows


def normalise(text):
    return re.sub(r"\s+", " ", text).strip()


def word_ngrams(text, n):
    words = normalise(text).split()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def load_novelty_reference(path, ngram):
    if not os.path.exists(path):
        print(f"novelty disabled: no training corpus at {path}")
        return None
    exact, grams, count = set(), set(), 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                source = json.loads(line)["bs"]
            except (ValueError, KeyError):
                continue
            exact.add(normalise(source))
            grams.update(word_ngrams(source, ngram))
            count += 1
    print(f"novelty reference: {count:,} programs, {len(grams):,} {ngram}-grams")
    return exact, grams


def novelty(source, reference, ngram, threshold):
    if reference is None:
        return None, None
    exact, corpus_grams = reference
    norm = normalise(source)
    mine = word_ngrams(source, ngram)
    if norm in exact:
        containment = 1.0
    elif not mine:
        return None, None
    else:
        containment = len(mine & corpus_grams) / len(mine)
    return containment, containment < threshold


class BSCfg:
    def __init__(self, model):
        self.vocab_size = model.vocab
        self.block_size = model.block


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "cfg" in checkpoint:
        cfg = GPTConfig(**checkpoint["cfg"])
        model = Brittain(cfg).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model, cfg, load_tokenizer(checkpoint), True
    model, tokenizer = load_bs(path, device)
    return model, BSCfg(model), tokenizer, False


@torch.no_grad()
def generate(model, cfg, tokenizer, is_brittain, prompt, count, max_tokens,
             args, seed, device):
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.eot]
    if len(prompt_ids) >= cfg.block_size:
        prompt_ids = prompt_ids[-(cfg.block_size - 1):]

    outputs = []
    torch.manual_seed(seed)
    random.seed(seed)
    while len(outputs) < count:
        size = min(args.batch, count - len(outputs))
        ids = torch.tensor([prompt_ids] * size, dtype=torch.long, device=device)
        if is_brittain:
            done = model.generate(
                ids, max_tokens, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
        else:
            done = model.generate(
                ids, max_tokens, temperature=args.temperature,
                top_p=args.top_p, repetition_penalty=args.repetition_penalty,
            )
        for row in done.tolist():
            body = row[len(prompt_ids):]
            if tokenizer.eot in body:
                body = body[:body.index(tokenizer.eot)]
            outputs.append(tokenizer.decode(body))
    return outputs


def diagnostic_lines(result):
    return result.stdout.splitlines() + (result.error or "").splitlines()


def classify_execution(result):
    lines = [line.strip() for line in diagnostic_lines(result)]
    syntax_ok = not any(line.startswith(SYNTAX_PREFIXES) for line in lines)
    runtime_ok = (result.ok and syntax_ok
                  and not any(line.startswith(ERROR_PREFIXES) for line in lines))
    if result.timed_out:
        verdict = "timeout"
    elif not syntax_ok:
        verdict = "syntax error"
    elif not result.ok:
        verdict = "crash"
    elif not runtime_ok:
        verdict = "runtime error"
    else:
        verdict = "runs"
    return syntax_ok, runtime_ok, verdict


def validate_assets(tasks, translations, run_bs, compare):
    failures = []
    for task in tasks:
        source = task["prompt"] + task["reference"] + task["harness"]
        result = run_bs(source, timeout=5)
        if not result.ok or result.stdout != task["expected"]:
            failures.append(
                f"task {task['id']}: expected {task['expected']!r}, "
                f"got {result.stdout!r}, error={result.error!r}"
            )
    for task in translations:
        ok, _, _, reason = compare(task["python"], task["reference_bs"], timeout=5)
        if not ok:
            failures.append(f"translation {task['id']}: {reason}")
    if failures:
        raise SystemExit("invalid benchmark assets:\n  " + "\n  ".join(failures))


def load_human_sources(directory):
    root = Path(directory)
    if not root.exists():
        return []
    return [(path.name, path.read_text(encoding="utf-8"))
            for path in sorted(root.glob("*.bs"))]


@torch.no_grad()
def bits_per_byte(model, cfg, tokenizer, is_brittain, sources, device):
    total_nll = 0.0
    total_bytes = 0
    for _, text in sources:
        ids = tokenizer.encode(text)
        total_bytes += len(text.encode("utf-8"))
        for start in range(0, len(ids) - 1, cfg.block_size):
            chunk = ids[start:start + cfg.block_size + 1]
            if len(chunk) < 2:
                continue
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            if is_brittain:
                logits, _ = model(x, y)
            else:
                logits = model(x)
            total_nll += torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
            ).item()
    if not total_bytes:
        return None
    return total_nll / (math.log(2) * total_bytes)


def evaluate_human_sources(sources, run_bs, timeout):
    rows = []
    for name, source in sources:
        if UNSAFE.search(source):
            rows.append({"file": name, "verdict": "unsafe"})
            continue
        result = run_bs(source, timeout=timeout)
        syntax_ok, runtime_ok, verdict = classify_execution(result)
        rows.append({"file": name, "syntax": syntax_ok, "runtime": runtime_ok,
                     "verdict": verdict})
    return rows


def evaluate_functional(model_name, model, cfg, tokenizer, is_brittain,
                        tasks, reference, args, device, run_bs):
    rows = []
    for task_index, task in enumerate(tasks):
        samples = generate(
            model, cfg, tokenizer, is_brittain, task["prompt"], args.samples,
            args.max_tokens, args, args.seed + task_index, device,
        )
        for sample_index, completion in enumerate(samples):
            candidate = task["prompt"] + completion
            containment, novel = novelty(candidate, reference, args.ngram,
                                         args.novel_below)
            row = {"kind": "functional", "model": model_name,
                   "task": task["id"], "category": task["category"],
                   "sample": sample_index,
                   "completion": completion, "containment": containment,
                   "novel": novel, "syntax": False, "runtime": False,
                   "functional": False}
            program = candidate + "\n" + task["harness"]
            if UNSAFE.search(program):
                row["verdict"] = "unsafe"
            else:
                result = run_bs(program, timeout=args.timeout)
                syntax_ok, runtime_ok, verdict = classify_execution(result)
                row.update({"syntax": syntax_ok, "runtime": runtime_ok,
                            "functional": runtime_ok and result.stdout == task["expected"],
                            "verdict": verdict, "stdout": result.stdout,
                            "error": result.error})
                if runtime_ok and not row["functional"]:
                    row["verdict"] = "wrong answer"
            rows.append(row)
        print(f"  functional {task_index + 1:>2}/{len(tasks)} {task['id']}", flush=True)
    return rows


def evaluate_translations(model_name, model, cfg, tokenizer, is_brittain,
                          tasks, args, device, run_python, run_bs):
    rows = []
    for task_index, task in enumerate(tasks):
        python_result = run_python(task["python"], timeout=args.timeout)
        prompt = format_prompt("Write this Python program in BrittainScript.",
                               task["python"])
        samples = generate(
            model, cfg, tokenizer, is_brittain, prompt, args.samples,
            args.translation_max_tokens, args, args.seed + 10_000 + task_index,
            device,
        )
        for sample_index, completion in enumerate(samples):
            row = {"kind": "translation", "model": model_name,
                   "task": task["id"], "sample": sample_index,
                   "completion": completion, "syntax": False,
                   "runtime": False, "functional": False}
            if UNSAFE.search(completion):
                row["verdict"] = "unsafe"
            else:
                result = run_bs(completion, timeout=args.timeout)
                syntax_ok, runtime_ok, verdict = classify_execution(result)
                equivalent = (python_result.ok and runtime_ok
                              and result.stdout == python_result.stdout)
                row.update({"syntax": syntax_ok, "runtime": runtime_ok,
                            "functional": equivalent, "verdict": verdict,
                            "stdout": result.stdout, "expected": python_result.stdout,
                            "error": result.error})
                if runtime_ok and not equivalent:
                    row["verdict"] = "stdout mismatch"
            rows.append(row)
        print(f"  translate  {task_index + 1:>2}/{len(tasks)} {task['id']}", flush=True)
    return rows


def rate(rows, key):
    return sum(bool(row.get(key)) for row in rows) / max(1, len(rows))


def task_pass(rows, first_only=False):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    passed = 0
    for task_rows in grouped.values():
        task_rows.sort(key=lambda row: row["sample"])
        selected = task_rows[:1] if first_only else task_rows
        passed += any(row.get("functional") for row in selected)
    return passed / max(1, len(grouped))


def summarize(model_name, functional, translations, human_bpb):
    eligible = [row for row in functional if row.get("novel") is not None]
    summary = {
        "kind": "summary",
        "model": model_name,
        "human_bpb": human_bpb,
        "functional_samples": len(functional),
        "syntax_rate": rate(functional, "syntax"),
        "runtime_rate": rate(functional, "runtime"),
        "functional_sample_rate": rate(functional, "functional"),
        "functional_pass_at_1": task_pass(functional, first_only=True),
        "functional_pass_at_k": task_pass(functional),
        "novel_rate": rate(eligible, "novel") if eligible else None,
        "categories": {},
    }
    categories = sorted({row["category"] for row in functional})
    for category in categories:
        category_rows = [row for row in functional
                         if row["category"] == category]
        summary["categories"][category] = {
            "tasks": len({row["task"] for row in category_rows}),
            "pass_at_1": task_pass(category_rows, first_only=True),
            "pass_at_k": task_pass(category_rows),
        }
    if translations:
        summary.update({
            "translation_samples": len(translations),
            "translation_syntax_rate": rate(translations, "syntax"),
            "translation_runtime_rate": rate(translations, "runtime"),
            "translation_sample_rate": rate(translations, "functional"),
            "translation_pass_at_1": task_pass(translations, first_only=True),
            "translation_pass_at_k": task_pass(translations),
        })
    return summary


def pct(value):
    return "  n/a" if value is None else f"{100 * value:5.1f}%"


def print_summary(summary, samples):
    bpb = summary["human_bpb"]
    print(f"\n{summary['model']}")
    print(f"  human BS BPB          {'n/a' if bpb is None else f'{bpb:.3f}'}")
    print(f"  prompted syntax       {pct(summary['syntax_rate'])}")
    print(f"  prompted runtime      {pct(summary['runtime_rate'])}")
    print(f"  functional pass@1     {pct(summary['functional_pass_at_1'])}")
    if samples > 1:
        print(f"  functional pass@{samples:<2}    {pct(summary['functional_pass_at_k'])}")
    print(f"  functional samples    {pct(summary['functional_sample_rate'])}")
    print(f"  novel completions     {pct(summary['novel_rate'])}")
    if "translation_samples" in summary:
        print(f"  translation syntax    {pct(summary['translation_syntax_rate'])}")
        print(f"  translation runtime   {pct(summary['translation_runtime_rate'])}")
        print(f"  translation pass@1    {pct(summary['translation_pass_at_1'])}")
        if samples > 1:
            print(f"  translation pass@{samples:<2}   {pct(summary['translation_pass_at_k'])}")


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(os.path.expanduser(args.py2bs_path)))
    try:
        from py2bs.verify import compare, run_brittainscript, run_python
    except ImportError as exc:
        raise SystemExit(f"cannot import py2bs from {args.py2bs_path}: {exc}") from exc

    tasks = read_jsonl(args.tasks)
    translations = [] if args.skip_translations else read_jsonl(args.translations)
    validate_assets(tasks, translations, run_brittainscript, compare)
    print(f"validated {len(tasks)} functional tasks and "
          f"{len(translations)} translation tasks")

    reference = load_novelty_reference(args.corpus, args.ngram)
    human_sources = [] if args.skip_human else load_human_sources(args.human_dir)
    human_rows = evaluate_human_sources(human_sources, run_brittainscript,
                                        args.timeout)
    if human_sources:
        print(f"human set: {len(human_sources)} files, "
              f"{sum(row.get('runtime', False) for row in human_rows)} run cleanly")
    else:
        print(f"human set empty: add .bs files under {args.human_dir}")

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    all_rows = [{"kind": "human_file", **row} for row in human_rows]
    summaries = []
    started = time.time()
    for checkpoint in args.checkpoints:
        if not os.path.exists(checkpoint):
            print(f"[skip] {checkpoint} not found")
            continue
        name = Path(checkpoint).name
        print(f"\nloading {name} on {device.type} ...", flush=True)
        model, cfg, tokenizer, is_brittain = load_model(checkpoint, device)
        print(f"  {model.num_params():,} params | ctx {cfg.block_size} | "
              f"{tokenizer.name}")
        human_bpb = bits_per_byte(model, cfg, tokenizer, is_brittain,
                                  human_sources, device)
        functional = evaluate_functional(
            name, model, cfg, tokenizer, is_brittain, tasks, reference,
            args, device, run_brittainscript,
        )
        translated = []
        if translations:
            translated = evaluate_translations(
                name, model, cfg, tokenizer, is_brittain, translations,
                args, device, run_python, run_brittainscript,
            )
        summary = summarize(name, functional, translated, human_bpb)
        summaries.append(summary)
        all_rows.extend(functional)
        all_rows.extend(translated)
        all_rows.append(summary)
        print_summary(summary, args.samples)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nDetailed results -> {output}")
    print(f"Elapsed: {time.time() - started:.1f}s")
    if not summaries:
        raise SystemExit("no checkpoints evaluated")


if __name__ == "__main__":
    main()
