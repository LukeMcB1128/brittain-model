"""Universal generation benchmark engine for the BRITTAIN model family.

Evaluates models across five core generation tracks:
- Python coding generation (AST syntax, execution pass@k, indentation integrity)
- JavaScript coding generation (node --check syntax, execution pass@k)
- TypeScript coding generation (tsc --noEmit type/syntax check, execution pass@k)
- JSON structured generation (strict JSON validity, schema/key adherence, completion closure)
- Prose generation & quality (lexical diversity Distinct-1/2, repetition collapse, sentence closure)
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .loading import document_prefix, generate, load_any, resolve_device, strip_specials
from .metrics import pass_at_k, repetition_collapse
from .paths import BENCHMARK_PROMPTS_DIR, PROJECT_ROOT
from .prompts import format_prompt
from .verification_v3 import DEFAULT_TSC, backend_status, verify_program, verify_syntax

UNIVERSAL_DIR = PROJECT_ROOT / "benchmarks" / "universal"
TRACKS = ("python", "javascript", "typescript", "json", "prose")

# Bumped whenever a change alters what a score MEANS, so old rows in a merged
# results file can be spotted instead of silently compared against new ones.
#   1 -> original completion-only harness
#   2 -> instruct/SFT checkpoints prompted with the Alpaca template they were
#        trained on; prose no longer reports a fake pass@1
HARNESS_VERSION = 2

PROMPT_STYLES = ("completion", "instruct")

# Same convention serve.py and sample.py use to infer mode, so all three agree.
INSTRUCT_NAME_TAGS = ("sft", "instruct")


@dataclass
class UniversalTask:
    id: str
    track: str
    category: str
    prompt: str
    entry_point: Optional[str] = None
    tests: List[str] = field(default_factory=list)
    expected_keys: List[str] = field(default_factory=list)
    schema_type: str = "object"
    description: str = ""


def load_universal_tasks(tracks: Optional[List[str]] = None) -> List[UniversalTask]:
    """Load universal benchmark tasks for the specified tracks (or all tracks)."""
    selected_tracks = tracks or list(TRACKS)
    tasks: List[UniversalTask] = []
    for track in selected_tracks:
        track_file = UNIVERSAL_DIR / f"{track}.jsonl"
        if not track_file.exists():
            continue
        with track_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                tasks.append(UniversalTask(
                    id=data["id"],
                    track=data["track"],
                    category=data.get("category", "general"),
                    prompt=data["prompt"],
                    entry_point=data.get("entry_point"),
                    tests=data.get("tests", []),
                    expected_keys=data.get("expected_keys", []),
                    schema_type=data.get("schema_type", "object"),
                    description=data.get("description", ""),
                ))
    return tasks


def load_universal_references() -> Dict[str, str]:
    """Load ground-truth reference completions for all universal tasks."""
    ref_file = UNIVERSAL_DIR / "reference.jsonl"
    references: Dict[str, str] = {}
    if not ref_file.exists():
        return references
    with ref_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            references[data["id"]] = data["completion"]
    return references


# ---------------- Truncation & Normalization ----------------


def normalize_indentation(body: str) -> str:
    """Expand leading tabs to 4 spaces to avoid TabError in mixed Python blocks."""
    lines = []
    for line in body.splitlines(keepends=True):
        stripped = line.lstrip("\t ")
        leading = line[:len(line) - len(stripped)]
        lines.append(leading.replace("\t", "    ") + stripped)
    return "".join(lines)


def truncate_python_completion(completion: str) -> str:
    """Truncate Python function completion at the end of the indented block."""
    kept = []
    for line in completion.splitlines(keepends=True):
        if line.strip() and not line[0].isspace():
            break
        kept.append(line)
    return "".join(kept)


def truncate_braced_completion(completion: str) -> str:
    """Truncate JS/TS/JSON completion at the end of the balanced outer block."""
    depth = 1
    quote: Optional[str] = None
    escaped = False
    line_comment = False
    block_comment = False
    index = 0
    while index < len(completion):
        ch = completion[index]
        nxt = completion[index + 1] if index + 1 < len(completion) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
        elif block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                index += 1
        elif quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
        elif ch == "/" and nxt == "/":
            line_comment = True
            index += 1
        elif ch == "/" and nxt == "*":
            block_comment = True
            index += 1
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch in ("{", "["):
            depth += 1
        elif ch in ("}", "]"):
            depth -= 1
            if depth == 0:
                return completion[:index + 1]
        index += 1
    return completion


def truncate_prose_completion(completion: str) -> str:
    """Truncate prose completion at the last sentence-ending punctuation mark if available."""
    last_punct = -1
    for i, ch in enumerate(completion):
        if ch in (".", "!", "?"):
            last_punct = i
    if last_punct != -1 and last_punct + 1 < len(completion):
        # Keep quotes or trailing whitespace if present
        end = last_punct + 1
        while end < len(completion) and completion[end] in ("'", '"', "”", "’", " ", "\n"):
            end += 1
        return completion[:end]
    return completion


# ---------------- JSON Validation ----------------


@dataclass
class JSONValidationResult:
    is_valid_json: bool
    schema_valid: bool
    parsed_object: Any = None
    error: str = ""


def truncate_json_completion(prompt: str, body: str) -> str:
    """Truncate JSON completion when the root object/array opened in prompt reaches depth 0."""
    full = prompt + body
    start = -1
    for i, c in enumerate(full):
        if c in ("{", "["):
            start = i
            break
    if start == -1:
        return body

    depth = 0
    quote = None
    escaped = False
    for i in range(start, len(full)):
        c = full[i]
        if quote:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == quote:
                quote = None
        elif c in ('"', "'"):
            quote = c
        elif c in ("{", "["):
            depth += 1
        elif c in ("}", "]"):
            depth -= 1
            if depth == 0:
                end = i + 1
                return full[len(prompt):end] if end > len(prompt) else ""
    return body


def verify_json_completion(prompt: str, body: str, expected_keys: Optional[List[str]] = None,
                           schema_type: str = "object") -> JSONValidationResult:
    """Verify JSON completion for valid syntax and required schema keys."""
    truncated = truncate_json_completion(prompt, body)
    full_text = prompt + truncated
    # Strip trailing whitespace or stray commas before closing brackets
    clean_text = re.sub(r",\s*([}\]])", r"\1", full_text.strip())

    parsed = None
    try:
        parsed = json.loads(clean_text)
    except json.JSONDecodeError as exc:
        # Try raw full_text
        try:
            parsed = json.loads(full_text.strip())
        except json.JSONDecodeError:
            return JSONValidationResult(is_valid_json=False, schema_valid=False, error=str(exc))

    if schema_type == "object" and not isinstance(parsed, dict):
        return JSONValidationResult(is_valid_json=True, schema_valid=False, parsed_object=parsed,
                                    error="Expected JSON object dict")
    if schema_type == "array" and not isinstance(parsed, list):
        return JSONValidationResult(is_valid_json=True, schema_valid=False, parsed_object=parsed,
                                    error="Expected JSON array list")

    if expected_keys:
        if isinstance(parsed, dict):
            keys = set(parsed.keys())
            missing = [k for k in expected_keys if k not in keys]
            if missing:
                return JSONValidationResult(is_valid_json=True, schema_valid=False, parsed_object=parsed,
                                            error=f"Missing keys: {missing}")
        elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            # Check items in array for expected keys
            first_keys = set(parsed[0].keys())
            missing = [k for k in expected_keys if k not in first_keys]
            if missing:
                return JSONValidationResult(is_valid_json=True, schema_valid=False, parsed_object=parsed,
                                            error=f"Missing keys in array items: {missing}")

    return JSONValidationResult(is_valid_json=True, schema_valid=True, parsed_object=parsed)


# ---------------- Prose Quality Metrics ----------------


@dataclass
class ProseMetrics:
    distinct_1: float
    distinct_2: float
    is_repetition_collapse: bool
    sentence_closed: bool
    word_count: int


def evaluate_prose_completion(body: str) -> ProseMetrics:
    """Evaluate lexical diversity, repetition collapse, and closure on generated prose."""
    words = re.findall(r"\b[a-zA-Z0-9'-]+\b", body.lower())
    if not words:
        return ProseMetrics(distinct_1=0.0, distinct_2=0.0, is_repetition_collapse=False,
                            sentence_closed=False, word_count=0)

    # Distinct-1
    d1 = len(set(words)) / len(words)

    # Distinct-2
    if len(words) > 1:
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        d2 = len(set(bigrams)) / len(bigrams)
    else:
        d2 = 1.0

    # Repetition collapse
    collapse = False
    if len(words) >= 12:
        fourgrams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
        counts = Counter(fourgrams)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count >= 3 and (most_common_count * 4) / len(words) > 0.4:
            collapse = True

    # Sentence closure
    stripped = body.strip()
    sentence_closed = bool(stripped and stripped[-1] in (".", "!", "?", '"', "”", "'"))

    return ProseMetrics(
        distinct_1=d1,
        distinct_2=d2,
        is_repetition_collapse=collapse,
        sentence_closed=sentence_closed,
        word_count=len(words),
    )


# ---------------- Cross-Tokenizer Bits Per Byte ----------------


@torch.no_grad()
def calculate_bpb(model, tokenizer, block_size: int, text: str, frame: str = "",
                  device: torch.device = torch.device("cpu")) -> float:
    """Calculate Bits Per Byte (BPB) on text, normalized by UTF-8 bytes."""
    n_bytes = len(text.encode("utf-8"))
    if n_bytes == 0:
        return float("nan")
    frame_ids = tokenizer.encode(frame) if frame else []
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return float("nan")
    content_len = block_size - len(frame_ids)
    if content_len < 8:
        return float("nan")

    total_nll = 0.0
    for start in range(0, len(ids) - 1, content_len):
        chunk = ids[start:start + content_len + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([frame_ids + chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([[-100] * len(frame_ids) + chunk[1:]], dtype=torch.long, device=device)
        out = model(x, y) if hasattr(model, "cfg") else model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1),
            reduction="sum", ignore_index=-100
        )
        total_nll += loss.item()
    return total_nll / (math.log(2) * n_bytes)



# ---------------- Prompt style (base vs instruction-tuned) ----------------


def infer_prompt_style(checkpoint_name: str) -> str:
    """Guess whether a checkpoint expects the Alpaca template, by filename.

    Identical convention to serve.py / sample.py so the three never disagree.
    An instruction-tuned model handed a bare code prefix does not complete it —
    it answers in prose, or restates the signature, and the harness scores that
    as broken syntax. That is a measurement of the prompt format, not the model.
    """
    lowered = Path(checkpoint_name).name.lower()
    return "instruct" if any(tag in lowered for tag in INSTRUCT_NAME_TAGS) else "completion"


def build_instruction(task: "UniversalTask") -> Tuple[str, str]:
    """Turn a completion-style task into an (instruction, input) pair.

    The suite is authored as prefixes to continue, so the instruction is
    synthesized from the task's leading comment plus its signature. This is a
    different prompt from the completion track by necessity — instruct and base
    numbers are directionally comparable, not identical measurements.
    """
    prompt = task.prompt
    if task.track in ("python", "javascript", "typescript"):
        comment_lines, code_lines = [], []
        for line in prompt.splitlines():
            stripped = line.strip()
            is_comment = stripped.startswith("#") or stripped.startswith("//")
            if is_comment and not code_lines:
                comment_lines.append(stripped.lstrip("#/ ").strip())
            elif stripped:
                code_lines.append(line)
        described = " ".join(comment_lines).strip()
        lang = {"python": "Python", "javascript": "JavaScript", "typescript": "TypeScript"}[task.track]
        name = task.entry_point or "the function"
        instruction = (
            f"Write a {lang} function `{name}`. {described}".strip()
            if described else f"Write a {lang} function `{name}` that completes the following signature."
        )
        return instruction, "\n".join(code_lines)
    if task.track == "json":
        return "Complete this JSON document. Reply with JSON only.", prompt
    return "Continue the following passage in the same style.", prompt


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)(?:```|\Z)", re.DOTALL)
_TEMPLATE_MARKERS = ("### Instruction:", "### Input:", "### Response:",
                     "Below is an instruction")


def strip_instruct_response(raw: str) -> str:
    """Cut an Alpaca response at the next template section, then unwrap fences."""
    for marker in _TEMPLATE_MARKERS:
        idx = raw.find(marker)
        if idx != -1:
            raw = raw[:idx]
    match = _FENCE_RE.search(raw)
    if match:
        return match.group(1)
    return raw.replace("```", "")


def defines_entry_point(track: str, body: str, entry_point: Optional[str]) -> bool:
    """True if `body` already declares the task's entry point at its own top level."""
    if not entry_point:
        return False
    name = re.escape(entry_point)
    if track == "python":
        return re.search(rf"^[ \t]*(?:async\s+)?def\s+{name}\s*\(", body, re.M) is not None
    return re.search(rf"\b(?:function|class|const|let|var)\s+{name}\b", body) is not None


def truncate_python_definition(completion: str, entry_point: Optional[str]) -> str:
    """Keep one standalone top-level `def` block out of an instruct response.

    truncate_python_completion() assumes an indented continuation and stops at
    the first column-0 line, so it returns "" for a whole function definition.
    """
    lines = completion.splitlines(keepends=True)
    start = None
    if entry_point:
        pattern = re.compile(rf"^[ \t]*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(")
        for i, line in enumerate(lines):
            if pattern.match(line):
                start = i
                break
    if start is None:
        for i, line in enumerate(lines):
            if re.match(r"^(?:async\s+)?(?:def|class)\s", line):
                start = i
                break
    if start is None:
        return ""
    kept = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and not line[0].isspace():
            break
        kept.append(line)
    return "".join(kept)


def assemble_program(task: "UniversalTask", body: str) -> str:
    """Full source to compile: the body alone if it is self-contained, else prompt+body."""
    if defines_entry_point(task.track, body, task.entry_point):
        return body
    return task.prompt + body



def truncate_braced_definition(body: str) -> str:
    """Balance a standalone braced definition from depth 0 (instruct responses).

    truncate_braced_completion() starts at depth 1 because the completion-track
    prompt ends with the opening brace. An instruct response contains the whole
    function, so it must start from the first brace it sees.
    """
    start = body.find("{")
    if start == -1:
        return body
    tail = truncate_braced_completion(body[start + 1:])
    return body[:start + 1] + tail


def parse_quiet(source: str) -> None:
    """ast.parse without leaking the model's own SyntaxWarnings to our stdout."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ast.parse(source)


def longest_parseable_python(task: "UniversalTask", body: str, limit: int = 4000) -> str:
    """Longest prefix of `body` whose assembled program is valid Python.

    These models emit no stop token. They finish the answer and run straight
    into the next document of their training data, frequently glued to the final
    token with no separator at all:

        return lowimport React from 'react';

    Line-based truncation cannot see that boundary — the poison is inside an
    otherwise well-indented line — so a correct function was scored as a syntax
    error. This is the same move truncate_braced_completion() already makes for
    JS/TS, which is why braced tracks were unaffected and Python was not: a
    closing brace marks the end of the answer, a dedent does not.

    Cutting at the last valid point is generous, but it is generous identically
    for every checkpoint and both prompt styles, so comparisons stay honest.
    """
    if not body.strip():
        return body
    candidate = body[:limit]
    while candidate:
        try:
            parse_quiet(assemble_program(task, candidate))
            return candidate
        except SyntaxError:
            candidate = candidate[:-1]
        except Exception:
            return ""
    return ""


# ---------------- Model Evaluation Runner ----------------


class UniversalBenchmarkRunner:
    def __init__(self, tsc_path: Path = DEFAULT_TSC, timeout: float = 10.0,
                 device: Optional[str] = None):
        self.tsc_path = tsc_path
        self.timeout = timeout
        self.device = resolve_device(device)
        self.toolchains = backend_status(self.tsc_path)

    def evaluate_checkpoint(self, checkpoint_path: str | Path, tasks: List[UniversalTask],
                            samples_per_task: int = 5, max_new_tokens: int = 128,
                            temperature: float = 0.4, top_p: float = 0.95,
                            repetition_penalty: float = 1.12, seed: int = 1337,
                            prompt_style: str = "auto") -> Dict[str, Any]:
        """Run the universal benchmark suite on a loaded checkpoint.

        prompt_style: "completion" feeds the task prefix raw; "instruct" wraps it
        in the Alpaca template the SFT models were trained on; "auto" picks by
        checkpoint name.
        """
        torch.manual_seed(seed)
        path = Path(checkpoint_path)
        model, block_size, tokenizer = load_any(path, self.device)

        style = infer_prompt_style(path.name) if prompt_style == "auto" else prompt_style
        if style not in PROMPT_STYLES:
            raise ValueError(f"Unknown prompt_style {style!r}; expected one of {PROMPT_STYLES}")
        if style == "instruct":
            print(f"    prompt style: instruct (Alpaca template)", flush=True)

        # Track-specific framing path
        framing_paths = {
            "python": "src/module.py",
            "javascript": "src/script.js",
            "typescript": "src/index.ts",
            "json": "config.json",
            "prose": "docs/article.txt",
        }

        # Track aggregates
        track_results: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_tasks": 0,
            "generations": 0,
            "syntax_ok": 0,
            "solved": 0,
            "schema_ok": 0,
            "collapses": 0,
            "empty": 0,
            "pass@1_list": [],
            "distinct_1_list": [],
            "distinct_2_list": [],
            "sentence_closed": 0,
        })

        per_task_records = []

        for task in tasks:
            track = task.track
            t_res = track_results[track]
            t_res["total_tasks"] += 1

            # Prepare prompt framing for Brittain3 if applicable
            subpath = framing_paths.get(track, "file.txt")
            prefix = document_prefix(tokenizer, "universal_bench", subpath)
            if style == "instruct":
                instruction, inp = build_instruction(task)
                full_prompt = prefix + format_prompt(instruction, inp)
            else:
                full_prompt = prefix + task.prompt
            prompt_ids = tokenizer.encode(full_prompt)

            context = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            if context.size(1) >= block_size:
                continue

            correct_count = 0
            for _ in range(samples_per_task):
                with torch.no_grad():
                    out = generate(
                        model, context, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                new_ids = strip_specials(tokenizer, out[0, context.size(1):].tolist())
                t_res["generations"] += 1

                if repetition_collapse(new_ids):
                    t_res["collapses"] += 1

                raw_body = tokenizer.decode(new_ids)
                if not raw_body.strip():
                    t_res["empty"] += 1
                    continue

                if track == "python":
                    if style == "instruct":
                        cleaned = strip_instruct_response(raw_body)
                        body = normalize_indentation(
                            truncate_python_definition(cleaned, task.entry_point)
                        )
                    else:
                        body = normalize_indentation(truncate_python_completion(raw_body))
                    body = longest_parseable_python(task, body)
                    if not body.strip():
                        t_res["empty"] += 1
                        continue
                    program = assemble_program(task, body)
                    try:
                        parse_quiet(program)
                        t_res["syntax_ok"] += 1
                        if task.tests:
                            checked = verify_program(
                                "python", program, "\n".join(task.tests),
                                timeout=self.timeout, tsc=self.tsc_path
                            )
                            if checked.ok:
                                correct_count += 1
                    except Exception:
                        pass

                elif track == "javascript":
                    if style == "instruct":
                        body = strip_instruct_response(raw_body)
                        if defines_entry_point(track, body, task.entry_point):
                            body = truncate_braced_definition(body)
                        else:
                            body = truncate_braced_completion(body)
                        if not body.strip():
                            t_res["empty"] += 1
                            continue
                    else:
                        body = truncate_braced_completion(raw_body)
                    program = assemble_program(task, body)
                    syntax = verify_syntax("javascript", program, timeout=self.timeout)
                    if syntax.ok:
                        t_res["syntax_ok"] += 1
                        if task.tests:
                            checked = verify_program(
                                "javascript", program, "\n".join(task.tests),
                                timeout=self.timeout, tsc=self.tsc_path
                            )
                            if checked.ok:
                                correct_count += 1

                elif track == "typescript":
                    if style == "instruct":
                        body = strip_instruct_response(raw_body)
                        if defines_entry_point(track, body, task.entry_point):
                            body = truncate_braced_definition(body)
                        else:
                            body = truncate_braced_completion(body)
                        if not body.strip():
                            t_res["empty"] += 1
                            continue
                    else:
                        body = truncate_braced_completion(raw_body)
                    program = assemble_program(task, body)
                    syntax = verify_syntax("typescript", program,
                                           timeout=self.timeout, tsc=self.tsc_path)
                    if syntax.ok:
                        t_res["syntax_ok"] += 1
                        if task.tests:
                            checked = verify_program(
                                "typescript", program, "\n".join(task.tests),
                                timeout=self.timeout, tsc=self.tsc_path
                            )
                            if checked.ok:
                                correct_count += 1

                elif track == "json":
                    if style == "instruct":
                        # An instruct model returns the whole document, not a tail.
                        cleaned = strip_instruct_response(raw_body).strip()
                        jres = verify_json_completion("", cleaned, task.expected_keys, task.schema_type)
                    else:
                        jres = verify_json_completion(task.prompt, raw_body, task.expected_keys, task.schema_type)
                    if jres.is_valid_json:
                        t_res["syntax_ok"] += 1
                    if jres.schema_valid:
                        t_res["schema_ok"] += 1
                        correct_count += 1

                elif track == "prose":
                    prose_body = strip_instruct_response(raw_body) if style == "instruct" else raw_body
                    pmetrics = evaluate_prose_completion(prose_body)
                    t_res["distinct_1_list"].append(pmetrics.distinct_1)
                    t_res["distinct_2_list"].append(pmetrics.distinct_2)
                    if pmetrics.is_repetition_collapse:
                        t_res["collapses"] += 1
                    if pmetrics.sentence_closed:
                        t_res["sentence_closed"] += 1
                    t_res["syntax_ok"] += 1  # Prose has no syntax failure
                    correct_count += int(pmetrics.sentence_closed and not pmetrics.is_repetition_collapse)

            p1 = pass_at_k(samples_per_task, correct_count, 1)
            t_res["pass@1_list"].append(p1)
            if correct_count > 0:
                t_res["solved"] += 1
            per_task_records.append({
                "id": task.id,
                "track": track,
                "correct": correct_count,
                "n": samples_per_task,
                "pass@1": p1,
            })

        # Calculate BPB on static fixtures
        code_ref_path = BENCHMARK_PROMPTS_DIR / "code.py"
        prose_ref_path = BENCHMARK_PROMPTS_DIR / "prose.txt"
        bpb_code = float("nan")
        bpb_prose = float("nan")
        try:
            if code_ref_path.exists():
                code_text = code_ref_path.read_text(encoding="utf-8", errors="ignore")[:50_000]
                bpb_code = calculate_bpb(model, tokenizer, block_size, code_text,
                                        document_prefix(tokenizer, "bench", "code.py"), self.device)
            if prose_ref_path.exists():
                prose_text = prose_ref_path.read_text(encoding="utf-8", errors="ignore")[:50_000]
                bpb_prose = calculate_bpb(model, tokenizer, block_size, prose_text,
                                         document_prefix(tokenizer, "bench", "prose.txt"), self.device)
        except Exception:
            pass

        # Build clean summary
        summary: Dict[str, Any] = {
            "checkpoint": path.name,
            "params": model.num_params() if hasattr(model, "num_params") else None,
            "block_size": block_size,
            "tokenizer": getattr(tokenizer, "name", "unknown"),
            "prompt_style": style,
            "harness_version": HARNESS_VERSION,
            "bpb_code": bpb_code,
            "bpb_prose": bpb_prose,
            "tracks": {},
            "per_task": per_task_records,
        }

        for track, r in track_results.items():
            gens = max(1, r["generations"])
            tasks_cnt = max(1, r["total_tasks"])
            summary["tracks"][track] = {
                "tasks": r["total_tasks"],
                "syntax_validity": r["syntax_ok"] / gens,
                "solved_tasks": r["solved"],
                "repetition_collapse": r["collapses"] / gens,
                "empty_rate": r["empty"] / gens,
            }
            rate = sum(r["pass@1_list"]) / tasks_cnt if r["pass@1_list"] else 0.0
            if track == "prose":
                # Prose has no execution oracle. What was reported as pass@1 was
                # "closed a sentence and did not collapse" — a restatement of two
                # metrics already in this dict, not a capability score. Naming it
                # pass@1 invited exactly the comparison it cannot support.
                summary["tracks"][track]["clean_completion_rate"] = rate
            else:
                summary["tracks"][track]["pass@1"] = rate
            if track == "json":
                summary["tracks"][track]["schema_validity"] = r["schema_ok"] / gens
            if track == "prose":
                summary["tracks"][track]["distinct_1"] = (
                    sum(r["distinct_1_list"]) / len(r["distinct_1_list"]) if r["distinct_1_list"] else 0.0
                )
                summary["tracks"][track]["distinct_2"] = (
                    sum(r["distinct_2_list"]) / len(r["distinct_2_list"]) if r["distinct_2_list"] else 0.0
                )
                summary["tracks"][track]["sentence_closed_rate"] = r["sentence_closed"] / gens

        del model
        if self.device.type == "mps":
            torch.mps.empty_cache()

        return summary


def validate_universal_suite(timeout: float = 10.0, tsc: str | Path = DEFAULT_TSC) -> Tuple[bool, List[str]]:
    """Validate all reference completions against their corresponding tasks."""
    tasks = load_universal_tasks()
    references = load_universal_references()
    failures: List[str] = []

    missing = [task.id for task in tasks if task.id not in references]
    if missing:
        failures.append(f"Missing reference solutions for {len(missing)} tasks: {missing}")
        return False, failures

    for task in tasks:
        completion = references[task.id]
        if task.track == "python":
            full = task.prompt + completion
            try:
                ast.parse(full)
            except Exception as e:
                failures.append(f"{task.id}: Python syntax error: {e}")
                continue
            if task.tests:
                res = verify_program("python", full, "\n".join(task.tests), timeout=timeout, tsc=tsc)
                if not res.ok:
                    failures.append(f"{task.id}: Python test failed ({res.phase}): {res.detail}")

        elif task.track == "javascript":
            full = task.prompt + completion
            syn = verify_syntax("javascript", full, timeout=timeout)
            if not syn.ok:
                failures.append(f"{task.id}: JS syntax error: {syn.detail}")
                continue
            if task.tests:
                res = verify_program("javascript", full, "\n".join(task.tests), timeout=timeout, tsc=tsc)
                if not res.ok:
                    failures.append(f"{task.id}: JS test failed ({res.phase}): {res.detail}")

        elif task.track == "typescript":
            full = task.prompt + completion
            syn = verify_syntax("typescript", full, timeout=timeout, tsc=tsc)
            if not syn.ok:
                failures.append(f"{task.id}: TS compiler error: {syn.detail}")
                continue
            if task.tests:
                res = verify_program("typescript", full, "\n".join(task.tests), timeout=timeout, tsc=tsc)
                if not res.ok:
                    failures.append(f"{task.id}: TS test failed ({res.phase}): {res.detail}")

        elif task.track == "json":
            jres = verify_json_completion(task.prompt, completion, task.expected_keys, task.schema_type)
            if not jres.is_valid_json:
                failures.append(f"{task.id}: JSON invalid: {jres.error}")
            elif not jres.schema_valid:
                failures.append(f"{task.id}: JSON schema violation: {jres.error}")

        elif task.track == "prose":
            pmetrics = evaluate_prose_completion(completion)
            if not pmetrics.sentence_closed:
                failures.append(f"{task.id}: Prose reference does not end in terminal sentence punctuation.")

    return len(failures) == 0, failures
