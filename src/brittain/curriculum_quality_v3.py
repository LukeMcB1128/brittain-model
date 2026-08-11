"""Contamination and duplicate guards for generated Brittain3 curriculum data."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .paths import PROJECT_ROOT


WORD = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
TOKEN = re.compile(
    r"(?P<block_comment>/\*.*?\*/)"
    r"|(?P<line_comment>//[^\n]*)"
    r"|(?P<string>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
    r"|(?P<number>\b(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?)\b)"
    r"|(?P<identifier>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<operator>[^\s])",
    re.DOTALL,
)

KEYWORDS = {
    "and", "as", "assert", "async", "await", "bool", "break", "case", "catch",
    "char", "class", "const", "continue", "def", "default", "defer", "delete", "do",
    "double", "else", "enum", "export", "extends", "false", "finally", "float", "fn",
    "for", "from", "func", "function", "go", "if", "impl", "import", "in", "include",
    "int", "interface", "let", "long", "map", "match", "mod", "mut", "new", "nil",
    "none", "not", "null", "or", "package", "pass", "private", "pub", "public",
    "raise", "range", "return", "short", "signed", "sizeof", "static", "str", "struct",
    "super", "switch", "this", "throw", "true", "try", "type", "typedef", "uint",
    "unsigned", "use", "var", "vec", "void", "while", "with", "yield",
}


# These rules describe behavior held out by novice-v1. Each tuple contains
# fragments that must all occur in a generated task brief. The rules are
# intentionally language-independent: translating a held-out Python task to
# Rust is still evaluation contamination.
HELD_OUT_BEHAVIOR_RULES: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    ("clamp_value", (
        re.compile(r"\bclamp\w*\b"),
        re.compile(r"\b(?:minimum|maximum|min|max|lower|upper|bound|range)\w*\b"),
    )),
    ("unique_preserve_order", (
        re.compile(r"\b(?:unique|duplicate|deduplicat)\w*\b"),
        re.compile(r"\b(?:preserv|first[- ]seen|original)\w*\b"),
        re.compile(r"\border\w*\b"),
    )),
    ("count_words", (
        re.compile(r"\bword\w*\b"),
        re.compile(r"\b(?:count|frequency|frequencies)\w*\b"),
        re.compile(r"\b(?:dict|dictionary|map|mapping|object)\w*\b"),
    )),
    ("extract_digits", (
        re.compile(
            r"\b(?:(?:extract|filter|keep)\w*(?:\W+\w+){0,5}\W+digits?"
            r"|(?:return|produce|output)\w*(?:\W+\w+){0,4}\W+only"
            r"(?:\W+\w+){0,3}\W+digits?)\b"
        ),
        re.compile(r"\b(?:string|text|character|input)\w*\b"),
    )),
    ("running_total", (
        re.compile(r"\b(?:running|cumulative)\w*\b"),
        re.compile(r"\b(?:sum|total)\w*\b"),
        re.compile(r"\b(?:list|array|sequence|value)\w*\b"),
    )),
    ("count_vowels", (
        re.compile(r"\bvowel\w*\b"),
        re.compile(r"\bcount\w*\b"),
    )),
    ("parse_boolean", (
        re.compile(r"\b(?:parse|convert)\w*\b"),
        re.compile(r"\b(?:bool|boolean|true|false)\w*\b"),
    )),
    ("factorial", (re.compile(r"\bfactorial\w*\b"),)),
)


def normalized_text(text: str) -> str:
    return " ".join(text.lower().split())


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_set(text: str) -> set[str]:
    return {word.lower() for word in WORD.findall(text) if len(word) > 2}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def structural_fingerprint(language: str, code: str) -> str:
    """Hash code shape after replacing literals and user-selected names."""
    if language == "python":
        code = re.sub(r"(?m)#.*$", "", code)
    identifiers: dict[str, str] = {}
    tokens = []
    for match in TOKEN.finditer(code):
        kind = match.lastgroup
        value = match.group()
        if kind in ("block_comment", "line_comment"):
            continue
        if kind == "string":
            tokens.append("<str>")
        elif kind == "number":
            tokens.append("<num>")
        elif kind == "identifier":
            lowered = value.lower()
            if lowered in KEYWORDS:
                tokens.append(lowered)
            else:
                identifiers.setdefault(value, f"id{len(identifiers)}")
                tokens.append(identifiers[value])
        else:
            tokens.append(value)
    return text_digest(language + "\n" + " ".join(tokens))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class EvaluationGuard:
    entry_points: frozenset[str]
    exact_hashes: frozenset[str]
    assertions: tuple[str, ...]
    descriptions: tuple[str, ...]
    task_words: tuple[set[str], ...]
    behavior_summaries: tuple[str, ...]

    @classmethod
    def novice_v1(cls) -> "EvaluationGuard":
        tasks_path = PROJECT_ROOT / "benchmarks" / "novice" / "tasks.jsonl"
        references_path = PROJECT_ROOT / "benchmarks" / "novice" / "reference.jsonl"
        tasks = _read_jsonl(tasks_path)
        references = {row["id"]: row["body"] for row in _read_jsonl(references_path)}
        hashes: set[str] = set()
        assertions: set[str] = set()
        descriptions: set[str] = set()
        task_words = []
        behavior_summaries = []
        for task in tasks:
            prompt = task["prompt"]
            body = references[task["id"]]
            for value in (prompt, body, prompt + body):
                hashes.add(text_digest(value))
                hashes.add(text_digest(normalized_text(value)))
            assertions.update(value.strip() for value in task["tests"] if len(value.strip()) >= 20)
            prose = []
            for line in prompt.splitlines():
                value = line.lstrip("#").strip()
                if len(value) >= 30:
                    descriptions.add(normalized_text(value))
                    prose.append(value)
                if value and "->" not in value and not value.startswith(("def ", "class ")):
                    behavior_summaries.append(value)
            task_words.append(word_set(" ".join(prose)))
        return cls(
            entry_points=frozenset(task["entry_point"] for task in tasks),
            exact_hashes=frozenset(hashes),
            assertions=tuple(sorted(assertions)),
            descriptions=tuple(sorted(descriptions)),
            task_words=tuple(task_words),
            behavior_summaries=tuple(dict.fromkeys(behavior_summaries)),
        )

    def reason(self, entry_point: str, texts: list[str], semantic_text: str) -> str | None:
        if entry_point.strip() in self.entry_points:
            return "evaluation_entry_point"
        combined = "\n".join(texts)
        normalized = normalized_text(combined)
        for value in texts + [combined]:
            if text_digest(value) in self.exact_hashes or text_digest(normalized_text(value)) in self.exact_hashes:
                return "evaluation_exact_text"
        if any(assertion in combined for assertion in self.assertions):
            return "evaluation_assertion"
        if any(description in normalized for description in self.descriptions):
            return "evaluation_description"
        semantic_normalized = normalized_text(semantic_text)
        for name, fragments in HELD_OUT_BEHAVIOR_RULES:
            if all(fragment.search(semantic_normalized) for fragment in fragments):
                return f"evaluation_behavior_{name}"
        candidate_words = word_set(semantic_text)
        if len(candidate_words) >= 6 and any(jaccard(candidate_words, held) >= 0.82 for held in self.task_words):
            return "evaluation_semantic_near_duplicate"
        return None


@dataclass
class DuplicateGuard:
    semantic_threshold: float = 0.82
    exact_hashes: set[str] = field(default_factory=set)
    structural_hashes: set[str] = field(default_factory=set)
    semantics: list[tuple[str, set[str]]] = field(default_factory=list)

    def reason(self, language: str, semantic_text: str, code: str) -> str | None:
        exact = text_digest(normalized_text(code))
        if exact in self.exact_hashes:
            return "duplicate_exact_code"
        structural = structural_fingerprint(language, code)
        if structural in self.structural_hashes:
            return "duplicate_structure"
        words = word_set(semantic_text)
        for previous_language, previous_words in self.semantics:
            if previous_language == language and jaccard(words, previous_words) >= self.semantic_threshold:
                return "duplicate_semantics"
        return None

    def add(self, language: str, semantic_text: str, code: str) -> dict[str, str]:
        exact = text_digest(normalized_text(code))
        structural = structural_fingerprint(language, code)
        self.exact_hashes.add(exact)
        self.structural_hashes.add(structural)
        self.semantics.append((language, word_set(semantic_text)))
        return {"exact": exact, "structural": structural}
