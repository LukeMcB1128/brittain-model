"""Brittain3 tokenizer interface and validation helpers."""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Iterable

from .paths import BRITTAIN3_TOKENIZER, PROJECT_ROOT
from .tokenizer import _byte_decoder


BRITTAIN3_SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|pad|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|repo_start|>",
    "<|repo_end|>",
    "<|file_start|>",
    "<|file_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|end_message|>",
)


class Brittain3Tokenizer:
    """Small common interface for the Brittain3 byte-level BPE."""

    name = "brittain3_bpe"

    def __init__(self, path: str | Path = BRITTAIN3_TOKENIZER):
        from tokenizers import Tokenizer

        candidate = Path(path).expanduser()
        if not candidate.is_absolute() and (PROJECT_ROOT / candidate).exists():
            candidate = PROJECT_ROOT / candidate
        elif not candidate.exists() and BRITTAIN3_TOKENIZER.exists():
            candidate = BRITTAIN3_TOKENIZER
        self.path = candidate.resolve()
        self._tok = Tokenizer.from_file(str(self.path))
        self.vocab_size = self._tok.get_vocab_size()
        self._byte_decoder = _byte_decoder()
        self.special_ids = {
            token: self._tok.token_to_id(token) for token in BRITTAIN3_SPECIAL_TOKENS
        }
        missing = [token for token, token_id in self.special_ids.items() if token_id is None]
        if missing:
            raise ValueError(f"Brittain3 tokenizer is missing special tokens: {missing}")
        self.eot = self.special_ids["<|endoftext|>"]
        self.pad = self.special_ids["<|pad|>"]
        self.fim_prefix = self.special_ids["<|fim_prefix|>"]
        self.fim_suffix = self.special_ids["<|fim_suffix|>"]
        self.fim_middle = self.special_ids["<|fim_middle|>"]
        self.repo_start = self.special_ids["<|repo_start|>"]
        self.repo_end = self.special_ids["<|repo_end|>"]
        self.file_start = self.special_ids["<|file_start|>"]
        self.file_end = self.special_ids["<|file_end|>"]

    @property
    def has_fim(self) -> bool:
        return True

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_bytes(self, token_id: int) -> bytes:
        token = self._tok.id_to_token(token_id)
        if token is None:
            return b""
        if token in self.special_ids:
            return token.encode("utf-8")
        try:
            return bytes(self._byte_decoder[character] for character in token)
        except KeyError:
            return self.decode([token_id]).encode("utf-8", errors="replace")


VALIDATION_SAMPLES = {
    "code": "def fibonacci(n: int) -> int:\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)\n",
    "english": "A coding assistant must inspect the project before it changes a file.\n",
    "indentation": "if ready:\n    for item in values:\n        print(item)\n",
    "identifiers": "parseHTTPResponse user_profile_id MAX_RETRY_COUNT __init__\n",
    "json": '{"name":"find_symbol","arguments":{"name":"parseConfig"}}',
    "tool_call": '<|assistant|><|tool_call|>{"name":"read_file","arguments":{"path":"src/app.js"}}<|end_message|>',
    "unicode": "naïve café — 東京 — 🧪\n",
}


def validate_tokenizer(
    tokenizer: Brittain3Tokenizer,
    *,
    reference=None,
) -> dict:
    """Validate required behavior and return comparable efficiency metrics."""
    if tokenizer.vocab_size != 24_576:
        raise ValueError(f"Brittain3 tokenizer vocab must be 24576, got {tokenizer.vocab_size}")
    for token, token_id in tokenizer.special_ids.items():
        encoded = tokenizer.encode(token)
        if encoded != [token_id]:
            raise ValueError(f"special token is not atomic: {token!r} -> {encoded}")

    metrics = {}
    for name, text in VALIDATION_SAMPLES.items():
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != text:
            raise ValueError(f"byte round trip failed for {name}: {decoded!r} != {text!r}")
        row = {
            "bytes": len(text.encode("utf-8")),
            "tokens": len(ids),
        }
        if reference is not None:
            reference_tokens = len(reference.encode(text))
            row["reference_tokens"] = reference_tokens
            row["token_change_fraction"] = (
                len(ids) - reference_tokens
            ) / max(1, reference_tokens)
        metrics[name] = row
    return {
        "name": tokenizer.name,
        "path": str(tokenizer.path),
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": tokenizer.special_ids,
        "samples": metrics,
    }


def save_validation_report(report: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def evaluate_tokenizer_corpus(
    tokenizer: Brittain3Tokenizer,
    paths: Iterable[str | Path],
    *,
    reference=None,
    maximum_bytes: int = 10_000_000,
) -> dict:
    """Measure token efficiency on provenance-rich held-out JSONL rows."""
    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    totals = defaultdict(lambda: {"documents": 0, "bytes": 0, "tokens": 0, "reference_tokens": 0})
    code_languages = defaultdict(
        lambda: {"documents": 0, "bytes": 0, "tokens": 0, "reference_tokens": 0}
    )
    consumed = 0
    for value in paths:
        path = Path(value)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if consumed >= maximum_bytes:
                    break
                try:
                    row = json.loads(line)
                    text = row["text"]
                    category = str(row.get("category", "unknown"))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"invalid evaluation row at {path}:{line_number}") from exc
                if not isinstance(text, str) or not text:
                    continue
                encoded = tokenizer.encode(text)
                if tokenizer.decode(encoded, skip_special_tokens=False) != text:
                    raise ValueError(f"byte round trip failed at {path}:{line_number}")
                size = len(text.encode("utf-8"))
                local = totals[category]
                local["documents"] += 1
                local["bytes"] += size
                local["tokens"] += len(encoded)
                reference_tokens = len(reference.encode(text)) if reference is not None else 0
                local["reference_tokens"] += reference_tokens
                if category == "code":
                    language = str(row.get("language", "other"))
                    language_row = code_languages[language]
                    language_row["documents"] += 1
                    language_row["bytes"] += size
                    language_row["tokens"] += len(encoded)
                    language_row["reference_tokens"] += reference_tokens
                consumed += size
            if consumed >= maximum_bytes:
                break
    def render(rows):
        result = {}
        for name, row in sorted(rows.items()):
            metrics = dict(row)
            metrics["bytes_per_token"] = row["bytes"] / max(1, row["tokens"])
            if reference is not None:
                metrics["token_change_fraction"] = (
                    row["tokens"] - row["reference_tokens"]
                ) / max(1, row["reference_tokens"])
            result[name] = metrics
        return result

    rendered = render(totals)
    rendered_languages = render(code_languages)
    return {
        "maximum_bytes": maximum_bytes,
        "evaluated_bytes": consumed,
        "categories": rendered,
        "code_languages": rendered_languages,
    }
