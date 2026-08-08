"""Document-aware Brittain3 data preparation primitives."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Document:
    repository: str
    path: str
    text: str
    source: str
    is_code: bool


@dataclass(frozen=True)
class FIMSettings:
    rate: float = 0.40
    psm_rate: float = 0.50
    line_rate: float = 0.50
    block_rate: float = 0.25

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.line_rate + self.block_rate > 1.0:
            raise ValueError("line_rate + block_rate cannot exceed 1")


@dataclass
class EncodedSegment:
    ids: list[int]
    repository: str
    path: str
    source: str
    is_fim: bool
    fim_order: str | None = None
    hole_kind: str | None = None


def repository_in_validation(repository: str, fraction: float, seed: int) -> bool:
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must be in (0, 1)")
    digest = hashlib.sha256(f"{seed}:{repository}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < fraction


def split_by_repository(
    documents: Iterable[Document], validation_fraction: float, seed: int
) -> tuple[list[Document], list[Document]]:
    train, validation = [], []
    assignments: dict[str, bool] = {}
    for document in documents:
        is_validation = assignments.setdefault(
            document.repository,
            repository_in_validation(document.repository, validation_fraction, seed),
        )
        (validation if is_validation else train).append(document)
    train_repositories = {document.repository for document in train}
    validation_repositories = {document.repository for document in validation}
    if train_repositories & validation_repositories:
        raise AssertionError("a repository appeared in both splits")
    return train, validation


def _line_hole(text: str, rng: random.Random) -> tuple[str, str, str] | None:
    lines = text.splitlines(keepends=True)
    if len(lines) < 3:
        return None
    start = rng.randrange(1, len(lines) - 1)
    max_lines = max(1, min(len(lines) - start - 1, max(1, len(lines) // 3)))
    count = rng.randint(1, max_lines)
    return "".join(lines[:start]), "".join(lines[start:start + count]), "".join(lines[start + count:])


def _block_hole(text: str, rng: random.Random) -> tuple[str, str, str] | None:
    lines = text.splitlines(keepends=True)
    starts = []
    pattern = re.compile(
        r"^(?P<indent>\s*)(?:(?:async\s+)?def\b|class\b|function\b|"
        r"(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)"
    )
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            starts.append((index, len(match.group("indent"))))
    if not starts:
        return None
    start, indentation = rng.choice(starts)
    declaration = lines[start].lstrip()
    python_block = bool(re.match(r"(?:async\s+)?def\b|class\b", declaration))
    end = len(lines)
    brace_depth = 0
    saw_brace = False
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped:
            continue
        current = len(lines[index]) - len(lines[index].lstrip())
        if python_block and current <= indentation:
            end = index
            break
        if not python_block:
            if not saw_brace:
                brace_depth += lines[start].count("{") - lines[start].count("}")
                saw_brace = "{" in lines[start]
            brace_depth += lines[index].count("{") - lines[index].count("}")
            saw_brace = saw_brace or "{" in lines[index]
            if saw_brace and brace_depth <= 0:
                end = index + 1
                break
            match = pattern.search(lines[index])
            if not saw_brace and match and len(match.group("indent")) <= indentation:
                end = index
                break
    if end <= start + 1:
        return None
    return "".join(lines[:start]), "".join(lines[start:end]), "".join(lines[end:])


def _random_hole(text: str, rng: random.Random) -> tuple[str, str, str] | None:
    if len(text) < 8:
        return None
    first, second = sorted(rng.sample(range(1, len(text)), 2))
    if not text[first:second].strip():
        return None
    return text[:first], text[first:second], text[second:]


def select_hole(
    text: str, settings: FIMSettings, rng: random.Random
) -> tuple[str, str, str, str] | None:
    draw = rng.random()
    if draw < settings.line_rate:
        selected = _line_hole(text, rng)
        kind = "line"
    elif draw < settings.line_rate + settings.block_rate:
        selected = _block_hole(text, rng)
        kind = "block"
    else:
        selected = _random_hole(text, rng)
        kind = "random"
    if selected is None:
        selected = _random_hole(text, rng)
        kind = "random"
    return (*selected, kind) if selected is not None else None


def _special(tokenizer, token: str) -> int:
    try:
        return tokenizer.special_ids[token]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"tokenizer does not define {token}") from exc


def _bounded_text(tokenizer, text: str, budget: int, rng: random.Random) -> str:
    ids = tokenizer.encode(text)
    if len(ids) <= budget:
        return text
    start = rng.randrange(0, len(ids) - budget + 1)
    return tokenizer.decode(ids[start:start + budget], skip_special_tokens=False)


def encode_document(
    document: Document,
    tokenizer,
    block_size: int,
    settings: FIMSettings,
    rng: random.Random,
) -> EncodedSegment:
    """Encode one indivisible document or FIM segment for later packing."""
    if block_size < 32:
        raise ValueError("block_size must be at least 32")
    repo_ids = tokenizer.encode(document.repository)
    path_ids = tokenizer.encode(document.path)
    fixed = [
        _special(tokenizer, "<|repo_start|>"),
        *repo_ids,
        _special(tokenizer, "<|file_start|>"),
        *path_ids,
    ]
    tail = [
        _special(tokenizer, "<|file_end|>"),
        _special(tokenizer, "<|repo_end|>"),
        _special(tokenizer, "<|endoftext|>"),
    ]
    # Reserve three FIM sentinels and one token of split retokenization slack.
    content_budget = block_size + 1 - len(fixed) - len(tail) - 4
    if content_budget < 8:
        raise ValueError("repository and path metadata leave no content space")
    text = _bounded_text(tokenizer, document.text, content_budget, rng)

    use_fim = document.is_code and rng.random() < settings.rate
    hole_kind = order = None
    content_ids: list[int]
    if use_fim:
        selected = select_hole(text, settings, rng)
        if selected is not None:
            prefix, middle, suffix, hole_kind = selected
            prefix_ids = tokenizer.encode(prefix)
            middle_ids = tokenizer.encode(middle)
            suffix_ids = tokenizer.encode(suffix)
            pre = _special(tokenizer, "<|fim_prefix|>")
            suf = _special(tokenizer, "<|fim_suffix|>")
            mid = _special(tokenizer, "<|fim_middle|>")
            if rng.random() < settings.psm_rate:
                order = "PSM"
                content_ids = [pre, *prefix_ids, suf, *suffix_ids, mid, *middle_ids]
            else:
                order = "SPM"
                content_ids = [pre, suf, *suffix_ids, mid, *prefix_ids, *middle_ids]
            if len(content_ids) > content_budget + 3:
                # Retokenization at the two cuts can add a few tokens. Shrink once
                # and retry as a random span rather than split the FIM structure.
                text = _bounded_text(tokenizer, text, max(8, content_budget - 8), rng)
                selected = _random_hole(text, rng)
                if selected is None:
                    use_fim = False
                else:
                    prefix, middle, suffix = selected
                    prefix_ids, middle_ids, suffix_ids = map(
                        tokenizer.encode, (prefix, middle, suffix)
                    )
                    content_ids = [pre, *prefix_ids, suf, *suffix_ids, mid, *middle_ids]
                    order, hole_kind = "PSM", "random"
        else:
            use_fim = False
    if not use_fim:
        content_ids = tokenizer.encode(text)
        order = hole_kind = None

    ids = [*fixed, *content_ids, *tail]
    if len(ids) > block_size + 1:
        raise ValueError(
            f"encoded segment has {len(ids)} tokens but limit is {block_size + 1}"
        )
    return EncodedSegment(
        ids=ids,
        repository=document.repository,
        path=document.path,
        source=document.source,
        is_fim=bool(use_fim),
        fim_order=order,
        hole_kind=hole_kind,
    )


def pack_segments(
    segments: Sequence[EncodedSegment], block_size: int, pad_id: int
) -> tuple[np.ndarray, np.ndarray, list[list[dict]]]:
    """Pack complete segments without splitting one across training rows."""
    rows: list[list[int]] = []
    spans: list[list[dict]] = []
    current: list[int] = []
    current_spans: list[dict] = []
    limit = block_size + 1
    for segment in segments:
        if len(segment.ids) > limit:
            raise ValueError("a segment exceeds the packer row limit")
        if current and len(current) + len(segment.ids) > limit:
            rows.append(current)
            spans.append(current_spans)
            current, current_spans = [], []
        start = len(current)
        current.extend(segment.ids)
        current_spans.append({
            "start": start,
            "end": len(current),
            "repository": segment.repository,
            "path": segment.path,
            "source": segment.source,
            "is_fim": segment.is_fim,
            "fim_order": segment.fim_order,
            "hole_kind": segment.hole_kind,
        })
    if current:
        rows.append(current)
        spans.append(current_spans)
    input_rows, label_rows = [], []
    for row in rows:
        padding = limit - len(row)
        padded = row + [pad_id] * padding
        input_rows.append(padded[:-1])
        labels = padded[1:]
        graded = max(0, len(row) - 1)
        labels[graded:] = [-100] * (block_size - graded)
        label_rows.append(labels)
    return (
        np.asarray(input_rows, dtype=np.uint16),
        np.asarray(label_rows, dtype=np.int32),
        spans,
    )


def token_controlled_mix(
    groups: dict[str, Sequence[Document]],
    weights: dict[str, float],
    token_length: Callable[[Document], int],
    seed: int,
) -> list[Document]:
    """Interleave finite source groups by their running token share."""
    if set(groups) != set(weights):
        raise ValueError("groups and weights must have the same source names")
    if any(value <= 0 for value in weights.values()):
        raise ValueError("all mixture weights must be positive")
    total_weight = sum(weights.values())
    normalized = {name: value / total_weight for name, value in weights.items()}
    rng = random.Random(seed)
    queues = {name: list(values) for name, values in groups.items()}
    for queue in queues.values():
        rng.shuffle(queue)
    emitted = {name: 0 for name in groups}
    result = []
    while any(queues.values()):
        available = [name for name, queue in queues.items() if queue]
        name = min(available, key=lambda item: emitted[item] / normalized[item])
        document = queues[name].pop()
        result.append(document)
        emitted[name] += max(1, token_length(document))
    return result


def write_dataset(
    output: str | Path,
    inputs: np.ndarray,
    labels: np.ndarray,
    metadata: dict,
) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, input_ids=inputs, labels=labels)
    destination.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
