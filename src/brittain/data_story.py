"""Window, tag, and pack narrative prose for brittain-shakespeare.

``data_v3.encode_document`` truncates a document to a single block, which is
right for source files and wrong for novels: it would keep the first 1,024 tokens
of Bleak House and discard the rest. This module replaces that step with
windowing at chapter and paragraph boundaries.

It also replaces ``data_v3.pack_segments``. That packer derives labels purely by
shifting, with no way to exclude a span from the loss, and excluding the tag
block from the loss on most examples is a deliberate part of the design.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .story_tagger import length_from_tokens
from .tags import TagPolicy, apply_policy, render

# Headings that mark a real division. Matched on a line of their own so a
# mention of "chapter" inside prose does not split a story.
_HEADING = re.compile(
    r"^[ \t]*(?:CHAPTER|Chapter|CHAP\.|BOOK|Book|PART|Part|ACT|Act|SCENE|Scene)"
    r"[ \t]+[IVXLCDM\d][IVXLCDM\d\w.\-—:' ]{0,80}$",
    re.MULTILINE,
)
_PARAGRAPH_SPLIT = re.compile(r"\n[ \t]*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[ \t]+")


@dataclass(frozen=True)
class StorySettings:
    """Windowing and packing sizes, mirrored from the corpus config."""

    block_size: int = 1024
    target_tokens: int = 1400
    minimum_tokens: int = 200
    maximum_tokens: int = 3800
    prefer_chapter_boundaries: bool = True
    policy: TagPolicy = field(default_factory=TagPolicy)

    def __post_init__(self) -> None:
        if self.minimum_tokens < 32:
            raise ValueError("minimum_tokens must be at least 32")
        if self.target_tokens < self.minimum_tokens:
            raise ValueError("target_tokens must be at least minimum_tokens")
        if self.maximum_tokens < self.target_tokens:
            raise ValueError("maximum_tokens must be at least target_tokens")


@dataclass
class EncodedStory:
    """One packable story segment and its per-token loss mask."""

    ids: list[int]
    # supervised[i] is False where token i must not contribute to the loss.
    supervised: list[bool]
    repository: str
    path: str
    source: str
    tags: dict[str, str]
    tags_masked: bool
    tags_reversed: bool

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.supervised):
            raise ValueError("ids and supervised must be the same length")


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #

def split_chapters(text: str) -> list[str]:
    """Split on chapter-like headings, keeping each heading with its body."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [text]
    chunks = []
    # Anything before the first heading is front matter or a preface body.
    if matches[0].start() > 0:
        head = text[: matches[0].start()].strip()
        if head:
            chunks.append(head)
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = text[match.start(): end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _split_long_paragraph(paragraph: str, tokenizer, limit: int) -> list[str]:
    """Break one oversized paragraph at sentence boundaries, never mid-sentence."""
    sentences = _SENTENCE_SPLIT.split(paragraph)
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*current, sentence])
        if current and len(tokenizer.encode(candidate)) > limit:
            pieces.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _flush(buffer: list[str], tokenizer, settings: StorySettings,
           windows: list[str]) -> list[str]:
    """Emit the buffer as one window, returning whatever did not fit.

    Token counts are accumulated per paragraph for speed, but joining paragraphs
    costs a little more than the sum of their parts because the tokenizer merges
    across the join. The approximate sum is fine for deciding when to flush; it
    is not sound for a hard bound, so the limit is checked exactly here.
    """
    leftover: list[str] = []
    while buffer:
        text = "\n\n".join(buffer)
        if len(tokenizer.encode(text)) <= settings.maximum_tokens:
            windows.append(text)
            break
        if len(buffer) == 1:
            windows.extend(
                _split_long_paragraph(buffer[0], tokenizer, settings.maximum_tokens)
            )
            break
        leftover.insert(0, buffer.pop())
    return leftover


def window_text(text: str, tokenizer, settings: StorySettings) -> list[str]:
    """Cut a book into story-sized windows.

    Chapter boundaries are preferred, paragraph boundaries are the fallback, and
    a sentence boundary is the last resort. A window is never cut mid-sentence.
    """
    sources = split_chapters(text) if settings.prefer_chapter_boundaries else [text]

    windows: list[str] = []
    for chunk in sources:
        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(chunk) if p.strip()]
        if not paragraphs:
            continue
        buffer: list[str] = []
        counts: list[int] = []

        def buffered() -> int:
            return sum(counts)

        for paragraph in paragraphs:
            count = len(tokenizer.encode(paragraph))
            if count > settings.maximum_tokens:
                if buffer:
                    buffer = _flush(buffer, tokenizer, settings, windows)
                    counts = [len(tokenizer.encode(item)) for item in buffer]
                windows.extend(
                    _split_long_paragraph(paragraph, tokenizer, settings.maximum_tokens)
                )
                continue
            if buffer and buffered() + count > settings.maximum_tokens:
                buffer = _flush(buffer, tokenizer, settings, windows)
                counts = [len(tokenizer.encode(item)) for item in buffer]
            buffer.append(paragraph)
            counts.append(count)
            if buffered() >= settings.target_tokens:
                buffer = _flush(buffer, tokenizer, settings, windows)
                counts = [len(tokenizer.encode(item)) for item in buffer]
        while buffer:
            buffer = _flush(buffer, tokenizer, settings, windows)

    # A trailing scrap is merged backwards rather than dropped, so the last
    # paragraphs of a chapter are not silently lost.
    merged: list[str] = []
    for window in windows:
        if (
            merged
            and len(tokenizer.encode(window)) < settings.minimum_tokens
            and len(tokenizer.encode(merged[-1] + "\n\n" + window))
            <= settings.maximum_tokens
        ):
            merged[-1] = merged[-1] + "\n\n" + window
        else:
            merged.append(window)
    return [
        window
        for window in merged
        if len(tokenizer.encode(window)) >= settings.minimum_tokens
    ]


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

def _special(tokenizer, name: str) -> int:
    token_id = tokenizer.special_ids.get(name)
    if token_id is None:
        raise ValueError(f"tokenizer is missing the {name} token")
    return token_id


def encode_story(
    window: str,
    tags: dict[str, str],
    tokenizer,
    settings: StorySettings,
    rng: random.Random,
    *,
    repository: str = "",
    path: str = "",
    source: str = "",
) -> EncodedStory:
    """Encode one window with its tag block, applying the randomization policy.

    ``Length`` is recomputed here rather than taken from the caller. It describes
    the window, and the window is only known after windowing, so a value carried
    over from the whole book would be wrong.
    """
    text_ids = tokenizer.encode(window)
    tags = dict(tags)
    tags["Length"] = length_from_tokens(len(text_ids))

    kept, order, masked, reversed_block = apply_policy(tags, settings.policy, rng)

    story_start = _special(tokenizer, "<|story_start|>")
    story_end = _special(tokenizer, "<|story_end|>")
    eot = _special(tokenizer, "<|endoftext|>")

    block_ids: list[int] = []
    if kept:
        block_ids = [
            _special(tokenizer, "<|tags|>"),
            *tokenizer.encode(render(kept, order=order)),
            _special(tokenizer, "<|end_tags|>"),
        ]

    # Trim the story, never the tag block: a truncated tag block would be a
    # malformed condition, while a shorter story is merely a shorter story.
    budget = settings.block_size + 1 - len(block_ids) - 3
    if budget < 16:
        raise ValueError("tag block leaves no room for a story")
    if len(text_ids) > budget:
        text_ids = text_ids[:budget]

    if reversed_block and block_ids:
        ids = [story_start, *text_ids, *block_ids, story_end, eot]
        block_at = 1 + len(text_ids)
    else:
        ids = [story_start, *block_ids, *text_ids, story_end, eot]
        block_at = 1

    supervised = [True] * len(ids)
    if masked and block_ids:
        # Predicting the tags from nothing is a high-entropy task that wastes
        # capacity at this scale. The block is still read as context; it just
        # does not have to be generated. The unmasked share keeps reverse
        # tagging and tag completion learnable.
        for index in range(block_at, block_at + len(block_ids)):
            supervised[index] = False

    return EncodedStory(
        ids=ids,
        supervised=supervised,
        repository=repository,
        path=path,
        source=source,
        tags=kept,
        tags_masked=bool(masked and block_ids),
        tags_reversed=bool(reversed_block and block_ids),
    )


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #

def pack_story_segments(
    segments: Sequence[EncodedStory], block_size: int, pad_id: int
) -> tuple[np.ndarray, np.ndarray, list[list[dict]]]:
    """Pack whole stories into rows, honouring each segment's loss mask.

    A story is never split across two rows. Padding and masked spans both become
    ``-100`` in the labels.
    """
    limit = block_size + 1
    rows: list[list[int]] = []
    row_masks: list[list[bool]] = []
    spans: list[list[dict]] = []
    current_ids: list[int] = []
    current_mask: list[bool] = []
    current_spans: list[dict] = []

    for segment in segments:
        if len(segment.ids) > limit:
            raise ValueError("a story segment exceeds the packer row limit")
        if current_ids and len(current_ids) + len(segment.ids) > limit:
            rows.append(current_ids)
            row_masks.append(current_mask)
            spans.append(current_spans)
            current_ids, current_mask, current_spans = [], [], []
        start = len(current_ids)
        current_ids.extend(segment.ids)
        current_mask.extend(segment.supervised)
        current_spans.append({
            "start": start,
            "end": len(current_ids),
            "repository": segment.repository,
            "path": segment.path,
            "source": segment.source,
            "tags": segment.tags,
            "tags_masked": segment.tags_masked,
            "tags_reversed": segment.tags_reversed,
        })
    if current_ids:
        rows.append(current_ids)
        row_masks.append(current_mask)
        spans.append(current_spans)

    input_rows, label_rows = [], []
    for ids, mask in zip(rows, row_masks):
        padding = limit - len(ids)
        padded_ids = ids + [pad_id] * padding
        padded_mask = mask + [False] * padding
        input_rows.append(padded_ids[:-1])
        label_rows.append([
            padded_ids[index + 1] if padded_mask[index + 1] else -100
            for index in range(block_size)
        ])
    return (
        np.asarray(input_rows, dtype=np.uint16),
        np.asarray(label_rows, dtype=np.int32),
        spans,
    )
