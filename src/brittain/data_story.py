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
from array import array
from dataclasses import dataclass, field
from pathlib import Path
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
# Labels carry token ids (vocab 8,192) and the ignore sentinel, both of which fit
# int16. At 4K that halves the largest file the pipeline writes.
LABEL_DTYPE = np.int16
IGNORE_INDEX = -100

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
    """One packable story segment and its per-token loss mask.

    Preparation holds every segment of the corpus in memory before packing, so
    the storage type is not a detail. A Python list costs about 44 bytes per
    token once the int objects are counted, which is fine for a 220M-token pilot
    and fatal at 1.7B: roughly 75GB. Held as a typed array and a bytearray the
    same corpus is about 5GB.

    Callers may pass ordinary lists; they are converted on construction.
    """

    ids: array
    # supervised[i] is 0 where token i must not contribute to the loss.
    supervised: bytearray
    repository: str
    path: str
    source: str
    tags: dict[str, str]
    tags_masked: bool
    tags_reversed: bool
    continues: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.ids, array):
            self.ids = array("H", self.ids)
        if not isinstance(self.supervised, (bytearray, bytes)):
            self.supervised = bytearray(1 if value else 0 for value in self.supervised)
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


class TokenLengths:
    """Memoized token counts for one document.

    Windowing asks for the length of the same text many times over: once per
    paragraph, again for every buffer it rebuilds after a flush, again when
    merging trailing scraps, and again when filtering by the minimum. Measured
    over real books, six characters went through the tokenizer for every one in
    the corpus, and tokenizing is the whole cost of the preparation pass.

    The cache is per document and thrown away with it, so it holds counts for one
    book rather than for the corpus.
    """

    __slots__ = ("tokenizer", "_cache")

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._cache: dict[str, int] = {}

    def __call__(self, text: str) -> int:
        length = self._cache.get(text)
        if length is None:
            length = len(self.tokenizer.encode(text))
            self._cache[text] = length
        return length


def _split_long_paragraph(paragraph: str, lengths: TokenLengths, limit: int) -> list[str]:
    """Break one oversized paragraph at sentence boundaries, never mid-sentence."""
    sentences = _SENTENCE_SPLIT.split(paragraph)
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*current, sentence])
        if current and lengths(candidate) > limit:
            pieces.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _flush(buffer: list[str], lengths: TokenLengths, settings: StorySettings,
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
        if lengths(text) <= settings.maximum_tokens:
            windows.append(text)
            break
        if len(buffer) == 1:
            windows.extend(
                _split_long_paragraph(buffer[0], lengths, settings.maximum_tokens)
            )
            break
        leftover.insert(0, buffer.pop())
    return leftover


def window_text(
    text: str, tokenizer, settings: StorySettings,
    lengths: TokenLengths | None = None,
) -> list[str]:
    """Cut a book into story-sized windows.

    Chapter boundaries are preferred, paragraph boundaries are the fallback, and
    a sentence boundary is the last resort. A window is never cut mid-sentence.

    A caller that also needs the token count of each window should pass its own
    ``TokenLengths`` so the count it asks for afterwards is already cached.
    """
    if lengths is None:
        lengths = TokenLengths(tokenizer)
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
            count = lengths(paragraph)
            if count > settings.maximum_tokens:
                if buffer:
                    buffer = _flush(buffer, lengths, settings, windows)
                    counts = [lengths(item) for item in buffer]
                windows.extend(
                    _split_long_paragraph(paragraph, lengths, settings.maximum_tokens)
                )
                continue
            if buffer and buffered() + count > settings.maximum_tokens:
                buffer = _flush(buffer, lengths, settings, windows)
                counts = [lengths(item) for item in buffer]
            buffer.append(paragraph)
            counts.append(count)
            if buffered() >= settings.target_tokens:
                buffer = _flush(buffer, lengths, settings, windows)
                counts = [lengths(item) for item in buffer]
        while buffer:
            buffer = _flush(buffer, lengths, settings, windows)

    # A trailing scrap is merged backwards rather than dropped, so the last
    # paragraphs of a chapter are not silently lost.
    merged: list[str] = []
    for window in windows:
        if (
            merged
            and lengths(window) < settings.minimum_tokens
            and lengths(merged[-1] + "\n\n" + window) <= settings.maximum_tokens
        ):
            merged[-1] = merged[-1] + "\n\n" + window
        else:
            merged.append(window)
    return [
        window
        for window in merged
        if lengths(window) >= settings.minimum_tokens
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
    continues: bool = False,
) -> EncodedStory:
    """Encode one window with its tag block, applying the randomization policy.

    ``Length`` is recomputed here rather than taken from the caller. It describes
    the window, and the window is only known after windowing, so a value carried
    over from the whole book would be wrong.

    ``continues`` marks a window that follows another from the same book. Those
    windows are packed contiguously and in order, so the preceding text is
    genuinely in the context; stamping every window with ``<|story_start|>``
    told the model to treat it as a fresh start and ignore what came before,
    which is the opposite of what long-range character tracking needs. A
    continuation window omits the marker instead.
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
    budget = settings.block_size + 1 - len(block_ids) - (2 if continues else 3)
    if budget < 16:
        raise ValueError("tag block leaves no room for a story")
    if len(text_ids) > budget:
        text_ids = text_ids[:budget]

    opening = [] if continues else [story_start]
    if reversed_block and block_ids:
        ids = [*opening, *text_ids, *block_ids, story_end, eot]
        block_at = len(opening) + len(text_ids)
    else:
        ids = [*opening, *block_ids, *text_ids, story_end, eot]
        block_at = len(opening)

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
        continues=continues,
    )


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #

def assign_rows(segments: Sequence[EncodedStory], limit: int) -> list[tuple[int, int]]:
    """Greedily group segments into rows, returning half-open index ranges.

    A story is never split across two rows, and packing is sequential, so every
    row is a contiguous run of segments and one pair of indices describes it.
    Returning ranges rather than materialized rows is what lets the writer below
    fill a memory-mapped array without ever holding the corpus twice.
    """
    ranges: list[tuple[int, int]] = []
    start = 0
    length = 0
    for index, segment in enumerate(segments):
        size = len(segment.ids)
        if size > limit:
            raise ValueError("a story segment exceeds the packer row limit")
        if length and length + size > limit:
            ranges.append((start, index))
            start, length = index, 0
        length += size
    if length:
        ranges.append((start, len(segments)))
    return ranges


def _fill_row(
    segments: Sequence[EncodedStory],
    first: int,
    last: int,
    limit: int,
    pad_id: int,
    ids_buffer: np.ndarray,
    mask_buffer: np.ndarray,
) -> None:
    """Write one row's token ids and supervision mask into the scratch buffers."""
    ids_buffer[:] = pad_id
    mask_buffer[:] = False
    offset = 0
    for segment in segments[first:last]:
        size = len(segment.ids)
        ids_buffer[offset:offset + size] = np.frombuffer(segment.ids, dtype=np.uint16)
        mask_buffer[offset:offset + size] = np.frombuffer(
            segment.supervised, dtype=np.uint8
        ).astype(bool)
        offset += size
    mask_buffer[offset:] = False
    assert offset <= limit


def _span(segment: EncodedStory, start: int) -> dict:
    return {
        "start": start,
        "end": start + len(segment.ids),
        "repository": segment.repository,
        "path": segment.path,
        "source": segment.source,
        "tags": segment.tags,
        "tags_masked": segment.tags_masked,
        "tags_reversed": segment.tags_reversed,
        "continues": segment.continues,
    }


def pack_story_segments(
    segments: Sequence[EncodedStory], block_size: int, pad_id: int
) -> tuple[np.ndarray, np.ndarray, list[list[dict]]]:
    """Pack whole stories into rows, honouring each segment's loss mask.

    A story is never split across two rows. Padding and masked spans both become
    ``-100`` in the labels. Everything is held in memory, so this is for tests
    and small sets; ``write_packed_stories`` is the one to use on a real corpus.
    """
    limit = block_size + 1
    ranges = assign_rows(segments, limit)
    inputs = np.empty((len(ranges), block_size), dtype=np.uint16)
    labels = np.empty((len(ranges), block_size), dtype=LABEL_DTYPE)
    ids_buffer = np.empty(limit, dtype=np.uint16)
    mask_buffer = np.empty(limit, dtype=bool)
    spans: list[list[dict]] = []
    for row, (first, last) in enumerate(ranges):
        _fill_row(segments, first, last, limit, pad_id, ids_buffer, mask_buffer)
        inputs[row] = ids_buffer[:-1]
        labels[row] = np.where(
            mask_buffer[1:], ids_buffer[1:].astype(LABEL_DTYPE), IGNORE_INDEX
        )
        offset = 0
        row_spans = []
        for segment in segments[first:last]:
            row_spans.append(_span(segment, offset))
            offset += len(segment.ids)
        spans.append(row_spans)
    return inputs, labels, spans


def write_packed_stories(
    segments: Sequence[EncodedStory],
    block_size: int,
    pad_id: int,
    destination: Path | str,
) -> dict:
    """Pack segments straight into memory-mapped ``.npy`` files on disk.

    The in-memory packer needs the whole packed corpus resident twice over — once
    as Python lists and again as the arrays they are converted into — which is
    tens of gigabytes at 2K and 4K. Writing through ``open_memmap`` keeps the
    peak cost at one row, and the training loader mmaps the same files back, so
    a stage no longer has to fit in RAM to be trained on either.
    """
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    limit = block_size + 1
    ranges = assign_rows(segments, limit)
    if not ranges:
        raise ValueError("no segments to pack")

    inputs = np.lib.format.open_memmap(
        directory / "input_ids.npy", mode="w+",
        dtype=np.uint16, shape=(len(ranges), block_size),
    )
    labels = np.lib.format.open_memmap(
        directory / "labels.npy", mode="w+",
        dtype=LABEL_DTYPE, shape=(len(ranges), block_size),
    )
    ids_buffer = np.empty(limit, dtype=np.uint16)
    mask_buffer = np.empty(limit, dtype=bool)
    supervised = 0
    try:
        for row, (first, last) in enumerate(ranges):
            _fill_row(segments, first, last, limit, pad_id, ids_buffer, mask_buffer)
            inputs[row] = ids_buffer[:-1]
            labels[row] = np.where(
                mask_buffer[1:], ids_buffer[1:].astype(LABEL_DTYPE), IGNORE_INDEX
            )
            supervised += int(mask_buffer[1:].sum())
        inputs.flush()
        labels.flush()
    finally:
        del inputs, labels
    return {
        "rows": len(ranges),
        "supervised_tokens": supervised,
        "block_size": block_size,
    }
