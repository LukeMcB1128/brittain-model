"""Conditioning-tag schema for brittain-shakespeare.

Tags are rendered as ordinary bracketed text between two special tokens:

    <|tags|>[Voice: Modern] [Genre: Mystery] [POV: First]<|end_tags|>

The values are deliberately not special tokens. A closed vocabulary controls the
training data, but the model reads each value as ordinary text, so an unseen
value such as ``[Setting: Lighthouse]`` still composes at inference.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

TAGS_START = "<|tags|>"
TAGS_END = "<|end_tags|>"

# Canonical order. Data preparation renders tags in this order unless it shuffles.
TAG_ORDER = (
    "Voice",
    "Genre",
    "POV",
    "Tense",
    "Setting",
    "Tone",
    "Cast",
    "Length",
    "Twist",
)

TAG_VALUES: dict[str, tuple[str, ...]] = {
    "Voice": ("Shakespearean", "Victorian", "Modern"),
    "Genre": (
        "Tragedy",
        "Comedy",
        "Romance",
        "Mystery",
        "Ghost",
        "Adventure",
        "Fable",
        "Drama",
    ),
    "POV": ("First", "Third-Limited", "Third-Omniscient"),
    "Tense": ("Past", "Present"),
    "Setting": (
        "Tavern",
        "Castle",
        "Sea",
        "Forest",
        "City",
        "Household",
        "Court",
        "Road",
        "Battlefield",
    ),
    "Tone": ("Dark", "Wry", "Tender", "Bleak", "Rousing", "Uneasy"),
    "Cast": ("Solo", "Pair", "Ensemble"),
    "Length": ("Flash", "Short", "Long"),
    "Twist": ("Betrayal", "Revelation", "Reversal", "Death", "Reunion", "None"),
}

assert set(TAG_ORDER) == set(TAG_VALUES), "TAG_ORDER and TAG_VALUES must agree"

# Tags a deterministic extractor scores exactly on generated text. The adherence
# evaluation reports these separately from the interpretive tags.
OBJECTIVE_TAGS = ("Voice", "POV", "Tense", "Setting", "Cast", "Length")
INTERPRETIVE_TAGS = ("Genre", "Tone", "Twist")


@dataclass(frozen=True)
class TagPolicy:
    """Randomization applied to one training example's tag block."""

    # Probability that any single tag is dropped from the block.
    tag_dropout: float = 0.30
    # Probability that the whole block is dropped, leaving an unconditional example.
    block_dropout: float = 0.10
    # Probability that the surviving tags are emitted out of canonical order.
    shuffle_rate: float = 0.15
    # Probability that the tag block is excluded from the training loss.
    mask_rate: float = 0.80
    # Probability that the block follows the story instead of preceding it.
    reverse_rate: float = 0.05

    def __post_init__(self) -> None:
        for name in ("tag_dropout", "block_dropout", "shuffle_rate", "mask_rate", "reverse_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")


def validate(tags: dict[str, str]) -> None:
    """Raise if any tag name or value falls outside the closed vocabulary."""
    for name, value in tags.items():
        if name not in TAG_VALUES:
            raise ValueError(f"unknown tag name: {name!r}")
        if value not in TAG_VALUES[name]:
            raise ValueError(f"unknown value for {name}: {value!r}")


def render(tags: dict[str, str], *, order: tuple[str, ...] | None = None) -> str:
    """Render a tag block body without the surrounding special tokens."""
    validate(tags)
    names = order if order is not None else TAG_ORDER
    if order is not None and set(order) != set(tags):
        raise ValueError("explicit order must name exactly the supplied tags")
    return " ".join(f"[{name}: {tags[name]}]" for name in names if name in tags)


def parse(text: str) -> dict[str, str]:
    """Parse a rendered tag block back into a mapping.

    Accepts the body alone or a full block wrapped in the special tokens. Unknown
    names and values are rejected so a malformed block never silently becomes an
    empty condition.
    """
    body = text.strip()
    if body.startswith(TAGS_START):
        body = body[len(TAGS_START):]
    if body.endswith(TAGS_END):
        body = body[: -len(TAGS_END)]
    tags: dict[str, str] = {}
    for chunk in body.split("]"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.startswith("["):
            raise ValueError(f"malformed tag block near {chunk!r}")
        name, separator, value = chunk[1:].partition(":")
        if not separator:
            raise ValueError(f"tag is missing a value: {chunk!r}")
        name, value = name.strip(), value.strip()
        if name in tags:
            raise ValueError(f"duplicate tag: {name!r}")
        tags[name] = value
    validate(tags)
    return tags


def parse_request(text: str) -> dict[str, str]:
    """Parse a relaxed command-line spec such as ``Genre: Tragedy, Setting: Tavern``.

    Bracketed input is accepted too, so a user can paste a block back in.
    """
    if "[" in text:
        return parse(text)
    tags: dict[str, str] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, separator, value = chunk.partition(":")
        if not separator:
            raise ValueError(f"tag is missing a value: {chunk!r}")
        name = name.strip()
        # Reject duplicates rather than letting the last one win. Writing
        # "Tone: Past" when Tense was meant should be an error, not a silently
        # discarded tag.
        if name in tags:
            raise ValueError(f"duplicate tag: {name!r}")
        tags[name] = value.strip()
    validate(tags)
    return tags


def apply_policy(
    tags: dict[str, str], policy: TagPolicy, rng: random.Random
) -> tuple[dict[str, str], tuple[str, ...], bool, bool]:
    """Sample one example's tag presentation.

    Returns the surviving tags, the order to render them in, whether the block is
    excluded from the loss, and whether it follows the story.
    """
    validate(tags)
    masked = rng.random() < policy.mask_rate
    reversed_block = rng.random() < policy.reverse_rate
    if rng.random() < policy.block_dropout:
        return {}, (), masked, reversed_block
    kept = {
        name: value
        for name, value in tags.items()
        if rng.random() >= policy.tag_dropout
    }
    order = tuple(name for name in TAG_ORDER if name in kept)
    if rng.random() < policy.shuffle_rate:
        shuffled = list(order)
        rng.shuffle(shuffled)
        order = tuple(shuffled)
    return kept, order, masked, reversed_block
