"""Project Gutenberg catalog reading and the brittain-shakespeare fiction filter.

The catalog is read from Gutenberg's own RDF feed rather than a Hugging Face
mirror. The fiction filter and the ``Genre`` tag both run on Library of Congress
metadata, and the mirrors generally ship the text with that metadata stripped.

Nothing here downloads anything. ``iter_catalog`` reads a local
``rdf-files.tar.bz2`` that the operator has already fetched.
"""
from __future__ import annotations

import bz2
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

from . import story_tagger

# Namespaces as declared on rdf:RDF in the Gutenberg feed.
NS = {
    "dcterms": "http://purl.org/dc/terms/",
    "pgterms": "http://www.gutenberg.org/2009/pgterms/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dcam": "http://purl.org/dc/dcam/",
}
LCSH_VOCABULARY = "http://purl.org/dc/terms/LCSH"
LCC_VOCABULARY = "http://purl.org/dc/terms/LCC"

_EBOOK_ID = re.compile(r"ebooks/(\d+)")


@dataclass
class BookRecord:
    """One Gutenberg ebook's metadata, before any text is read."""

    book_id: int
    title: str = ""
    author: str = ""
    birth_year: int | None = None
    death_year: int | None = None
    language: str = ""
    rights: str = ""
    lcc: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    bookshelves: list[str] = field(default_factory=list)


def _text(node) -> str:
    return (node.text or "").strip() if node is not None else ""


def _year(node) -> int | None:
    value = _text(node)
    try:
        return int(value)
    except ValueError:
        return None


def _values_by_vocabulary(ebook, tag: str) -> dict[str, list[str]]:
    """Collect rdf:value strings under ``tag``, grouped by their dcam:memberOf."""
    grouped: dict[str, list[str]] = {}
    for holder in ebook.findall(tag, NS):
        for description in holder.findall("rdf:Description", NS):
            member = description.find("dcam:memberOf", NS)
            vocabulary = (
                member.get(f"{{{NS['rdf']}}}resource", "") if member is not None else ""
            )
            value = _text(description.find("rdf:value", NS))
            if value:
                grouped.setdefault(vocabulary, []).append(value)
    return grouped


def parse_rdf(source) -> BookRecord | None:
    """Parse one pg<id>.rdf document into a record, or None if it has no ebook."""
    try:
        root = ElementTree.parse(source).getroot()
    except ElementTree.ParseError:
        return None
    ebook = root.find("pgterms:ebook", NS)
    if ebook is None:
        return None
    match = _EBOOK_ID.search(ebook.get(f"{{{NS['rdf']}}}about", ""))
    if match is None:
        return None

    record = BookRecord(book_id=int(match.group(1)))
    record.title = _text(ebook.find("dcterms:title", NS))
    record.rights = _text(ebook.find("dcterms:rights", NS))

    language = ebook.find("dcterms:language/rdf:Description/rdf:value", NS)
    record.language = _text(language)

    agent = ebook.find("dcterms:creator/pgterms:agent", NS)
    if agent is not None:
        record.author = _text(agent.find("pgterms:name", NS))
        record.birth_year = _year(agent.find("pgterms:birthdate", NS))
        record.death_year = _year(agent.find("pgterms:deathdate", NS))

    subjects = _values_by_vocabulary(ebook, "dcterms:subject")
    record.subjects = subjects.get(LCSH_VOCABULARY, [])
    record.lcc = subjects.get(LCC_VOCABULARY, [])

    shelves = _values_by_vocabulary(ebook, "pgterms:bookshelf")
    record.bookshelves = [value for values in shelves.values() for value in values]

    return record


def iter_catalog(archive: str | Path, *, limit: int | None = None) -> Iterator[BookRecord]:
    """Yield every record in a local ``rdf-files.tar.bz2``.

    The archive holds one small RDF file per ebook. It is streamed rather than
    extracted so the whole catalog never lands on disk twice.
    """
    path = Path(archive)
    seen = 0
    with tarfile.open(path, "r:bz2") as handle:
        for member in handle:
            if not member.isfile() or not member.name.endswith(".rdf"):
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                continue
            record = parse_rdf(extracted)
            if record is None:
                continue
            yield record
            seen += 1
            if limit is not None and seen >= limit:
                return


def read_catalog_directory(directory: str | Path) -> Iterator[BookRecord]:
    """Yield records from a directory of loose .rdf files, for tests and dry runs."""
    for path in sorted(Path(directory).rglob("*.rdf")):
        with path.open("rb") as handle:
            record = parse_rdf(handle)
        if record is not None:
            yield record


# --------------------------------------------------------------------------- #
# Metadata filter
# --------------------------------------------------------------------------- #

# Ordered. The funnel report counts rejections against the first rule a book
# fails, so the order determines what the report attributes a loss to.
REJECTION_REASONS = (
    "language",
    "rights",
    "lcc_class",
    "excluded_subject",
    "no_narrative_subject",
    "no_author_dates",
)


@dataclass
class TextureConfig:
    """The world-texture path: narrative prose shelved under history or religion.

    Legends, folk tales, saints' lives, first-person war memoirs, and Bible-story
    retellings give a story about the Crusades or a war its setting texture. This
    is not a route to encyclopedic knowledge: it is capped, and the text floors
    are stricter than on the fiction path because these classes carry far more
    expository prose.
    """

    lcc_prefixes: tuple[str, ...] = ()
    required_subject_markers: tuple[str, ...] = ()
    excluded_subject_patterns: tuple[str, ...] = ()
    minimum_speech_fraction: float = 0.0
    minimum_narrative_verb_ratio: float = 0.0
    maximum_corpus_share: float = 0.05

    @classmethod
    def from_config(cls, config: dict) -> "TextureConfig | None":
        texture = config.get("world_texture")
        if not texture:
            return None
        return cls(
            # Longest first so DA is tested before D.
            lcc_prefixes=tuple(
                sorted(
                    (code.upper() for code in texture.get("lcc_prefixes", ())),
                    key=len,
                    reverse=True,
                )
            ),
            required_subject_markers=tuple(
                marker.lower() for marker in texture.get("required_subject_markers", ())
            ),
            excluded_subject_patterns=tuple(
                pattern.lower()
                for pattern in texture.get("excluded_subject_patterns", ())
            ),
            minimum_speech_fraction=float(
                texture.get("minimum_speech_fraction", 0.0)
            ),
            minimum_narrative_verb_ratio=float(
                texture.get("minimum_narrative_verb_ratio", 0.0)
            ),
            maximum_corpus_share=float(texture.get("maximum_corpus_share", 0.05)),
        )


@dataclass
class FilterConfig:
    """The metadata half of the fiction filter, loaded from the corpus config."""

    language: str = "en"
    allowed_rights: tuple[str, ...] = ()
    required_lcc_classes: frozenset[str] = frozenset()
    required_subject_patterns: tuple[str, ...] = ()
    excluded_subject_patterns: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: dict) -> "FilterConfig":
        fiction = config.get("fiction_filter", {})
        return cls(
            language=config.get("language", "en"),
            allowed_rights=tuple(config.get("allowed_rights", ())),
            required_lcc_classes=frozenset(
                code.upper() for code in fiction.get("required_lcc_classes", ())
            ),
            required_subject_patterns=tuple(
                pattern.lower()
                for pattern in fiction.get("required_subject_patterns", ())
            ),
            excluded_subject_patterns=tuple(
                pattern.lower()
                for pattern in config.get("excluded_subject_patterns", ())
            ),
        )


def rejection_reason(record: BookRecord, config: FilterConfig) -> str | None:
    """Return the first rule the book fails, or None if it passes every one.

    This is the metadata half of the filter only. The dialogue-density and
    narrative-verb floors need the text and are applied later, once a book has
    earned the cost of being downloaded.
    """
    if config.language and record.language != config.language:
        return "language"
    if config.allowed_rights and record.rights not in config.allowed_rights:
        return "rights"
    if config.required_lcc_classes:
        codes = {code.strip().upper() for code in record.lcc}
        if not codes & config.required_lcc_classes:
            return "lcc_class"

    haystack = [value.lower() for value in (*record.subjects, *record.bookshelves)]
    if any(
        pattern in value
        for pattern in config.excluded_subject_patterns
        for value in haystack
    ):
        return "excluded_subject"
    if config.required_subject_patterns and not any(
        pattern in value
        for pattern in config.required_subject_patterns
        for value in haystack
    ):
        return "no_narrative_subject"

    # Voice falls back to archaic morphology in the text, so missing dates are not
    # fatal on their own. They are only fatal when nothing else can place the
    # register either, which is checked once the text is available.
    if record.birth_year is None and record.death_year is None:
        return "no_author_dates"
    return None


FICTION_CATEGORY = "gutenberg_fiction"
TEXTURE_CATEGORY = "world_texture"


def texture_rejection_reason(
    record: BookRecord, config: FilterConfig, texture: TextureConfig
) -> str | None:
    """Metadata rules for the world-texture path."""
    if config.language and record.language != config.language:
        return "language"
    if config.allowed_rights and record.rights not in config.allowed_rights:
        return "rights"

    codes = [code.strip().upper() for code in record.lcc]
    if not any(
        code.startswith(prefix) for code in codes for prefix in texture.lcc_prefixes
    ):
        return "texture_lcc_class"

    haystack = [value.lower() for value in (*record.subjects, *record.bookshelves)]
    if any(
        pattern in value
        for pattern in texture.excluded_subject_patterns
        for value in haystack
    ):
        return "texture_excluded_subject"
    if not any(
        marker in value
        for marker in texture.required_subject_markers
        for value in haystack
    ):
        return "texture_not_narrative"
    if record.birth_year is None and record.death_year is None:
        return "no_author_dates"
    return None


def classify(
    record: BookRecord, config: FilterConfig, texture: TextureConfig | None
) -> tuple[str | None, str | None]:
    """Assign a record to a corpus category, or explain why it is rejected.

    The fiction path is tried first so a novel is never counted as texture. A
    book rejected by fiction is reported against the texture rules only when it
    could plausibly belong there, so the funnel keeps attributing ordinary
    non-fiction to the fiction rule that caught it.
    """
    reason = rejection_reason(record, config)
    if reason is None:
        return FICTION_CATEGORY, None
    if texture is None:
        return None, reason
    texture_reason = texture_rejection_reason(record, config, texture)
    if texture_reason is None:
        return TEXTURE_CATEGORY, None
    return None, reason


def book_tags(record: BookRecord) -> dict[str, str]:
    """The tags derivable from metadata alone, before the text is read."""
    tags: dict[str, str] = {}
    voice = story_tagger.voice_from_author_dates(record.birth_year, record.death_year)
    if voice is not None:
        tags["Voice"] = voice
    genre = story_tagger.genre_from_metadata(record.subjects, record.bookshelves)
    if genre is not None:
        tags["Genre"] = genre
    return tags


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #

_START_MARKER = re.compile(
    r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_END_MARKER = re.compile(
    r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TRANSCRIBER = re.compile(
    r"^\[?(?:Transcriber'?s?|Editor'?s?) Note.*?(?:\]|\n\s*\n)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_BLANK_RUN = re.compile(r"\n{3,}")


def strip_boilerplate(text: str) -> str:
    """Remove the Gutenberg header, footer, and transcriber notes."""
    start = _START_MARKER.search(text)
    if start is not None:
        text = text[start.end():]
    end = _END_MARKER.search(text)
    if end is not None:
        text = text[: end.start()]
    text = _TRANSCRIBER.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _BLANK_RUN.sub("\n\n", text).strip()


_DIALOGUE_LINE = re.compile(r"[\"“”«»]")

# Speech does not always arrive in quotation marks, and assuming it does is a
# trap: Hamlet contains no quotation marks at all. Drama puts the speaker on its
# own line, Continental and Irish printings open with an em dash, and British
# printings often use single quotes.
_DRAMA_SPEAKER = re.compile(r"^[ \t]*[A-Z][A-Z'’.\- ]{1,28}\.[ \t]*$")
_DRAMA_SPEAKER_TITLED = re.compile(r"^[ \t]*[A-Z][a-z]{2,14}\.[ \t]*$")
_STAGE_DIRECTION = re.compile(r"^[ \t]*\[_.*_\.?\]")
_EM_DASH_SPEECH = re.compile(r"^[ \t]*[—–][ \t]*[A-Z]")
# A single-quoted span long enough not to be an apostrophe or a scare quote.
_SINGLE_QUOTE_SPEECH = re.compile(r"(?<![A-Za-z])'[A-Z][^']{8,}?[,.!?]'")


def dialogue_fraction(text: str) -> float:
    """Share of non-empty lines carrying a quotation mark."""
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return 0.0
    return sum(1 for line in lines if _DIALOGUE_LINE.search(line)) / len(lines)


def drama_fraction(text: str) -> float:
    """Share of non-empty lines that are a speaker cue or a stage direction."""
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return 0.0
    hits = sum(
        1
        for line in lines
        if _DRAMA_SPEAKER.match(line)
        or _DRAMA_SPEAKER_TITLED.match(line)
        or _STAGE_DIRECTION.match(line)
    )
    return hits / len(lines)


def speech_fraction(text: str) -> float:
    """Share of non-empty lines carrying speech in any printing convention.

    This is the narrative signal the corpus floors use. Fiction and drama have
    speech; treatises do not.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return 0.0
    hits = 0
    for line in lines:
        if (
            _DIALOGUE_LINE.search(line)
            or _DRAMA_SPEAKER.match(line)
            or _DRAMA_SPEAKER_TITLED.match(line)
            or _STAGE_DIRECTION.match(line)
            or _EM_DASH_SPEECH.match(line)
            or _SINGLE_QUOTE_SPEECH.search(line)
        ):
            hits += 1
    return hits / len(lines)


_NARRATIVE_VERBS = frozenset(
    """said asked replied answered cried whispered shouted muttered
    murmured exclaimed laughed smiled nodded turned walked ran looked
    watched waited stood sat rose went came saw took gave told""".split()
)


def narrative_verb_ratio(text: str) -> float:
    """Share of words that are common narrative verbs."""
    tokens = [token.lower() for token in story_tagger.words(text)]
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token in _NARRATIVE_VERBS) / len(tokens)


def text_rejection_reason(
    text: str, *, minimum_speech: float, minimum_verb_ratio: float
) -> str | None:
    """Apply the narrative floor. Returns the failed rule or None.

    The two measures are combined with OR, not AND, because they detect
    different forms of narrative and neither is present in both. Drama is almost
    entirely speech and has nearly no narrative verbs: Hamlet scores 0.248 on
    speech and 0.0023 on verbs. Sparse-dialogue literary prose is the reverse.
    Requiring both would reject Shakespeare from a model named after him, and
    that failure would have been invisible in the corpus counts.
    """
    if speech_fraction(text) >= minimum_speech:
        return None
    if narrative_verb_ratio(text) >= minimum_verb_ratio:
        return None
    return "narrative_floor"


def read_book_text(path: str | Path) -> str:
    """Read one Gutenberg .txt, plain or bz2, and strip its boilerplate."""
    path = Path(path)
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return strip_boilerplate(handle.read())
