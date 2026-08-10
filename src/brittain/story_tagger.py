"""Deterministic conditioning-tag extractors for brittain-shakespeare.

Every extractor is a pure function of the text or its bibliographic metadata. No
model is involved. That has two consequences the project depends on:

1. Training labels are correct by construction. A tag the model cannot verify
   against the text teaches it that tags are noise, and the control levers go dead.
2. The same code scores generated samples, so tag adherence is measurable rather
   than a matter of opinion. See ``scripts/evaluate/tag_adherence.py``.

Every extractor returns ``None`` when the evidence is weak. An absent tag is
already in distribution because of tag dropout, so silence is always cheaper than
a wrong label.

``Genre``, ``Tone``, and ``Twist`` are interpretive. ``Genre`` comes from curated
subject headings rather than the text. ``Tone`` uses an affect lexicon with a wide
margin requirement. ``Twist`` has no extractor at all and is supplied only by the
synthetic story set.
"""
from __future__ import annotations

import re
from collections import Counter

# --------------------------------------------------------------------------- #
# Text segmentation
# --------------------------------------------------------------------------- #

# Straight and curly double quotes, plus the em-dash dialogue convention used by
# many Gutenberg texts is not handled here; only quoted spans are removed.
_DIALOGUE = re.compile(r"[\"“”«»][^\"“”«»]{0,2000}?[\"“”«»]")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


def narration(text: str) -> str:
    """Return the text with quoted dialogue removed.

    Dialogue is first person and present tense by nature. Measuring point of view
    or tense without removing it reports the speech, not the narrator.
    """
    return _DIALOGUE.sub(" ", text)


def words(text: str) -> list[str]:
    return _WORD.findall(text)


# --------------------------------------------------------------------------- #
# Voice, from author life dates
# --------------------------------------------------------------------------- #

# Deliberately not publication year. Project Gutenberg's dcterms:issued is the
# date the book was released by Gutenberg, not when it was written: Dracula is
# stamped 1995-10-01. Using it would label nearly the whole corpus "Modern" and
# quietly train the register lever on noise. The RDF does carry author birth and
# death dates, and an author's death year approximates the end of their career
# well enough to place their register.

def voice_from_author_dates(
    birth_year: int | None, death_year: int | None
) -> str | None:
    """Map an author's life dates to a prose register."""
    effective = death_year
    if effective is None and birth_year is not None:
        # A living or undated author: assume a career ending in later life.
        effective = birth_year + 60
    if effective is None:
        return None
    if effective <= 1700:
        return "Shakespearean"
    if 1820 <= effective <= 1930:
        return "Victorian"
    if effective > 1930:
        return "Modern"
    # The long eighteenth century belongs to no register in this schema.
    return None


# Fallback when metadata carries no usable year: archaic morphology is distinctive
# enough to identify Early Modern English on its own.
_ARCHAIC = frozenset(
    """thou thee thy thine ye hast hath doth dost art wilt shalt canst
    didst wouldst couldst shouldst mayst prithee anon methinks nay yea
    forsooth wherefore whence whither hither thither betwixt ere oft
    'tis 'twas o'er e'er ne'er""".split()
)


def voice_from_text(text: str) -> str | None:
    """Detect the Shakespearean register from archaic morphology alone."""
    tokens = [word.lower() for word in words(text)]
    if len(tokens) < 120:
        return None
    hits = sum(1 for token in tokens if token in _ARCHAIC)
    archaic_rate = hits / len(tokens)
    # Also count second-person singular verb inflection, which survives contraction.
    inflected = len(re.findall(r"\b\w+(?:est|eth)\b", text.lower()))
    if archaic_rate >= 0.008 or (archaic_rate >= 0.004 and inflected >= 3):
        return "Shakespearean"
    return None


# --------------------------------------------------------------------------- #
# Fiction gate, from Library of Congress classification
# --------------------------------------------------------------------------- #

# Literature classes that actually hold narrative prose and drama. This is a
# structural signal and a far cleaner first gate than matching the word "fiction"
# against free-text headings. It is necessary, not sufficient: PR also holds
# poetry, criticism, and essays, so the corpus builder still applies the subject
# exclusions and the dialogue-density floor on top.
FICTION_LCC_CLASSES = frozenset({"PR", "PS", "PQ", "PT", "PZ"})


def is_fiction_lcc(codes: list[str]) -> bool:
    """Return whether any Library of Congress class is a literature class."""
    return any(code.strip().upper() in FICTION_LCC_CLASSES for code in codes)


# --------------------------------------------------------------------------- #
# Genre, from curated bookshelves and subject headings
# --------------------------------------------------------------------------- #

# Gutenberg's curated bookshelves are cleaner than raw LCSH strings, so they are
# consulted first. Ordered most specific first; a book matching several takes the
# first, which is why Ghost precedes Mystery (Dracula carries both).
_BOOKSHELF_GENRE: tuple[tuple[str, str], ...] = (
    ("horror", "Ghost"),
    ("gothic", "Ghost"),
    ("ghost", "Ghost"),
    ("crime, thrillers and mystery", "Mystery"),
    ("mystery", "Mystery"),
    ("detective", "Mystery"),
    ("fairy tales", "Fable"),
    ("folklore", "Fable"),
    ("mythology", "Fable"),
    ("fables", "Fable"),
    ("tragedy", "Tragedy"),
    ("humour", "Comedy"),
    ("humor", "Comedy"),
    ("comedy", "Comedy"),
    ("satire", "Comedy"),
    ("romance", "Romance"),
    ("love", "Romance"),
    ("science-fiction & fantasy", "Adventure"),
    ("science fiction", "Adventure"),
    ("fantasy", "Adventure"),
    ("pirates", "Adventure"),
    ("nautical", "Adventure"),
    ("western", "Adventure"),
    ("adventure", "Adventure"),
    ("plays", "Drama"),
    ("drama", "Drama"),
    ("theatre", "Drama"),
)

# Ordered most specific first. A heading matching several patterns takes the first.
_SUBJECT_GENRE: tuple[tuple[str, str], ...] = (
    ("ghost stories", "Ghost"),
    ("horror tales", "Ghost"),
    ("supernatural", "Ghost"),
    ("detective and mystery stories", "Mystery"),
    ("mystery", "Mystery"),
    ("detective", "Mystery"),
    ("tragedies", "Tragedy"),
    ("tragedy", "Tragedy"),
    ("comedies", "Comedy"),
    ("comedy", "Comedy"),
    ("humorous stories", "Comedy"),
    ("satire", "Comedy"),
    ("love stories", "Romance"),
    ("courtship", "Romance"),
    ("romance", "Romance"),
    ("adventure stories", "Adventure"),
    ("sea stories", "Adventure"),
    ("western stories", "Adventure"),
    ("science fiction", "Adventure"),
    ("fantasy", "Adventure"),
    ("adventure", "Adventure"),
    ("fairy tales", "Fable"),
    ("folklore", "Fable"),
    ("fables", "Fable"),
    ("legends", "Fable"),
    ("drama", "Drama"),
    ("plays", "Drama"),
)


# Most specific first. A book usually matches several genres across its headings
# and bookshelves, and the specific label is the informative one: every
# Shakespeare tragedy carries the heading "Tragedies (Drama)" and the bookshelf
# "Category: Plays/Films/Dramas", so a rule that simply preferred bookshelves
# labelled Hamlet, Lear, Macbeth, and Othello as Drama and left Tragedy with
# almost no examples. Specificity decides, not which field the match came from.
GENRE_PRECEDENCE = (
    "Ghost",
    "Tragedy",
    "Mystery",
    "Fable",
    "Comedy",
    "Romance",
    "Adventure",
    "Drama",
)


def genre_from_metadata(
    subjects: list[str] | None = None,
    bookshelves: list[str] | None = None,
) -> str | None:
    """Map bookshelves and subject headings to the most specific genre matched."""
    matched: set[str] = set()
    for values, table in (
        (bookshelves, _BOOKSHELF_GENRE),
        (subjects, _SUBJECT_GENRE),
    ):
        if not values:
            continue
        lowered = [value.lower() for value in values]
        for needle, genre in table:
            if any(needle in value for value in lowered):
                matched.add(genre)
    for genre in GENRE_PRECEDENCE:
        if genre in matched:
            return genre
    return None


# --------------------------------------------------------------------------- #
# Point of view
# --------------------------------------------------------------------------- #

_FIRST_PERSON = frozenset("i me my mine we us our ours myself ourselves".split())
_THIRD_PERSON = frozenset(
    "he him his she her hers they them their theirs himself herself themselves".split()
)
_INTERIORITY = re.compile(
    r"\b([A-Z][a-z]{2,})\s+(?:thought|felt|knew|wondered|realized|realised|"
    r"remembered|feared|hoped|believed|suspected)\b"
)


def point_of_view(text: str) -> str | None:
    """Classify narrative point of view over narration only."""
    body = narration(text)
    tokens = [word.lower() for word in words(body)]
    if len(tokens) < 150:
        return None
    counts = Counter(tokens)
    first = sum(counts[word] for word in _FIRST_PERSON)
    third = sum(counts[word] for word in _THIRD_PERSON)
    total = first + third
    if total < 20:
        return None
    if first / total >= 0.55:
        return "First"
    if third / total < 0.80:
        # Mixed narration. Neither label is defensible.
        return None
    # Third person. Interiority attributed to several distinct people reads as
    # omniscient; a single centre of consciousness reads as limited. This is the
    # weakest extractor in the set, so it demands clear evidence either way.
    subjects = {match.group(1) for match in _INTERIORITY.finditer(body)}
    if len(subjects) >= 3:
        return "Third-Omniscient"
    if len(subjects) == 1:
        return "Third-Limited"
    return None


# --------------------------------------------------------------------------- #
# Tense
# --------------------------------------------------------------------------- #

_PAST_IRREGULAR = frozenset(
    """was were had did said went came saw took got made knew thought felt
    found gave told became left began kept held stood heard let meant set
    put ran brought sat spoke lay led read grew lost fell sent built
    understood drew broke spent cut rose drove bought wore chose""".split()
)
_PRESENT_MARKERS = frozenset(
    """is are am has have does do says goes comes sees takes gets makes
    knows thinks feels finds gives tells becomes leaves begins keeps
    holds stands hears lets means sets puts runs brings sits speaks""".split()
)


def tense(text: str) -> str | None:
    """Classify narrative tense over narration only."""
    tokens = [word.lower() for word in words(narration(text))]
    if len(tokens) < 150:
        return None
    past = sum(1 for token in tokens if token in _PAST_IRREGULAR)
    past += sum(1 for token in tokens if len(token) > 4 and token.endswith("ed"))
    present = sum(1 for token in tokens if token in _PRESENT_MARKERS)
    total = past + present
    if total < 15:
        return None
    if past / total >= 0.75:
        return "Past"
    if present / total >= 0.60:
        return "Present"
    return None


# --------------------------------------------------------------------------- #
# Setting
# --------------------------------------------------------------------------- #

_SETTING_LEXICON: dict[str, tuple[str, ...]] = {
    "Tavern": (
        "tavern", "inn", "innkeeper", "alehouse", "ale", "tapster", "landlord",
        "barmaid", "bar", "saloon", "pub", "public-house", "hostelry", "tankard",
        "flagon", "beer", "brandy", "wine", "drinkers", "counter", "stool",
    ),
    "Castle": (
        "castle", "keep", "battlement", "battlements", "turret", "turrets",
        "portcullis", "drawbridge", "moat", "rampart", "ramparts", "dungeon",
        "tower", "fortress", "citadel", "gatehouse", "parapet", "buttress",
    ),
    "Sea": (
        "sea", "ship", "shipboard", "deck", "mast", "sail", "sails", "harbour",
        "harbor", "wave", "waves", "tide", "shore", "captain", "crew", "sailor",
        "sailors", "voyage", "ocean", "helm", "rigging", "bow", "stern", "port",
        "anchor", "schooner", "brig", "quay",
    ),
    "Forest": (
        "forest", "wood", "woods", "woodland", "thicket", "glade", "clearing",
        "bracken", "undergrowth", "pine", "oak", "birch", "moss", "trail",
        "branches", "canopy", "brambles", "fern", "ferns",
    ),
    "City": (
        "city", "street", "streets", "pavement", "alley", "square", "market",
        "shop", "shops", "traffic", "omnibus", "cab", "lamplight", "crowd",
        "crowds", "boulevard", "terrace", "quarter", "district", "gutter",
    ),
    "Household": (
        "kitchen", "parlour", "parlor", "hearth", "fireside", "cottage",
        "bedroom", "hallway", "staircase", "drawing-room", "sitting-room",
        "supper", "kettle", "mantelpiece", "doorstep", "garden", "nursery",
    ),
    "Court": (
        "court", "courtier", "courtiers", "throne", "king", "queen", "prince",
        "princess", "duke", "duchess", "lord", "lady", "majesty", "highness",
        "crown", "royal", "chamberlain", "herald", "audience-chamber",
    ),
    "Road": (
        "road", "highway", "lane", "track", "coach", "carriage", "horseback",
        "saddle", "traveller", "traveler", "milestone", "wayside", "journey",
        "footpath", "crossroads", "cart", "wagon", "stagecoach",
    ),
    "Battlefield": (
        "battle", "battlefield", "army", "armies", "soldier", "soldiers",
        "sword", "swords", "spear", "banner", "banners", "trench", "trenches",
        "cannon", "musket", "rifle", "charge", "retreat", "siege", "slain",
        "wounded", "regiment", "camp",
    ),
}

_SETTING_INDEX: dict[str, str] = {
    term: setting
    for setting, terms in _SETTING_LEXICON.items()
    for term in terms
}


def setting(text: str) -> str | None:
    """Pick the dominant physical setting, or nothing if no setting dominates."""
    tokens = [word.lower().strip("'’") for word in words(text)]
    if len(tokens) < 100:
        return None
    counts: Counter[str] = Counter()
    for token in tokens:
        name = _SETTING_INDEX.get(token)
        if name is not None:
            counts[name] += 1
    if not counts:
        return None
    ranked = counts.most_common(2)
    best, best_count = ranked[0]
    # Require real evidence and a clear margin over the runner-up. A story that
    # touches several places has no single setting, and guessing one would teach
    # the model that the tag does not mean anything.
    if best_count < 5:
        return None
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if best_count < 2 * runner_up:
        return None
    return best


# --------------------------------------------------------------------------- #
# Tone
# --------------------------------------------------------------------------- #

_TONE_LEXICON: dict[str, tuple[str, ...]] = {
    "Dark": (
        "blood", "corpse", "murder", "grave", "shadow", "shadows", "terror",
        "dread", "cruel", "cruelty", "rot", "wound", "knife", "scream",
        "screamed", "darkness", "vengeance", "curse", "corpses", "gallows",
    ),
    "Wry": (
        "absurd", "amused", "amusing", "irony", "ironic", "smirk", "grinned",
        "chuckled", "ridiculous", "preposterous", "comic", "wry", "drily",
        "dryly", "solemnly", "pompous", "quipped", "jest",
    ),
    "Tender": (
        "gentle", "gently", "tender", "tenderly", "kindness", "kindly",
        "embrace", "embraced", "warmth", "beloved", "dear", "caress",
        "comfort", "comforted", "affection", "smiled", "soothed", "cradled",
    ),
    "Bleak": (
        "grey", "gray", "empty", "emptiness", "cold", "barren", "desolate",
        "silence", "silent", "weary", "wearily", "hopeless", "ruin", "ruined",
        "ash", "ashes", "bitter", "hollow", "nothing", "abandoned",
    ),
    "Rousing": (
        "triumph", "triumphant", "courage", "brave", "bravely", "glory",
        "victory", "cheered", "cheering", "banner", "rally", "rallied",
        "defiance", "defiant", "roared", "onward", "valiant", "hero",
    ),
    "Uneasy": (
        "uneasy", "uneasily", "nervous", "nervously", "hesitated", "watchful",
        "suspicion", "suspicious", "strange", "strangely", "wary", "warily",
        "trembled", "flinched", "waiting", "listened", "whispered", "shivered",
    ),
}

_TONE_INDEX: dict[str, str] = {
    term: tone
    for tone, terms in _TONE_LEXICON.items()
    for term in terms
}


def tone(text: str) -> str | None:
    """Pick the dominant affect, or nothing if no affect dominates.

    This is a lexicon, not a reader. It demands a wide margin because a wrong tone
    label is worse for the control lever than an absent one.
    """
    tokens = [word.lower() for word in words(text)]
    if len(tokens) < 200:
        return None
    counts: Counter[str] = Counter()
    for token in tokens:
        name = _TONE_INDEX.get(token)
        if name is not None:
            counts[name] += 1
    if not counts:
        return None
    ranked = counts.most_common(2)
    best, best_count = ranked[0]
    if best_count < 6:
        return None
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if best_count < 2 * runner_up:
        return None
    return best


# --------------------------------------------------------------------------- #
# Cast
# --------------------------------------------------------------------------- #

# Capitalized words that are not character names.
_NOT_A_NAME = frozenset(
    """the a an and but or if so then when while as for nor yet of to in on at
    by from with about into over after before under above i we he she it they
    you there here this that these those what who whom whose why how all any
    both each few more most other some such no not only own same than too very
    can will just now mr mrs miss dr sir lady lord god heaven hell chapter
    monday tuesday wednesday thursday friday saturday sunday january february
    march april may june july august september october november december
    yes oh ah well come go let""".split()
)
_CAPITALIZED = re.compile(r"\b([A-Z][a-z]{2,})\b")


def cast(text: str) -> str | None:
    """Estimate how many people carry the story."""
    tokens = words(text)
    if len(tokens) < 150:
        return None
    # Sentence-initial capitalization carries no information, but discarding it
    # would discard most of the evidence: a protagonist's name usually opens
    # sentences. Instead, treat a capitalized word as a name only when its
    # lowercase form never appears on its own anywhere in the text. That removes
    # ordinary words caught at a sentence boundary without losing the name.
    lowercase_seen = {token.lower() for token in tokens if token[0].islower()}
    candidates: Counter[str] = Counter()
    for match in _CAPITALIZED.finditer(text):
        word = match.group(1)
        folded = word.lower()
        if folded in _NOT_A_NAME or folded in lowercase_seen:
            continue
        candidates[word] += 1
    # A real character is named repeatedly. A place mentioned once is not.
    names = [name for name, count in candidates.items() if count >= 3]
    if not names:
        return None
    if len(names) == 1:
        return "Solo"
    if len(names) == 2:
        return "Pair"
    if len(names) >= 4:
        return "Ensemble"
    # Exactly three named people sits on the boundary between the two labels.
    return None


# --------------------------------------------------------------------------- #
# Length
# --------------------------------------------------------------------------- #

FLASH_MAX_TOKENS = 600
SHORT_MAX_TOKENS = 2000


def length_from_tokens(token_count: int) -> str:
    """Bucket a window by its exact token count. Always returns a value."""
    if token_count <= FLASH_MAX_TOKENS:
        return "Flash"
    if token_count <= SHORT_MAX_TOKENS:
        return "Short"
    return "Long"


# --------------------------------------------------------------------------- #
# Combined extraction
# --------------------------------------------------------------------------- #

def extract(
    text: str,
    *,
    token_count: int | None = None,
    birth_year: int | None = None,
    death_year: int | None = None,
    subjects: list[str] | None = None,
    bookshelves: list[str] | None = None,
) -> dict[str, str]:
    """Run every extractor and return only the tags with usable evidence.

    ``Twist`` is never produced here. It has no deterministic extractor and is
    supplied only by the synthetic story set.
    """
    tags: dict[str, str] = {}

    # Archaic morphology in the text itself outranks the author's dates: a modern
    # author writing in period voice should be labelled by what they wrote.
    voice = voice_from_text(text) or voice_from_author_dates(birth_year, death_year)
    if voice is not None:
        tags["Voice"] = voice

    genre = genre_from_metadata(subjects, bookshelves)
    if genre is not None:
        tags["Genre"] = genre

    for name, value in (
        ("POV", point_of_view(text)),
        ("Tense", tense(text)),
        ("Setting", setting(text)),
        ("Tone", tone(text)),
        ("Cast", cast(text)),
    ):
        if value is not None:
            tags[name] = value

    if token_count is not None:
        tags["Length"] = length_from_tokens(token_count)

    return tags
