"""Prompt construction and verification for the synthetic story set.

The synthetic stories are the only in-domain data in the project. Every other
document is a window cut from the middle of a novel, which teaches prose texture
and cannot teach narrative structure: there is no setup, no turn, and no close in
the middle of a chapter. These stories are the only place the model ever sees a
story shaped like a story.

That is also why the exemplars below matter more than the generator model does. A
cheap model pattern-matching against three well-built examples produces better
structure than an expensive one improvising, so the examples carry the design.

Generated stories are verified with the same deterministic extractors that
labelled the real corpus. A story that contradicts its own tags is discarded
rather than trusted, which preserves the correct-by-construction property the
control levers depend on.
"""
from __future__ import annotations

import random

from . import story_tagger
from .tags import TAG_VALUES, render

# Every synthetic story is modern prose. Binding the generator's voice to an
# explicit register is what stops it leaking everywhere: the model learns this
# flavour belongs to Modern, so asking for Shakespearean steers away from it.
SYNTHETIC_VOICE = "Modern"

# Values the real corpus barely covers, so the generator is pushed at them.
# Twist has no extractor at all and appears nowhere in the Gutenberg data.
WEIGHTED_VALUES: dict[str, dict[str, float]] = {
    "Twist": {"Betrayal": 1.0, "Revelation": 1.0, "Reversal": 1.0,
              "Death": 1.0, "Reunion": 1.0, "None": 0.4},
    "Tone": {"Dark": 1.0, "Wry": 1.4, "Tender": 1.2, "Bleak": 1.0,
             "Rousing": 1.4, "Uneasy": 1.0},
    "Setting": {"Tavern": 1.5, "Castle": 1.5, "Sea": 0.6, "Forest": 1.5,
                "City": 1.4, "Household": 1.5, "Court": 0.7, "Road": 1.4,
                "Battlefield": 0.9},
    "POV": {"First": 1.0, "Third-Limited": 1.0, "Third-Omniscient": 0.5},
    "Cast": {"Solo": 0.7, "Pair": 1.4, "Ensemble": 1.5},
    "Genre": {name: 1.0 for name in TAG_VALUES["Genre"]},
    "Tense": {"Past": 1.0, "Present": 1.1},
    # Long is deliberately absent: it would not fit one 1K training row, and the
    # pretraining corpus already supplies long-form prose in abundance.
    "Length": {"Flash": 1.0, "Short": 1.2, "Long": 0.0},
}

NAME_POOL = (
    "Alder", "Bramwell", "Calder", "Dunmore", "Elias", "Fenwick", "Greer",
    "Harrow", "Ivo", "Jessamy", "Keziah", "Lark", "Merrick", "Nessa", "Orrin",
    "Peregrine", "Quill", "Rowan", "Selby", "Thea", "Ursa", "Vance", "Wren",
    "Yarrow", "Zeb", "Auberon", "Bridie", "Cassia", "Dorian", "Esme",
)

LENGTH_WORDS = {"Flash": (250, 400), "Short": (500, 850), "Long": (900, 1200)}

SYSTEM_PROMPT = """You write complete short stories to specification.

Rules, in order of importance:

1. Write a COMPLETE story. It needs a beginning that establishes someone who
   wants something, a middle where that meets an obstacle or a turn, and an
   ending that closes. The last line must feel like an ending, not a stopping
   point. This matters more than beauty.
2. Obey every tag exactly. The tags are constraints, not suggestions.
3. Use the named characters, by those names, repeatedly. Do not rename them and
   do not add other named characters.
4. Write only the story. No title, no preamble, no commentary, no tag echo.
5. Plain prose. No markdown, no headings, no bullet points.
"""


def _exemplar(tags, story):
    return {"tags": tags, "story": story}


_TAVERN = (
    "Merrick had the money in his coat and he kept his hand on it while he "
    "waited. The tavern was almost empty. Rain came off the roof in a steady "
    "line past the window, and the landlord wiped the same glass he had been "
    "wiping for an hour.\n\n"
    "Calder came in at nine, shook the water off his shoulders, and sat down "
    "across from him like a man with nothing to hide.\n\n"
    "“You brought it,” Calder said.\n\n"
    "“I brought it.”\n\n"
    "“Then it is done. They will take the debt off the house and you can go "
    "home and sleep.” Calder smiled. He had a good smile. Merrick had "
    "trusted it for eleven years, since they were boys hauling nets on the same "
    "boat.\n\n"
    "Merrick put the packet on the table. Calder covered it with his hand and "
    "did not move it.\n\n"
    "“There is one thing,” Calder said. “They wanted a name as "
    "well. Someone to carry it. I had to give them one.”\n\n"
    "The rain went on. Merrick understood before Calder had finished speaking, "
    "and what surprised him was not the betrayal but how little of him was "
    "surprised.\n\n"
    "“How long have they had my name?”\n\n"
    "“Since spring.”\n\n"
    "Merrick stood. He left the money where it lay, because it was not his any "
    "more and had not been for months. At the door he looked back once. Calder "
    "was still sitting with his hand on the packet, watching him go, and he did "
    "not say anything at all."
)

_HOUSE = (
    "I took the house in October because it was cheap and I did not ask why. "
    "Esme helped me carry the boxes in and said the hall smelled of somebody "
    "else's cooking, and I said that all old houses do.\n\n"
    "The first week I heard someone on the stairs at night. I told myself it was "
    "the pipes. The second week I found the kitchen door standing open twice, "
    "and I started locking it, and it went on standing open.\n\n"
    "I asked the woman next door. She was careful with her answer.\n\n"
    "“There was a girl here,” she said. “Before. She used to sit "
    "on the stairs.”\n\n"
    "“What happened to her?”\n\n"
    "“Nothing happened to her. She moved away.” The woman looked at me "
    "a moment longer than she needed to. “You look like her, a little."
    "”\n\n"
    "I went home and I sat on the stairs myself, in the dark, because I wanted "
    "to know what she had been listening for. And I heard it. A woman below me "
    "in the kitchen, moving about, opening the door.\n\n"
    "I have thought about it every night since. She was not haunting the house. "
    "She was sitting where I am sitting, hearing what I am hearing, and the "
    "thing in the kitchen was never hers and it is not mine.\n\n"
    "Bridie asks me why I do not simply leave. I tell her that I will. I have "
    "been telling her that since October."
)

_HILL = (
    "Thea counts eleven of them coming up the slope and decides that eleven is a "
    "number she can work with.\n\n"
    "“We run,” Orrin says. He is bleeding above the eye and he keeps "
    "wiping at it as though it is the thing that matters. “Thea. We run."
    "”\n\n"
    "“Downhill into open ground, with horses behind us.”\n\n"
    "“Then what?”\n\n"
    "She looks at the wall. It is a bad wall, waist-high, falling apart at the "
    "western end, and it has one virtue: it is the only cover on this hill, and "
    "the men coming up know it.\n\n"
    "“We give it to them,” she says.\n\n"
    "Orrin stares at her. “We give them the wall.”\n\n"
    "“We leave it, they take it, and then they are eleven men crowded "
    "behind a wall that ends. We come along the western side where it falls "
    "down, and they cannot turn.”\n\n"
    "They go. It costs Thea everything she has to walk away from the only cover "
    "on the hill, and she does it slowly, so that it looks like a rout.\n\n"
    "The eleven take the wall, as men do. They crowd in behind it. They are "
    "pleased with themselves.\n\n"
    "It takes Thea and Orrin four minutes to come round the broken end. By then "
    "the wall belongs to nobody, and the hill is quiet, and Orrin is laughing in "
    "a way that will embarrass him later."
)

# Three exemplars spanning first and third person, past and present, and three
# different twists, so the pattern copied is "a complete story" rather than
# "a story like that one".
EXEMPLARS: tuple[dict, ...] = (
    _exemplar(
        {"Voice": "Modern", "Genre": "Tragedy", "POV": "Third-Limited",
         "Tense": "Past", "Setting": "Tavern", "Tone": "Dark", "Cast": "Pair",
         "Characters": "Merrick; Calder", "Length": "Flash",
         "Twist": "Betrayal"},
        _TAVERN,
    ),
    _exemplar(
        {"Voice": "Modern", "Genre": "Ghost", "POV": "First", "Tense": "Past",
         "Setting": "Household", "Tone": "Uneasy", "Cast": "Pair",
         "Characters": "Esme; Bridie", "Length": "Flash",
         "Twist": "Revelation"},
        _HOUSE,
    ),
    _exemplar(
        {"Voice": "Modern", "Genre": "Adventure", "POV": "Third-Limited",
         "Tense": "Present", "Setting": "Battlefield", "Tone": "Rousing",
         "Cast": "Pair", "Characters": "Thea; Orrin", "Length": "Flash",
         "Twist": "Reversal"},
        _HILL,
    ),
)


def sample_tags(rng: random.Random) -> dict[str, str]:
    """Sample one tag combination, weighted toward the thin values."""
    tags = {"Voice": SYNTHETIC_VOICE}
    for name in ("Genre", "POV", "Tense", "Setting", "Tone", "Cast", "Length",
                 "Twist"):
        weights = WEIGHTED_VALUES[name]
        values = [value for value, weight in weights.items() if weight > 0]
        tags[name] = rng.choices(values, weights=[weights[v] for v in values])[0]
    # Omniscient narration means several minds on the page, which needs both a
    # cast to hold them and the length to reach them. Sampled independently it
    # was mostly requested for solo flash fiction and could never be produced.
    if tags["POV"] == "Third-Omniscient":
        tags["Cast"] = "Ensemble"
        tags["Length"] = "Short"
    # cast() treats exactly three names as undecidable and reports Ensemble only
    # at four or more, so asking for three made Ensemble unreachable: it could
    # come back Pair, Solo or nothing, but never what was requested.
    count = {"Solo": 1, "Pair": 2, "Ensemble": 4}[tags["Cast"]]
    tags["Characters"] = "; ".join(rng.sample(NAME_POOL, count))
    return tags


def _request(tags: dict[str, str]) -> str:
    low, high = LENGTH_WORDS[tags["Length"]]
    names = tags["Characters"].replace(";", " and")
    lines = [
        render(tags),
        "",
        f"Write a complete story of about {low} to {high} words, using {names}.",
    ]
    # The two tags every model quietly ignores. Past tense and a single
    # viewpoint are the defaults they fall back into, so they get said twice.
    if tags["Tense"] == "Present":
        lines.append(
            "Write it in the PRESENT tense throughout: 'she walks', not 'she walked'."
        )
    if tags["POV"] == "Third-Omniscient":
        lines.append(
            "Narrate from OUTSIDE all of them, entering the thoughts of at least "
            "three of the named characters by name."
        )
    return "\n".join(lines)


def build_messages(tags: dict[str, str]) -> list[dict[str, str]]:
    """Build the chat messages for one story request."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for exemplar in EXEMPLARS:
        messages.append({"role": "user", "content": _request(exemplar["tags"])})
        messages.append({"role": "assistant", "content": exemplar["story"]})
    messages.append({"role": "user", "content": _request(tags)})
    return messages


def verify(story: str, tags: dict[str, str]) -> tuple[bool, str]:
    """Check a generated story against the tags it was asked for.

    Contradiction rejects; silence does not. The extractors decline whenever the
    evidence is thin, and a short story often gives them too little to work with,
    so treating "no opinion" as failure would throw away most of the set. Only an
    extractor that positively disagrees is grounds for discarding.
    """
    words = story_tagger.words(story)
    if len(words) < 120:
        return False, "too short"
    if len(words) > 1600:
        return False, "too long"
    # A story that stops mid-sentence teaches exactly the failure this whole set
    # exists to fix, so it is worse than no story at all.
    if story.rstrip()[-1:] not in tuple('.!?"”’—'):
        return False, "does not end on a sentence"

    for name, value in (
        ("POV", story_tagger.point_of_view(story)),
        ("Tense", story_tagger.tense(story)),
        ("Setting", story_tagger.setting(story)),
        ("Cast", story_tagger.cast(story)),
    ):
        if value is not None and value != tags[name]:
            return False, f"{name} is {value}, asked for {tags[name]}"

    # The named characters have to actually appear, or the Characters lever is
    # being trained on a lie.
    for name in tags["Characters"].split(";"):
        name = name.strip()
        if name and name not in story:
            return False, f"missing character {name!r}"

    if story_tagger.voice_from_text(story) == "Shakespearean":
        return False, "archaic voice in a Modern story"
    return True, ""
