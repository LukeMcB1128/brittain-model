"""Turn tagged stories into instruction/response examples.

Pretraining taught the tags as native structure: the model reads a tag block and
writes to it. What it cannot do is take the request in words. This module builds
the layer that maps one to the other, by inverting data that already exists --
every synthetic story was generated from a known tag combination, so the tags
are a specification the story is already known to satisfy.

The frame is:

    <|user|>Write me a short tragedy set in a tavern...<|end_message|>
    <|assistant|><|tags|>[...]<|end_tags|>{story}<|story_end|><|end_message|>

The tag block sits inside the response, not the prompt. That is deliberate. The
model is asked to produce the tags itself and then write to them, which keeps the
lever it already learned in the loop instead of routing around it, and means a
user can still override by supplying tags directly.

Only response tokens are graded. The prompt is masked to -100 so the model reads
the instruction as context and is scored on the story.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .tags import CHARACTER_SEPARATOR, TAG_ORDER, render

# How each tag reads in an instruction. The point is that a user writes "a
# tragedy", not "[Genre: Tragedy]", so the phrasings have to be things a person
# would actually type.
PHRASINGS: dict[str, dict[str, tuple[str, ...]]] = {
    "Genre": {
        "Tragedy": ("a tragedy", "a tragic story", "something tragic"),
        "Comedy": ("a comedy", "something funny", "a comic story"),
        "Romance": ("a romance", "a love story", "something romantic"),
        "Mystery": ("a mystery", "a whodunit", "something mysterious"),
        "Ghost": ("a ghost story", "something haunted", "a supernatural story"),
        "Adventure": ("an adventure", "an adventure story", "something exciting"),
        "Fable": ("a fable", "a moral tale", "something fable-like"),
        "Drama": ("a drama", "a dramatic story", "something dramatic"),
    },
    "Setting": {
        "Tavern": ("in a tavern", "set in a pub", "in an inn"),
        "Castle": ("in a castle", "set in a castle", "inside a fortress"),
        "Sea": ("at sea", "on a ship", "out on the water"),
        "Forest": ("in a forest", "in the woods", "deep in the trees"),
        "City": ("in a city", "set in a city", "on city streets"),
        "Household": ("in a house", "at home", "inside a family home"),
        "Court": ("at court", "in a royal court", "among courtiers"),
        "Road": ("on the road", "while travelling", "on a journey"),
        "Battlefield": ("on a battlefield", "during a battle", "in the middle of a war"),
    },
    "Tone": {
        "Dark": ("dark", "grim"), "Wry": ("wry", "dryly funny"),
        "Tender": ("tender", "gentle"), "Bleak": ("bleak", "hopeless"),
        "Rousing": ("rousing", "stirring"), "Uneasy": ("uneasy", "unsettling"),
    },
    "Twist": {
        "Betrayal": ("someone gets betrayed", "it ends in betrayal",
                     "there is a betrayal"),
        "Revelation": ("a secret comes out", "something hidden is revealed",
                       "there is a revelation"),
        "Reversal": ("everything turns around", "the situation reverses",
                     "there is a reversal"),
        "Death": ("someone dies", "it ends in a death", "there is a death"),
        "Reunion": ("two people find each other again", "it ends in a reunion",
                    "there is a reunion"),
        "None": (),          # nothing to ask for; the tag says "no twist"
    },
    "POV": {
        "First": ("in first person", "written as 'I'"),
        "Third-Limited": ("in third person", "from one character's point of view"),
        "Third-Omniscient": ("with an omniscient narrator",
                             "from everyone's point of view"),
    },
    "Tense": {
        "Past": ("in past tense",), "Present": ("in present tense",),
    },
    # Inline forms only. These are spliced in as adjectives ("a very short
    # comedy"), and a descriptive phrase there reads as nonsense: "a just a few
    # hundred words drama".
    "Length": {
        "Flash": ("very short", "short"),
        "Short": ("short",),
        "Long": ("long", "longer"),
    },
    "Voice": {
        "Shakespearean": ("in Shakespearean English", "in an archaic voice"),
        "Victorian": ("in a Victorian style",),
        "Modern": ("in modern prose", "in a modern voice"),
    },
}

OPENERS = (
    "Write me {body}.",
    "Can you write {body}?",
    "I'd like {body}.",
    "Give me {body}.",
    "Write {body}.",
    "{body_capitalized}, please.",
)


@dataclass(frozen=True)
class SFTExample:
    """One instruction/response pair, before tokenizing."""

    instruction: str
    tags: dict[str, str]
    story: str


def _cast_clause(tags: dict[str, str]) -> str:
    """Describe the cast by name.

    Pronouns were meant to go here -- "Anthony (he/him)" -- to teach the model
    that a name commits you to a gender. Recovering gender from the existing
    stories was measured at 55% and then 37% accurate, which would have taught
    the opposite, so the clause names people and says nothing about them. The
    measured pronoun consistency of the pretrained model is 75%, so this is a
    smaller gap than it looked from one bad sample.
    """
    names = [
        name.strip()
        for name in (tags.get("Characters") or "").split(CHARACTER_SEPARATOR)
        if name.strip()
    ]
    if not names:
        return ""
    if len(names) == 1:
        return f"about {names[0]}"
    return "about " + ", ".join(names[:-1]) + f" and {names[-1]}"


def build_instruction(tags: dict[str, str], rng: random.Random) -> str:
    """Phrase one request in words a person would plausibly type.

    Tags are dropped at random so the model sees requests of every specificity,
    for the same reason pretraining dropped them: a model shown only complete
    nine-tag specifications handles nothing else.
    """
    body: list[str] = []

    genre = tags.get("Genre")
    choices = PHRASINGS["Genre"].get(genre or "", ())
    body.append(rng.choice(choices) if choices else "a story")

    # Voice is deliberately absent: it is Modern on every synthetic story, so
    # asking for it would put the same constant in nearly every instruction.
    for name in ("Length", "Setting", "POV", "Tense"):
        value = tags.get(name)
        if not value or rng.random() < 0.35:
            continue
        options = PHRASINGS[name].get(value, ())
        if not options:
            continue
        phrase = rng.choice(options)
        # Length reads as an adjective on the genre, not a trailing clause.
        if name == "Length":
            body[0] = re.sub(r"^(a|an) ", lambda m: f"{m.group(1)} {phrase} ", body[0])
        else:
            body.append(phrase)

    cast = _cast_clause(tags)
    if cast and rng.random() >= 0.25:
        body.append(cast)

    tone = tags.get("Tone")
    if tone and rng.random() >= 0.4:
        body.append(f"and make it {rng.choice(PHRASINGS['Tone'][tone])}")

    sentence = " ".join(body)
    opener = rng.choice(OPENERS)
    text = opener.format(body=sentence, body_capitalized=sentence[:1].upper() + sentence[1:])

    twist = tags.get("Twist")
    twist_options = PHRASINGS["Twist"].get(twist or "", ())
    if twist_options and rng.random() >= 0.35:
        text += f" I want it to end so that {rng.choice(twist_options)}."
    return text


def example_from_story(
    story: str, tags: dict[str, str], rng: random.Random
) -> SFTExample | None:
    """Build one example, or None when the story cannot supply a clean one."""
    if not story.strip() or not tags:
        return None
    instruction = build_instruction(tags, rng)
    kept = {name: value for name, value in tags.items() if name in TAG_ORDER}
    return SFTExample(instruction=instruction, tags=kept, story=story.strip())


def encode_example(example: SFTExample, tokenizer) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels) with the prompt masked out of the loss."""
    special = tokenizer.special_ids
    prompt = [
        special["<|user|>"],
        *tokenizer.encode(example.instruction),
        special["<|end_message|>"],
        special["<|assistant|>"],
    ]
    response = [
        special["<|tags|>"],
        *tokenizer.encode(render(example.tags)),
        special["<|end_tags|>"],
        *tokenizer.encode(example.story),
        special["<|story_end|>"],
        special["<|end_message|>"],
    ]
    ids = [*prompt, *response]
    # -100 over the prompt: the model reads the instruction and is graded only on
    # the story it produces from it.
    labels = [-100] * len(prompt) + response
    return ids, labels
