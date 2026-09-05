import random
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.sft_story import (
    PHRASINGS,
    SFTExample,
    build_instruction,
    encode_example,
    example_from_story,
)
from brittain.tokenizer_story import STORY_SPECIAL_TOKENS, StoryTokenizer


@pytest.fixture()
def tokenizer(tmp_path):
    output = tmp_path / "tokenizer.json"
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=1024, special_tokens=list(STORY_SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    )
    tok.train_from_iterator(
        ["Write me a short tragedy set in a tavern about Alice and Edmund.\n",
         "The wooden door slammed and the room went very quiet.\n"] * 200,
        trainer=trainer,
    )
    tok.save(str(output))
    return StoryTokenizer(output)


TAGS = {
    "Voice": "Modern", "Genre": "Tragedy", "POV": "First", "Tense": "Past",
    "Setting": "Tavern", "Tone": "Bleak", "Cast": "Pair",
    "Characters": "Alice; Edmund", "Length": "Short", "Twist": "Betrayal",
}


def test_the_instruction_asks_for_the_genre_in_words():
    # Matched on the word, not the whole phrase: Length splices an adjective
    # into it, so "a tragic story" becomes "a short tragic story".
    rng = random.Random(0)
    for _ in range(20):
        instruction = build_instruction(TAGS, rng)
        assert "tragedy" in instruction or "tragic" in instruction
        assert "[Genre:" not in instruction


def test_length_splices_in_as_an_adjective():
    # "a just a few hundred words drama" was the bug: a descriptive phrase read
    # as an adjective is nonsense, so only inline forms belong in the table.
    for value in ("Flash", "Short", "Long"):
        for phrase in PHRASINGS["Length"][value]:
            assert " " not in phrase or phrase.startswith("very "), phrase


def test_voice_is_never_requested():
    # It is Modern on every synthetic story, so asking for it would put the same
    # constant into nearly every instruction and teach nothing.
    rng = random.Random(1)
    for _ in range(60):
        instruction = build_instruction(TAGS, rng)
        for phrase in PHRASINGS["Voice"]["Modern"]:
            assert phrase not in instruction


def test_dropping_tags_produces_requests_of_varying_specificity():
    rng = random.Random(2)
    lengths = {len(build_instruction(TAGS, rng)) for _ in range(80)}
    # A model shown only fully specified requests handles nothing else.
    assert len(lengths) > 10


def test_the_prompt_is_masked_and_the_response_is_graded(tokenizer):
    example = SFTExample(
        instruction="Write me a short tragedy set in a tavern.",
        tags={"Genre": "Tragedy", "Setting": "Tavern"},
        story="The wooden door slammed and the room went very quiet.",
    )
    ids, labels = encode_example(example, tokenizer)
    assert len(ids) == len(labels)

    assistant = ids.index(tokenizer.special_ids["<|assistant|>"])
    # Nothing up to and including the assistant turn contributes gradient.
    assert all(label == -100 for label in labels[: assistant + 1])
    # Everything after it does.
    assert all(label != -100 for label in labels[assistant + 1:])
    # And where graded, the label is the input itself: the shift into
    # next-token targets happens in the packer, not here.
    assert labels[assistant + 1:] == ids[assistant + 1:]


def test_the_response_opens_with_the_tag_block(tokenizer):
    # The model produces the tags and then writes to them, which keeps the lever
    # pretraining taught in the loop rather than routing around it.
    example = SFTExample("Write me a tragedy.", {"Genre": "Tragedy"}, "It ended badly.")
    ids, _ = encode_example(example, tokenizer)
    assistant = ids.index(tokenizer.special_ids["<|assistant|>"])
    assert ids[assistant + 1] == tokenizer.special_ids["<|tags|>"]


def test_a_story_without_tags_yields_no_example():
    assert example_from_story("Some prose.", {}, random.Random(0)) is None
    assert example_from_story("", TAGS, random.Random(0)) is None
