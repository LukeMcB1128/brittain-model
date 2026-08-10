import random
import sys
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.data_story import (
    EncodedStory,
    StorySettings,
    encode_story,
    pack_story_segments,
    split_chapters,
    window_text,
)
from brittain.tags import TagPolicy, parse
from brittain.tokenizer_story import STORY_SPECIAL_TOKENS, StoryTokenizer


@pytest.fixture()
def tokenizer(tmp_path):
    output = tmp_path / "tokenizer.json"
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=1024,
        special_tokens=list(STORY_SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    samples = [
        "The wooden door slammed, and the room went very quiet.\n",
        '"I am here," she said. "I have always been here."\n',
        "Thou art come betimes, and thy face doth tell me what thy tongue hides.\n",
        "[Voice: Modern] [Genre: Tragedy] [Setting: Tavern] [Length: Flash]\n",
        "He turned and walked to the window and looked out at the night.\n",
    ] * 120
    tok.train_from_iterator(samples, trainer=trainer)
    tok.save(str(output))
    return StoryTokenizer(output)


PARAGRAPH = (
    "He turned and walked to the window and looked out at the night. "
    "The water moved against the stones and he counted the minutes. "
    '"You must be tired," he said, and she did not answer him at all.'
)


def book(chapters=3, paragraphs=6):
    parts = []
    for index in range(chapters):
        parts.append(f"CHAPTER {'I' * (index + 1)}")
        parts.append("\n\n".join([PARAGRAPH] * paragraphs))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Chapter splitting
# --------------------------------------------------------------------------- #

def test_split_chapters_keeps_each_heading_with_its_body():
    chunks = split_chapters(book(chapters=3, paragraphs=1))
    assert len(chunks) == 3
    assert chunks[0].startswith("CHAPTER I")
    assert PARAGRAPH in chunks[0]


def test_split_chapters_keeps_front_matter_before_the_first_heading():
    text = "A preface paragraph.\n\nCHAPTER I\n\n" + PARAGRAPH
    chunks = split_chapters(text)
    assert chunks[0] == "A preface paragraph."
    assert chunks[1].startswith("CHAPTER I")


def test_split_chapters_returns_the_whole_text_when_there_are_no_headings():
    assert split_chapters(PARAGRAPH) == [PARAGRAPH]


def test_a_mention_of_chapter_inside_prose_does_not_split():
    text = "She read the chapter II again and closed the book."
    assert split_chapters(text) == [text]


def test_act_and_scene_headings_split_too():
    text = "ACT I\n\n" + PARAGRAPH + "\n\nSCENE II\n\n" + PARAGRAPH
    assert len(split_chapters(text)) == 2


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #

def test_windows_respect_the_maximum(tokenizer):
    settings = StorySettings(block_size=1024, target_tokens=120, minimum_tokens=40,
                             maximum_tokens=200)
    windows = window_text(book(), tokenizer, settings)
    assert windows
    for window in windows:
        assert len(tokenizer.encode(window)) <= settings.maximum_tokens


def test_windows_meet_the_minimum(tokenizer):
    settings = StorySettings(block_size=1024, target_tokens=120, minimum_tokens=40,
                             maximum_tokens=200)
    for window in window_text(book(), tokenizer, settings):
        assert len(tokenizer.encode(window)) >= settings.minimum_tokens


def test_windowing_never_cuts_mid_sentence(tokenizer):
    settings = StorySettings(block_size=1024, target_tokens=60, minimum_tokens=40,
                             maximum_tokens=120)
    for window in window_text(book(), tokenizer, settings):
        assert window.rstrip()[-1] in '.!?"'


def test_an_oversized_paragraph_is_split_at_sentences(tokenizer):
    giant = " ".join([PARAGRAPH] * 20)
    settings = StorySettings(block_size=1024, target_tokens=80, minimum_tokens=40,
                             maximum_tokens=120)
    windows = window_text(giant, tokenizer, settings)
    assert len(windows) > 1
    for window in windows:
        assert window.rstrip()[-1] in '.!?"'


def test_windowing_keeps_most_of_the_book(tokenizer):
    # The failure this guards against is silent truncation: encode_document would
    # have kept one block of a novel and dropped the rest.
    text = book(chapters=4, paragraphs=8)
    settings = StorySettings(block_size=1024, target_tokens=120, minimum_tokens=40,
                             maximum_tokens=250)
    windows = window_text(text, tokenizer, settings)
    kept = sum(len(tokenizer.encode(window)) for window in windows)
    assert kept > 0.9 * len(tokenizer.encode(text))


def test_settings_reject_impossible_sizes():
    with pytest.raises(ValueError):
        StorySettings(minimum_tokens=8)
    with pytest.raises(ValueError):
        StorySettings(target_tokens=100, minimum_tokens=200)
    with pytest.raises(ValueError):
        StorySettings(target_tokens=500, maximum_tokens=100)


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

DETERMINISTIC = TagPolicy(tag_dropout=0.0, block_dropout=0.0, shuffle_rate=0.0,
                          mask_rate=0.0, reverse_rate=0.0)


def settings_with(policy, block_size=1024):
    return StorySettings(block_size=block_size, target_tokens=120,
                         minimum_tokens=40, maximum_tokens=400, policy=policy)


def test_encoded_story_has_the_expected_frame(tokenizer):
    rng = random.Random(0)
    story = encode_story(PARAGRAPH, {"Genre": "Tragedy"}, tokenizer,
                         settings_with(DETERMINISTIC), rng)
    assert story.ids[0] == tokenizer.story_start
    assert story.ids[-2] == tokenizer.story_end
    assert story.ids[-1] == tokenizer.eot
    assert tokenizer.tags_start in story.ids and tokenizer.tags_end in story.ids


def test_tag_block_round_trips_through_the_tokenizer(tokenizer):
    rng = random.Random(0)
    tags = {"Voice": "Modern", "Genre": "Tragedy", "Setting": "Tavern"}
    story = encode_story(PARAGRAPH, tags, tokenizer, settings_with(DETERMINISTIC), rng)
    start = story.ids.index(tokenizer.tags_start)
    end = story.ids.index(tokenizer.tags_end)
    body = tokenizer.decode(story.ids[start + 1: end])
    recovered = parse(body)
    assert recovered == {**tags, "Length": recovered["Length"]}


def test_length_is_recomputed_from_the_window_not_the_caller(tokenizer):
    rng = random.Random(0)
    # A caller claiming "Long" for a short window must be overruled: Length
    # describes the window, which is only known after windowing.
    story = encode_story(PARAGRAPH, {"Length": "Long"}, tokenizer,
                         settings_with(DETERMINISTIC), rng)
    assert story.tags["Length"] == "Flash"


def test_masked_tag_block_is_excluded_from_the_loss(tokenizer):
    rng = random.Random(0)
    policy = TagPolicy(tag_dropout=0.0, block_dropout=0.0, shuffle_rate=0.0,
                       mask_rate=1.0, reverse_rate=0.0)
    story = encode_story(PARAGRAPH, {"Genre": "Tragedy"}, tokenizer,
                         settings_with(policy), rng)
    start = story.ids.index(tokenizer.tags_start)
    end = story.ids.index(tokenizer.tags_end)
    assert not any(story.supervised[start: end + 1])
    # The story itself stays supervised.
    assert all(story.supervised[end + 1:])
    assert story.supervised[0]


def test_unmasked_tag_block_is_supervised(tokenizer):
    rng = random.Random(0)
    story = encode_story(PARAGRAPH, {"Genre": "Tragedy"}, tokenizer,
                         settings_with(DETERMINISTIC), rng)
    assert all(story.supervised)


def test_reversed_examples_put_the_block_after_the_story(tokenizer):
    rng = random.Random(0)
    policy = TagPolicy(tag_dropout=0.0, block_dropout=0.0, shuffle_rate=0.0,
                       mask_rate=0.0, reverse_rate=1.0)
    story = encode_story(PARAGRAPH, {"Genre": "Tragedy"}, tokenizer,
                         settings_with(policy), rng)
    assert story.tags_reversed
    # The block sits between the story and the closing sentinels.
    assert story.ids.index(tokenizer.tags_start) > 1
    assert story.ids.index(tokenizer.tags_end) == len(story.ids) - 3


def test_a_dropped_block_produces_an_unconditional_example(tokenizer):
    rng = random.Random(0)
    policy = TagPolicy(tag_dropout=0.0, block_dropout=1.0, shuffle_rate=0.0,
                       mask_rate=0.0, reverse_rate=0.0)
    story = encode_story(PARAGRAPH, {"Genre": "Tragedy"}, tokenizer,
                         settings_with(policy), rng)
    assert story.tags == {}
    assert tokenizer.tags_start not in story.ids
    assert not story.tags_masked


def test_the_story_is_trimmed_but_the_tag_block_never_is(tokenizer):
    rng = random.Random(0)
    tags = {"Voice": "Modern", "Genre": "Tragedy", "Setting": "Tavern",
            "Tone": "Dark", "POV": "First"}
    long_text = " ".join([PARAGRAPH] * 40)
    settings = settings_with(DETERMINISTIC, block_size=128)
    story = encode_story(long_text, tags, tokenizer, settings, rng)
    assert len(story.ids) <= settings.block_size + 1
    start = story.ids.index(tokenizer.tags_start)
    end = story.ids.index(tokenizer.tags_end)
    assert parse(tokenizer.decode(story.ids[start + 1: end]))


def test_encoding_fails_loudly_when_the_block_cannot_fit(tokenizer):
    rng = random.Random(0)
    tags = {"Voice": "Shakespearean", "Genre": "Tragedy", "Setting": "Battlefield",
            "Tone": "Dark", "POV": "Third-Omniscient", "Cast": "Ensemble"}
    with pytest.raises(ValueError):
        encode_story(PARAGRAPH, tags, tokenizer, settings_with(DETERMINISTIC, 24), rng)


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #

def make_segment(ids, supervised=None):
    return EncodedStory(
        ids=list(ids),
        supervised=list(supervised) if supervised else [True] * len(ids),
        repository="gutenberg/345", path="Dracula", source="gutenberg_fiction",
        tags={}, tags_masked=False, tags_reversed=False,
    )


def test_encoded_story_rejects_a_mismatched_mask():
    with pytest.raises(ValueError):
        make_segment([1, 2, 3], [True, True])


def test_packing_shapes_and_no_split_across_rows():
    block_size = 16
    segments = [make_segment(range(1, 8)) for _ in range(4)]
    inputs, labels, spans = pack_story_segments(segments, block_size, pad_id=0)
    assert inputs.shape == labels.shape == (len(spans), block_size)
    for row in spans:
        for span in row:
            assert span["end"] - span["start"] == 7


def test_padding_becomes_ignore_index():
    inputs, labels, _ = pack_story_segments(
        [make_segment(range(1, 6))], block_size=16, pad_id=0
    )
    # Five tokens yield four supervised predictions; the rest is padding.
    assert (labels[0][:4] != -100).all()
    assert (labels[0][4:] == -100).all()


def test_masked_span_becomes_ignore_index():
    supervised = [True, False, False, True, True]
    inputs, labels, _ = pack_story_segments(
        [make_segment([5, 6, 7, 8, 9], supervised)], block_size=8, pad_id=0
    )
    # Labels are the next token, so label[i] follows supervised[i + 1].
    assert labels[0][0] == -100
    assert labels[0][1] == -100
    assert labels[0][2] == 8
    assert labels[0][3] == 9


def test_labels_are_inputs_shifted_by_one_where_supervised():
    segment = make_segment([3, 4, 5, 6])
    inputs, labels, _ = pack_story_segments([segment], block_size=8, pad_id=0)
    assert list(inputs[0][:3]) == [3, 4, 5]
    assert list(labels[0][:3]) == [4, 5, 6]


def test_packer_rejects_an_oversized_segment():
    with pytest.raises(ValueError):
        pack_story_segments([make_segment(range(40))], block_size=8, pad_id=0)


def test_packing_is_dtype_compatible_with_the_trainer():
    inputs, labels, _ = pack_story_segments(
        [make_segment(range(1, 6))], block_size=8, pad_id=0
    )
    assert inputs.dtype == np.uint16
    assert labels.dtype == np.int32


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_book_to_packed_rows(tokenizer):
    rng = random.Random(1337)
    settings = StorySettings(block_size=256, target_tokens=100, minimum_tokens=40,
                             maximum_tokens=200, policy=TagPolicy())
    windows = window_text(book(chapters=3, paragraphs=6), tokenizer, settings)
    segments = [
        encode_story(window, {"Voice": "Modern", "Genre": "Tragedy"}, tokenizer,
                     settings, rng, repository="gutenberg/345", path="Dracula",
                     source="gutenberg_fiction")
        for window in windows
    ]
    inputs, labels, spans = pack_story_segments(segments, settings.block_size,
                                                tokenizer.pad)
    assert inputs.shape[1] == settings.block_size
    assert inputs.shape[0] == len(spans)
    # Something is always supervised, and something is always ignored (padding).
    assert (labels != -100).any()
    assert all(span["repository"] == "gutenberg/345" for row in spans for span in row)


def test_policy_produces_a_mix_of_presentations(tokenizer):
    rng = random.Random(7)
    settings = settings_with(TagPolicy())
    tags = {"Voice": "Modern", "Genre": "Tragedy", "Setting": "Tavern",
            "POV": "First", "Tone": "Dark"}
    stories = [
        encode_story(PARAGRAPH, tags, tokenizer, settings, rng) for _ in range(400)
    ]
    # Partial specifications must occur, or the model only ever learns the full
    # nine-tag form and degrades on a two-tag request.
    sizes = {len(story.tags) for story in stories}
    assert len(sizes) > 3
    assert any(story.tags_masked for story in stories)
    assert any(story.tags_reversed for story in stories)
    assert any(not story.tags for story in stories)
