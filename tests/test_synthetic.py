import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain import synthetic
from brittain.tags import CARRIED_TAGS, TAG_ORDER, TAG_VALUES, validate


# --------------------------------------------------------------------------- #
# Exemplars
# --------------------------------------------------------------------------- #

def test_every_exemplar_has_valid_tags():
    for exemplar in synthetic.EXEMPLARS:
        validate(exemplar["tags"])
        assert set(exemplar["tags"]) == set(TAG_ORDER)


def test_every_exemplar_passes_its_own_verifier():
    # The examples carry the design, so an example the filter would reject is a
    # bug in the example, not a curiosity.
    for exemplar in synthetic.EXEMPLARS:
        ok, why = synthetic.verify(exemplar["story"], exemplar["tags"])
        assert ok, f"{exemplar['tags']['Twist']}: {why}"


def test_exemplars_span_the_shapes_they_are_meant_to_teach():
    povs = {exemplar["tags"]["POV"] for exemplar in synthetic.EXEMPLARS}
    tenses = {exemplar["tags"]["Tense"] for exemplar in synthetic.EXEMPLARS}
    twists = {exemplar["tags"]["Twist"] for exemplar in synthetic.EXEMPLARS}
    assert "First" in povs and len(povs) > 1
    assert tenses == {"Past", "Present"}
    assert len(twists) == len(synthetic.EXEMPLARS)


def test_exemplars_are_modern_voice():
    # Synthetic prose is bound to Modern so the generator's flavour cannot leak
    # into the archaic register.
    for exemplar in synthetic.EXEMPLARS:
        assert exemplar["tags"]["Voice"] == synthetic.SYNTHETIC_VOICE


# --------------------------------------------------------------------------- #
# Tag sampling
# --------------------------------------------------------------------------- #

def test_sampled_tags_are_always_valid_and_complete():
    rng = random.Random(0)
    for _ in range(300):
        tags = synthetic.sample_tags(rng)
        validate(tags)
        assert set(tags) == set(TAG_ORDER)


def test_sampling_covers_every_twist_value():
    # Twist has no extractor and appears nowhere in the real corpus, so the
    # synthetic set is its only source of supervision.
    rng = random.Random(1)
    seen = {synthetic.sample_tags(rng)["Twist"] for _ in range(500)}
    assert seen == set(TAG_VALUES["Twist"])


def test_sampling_never_asks_for_long():
    # A Long story would not fit one 1K training row.
    rng = random.Random(2)
    assert all(
        synthetic.sample_tags(rng)["Length"] != "Long" for _ in range(300)
    )


def test_character_count_matches_the_cast_bucket():
    # Four for Ensemble, not three: cast() calls exactly three undecidable and
    # only reports Ensemble at four, so three could never verify as requested.
    rng = random.Random(3)
    expected = {"Solo": 1, "Pair": 2, "Ensemble": 4}
    for _ in range(200):
        tags = synthetic.sample_tags(rng)
        names = [n for n in tags["Characters"].split(";") if n.strip()]
        assert len(names) == expected[tags["Cast"]]


def test_ensemble_requests_can_actually_verify_as_ensemble():
    from brittain import story_tagger

    names = ["Alder", "Bramwell", "Calder", "Dorian"]
    text = " ".join(f"{n} went out. {n} spoke once. {n} waited there." for n in names)
    text = text * 8  # cast() needs at least 150 words before it will decide
    assert story_tagger.cast(text) == "Ensemble"


def test_omniscient_requests_get_the_cast_and_length_they_need():
    # Three named minds do not fit in three hundred words, so an omniscient
    # request sampled alongside Solo and Flash could never be produced.
    rng = random.Random(11)
    seen = 0
    for _ in range(600):
        tags = synthetic.sample_tags(rng)
        if tags["POV"] == "Third-Omniscient":
            seen += 1
            assert tags["Cast"] == "Ensemble"
            assert tags["Length"] == "Short"
    assert seen > 20


def test_the_ignored_tags_are_restated_in_the_request():
    tags = dict(synthetic.EXEMPLARS[0]["tags"])
    tags.update({"Tense": "Present", "POV": "Third-Omniscient", "Cast": "Ensemble",
                 "Characters": "Thea; Orrin; Esme; Calder", "Length": "Short"})
    request = synthetic._request(tags)
    assert "PRESENT tense" in request
    assert "OUTSIDE" in request


def test_thin_values_are_sampled_more_often_than_common_ones():
    rng = random.Random(4)
    settings = [synthetic.sample_tags(rng)["Setting"] for _ in range(3000)]
    # Sea and Court are well covered by the real corpus; Tavern and Household
    # are not, and the weights push at the gap.
    assert settings.count("Tavern") > settings.count("Sea")
    assert settings.count("Household") > settings.count("Court")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

def test_messages_alternate_and_end_on_the_request():
    tags = synthetic.sample_tags(random.Random(5))
    messages = synthetic.build_messages(tags)
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert [m["role"] for m in messages[1:-1]] == ["user", "assistant"] * len(
        synthetic.EXEMPLARS
    )
    assert tags["Characters"].split(";")[0].strip() in messages[-1]["content"]


def test_prompt_stays_small_enough_to_be_cheap():
    # Input is billed per story, so the exemplars have to earn their size.
    messages = synthetic.build_messages(synthetic.sample_tags(random.Random(6)))
    characters = sum(len(m["content"]) for m in messages)
    assert characters < 8000, characters


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def base_tags(**overrides):
    tags = dict(synthetic.EXEMPLARS[0]["tags"])
    tags.update(overrides)
    return tags


def test_verify_rejects_a_contradicted_tag():
    story = synthetic.EXEMPLARS[0]["story"]
    ok, why = synthetic.verify(story, base_tags(Tense="Present"))
    assert not ok and "Tense" in why


def test_verify_tolerates_an_extractor_that_declines():
    # Silence is not failure. Most short stories give the lexicon extractors too
    # little to work with, and rejecting those would discard most of the set.
    story = synthetic.EXEMPLARS[0]["story"]
    ok, _ = synthetic.verify(story, base_tags(Tone="Rousing"))
    assert ok


def test_verify_requires_the_named_characters():
    story = synthetic.EXEMPLARS[0]["story"]
    ok, why = synthetic.verify(story, base_tags(Characters="Merrick; Nobody"))
    assert not ok and "Nobody" in why


def test_verify_rejects_length_extremes():
    tags = base_tags()
    assert not synthetic.verify("Too short.", tags)[0]
    assert not synthetic.verify("word " * 2000, tags)[0]


def test_verify_rejects_archaic_prose_in_a_modern_story():
    archaic = (
        "Thou art come betimes, and thy face doth tell me what thy tongue would "
        "hide. Wherefore dost thou stand so? Nay, prithee, speak. 'Tis not the "
        "hour for silence, and thou hast never kept it well. Methinks the night "
        "hath other business with us both. Hast thou the letter? Give it me, ere "
        "the watch comes round, and let us know the worst that may befall. "
        "Merrick and Calder shall bear it as they may, o'er and done. "
    ) * 3
    ok, why = synthetic.verify(archaic, base_tags(Characters="Merrick; Calder"))
    assert not ok and "archaic" in why


# --------------------------------------------------------------------------- #
# Integration with preparation
# --------------------------------------------------------------------------- #

def test_twist_is_carried_because_nothing_can_extract_it():
    # Preparation re-derives every tag it can from the text and carries only
    # these. Twist has no extractor at all, so without this the synthetic set's
    # main contribution would be silently dropped.
    assert "Twist" in CARRIED_TAGS
    from brittain.story_tagger import extract

    tags = extract(synthetic.EXEMPLARS[0]["story"], token_count=400)
    assert "Twist" not in tags


def test_exemplar_order_is_shuffled_per_request():
    # Fixed order put the one present-tense exemplar last every time, and the
    # model copied the nearest thing it had seen: past-tense requests came back
    # in the present.
    rng = random.Random(0)
    tags = synthetic.sample_tags(rng)
    lasts = {
        synthetic.build_messages(tags, rng)[-2]["content"][:60]
        for _ in range(60)
    }
    assert len(lasts) == len(synthetic.EXEMPLARS)


def test_request_states_the_exact_cast_and_setting():
    tags = dict(synthetic.EXEMPLARS[0]["tags"])
    request = synthetic._request(tags)
    assert "exactly 2 named people" in request
    assert "Give no one else a name" in request
    assert "setting: Tavern" in request


def test_story_ids_come_from_content_not_a_counter():
    # Indices are handed out per attempt, so accepted rows carry sparse ids.
    # Resuming from the accepted count renumbered into a range already used and
    # produced six collisions. A content hash cannot collide on a resume.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_stories", PROJECT_ROOT / "scripts/prepare/generate_stories.py"
    )
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)

    tags = dict(synthetic.EXEMPLARS[0]["tags"])
    first = generate.corpus_row(0, "A story that ends here.", tags)
    # The same text at a different index keeps the same identity...
    again = generate.corpus_row(999, "A story that ends here.", tags)
    assert first["repository"] == again["repository"]
    # ...and different text never shares one.
    other = generate.corpus_row(0, "A different story entirely.", tags)
    assert other["repository"] != first["repository"]
    assert first["repository"].startswith("synthetic/")


def test_genre_is_not_carried_from_an_unverifiable_source():
    # Real books get Genre from their Gutenberg bookshelves, which extract()
    # reads directly. Synthetic stories have none, so carrying it meant trusting
    # the generator: one story tagged Ghost is a darts match with no ghost.
    from brittain.story_tagger import extract

    assert "Genre" not in CARRIED_TAGS
    real = extract("word " * 400, token_count=400, birth_year=1847,
                   death_year=1912, bookshelves=["Horror"])
    assert real["Genre"] == "Ghost"
    synthetic_story = extract("word " * 400, token_count=400)
    assert "Genre" not in synthetic_story


def test_twist_and_voice_are_still_carried():
    # Twist has no extractor at all. Voice is checked a different way: verify()
    # rejects any synthetic story that reads archaic, so Modern is not a bare
    # claim.
    assert CARRIED_TAGS == {"Twist", "Voice"}


def test_the_generator_treats_hopeless_statuses_as_fatal():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_stories", PROJECT_ROOT / "scripts/prepare/generate_stories.py"
    )
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)

    # An exhausted balance produced 2,320,998 wasted attempts because the loop
    # only asked whether the target was met, never why an attempt failed.
    assert 402 in generate.FATAL_STATUS
    for status in (400, 401, 403, 404):
        assert status in generate.FATAL_STATUS
    # Rate limits and server faults stay retryable.
    for status in (429, 500, 503):
        assert status not in generate.FATAL_STATUS
    assert generate.CONSECUTIVE_FAILURE_LIMIT > 0
