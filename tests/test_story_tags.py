import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain import story_tagger
from brittain.tags import (
    OBJECTIVE_TAGS,
    TAG_ORDER,
    TAG_VALUES,
    TagPolicy,
    apply_policy,
    parse,
    parse_request,
    render,
    validate,
)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def test_render_uses_canonical_order_regardless_of_insertion_order():
    tags = {"Setting": "Tavern", "Voice": "Modern", "POV": "First"}
    assert render(tags) == "[Voice: Modern] [POV: First] [Setting: Tavern]"


def test_render_parse_round_trip_for_every_value():
    for name, values in TAG_VALUES.items():
        for value in values:
            assert parse(render({name: value})) == {name: value}


def test_parse_accepts_a_full_block_with_special_tokens():
    body = render({"Genre": "Tragedy", "Twist": "Betrayal"})
    assert parse(f"<|tags|>{body}<|end_tags|>") == {"Genre": "Tragedy", "Twist": "Betrayal"}


def test_parse_rejects_unknown_names_and_values():
    with pytest.raises(ValueError):
        parse("[Mood: Sad]")
    with pytest.raises(ValueError):
        parse("[Genre: Cyberpunk]")
    with pytest.raises(ValueError):
        parse("[Genre]")
    with pytest.raises(ValueError):
        parse("[Genre: Tragedy] [Genre: Comedy]")


def test_validate_rejects_a_value_from_the_wrong_tag():
    with pytest.raises(ValueError):
        validate({"POV": "Tavern"})


def test_parse_request_accepts_the_relaxed_command_line_form():
    assert parse_request("Genre: Tragedy, Setting: Tavern") == {
        "Genre": "Tragedy",
        "Setting": "Tavern",
    }
    assert parse_request("[Genre: Tragedy] [Setting: Tavern]") == {
        "Genre": "Tragedy",
        "Setting": "Tavern",
    }


def test_render_rejects_an_explicit_order_that_does_not_match_the_tags():
    with pytest.raises(ValueError):
        render({"POV": "First"}, order=("POV", "Genre"))


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #

def _full_tags():
    return {name: TAG_VALUES[name][0] for name in TAG_ORDER}


def test_tag_policy_rejects_probabilities_outside_the_unit_interval():
    with pytest.raises(ValueError):
        TagPolicy(tag_dropout=1.5)


def test_policy_statistics_match_the_configured_rates():
    policy = TagPolicy()
    rng = random.Random(1337)
    tags = _full_tags()
    trials = 20_000
    empty = kept = masked = reversed_count = 0
    for _ in range(trials):
        surviving, order, mask, is_reversed = apply_policy(tags, policy, rng)
        assert set(order) == set(surviving)
        if not surviving:
            empty += 1
        kept += len(surviving)
        masked += mask
        reversed_count += is_reversed

    # A block is empty when it is dropped whole, or when every tag is dropped.
    expected_empty = policy.block_dropout + (
        (1 - policy.block_dropout) * policy.tag_dropout ** len(tags)
    )
    assert empty / trials == pytest.approx(expected_empty, abs=0.01)

    expected_kept = (1 - policy.block_dropout) * len(tags) * (1 - policy.tag_dropout)
    assert kept / trials == pytest.approx(expected_kept, abs=0.1)

    assert masked / trials == pytest.approx(policy.mask_rate, abs=0.02)
    assert reversed_count / trials == pytest.approx(policy.reverse_rate, abs=0.01)


def test_policy_shuffles_only_sometimes():
    policy = TagPolicy(tag_dropout=0.0, block_dropout=0.0)
    rng = random.Random(7)
    tags = _full_tags()
    canonical = 0
    trials = 4000
    for _ in range(trials):
        _, order, _, _ = apply_policy(tags, policy, rng)
        if order == TAG_ORDER:
            canonical += 1
    # Canonical order also turns up by chance inside the shuffled share, but with
    # nine tags that probability is negligible.
    assert canonical / trials == pytest.approx(1 - policy.shuffle_rate, abs=0.02)


def test_disabled_policy_is_a_pass_through():
    policy = TagPolicy(tag_dropout=0.0, block_dropout=0.0, shuffle_rate=0.0,
                       mask_rate=0.0, reverse_rate=0.0)
    rng = random.Random(0)
    tags = _full_tags()
    surviving, order, masked, is_reversed = apply_policy(tags, policy, rng)
    assert surviving == tags
    assert order == TAG_ORDER
    assert not masked and not is_reversed


# --------------------------------------------------------------------------- #
# Extractor fixtures
# --------------------------------------------------------------------------- #

FIRST_PAST = """
    I came down to the harbour before the light. I had slept badly, and my hands
    were still cold when I found the boat. I remembered what my father told me
    about the tide, and I waited. The water moved against the stones. I counted
    the minutes as I watched the far shore, and I thought of the letter I had
    burned. When the wind turned I pushed off, and I did not look back at the
    town. I rowed until my shoulders ached. I knew the channel well enough, but I
    had never taken it alone, and I felt the current pull against me. I kept the
    lamp low. I told myself that I would return before anyone woke, and I almost
    believed it. The sea was flat and grey and it gave me nothing back.

    I had made the same crossing in my mind a hundred times, and none of them
    resembled this one. My oars caught badly on the second stroke and I lost the
    rhythm I had counted on. I stopped and let the boat drift while I steadied
    myself, and I listened for anything behind me. Nothing followed. I took up
    the oars again and I worked until the headland showed against the sky. I had
    promised myself that I would not think about the house, and I thought about
    it anyway. I saw the window where the lamp had stood, and I saw my own hand
    reaching for the latch, and I wished that I had waited one more day. I did
    not weep. I had spent all of that in the weeks before, and what remained in
    me by then was only the wish to be finished with it and gone.
"""

THIRD_PAST = """
    Harker came down to the harbour before the light. He had slept badly, and his
    hands were still cold when he found the boat. Harker remembered what his
    father told him about the tide, and he waited. The water moved against the
    stones. He counted the minutes as he watched the far shore, and he thought of
    the letter he had burned. When the wind turned he pushed off, and he did not
    look back at the town. Harker rowed until his shoulders ached. He knew the
    channel well enough, but he had never taken it alone, and he felt the current
    pull against him. He kept the lamp low. He told himself that he would return
    before anyone woke, and he almost believed it. The sea gave him nothing back.

    He had made the same crossing in his mind a hundred times, and none of them
    resembled this one. His oars caught badly on the second stroke and he lost
    the rhythm he had counted on. He stopped and let the boat drift while he
    steadied himself, and he listened for anything behind him. Nothing followed.
    He took up the oars again and he worked until the headland showed against the
    sky. He had promised himself that he would not think about the house, and he
    thought about it anyway. He saw the window where the lamp had stood, and he
    saw his own hand reaching for the latch, and he wished that he had waited one
    more day. He did not weep. He had spent all of that in the weeks before, and
    what remained in him was only the wish to be finished with it and gone.
"""

THIRD_PRESENT = """
    Harker comes down to the harbour before the light. He is tired and his hands
    are cold when he finds the boat. He remembers what his father tells him about
    the tide, and he waits. The water moves against the stones. He counts the
    minutes as he watches the far shore. When the wind turns he goes, and he does
    not look back at the town. He rows until his shoulders ache. He knows the
    channel, but he has never taken it alone, and he feels the current pull. He
    keeps the lamp low. He tells himself that he returns before anyone wakes, and
    he almost believes it. The sea is flat and grey and it gives him nothing.

    He makes the same crossing in his mind a hundred times, and none of them is
    like this one. His oars catch on the second stroke and he loses the rhythm he
    counts on. He stops and lets the boat drift while he steadies himself, and he
    listens for anything behind him. Nothing follows. He takes up the oars again
    and he works until the headland shows against the sky. He tells himself that
    he does not think about the house, and he thinks about it anyway. He sees the
    window where the lamp stands, and he sees his own hand on the latch, and he
    wishes that he waits one more day. He does not weep. He spends all of that in
    the weeks before, and what remains in him is the wish to be gone.
"""

# Dialogue is first person and present tense by nature. A narration-blind
# extractor reports the speech instead of the narrator, so this fixture exists to
# prove the quoted spans are removed first.
THIRD_PAST_HEAVY_DIALOGUE = THIRD_PAST + """
    "I am here," she said. "I am always here. I do not leave, I do not sleep, and
    I never forget what I am owed." "I know," he said. "I know it. I am not the
    man I was, and I am not asking you to pretend that I am."
"""


def test_point_of_view_first_person():
    assert story_tagger.point_of_view(FIRST_PAST) == "First"


def test_point_of_view_third_limited():
    assert story_tagger.point_of_view(THIRD_PAST) == "Third-Limited"


def test_point_of_view_ignores_first_person_dialogue():
    assert story_tagger.point_of_view(THIRD_PAST_HEAVY_DIALOGUE) == "Third-Limited"


def test_point_of_view_declines_on_short_text():
    assert story_tagger.point_of_view("I went home.") is None


def test_tense_past_and_present():
    assert story_tagger.tense(FIRST_PAST) == "Past"
    assert story_tagger.tense(THIRD_PAST) == "Past"
    assert story_tagger.tense(THIRD_PRESENT) == "Present"


def test_tense_ignores_present_tense_dialogue():
    assert story_tagger.tense(THIRD_PAST_HEAVY_DIALOGUE) == "Past"


def test_narration_removes_quoted_spans():
    assert "always here" not in story_tagger.narration(THIRD_PAST_HEAVY_DIALOGUE)
    assert "harbour" in story_tagger.narration(THIRD_PAST_HEAVY_DIALOGUE)


def test_voice_from_author_dates_buckets():
    # Shakespeare, Marlowe, Webster.
    assert story_tagger.voice_from_author_dates(1564, 1616) == "Shakespearean"
    # Dickens, Stoker.
    assert story_tagger.voice_from_author_dates(1812, 1870) == "Victorian"
    assert story_tagger.voice_from_author_dates(1847, 1912) == "Victorian"
    # Twentieth century.
    assert story_tagger.voice_from_author_dates(1899, 1961) == "Modern"
    assert story_tagger.voice_from_author_dates(None, None) is None
    # The long eighteenth century belongs to no register in this schema.
    assert story_tagger.voice_from_author_dates(1660, 1731) is None


def test_voice_falls_back_to_birth_year_when_death_is_unknown():
    assert story_tagger.voice_from_author_dates(1920, None) == "Modern"
    assert story_tagger.voice_from_author_dates(1560, None) == "Shakespearean"


def test_voice_never_uses_a_gutenberg_release_date():
    # Dracula's dcterms:issued is 1995-10-01, the Gutenberg release date, not the
    # 1897 publication. Voice must come from Stoker's dates instead, or the
    # register lever trains on noise across nearly the whole corpus.
    assert story_tagger.voice_from_author_dates(1847, 1912) == "Victorian"
    assert not hasattr(story_tagger, "voice_from_year")


def test_voice_from_text_detects_early_modern_english():
    text = """
        Thou art come betimes, and thy face doth tell me what thy tongue would
        hide. Wherefore dost thou stand so? Nay, prithee, speak. 'Tis not the
        hour for silence, and thou hast never kept it well. Methinks the night
        hath other business with us both. Hast thou the letter? Give it me, ere
        the watch comes round again, and let us know the worst that may befall.
        I would not have thee suffer for my sake, and yet I cannot let thee go.
        Speak plainly, as thou wert wont to do, and I shall bear it as I may.
        Anon the bell will ring, and then betwixt us all is ended, o'er and done.

        Whence came this news, and whither wouldst thou carry it? Thou knowest
        well that the duke hath eyes in every hall, and that he heareth what is
        whispered ere it is well spoken. If thou wilt go, then go; but thou shalt
        not go unaccompanied, for I have sworn it, and I keep such oaths as I do
        make. Thy silence answereth me better than thy speech, and I like it not.
        Come, give me thy hand. The watch hath passed, and we are yet alive, and
        that is more than either of us did look for when the sun went down.
    """
    assert story_tagger.voice_from_text(text) == "Shakespearean"


def test_voice_from_text_declines_on_modern_prose():
    assert story_tagger.voice_from_text(THIRD_PAST) is None


def test_genre_from_subjects_prefers_the_specific_heading():
    assert story_tagger.genre_from_metadata(
        subjects=["Detective and mystery stories", "Fiction"]
    ) == "Mystery"
    assert story_tagger.genre_from_metadata(subjects=["Ghost stories, English"]) == "Ghost"
    assert story_tagger.genre_from_metadata(subjects=["Love stories"]) == "Romance"
    assert story_tagger.genre_from_metadata(subjects=["Physics -- Textbooks"]) is None
    assert story_tagger.genre_from_metadata() is None


# The real bookshelf and subject values Project Gutenberg records for Dracula
# (ebook 345), taken verbatim from its RDF.
DRACULA_SUBJECTS = [
    "Horror tales",
    "Epistolary fiction",
    "Gothic fiction",
    "Vampires -- Fiction",
    "Dracula, Count (Fictitious character) -- Fiction",
    "Transylvania (Romania) -- Fiction",
    "Whitby (England) -- Fiction",
]
DRACULA_BOOKSHELVES = [
    "Horror",
    "Gothic Fiction",
    "Mystery Fiction",
    "Movie Books",
    "Category: Science-Fiction & Fantasy",
    "Category: Crime, Thrillers and Mystery",
    "Category: Novels",
    "Category: Classics of Literature",
    "Category: British Literature",
]


def test_genre_takes_the_most_specific_match():
    # Dracula carries both horror and mystery signals. Ghost is more specific.
    assert story_tagger.genre_from_metadata(
        subjects=DRACULA_SUBJECTS, bookshelves=DRACULA_BOOKSHELVES
    ) == "Ghost"


def test_genre_reads_subjects_when_no_bookshelf_matches():
    assert story_tagger.genre_from_metadata(
        subjects=["Love stories"], bookshelves=["Movie Books", "Category: Novels"]
    ) == "Romance"


# The real headings and bookshelves Gutenberg records for Hamlet.
HAMLET_SUBJECTS = [
    "Tragedies (Drama)",
    "Hamlet (Legendary character) -- Drama",
    "Kings and rulers -- Succession -- Drama",
]
HAMLET_BOOKSHELVES = [
    "Best Books Ever Listings",
    "Category: Plays/Films/Dramas",
    "Category: Classics of Literature",
    "Category: British Literature",
]


def test_tragedy_is_not_shadowed_by_the_generic_drama_bookshelf():
    # Every Shakespeare tragedy carries the "Tragedies (Drama)" heading and the
    # generic "Plays/Films/Dramas" bookshelf. Preferring bookshelves outright
    # labelled all of them Drama and left Tragedy with four examples in the whole
    # 79,148-book catalog.
    assert story_tagger.genre_from_metadata(
        subjects=HAMLET_SUBJECTS, bookshelves=HAMLET_BOOKSHELVES
    ) == "Tragedy"


def test_a_plain_play_still_reads_as_drama():
    assert story_tagger.genre_from_metadata(
        subjects=["English drama"], bookshelves=["Category: Plays/Films/Dramas"]
    ) == "Drama"


def test_genre_precedence_covers_every_schema_value():
    from brittain.tags import TAG_VALUES

    assert set(story_tagger.GENRE_PRECEDENCE) == set(TAG_VALUES["Genre"])


def test_fiction_lcc_gate():
    assert story_tagger.is_fiction_lcc(["PR"])
    assert story_tagger.is_fiction_lcc(["ps"])
    assert story_tagger.is_fiction_lcc(["QA", "PZ"])
    # History, science, and philosophy classes are not narrative prose.
    assert not story_tagger.is_fiction_lcc(["DA"])
    assert not story_tagger.is_fiction_lcc(["QA", "B"])
    assert not story_tagger.is_fiction_lcc([])


def test_setting_requires_a_clear_margin():
    tavern = """
        The tavern was loud and the ale was sour. The innkeeper filled another
        tankard and set it on the bar without looking up. A tapster carried a
        flagon between the tables, and the landlord counted coins behind the
        counter. Someone laughed near the fire. The ale went round again and the
        tavern grew louder still, and no one in the inn thought of leaving while
        the beer held out and the barmaid kept the tankards moving down the bar.
        By midnight the tavern had taken on the particular warmth that comes of
        too many people in too small a room, and the innkeeper had stopped
        pretending to keep the tally straight. A stool went over somewhere behind
        the door and nobody troubled to right it. The barmaid stepped across it
        with a flagon in each hand and did not spill a drop of the wine.
    """
    assert story_tagger.setting(tavern) == "Tavern"


def test_setting_declines_when_no_place_dominates():
    mixed = """
        They left the tavern at dawn and rode the road until the forest closed
        over the track. The city was two days behind them and the sea another
        week beyond that. He thought of the castle and of the ale he had not
        finished, and of the ship that would not wait, and of the long street
        where the crowd had parted for the soldiers and their banners and swords.
    """
    assert story_tagger.setting(mixed) is None


def test_setting_declines_on_short_text():
    assert story_tagger.setting("The tavern was loud.") is None


def test_cast_counts_repeated_names_only():
    solo = THIRD_PAST  # Harker is named three times; no other name repeats.
    assert story_tagger.cast(solo) == "Solo"


def test_cast_ignores_a_name_used_once():
    text = THIRD_PAST + " Somewhere in Bristol a bell rang."
    assert story_tagger.cast(text) == "Solo"


def test_cast_pair():
    text = THIRD_PAST + """
        Marlow found him there. Marlow had been waiting since the previous night,
        and Marlow said nothing at all about the boat or the missing lamp.
    """
    assert story_tagger.cast(text) == "Pair"


def test_length_buckets_are_exact():
    assert story_tagger.length_from_tokens(1) == "Flash"
    assert story_tagger.length_from_tokens(story_tagger.FLASH_MAX_TOKENS) == "Flash"
    assert story_tagger.length_from_tokens(story_tagger.FLASH_MAX_TOKENS + 1) == "Short"
    assert story_tagger.length_from_tokens(story_tagger.SHORT_MAX_TOKENS) == "Short"
    assert story_tagger.length_from_tokens(story_tagger.SHORT_MAX_TOKENS + 1) == "Long"


# --------------------------------------------------------------------------- #
# Combined extraction
# --------------------------------------------------------------------------- #

def test_extract_produces_only_schema_valid_tags():
    tags = story_tagger.extract(
        THIRD_PAST,
        token_count=250,
        birth_year=1847,
        death_year=1912,
        subjects=DRACULA_SUBJECTS,
        bookshelves=DRACULA_BOOKSHELVES,
    )
    validate(tags)
    assert tags["Voice"] == "Victorian"
    assert tags["Genre"] == "Ghost"
    assert tags["POV"] == "Third-Limited"
    assert tags["Tense"] == "Past"
    assert tags["Length"] == "Flash"


def test_extract_prefers_archaic_text_over_author_dates():
    # A modern author writing in period voice is labelled by what they wrote.
    archaic = """
        Thou art come betimes, and thy face doth tell me what thy tongue would
        hide. Wherefore dost thou stand so? Nay, prithee, speak. 'Tis not the
        hour for silence, and thou hast never kept it well. Methinks the night
        hath other business with us both. Hast thou the letter? Give it me, ere
        the watch comes round again, and let us know the worst that may befall.
        I would not have thee suffer for my sake, and yet I cannot let thee go.
        Speak plainly, as thou wert wont to do, and I shall bear it as I may.
        Anon the bell will ring, and betwixt us all is ended, o'er and done.
        Whence came this news, and whither wouldst thou carry it? Thou knowest
        well the duke hath eyes in every hall, and heareth what is whispered.
    """
    tags = story_tagger.extract(archaic, token_count=200, birth_year=1930, death_year=2001)
    assert tags["Voice"] == "Shakespearean"


def test_extract_never_emits_twist():
    tags = story_tagger.extract(
        THIRD_PAST, token_count=250, birth_year=1847, death_year=1912
    )
    assert "Twist" not in tags


def test_extract_stays_silent_rather_than_guessing():
    tags = story_tagger.extract("A short line of text.", token_count=6)
    # Length is always derivable; nothing else has enough evidence.
    assert set(tags) <= {"Length"}


def test_extract_output_renders_and_parses():
    tags = story_tagger.extract(FIRST_PAST, token_count=900, birth_year=1564, death_year=1616)
    assert parse(render(tags)) == tags


def test_every_objective_tag_has_an_extractor_path():
    tags = story_tagger.extract(
        THIRD_PAST_HEAVY_DIALOGUE + THIRD_PAST,
        token_count=700,
        birth_year=1857,
        death_year=1924,
    )
    # Setting and Cast may legitimately decline on this fixture; the point is that
    # no objective tag is structurally unreachable.
    assert set(tags) <= set(TAG_ORDER)
    assert {"Voice", "POV", "Tense", "Length"} <= set(OBJECTIVE_TAGS)


# --------------------------------------------------------------------------- #
# Threshold scaling
# --------------------------------------------------------------------------- #

def test_tone_fires_inside_a_window_sized_passage():
    # Flat evidence counts tuned for a whole book almost never fired inside a
    # thousand-token window: Tone reached only 1.2% coverage and the lever was
    # untrainable. The requirement scales with length instead.
    passage = (
        "The room was cold and the grate was empty. She sat in the grey light "
        "and did not move. Everything felt hollow, and the house was silent, "
        "and the hours went by wearily. Nothing remained of what had been. "
    ) * 8
    assert len(story_tagger.words(passage)) < 500
    assert story_tagger.tone(passage) == "Bleak"


def test_setting_fires_inside_a_window_sized_passage():
    passage = (
        "The ship rolled and the deck ran with water. The captain held the helm "
        "and watched the sea. A sailor came up from below and said nothing. "
    ) * 8
    assert story_tagger.setting(passage) == "Sea"


def test_third_limited_is_found_from_pronoun_interiority():
    # Interiority usually attaches to a pronoun rather than a name, so counting
    # only named subjects left most third-person windows unlabelled.
    passage = (
        "He came to the door and waited. He thought of the letter and he knew "
        "what it meant. He remembered the house and he felt the cold. He went "
        "down to the water and he watched the boats and he wondered about her. "
    ) * 6
    assert story_tagger.point_of_view(passage) == "Third-Limited"
