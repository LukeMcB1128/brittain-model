import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.corpus_story import (
    FICTION_CATEGORY,
    LCC_VOCABULARY,
    LCSH_VOCABULARY,
    TEXTURE_CATEGORY,
    BookRecord,
    FilterConfig,
    TextureConfig,
    book_tags,
    classify,
    dialogue_fraction,
    iter_catalog,
    narrative_verb_ratio,
    parse_rdf,
    read_catalog_directory,
    rejection_reason,
    strip_boilerplate,
    text_rejection_reason,
)

CORPUS_CONFIG = json.loads(
    (PROJECT_ROOT / "configs/data/shakespeare_corpus.json").read_text(encoding="utf-8")
)


def rdf_document(
    book_id=345,
    title="Dracula",
    name="Stoker, Bram",
    birth="1847",
    death="1912",
    language="en",
    rights="Public domain in the USA.",
    lcsh=("Horror tales", "Gothic fiction", "Vampires -- Fiction"),
    lcc=("PR",),
    shelves=("Horror", "Gothic Fiction", "Category: Novels"),
):
    """Build an RDF document matching the real Gutenberg feed structure."""
    def subject(vocabulary, value):
        return (
            "<dcterms:subject><rdf:Description>"
            f'<dcam:memberOf rdf:resource="{vocabulary}"/>'
            f"<rdf:value>{value}</rdf:value>"
            "</rdf:Description></dcterms:subject>"
        )

    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<rdf:RDF xmlns:dcterms="http://purl.org/dc/terms/"',
        ' xmlns:pgterms="http://www.gutenberg.org/2009/pgterms/"',
        ' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
        ' xmlns:dcam="http://purl.org/dc/dcam/">',
        f'<pgterms:ebook rdf:about="ebooks/{book_id}">',
        f"<dcterms:title>{title}</dcterms:title>",
        f"<dcterms:rights>{rights}</dcterms:rights>",
        "<dcterms:language><rdf:Description>"
        f"<rdf:value>{language}</rdf:value>"
        "</rdf:Description></dcterms:language>",
        '<dcterms:creator><pgterms:agent rdf:about="2009/agents/190">',
        f"<pgterms:name>{name}</pgterms:name>",
    ]
    if birth is not None:
        parts.append(
            '<pgterms:birthdate rdf:datatype="http://www.w3.org/2001/XMLSchema#integer"'
            f">{birth}</pgterms:birthdate>"
        )
    if death is not None:
        parts.append(
            '<pgterms:deathdate rdf:datatype="http://www.w3.org/2001/XMLSchema#integer"'
            f">{death}</pgterms:deathdate>"
        )
    parts.append("</pgterms:agent></dcterms:creator>")
    parts.extend(subject(LCSH_VOCABULARY, value) for value in lcsh)
    parts.extend(subject(LCC_VOCABULARY, value) for value in lcc)
    parts.extend(
        "<pgterms:bookshelf><rdf:Description>"
        '<dcam:memberOf rdf:resource="2009/pgterms/Bookshelf"/>'
        f"<rdf:value>{value}</rdf:value>"
        "</rdf:Description></pgterms:bookshelf>"
        for value in shelves
    )
    parts.append("</pgterms:ebook></rdf:RDF>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# RDF parsing
# --------------------------------------------------------------------------- #

def test_parse_rdf_reads_every_field():
    record = parse_rdf(io.StringIO(rdf_document()))
    assert record.book_id == 345
    assert record.title == "Dracula"
    assert record.author == "Stoker, Bram"
    assert record.birth_year == 1847
    assert record.death_year == 1912
    assert record.language == "en"
    assert record.rights == "Public domain in the USA."
    assert record.lcc == ["PR"]
    assert "Horror tales" in record.subjects
    assert "Gothic Fiction" in record.bookshelves


def test_parse_rdf_separates_lcsh_from_lcc():
    # Both live under dcterms:subject and are told apart only by dcam:memberOf.
    # Confusing them would put "PR" into the genre lexicon and headings into the
    # fiction gate.
    record = parse_rdf(io.StringIO(rdf_document()))
    assert "PR" not in record.subjects
    assert "Horror tales" not in record.lcc


def test_parse_rdf_tolerates_missing_author_dates():
    record = parse_rdf(io.StringIO(rdf_document(birth=None, death=None)))
    assert record.birth_year is None and record.death_year is None


def test_parse_rdf_tolerates_unparsable_dates():
    record = parse_rdf(io.StringIO(rdf_document(birth="circa 1840")))
    assert record.birth_year is None


def test_parse_rdf_returns_none_on_junk():
    assert parse_rdf(io.StringIO("<rdf:RDF/>")) is None
    assert parse_rdf(io.StringIO("not xml at all")) is None


def test_read_catalog_directory(tmp_path):
    (tmp_path / "pg345.rdf").write_text(rdf_document(345), encoding="utf-8")
    (tmp_path / "pg84.rdf").write_text(
        rdf_document(84, title="Frankenstein"), encoding="utf-8"
    )
    records = list(read_catalog_directory(tmp_path))
    assert {record.book_id for record in records} == {84, 345}


def test_iter_catalog_streams_a_tar_bz2(tmp_path):
    archive = tmp_path / "rdf-files.tar.bz2"
    with tarfile.open(archive, "w:bz2") as handle:
        for book_id in (345, 84, 1342):
            payload = rdf_document(book_id).encode("utf-8")
            info = tarfile.TarInfo(f"cache/epub/{book_id}/pg{book_id}.rdf")
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    records = list(iter_catalog(archive))
    assert {record.book_id for record in records} == {84, 345, 1342}


def test_iter_catalog_honours_limit(tmp_path):
    archive = tmp_path / "rdf-files.tar.bz2"
    with tarfile.open(archive, "w:bz2") as handle:
        for book_id in range(1, 11):
            payload = rdf_document(book_id).encode("utf-8")
            info = tarfile.TarInfo(f"cache/epub/{book_id}/pg{book_id}.rdf")
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    assert len(list(iter_catalog(archive, limit=4))) == 4


# --------------------------------------------------------------------------- #
# Filter
# --------------------------------------------------------------------------- #

@pytest.fixture()
def filter_config():
    return FilterConfig.from_config(CORPUS_CONFIG)


def record_from(**kwargs):
    return parse_rdf(io.StringIO(rdf_document(**kwargs)))


def test_the_shipped_config_accepts_dracula(filter_config):
    # The end-to-end check that the real configuration admits a real novel.
    assert rejection_reason(record_from(), filter_config) is None


def test_filter_rejects_other_languages(filter_config):
    assert rejection_reason(record_from(language="fr"), filter_config) == "language"


def test_filter_rejects_unlisted_rights(filter_config):
    assert rejection_reason(
        record_from(rights="Copyrighted. Read the copyright notice."), filter_config
    ) == "rights"


def test_filter_rejects_non_literature_lcc(filter_config):
    # QA is mathematics. The LCC gate is what keeps reference works out.
    assert rejection_reason(
        record_from(lcc=("QA",), lcsh=("Fiction",), shelves=()), filter_config
    ) == "lcc_class"


def test_filter_rejects_an_excluded_subject_even_inside_a_literature_class(
    filter_config,
):
    # PR holds criticism and essays as well as novels, which is exactly why the
    # LCC gate alone is not sufficient.
    assert rejection_reason(
        record_from(lcsh=("English fiction -- History and criticism",), shelves=()),
        filter_config,
    ) == "excluded_subject"


def test_filter_rejects_a_literature_class_with_no_narrative_subject(filter_config):
    assert rejection_reason(
        record_from(lcsh=("Sonnets, English",), shelves=()), filter_config
    ) == "no_narrative_subject"


def test_filter_rejects_missing_author_dates(filter_config):
    assert rejection_reason(
        record_from(birth=None, death=None), filter_config
    ) == "no_author_dates"


def test_rejection_reasons_are_reported_in_a_stable_order(filter_config):
    # A book failing several rules is attributed to the first, so funnel counts
    # stay comparable between runs.
    broken = record_from(language="de", rights="Copyrighted.", lcc=("QA",))
    assert rejection_reason(broken, filter_config) == "language"


# --------------------------------------------------------------------------- #
# World-texture path
# --------------------------------------------------------------------------- #

@pytest.fixture()
def texture_config():
    return TextureConfig.from_config(CORPUS_CONFIG)


def test_the_shipped_config_defines_the_texture_path(texture_config):
    assert texture_config is not None
    assert texture_config.maximum_corpus_share <= 0.05


def test_texture_lcc_prefixes_are_tested_longest_first(texture_config):
    # DA must be tested before D, or a two-letter class could never match its own
    # rule. Sorting by descending length is what guarantees it.
    lengths = [len(prefix) for prefix in texture_config.lcc_prefixes]
    assert lengths == sorted(lengths, reverse=True)


def test_texture_admits_a_book_of_legends(filter_config, texture_config):
    # "Tales and Legends of the English Lakes": DA, but narrative.
    record = record_from(
        title="Tales and Legends of the English Lakes",
        lcc=("DA",),
        lcsh=("Legends -- England -- Lake District", "Folklore -- England"),
        shelves=(),
    )
    assert classify(record, filter_config, texture_config)[0] == TEXTURE_CATEGORY


def test_texture_admits_a_first_person_war_memoir(filter_config, texture_config):
    record = record_from(
        title="Sketches From My Life",
        lcc=("D501",),
        lcsh=("Personal narratives", "Adventure and adventurers"),
        shelves=(),
    )
    assert classify(record, filter_config, texture_config)[0] == TEXTURE_CATEGORY


def test_texture_admits_bible_stories(filter_config, texture_config):
    record = record_from(
        title="Wee Ones' Bible Stories",
        lcc=("BS",),
        lcsh=("Bible stories, English",),
        shelves=(),
    )
    assert classify(record, filter_config, texture_config)[0] == TEXTURE_CATEGORY


def test_texture_rejects_a_history_of_the_crusades(filter_config, texture_config):
    # The narrative marker is what keeps campaign chronology out. Crusade
    # *novels* reach the corpus through the fiction path instead.
    record = record_from(
        title="The History of the Crusades", lcc=("D",), lcsh=("Crusades",), shelves=()
    )
    assert classify(record, filter_config, texture_config)[0] is None


def test_texture_rejects_doctrinal_and_reference_work(filter_config, texture_config):
    for subjects in (
        ("Theology, Doctrinal",),
        ("Bible -- Dictionaries",),
        ("English literature -- History and criticism", "Legends"),
    ):
        record = record_from(lcc=("BS",), lcsh=subjects, shelves=())
        assert classify(record, filter_config, texture_config)[0] is None, subjects


def test_texture_does_not_cover_philosophy_or_psychology(texture_config):
    # B, BF, BJ, BT, and BV are argument rather than narrative and must not be
    # reachable through a prefix match.
    for absent in ("BF", "BJ", "BT", "BV"):
        assert not any(
            absent.startswith(prefix) and prefix.startswith("B")
            for prefix in texture_config.lcc_prefixes
        ), absent


def test_a_novel_is_never_classified_as_texture(filter_config, texture_config):
    # The fiction path is tried first, so Dracula stays fiction even though its
    # headings would also satisfy the texture markers.
    assert classify(record_from(), filter_config, texture_config)[0] == FICTION_CATEGORY


def test_classify_reports_the_fiction_reason_for_ordinary_non_fiction(
    filter_config, texture_config
):
    # A mathematics textbook should be attributed to the fiction rule that caught
    # it, not to a texture rule, so the funnel stays readable.
    record = record_from(lcc=("QA",), lcsh=("Mathematics",), shelves=())
    assert classify(record, filter_config, texture_config) == (None, "lcc_class")


def test_texture_path_is_inert_when_unconfigured(filter_config):
    record = record_from(lcc=("DA",), lcsh=("Legends",), shelves=())
    assert classify(record, filter_config, None) == (None, "lcc_class")


def test_texture_floors_are_stricter_than_the_fiction_floors(texture_config):
    fiction = CORPUS_CONFIG["fiction_filter"]
    assert texture_config.minimum_narrative_verb_ratio > float(
        fiction["minimum_narrative_verb_ratio"]
    )


# --------------------------------------------------------------------------- #
# Metadata tags
# --------------------------------------------------------------------------- #

def test_book_tags_from_metadata_alone():
    tags = book_tags(record_from())
    assert tags == {"Voice": "Victorian", "Genre": "Ghost"}


def test_book_tags_omit_what_metadata_cannot_supply():
    tags = book_tags(record_from(birth=None, death=None, lcsh=("Fiction",), shelves=()))
    assert "Voice" not in tags
    assert "Twist" not in tags


# --------------------------------------------------------------------------- #
# Text cleaning and floors
# --------------------------------------------------------------------------- #

RAW_BOOK = """The Project Gutenberg eBook of Dracula

This ebook is for the use of anyone anywhere in the United States.

*** START OF THE PROJECT GUTENBERG EBOOK DRACULA ***

[Transcriber's Note: some obvious typographical errors have been corrected.]

"I am here," said the Count. "You are welcome."

He turned and walked to the window and looked out at the night.



"You must be tired," he said.

*** END OF THE PROJECT GUTENBERG EBOOK DRACULA ***

This eBook is distributed by Project Gutenberg. Please read the license.
"""


def test_strip_boilerplate_removes_header_footer_and_notes():
    cleaned = strip_boilerplate(RAW_BOOK)
    assert "PROJECT GUTENBERG EBOOK" not in cleaned
    assert "Transcriber" not in cleaned
    assert "distributed by Project Gutenberg" not in cleaned
    assert "You are welcome." in cleaned


def test_strip_boilerplate_collapses_blank_runs():
    assert "\n\n\n" not in strip_boilerplate(RAW_BOOK)


def test_strip_boilerplate_is_a_no_op_without_markers():
    assert strip_boilerplate("Plain text.\n") == "Plain text."


def test_dialogue_fraction_separates_fiction_from_a_treatise():
    fiction = strip_boilerplate(RAW_BOOK)
    treatise = "\n".join(
        [
            "The principle may be stated as follows.",
            "It follows that the second case is reducible to the first.",
            "We therefore conclude that the proposition holds generally.",
        ]
    )
    assert dialogue_fraction(fiction) > dialogue_fraction(treatise)
    assert dialogue_fraction(treatise) == 0.0


def test_dialogue_fraction_of_empty_text_is_zero():
    assert dialogue_fraction("") == 0.0
    assert narrative_verb_ratio("") == 0.0


def test_narrative_verb_ratio_separates_fiction_from_a_treatise():
    fiction = "He turned and walked to the window and looked out and waited."
    treatise = "The principle may be stated thus, and the proposition holds."
    assert narrative_verb_ratio(fiction) > narrative_verb_ratio(treatise)


def test_text_floors_reject_and_name_the_failed_rule():
    treatise = "The principle may be stated as follows and the proposition holds.\n" * 5
    assert text_rejection_reason(
        treatise, minimum_dialogue=0.04, minimum_verb_ratio=0.01
    ) == "dialogue_floor"
    # Passing the dialogue floor but not the verb floor names the second rule.
    quoted = '"The principle," he noted, "may be stated as follows."\n' * 5
    assert text_rejection_reason(
        quoted, minimum_dialogue=0.04, minimum_verb_ratio=0.5
    ) == "narrative_verb_floor"


def test_text_floors_accept_narrative_prose():
    fiction = strip_boilerplate(RAW_BOOK)
    assert text_rejection_reason(
        fiction, minimum_dialogue=0.04, minimum_verb_ratio=0.01
    ) is None
