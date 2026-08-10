import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.corpus_story import read_book_text

# The extractor is a script, not a package module.
_SPEC = importlib.util.spec_from_file_location(
    "extract_story_texts", PROJECT_ROOT / "scripts/prepare/extract_story_texts.py"
)
extract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(extract)


BOOK = (
    "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
    '"I am here," she said.\n\n'
    "He turned and walked to the window.\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
)


def build_archive(path, members):
    """Build a zip wrapping a tar, matching the real txt-files.tar.zip shape."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, payload in members:
            data = payload.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with zipfile.ZipFile(path, "w") as zip_handle:
        zip_handle.writestr("txt-files.tar", buffer.getvalue())


def test_member_book_id_handles_the_real_layouts():
    assert extract.member_book_id("cache/epub/345/pg345.txt") == 345
    assert extract.member_book_id("1/2/3/1234/1234.txt") == 1234
    assert extract.member_book_id("cache/epub/84/pg84-8.txt") == 84
    assert extract.member_book_id("cache/epub/345/pg345.rdf") is None
    assert extract.member_book_id("readme.html") is None


def test_read_wanted_accepts_both_line_forms(tmp_path):
    path = tmp_path / "ids.tsv"
    path.write_text("345\tgutenberg_fiction\n84\n\n1342\tworld_texture\n", encoding="utf-8")
    assert extract.read_wanted(path) == {84, 345, 1342}


def test_read_wanted_rejects_an_empty_list(tmp_path):
    path = tmp_path / "ids.tsv"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        extract.read_wanted(path)


def test_extractor_writes_only_the_wanted_books(tmp_path, monkeypatch, capsys):
    archive = tmp_path / "txt-files.tar.zip"
    build_archive(archive, [
        ("cache/epub/345/pg345.txt", BOOK),
        ("cache/epub/84/pg84.txt", BOOK),
        ("cache/epub/999/pg999.txt", BOOK),
    ])
    ids = tmp_path / "ids.tsv"
    ids.write_text("345\tgutenberg_fiction\n84\tworld_texture\n", encoding="utf-8")
    output = tmp_path / "text"

    monkeypatch.setattr(sys, "argv", [
        "extract_story_texts.py",
        "--archive", str(archive),
        "--ids", str(ids),
        "--output-dir", str(output),
    ])
    extract.main()

    written = sorted(path.name for path in output.iterdir())
    assert written == ["pg345.txt.bz2", "pg84.txt.bz2"]
    # The book nobody asked for is not on disk.
    assert not (output / "pg999.txt.bz2").exists()


def test_extracted_text_round_trips_through_the_corpus_reader(tmp_path, monkeypatch):
    archive = tmp_path / "txt-files.tar.zip"
    build_archive(archive, [("cache/epub/345/pg345.txt", BOOK)])
    ids = tmp_path / "ids.tsv"
    ids.write_text("345\n", encoding="utf-8")
    output = tmp_path / "text"

    monkeypatch.setattr(sys, "argv", [
        "extract_story_texts.py",
        "--archive", str(archive), "--ids", str(ids), "--output-dir", str(output),
    ])
    extract.main()

    text = read_book_text(output / "pg345.txt.bz2")
    assert "PROJECT GUTENBERG" not in text
    assert "I am here" in text


def test_missing_ids_are_reported(tmp_path, monkeypatch, capsys):
    archive = tmp_path / "txt-files.tar.zip"
    build_archive(archive, [("cache/epub/345/pg345.txt", BOOK)])
    ids = tmp_path / "ids.tsv"
    ids.write_text("345\n777\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "extract_story_texts.py",
        "--archive", str(archive), "--ids", str(ids),
        "--output-dir", str(tmp_path / "text"),
    ])
    extract.main()
    captured = capsys.readouterr().out
    assert "wanted but not in archive: 1" in captured


def test_plain_mode_writes_uncompressed(tmp_path, monkeypatch):
    archive = tmp_path / "txt-files.tar.zip"
    build_archive(archive, [("cache/epub/345/pg345.txt", BOOK)])
    ids = tmp_path / "ids.tsv"
    ids.write_text("345\n", encoding="utf-8")
    output = tmp_path / "text"

    monkeypatch.setattr(sys, "argv", [
        "extract_story_texts.py",
        "--archive", str(archive), "--ids", str(ids),
        "--output-dir", str(output), "--plain",
    ])
    extract.main()
    assert (output / "pg345.txt").exists()
    assert read_book_text(output / "pg345.txt").startswith('"I am here,"')
