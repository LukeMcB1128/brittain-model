"""Extract only the wanted Gutenberg books from the bulk text archive.

The bulk archive is ``txt-files.tar.zip``: a zip holding a tar of every plain-text
ebook, about 10GB. Only the books that survive the corpus filter are needed, so
this streams the archive and writes just those, discarding the rest.

Produce the wanted-id list first:

    python3 scripts/prepare/build_story_corpus.py \\
        --catalog data/raw/gutenberg/rdf-files.tar.bz2 \\
        --catalog-only --emit-ids data/raw/brittain-shakespeare/wanted-ids.tsv

Then:

    python3 scripts/prepare/extract_story_texts.py \\
        --archive data/raw/gutenberg/txt-files.tar.zip \\
        --ids data/raw/brittain-shakespeare/wanted-ids.tsv \\
        --output-dir data/raw/gutenberg/text

Text is stored bz2-compressed by default. ``corpus_story.read_book_text`` reads
either form, and compressed text costs a little CPU to save a lot of disk.
"""
from __future__ import annotations

import argparse
import bz2
import re
import sys
import tarfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Members look like "cache/epub/345/pg345.txt" or "1/2/3/1234/1234.txt".
_MEMBER_ID = re.compile(r"(?:^|/)(?:pg)?(\d+)(?:-8|-0)?\.txt$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, help="local txt-files.tar.zip")
    parser.add_argument("--ids", required=True, help="wanted-ids.tsv from --emit-ids")
    parser.add_argument("--output-dir", default="data/raw/gutenberg/text")
    parser.add_argument(
        "--plain", action="store_true", help="write .txt instead of .txt.bz2"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_wanted(path: Path) -> set[int]:
    """Read the id list. Accepts a bare id per line or an id<TAB>category line."""
    wanted: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            field = line.strip().split("\t")[0]
            if not field:
                continue
            try:
                wanted.add(int(field))
            except ValueError:
                continue
    if not wanted:
        raise SystemExit(f"no ids found in {path}")
    return wanted


def member_book_id(name: str) -> int | None:
    match = _MEMBER_ID.search(name)
    return int(match.group(1)) if match else None


def open_inner_tar(archive: Path):
    """Return a streaming handle to the tar inside the zip.

    The archive is a zip wrapping a single tar. Reading the tar as a stream keeps
    peak disk to the archive itself rather than the archive plus its contents.
    """
    zip_handle = zipfile.ZipFile(archive)
    names = [name for name in zip_handle.namelist() if name.endswith(".tar")]
    if not names:
        raise SystemExit(f"no .tar member inside {archive}")
    inner = zip_handle.open(names[0])
    return zip_handle, tarfile.open(fileobj=inner, mode="r|")


def main():
    args = parse_args()
    archive = project_path(args.archive)
    if not archive.exists():
        raise SystemExit(f"archive not found: {archive}")
    wanted = read_wanted(project_path(args.ids))
    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".txt" if args.plain else ".txt.bz2"
    written = skipped = seen = 0
    found: set[int] = set()

    zip_handle, tar = open_inner_tar(archive)
    try:
        for member in tar:
            if not member.isfile():
                continue
            seen += 1
            book_id = member_book_id(member.name)
            if book_id is None or book_id not in wanted or book_id in found:
                continue
            destination = output_dir / f"pg{book_id}{suffix}"
            if destination.exists() and not args.overwrite:
                found.add(book_id)
                skipped += 1
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            if args.plain:
                destination.write_bytes(payload)
            else:
                with bz2.open(destination, "wb") as out:
                    out.write(payload)
            found.add(book_id)
            written += 1
            if written % 1000 == 0:
                print(f"  written {written:,} of {len(wanted):,}", flush=True)
            if args.limit is not None and written >= args.limit:
                break
    finally:
        tar.close()
        zip_handle.close()

    missing = wanted - found
    print(f"archive members scanned: {seen:,}")
    print(f"written: {written:,}")
    print(f"already present: {skipped:,}")
    print(f"wanted but not in archive: {len(missing):,}")
    if missing:
        listing = project_path("data/raw/brittain-shakespeare/missing-ids.txt")
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text(
            "\n".join(str(value) for value in sorted(missing)) + "\n", encoding="utf-8"
        )
        print(f"missing ids written to {listing}")


if __name__ == "__main__":
    main()
