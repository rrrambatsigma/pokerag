"""Artifact bootstrap: integrity parsing, selection and extraction (no network)."""

from __future__ import annotations

import tarfile
from pathlib import Path

from pokerag.bootstrap import (
    IntegrityEntry,
    extract_indexes,
    organize,
    parse_integrity,
    select_paths,
    target_relative,
)
from pokerag.core.hashing import sha256_file


def test_parse_integrity_reads_sha256sum_format() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    text = f"{digest_a}  chunks/fixed.parquet\n{digest_b} *LICENSE\n"

    entries = parse_integrity(text)

    assert set(entries) == {"chunks/fixed.parquet", "LICENSE"}
    assert entries["chunks/fixed.parquet"] == IntegrityEntry(digest_a, "chunks/fixed.parquet")
    assert entries["LICENSE"].sha256 == digest_b


def test_parse_integrity_skips_malformed_lines() -> None:
    text = "not-a-digest  file\n\n# comment\n" + "c" * 64 + "  ok.bin\n"

    assert set(parse_integrity(text)) == {"ok.bin"}


def test_select_paths_default_set() -> None:
    paths = [
        "INTEGRITY.sha256",
        "LICENSE",
        "indexes.tar.gz",
        "documents.parquet",
        "chunks/fixed.parquet",
        "manifests/snapshot.json",
        "reports/A1-index.md",
        "raw/html/1.html",
        "embeddings/cache.parquet",
    ]

    selected = select_paths(paths)

    assert "chunks/fixed.parquet" in selected
    assert "manifests/snapshot.json" in selected
    assert "indexes.tar.gz" in selected
    assert "raw/html/1.html" not in selected
    assert "embeddings/cache.parquet" not in selected


def test_select_paths_all_includes_raw() -> None:
    paths = ["raw/html/1.html", "chunks/fixed.parquet"]

    assert set(select_paths(paths, include_all=True)) == set(paths)


def test_sha256_file_matches_known_value(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"pokerag")

    assert sha256_file(target) == "f6fb8320f6647ee8543912af2ee1f7bf73271ba49b1ac814f07f0eb2a5022c13"


def test_extract_indexes_and_skip_second_time(tmp_path: Path) -> None:
    archive = tmp_path / "indexes.tar.gz"
    payload = tmp_path / "meta.json"
    payload.write_text("{}", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="indexes/qdrant/meta.json")

    dest = tmp_path / "artifacts"
    dest.mkdir()

    assert extract_indexes(archive, dest) is True
    assert (dest / "indexes" / "qdrant" / "meta.json").is_file()
    assert extract_indexes(archive, dest) is False


def test_target_relative_layout() -> None:
    assert target_relative("chunks/fixed.parquet", "snap") == Path(
        "corpus/snap/chunks/fixed.parquet"
    )
    assert target_relative("documents.parquet", "snap") == Path("corpus/snap/documents.parquet")
    assert target_relative("manifests/snapshot.json", "snap") == Path(
        "corpus/snap/manifests/snapshot.json"
    )
    assert target_relative("indexes.tar.gz", "snap") is None


def test_organize_moves_verified_files(tmp_path: Path) -> None:
    staging = tmp_path / "raw"
    (staging / "chunks").mkdir(parents=True)
    (staging / "chunks" / "fixed.parquet").write_bytes(b"x")
    (staging / "documents.parquet").write_bytes(b"y")

    dest = tmp_path / "artifacts"
    moved = organize(
        staging,
        dest,
        "snap",
        ["chunks/fixed.parquet", "documents.parquet", "indexes.tar.gz"],
    )

    assert (dest / "corpus" / "snap" / "chunks" / "fixed.parquet").is_file()
    assert (dest / "corpus" / "snap" / "documents.parquet").is_file()
    assert not (staging / "chunks" / "fixed.parquet").exists()
    assert len(moved) == 2
