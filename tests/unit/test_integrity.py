import pytest

from pokerag.core.integrity import parse_integrity, sha256_file, verify_file
from pokerag.ingest.hf import select_paths


def test_parse_integrity_reads_sha256sum_format() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    text = f"{digest_a}  chunks/fixed.parquet\n{digest_b} *LICENSE\n"

    entries = parse_integrity(text)

    assert set(entries) == {"chunks/fixed.parquet", "LICENSE"}
    assert entries["chunks/fixed.parquet"].sha256 == digest_a
    assert entries["LICENSE"].sha256 == digest_b


def test_parse_integrity_skips_malformed_lines() -> None:
    text = "not-a-digest  file\n\n# comment\n" + "c" * 64 + "  ok.bin\n"

    entries = parse_integrity(text)

    assert set(entries) == {"ok.bin"}


def test_sha256_file_matches_known_value(tmp_path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"pokerag")

    assert sha256_file(target) == "f6fb8320f6647ee8543912af2ee1f7bf73271ba49b1ac814f07f0eb2a5022c13"


def test_verify_file_detects_mismatch(tmp_path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"pokerag")
    entry = parse_integrity(f"{'0' * 64}  payload.bin\n")["payload.bin"]

    assert verify_file(target, entry) is False


def test_verify_file_accepts_match(tmp_path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"pokerag")
    digest = sha256_file(target)
    entry = parse_integrity(f"{digest}  payload.bin\n")["payload.bin"]

    assert verify_file(target, entry) is True


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


@pytest.mark.parametrize("path", ["chunks/fixed.parquet", "manifests/snapshot.json"])
def test_default_set_keeps_expected_prefixes(path: str) -> None:
    assert path in select_paths([path])
