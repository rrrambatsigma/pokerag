"""Bootstrap the frozen corpus artifacts from Hugging Face.

Rebuilding the corpus needs scraping and a GPU. For a fresh machine, CI or the
API image, download the frozen artifacts instead: Qdrant index, BM25 sidecars,
chunks and manifests. Every file is verified against ``INTEGRITY.sha256`` before
it is used, so a truncated download cannot silently poison a run.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download

from pokerag.core.hashing import sha256_file
from pokerag.core.logging import get_logger

LOGGER = get_logger("pokerag.bootstrap")

DEFAULT_DATASET = "meivawijaya/pokerag-corpus-v3"
DEFAULT_REVISION = "main"
INTEGRITY_NAME = "INTEGRITY.sha256"

DEFAULT_FILES = (INTEGRITY_NAME, "LICENSE")
DEFAULT_PREFIXES = ("manifests/", "chunks/", "reports/")
DEFAULT_EXTRA = ("indexes.tar.gz", "documents.parquet")

INDEX_MARKER = Path("indexes") / "qdrant" / "meta.json"

SNAPSHOT_FILES = {"documents.parquet", "register.parquet", "cleaned.jsonl"}
PREFIX_LAYOUT = {
    "chunks/": "corpus/{snapshot}/chunks",
    "manifests/": "corpus/{snapshot}/manifests",
    "embeddings/": "embeddings",
    "reports/": "reports",
}


@dataclass(frozen=True)
class IntegrityEntry:
    sha256: str
    path: str


def parse_integrity(text: str) -> dict[str, IntegrityEntry]:
    """Parse an ``INTEGRITY.sha256`` document (``sha256sum`` format)."""
    entries: dict[str, IntegrityEntry] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, separator, path = line.partition(" ")
        path = path.strip().lstrip("*")
        if not separator or len(digest) != 64 or not path:
            continue
        entries[path] = IntegrityEntry(sha256=digest.lower(), path=path)
    return entries


def select_paths(
    paths: Iterable[str],
    *,
    include_all: bool = False,
    files: tuple[str, ...] = DEFAULT_FILES,
    prefixes: tuple[str, ...] = DEFAULT_PREFIXES,
    extra: tuple[str, ...] = DEFAULT_EXTRA,
) -> list[str]:
    """Pick the default artifact set: manifests, chunks, reports, index archive."""
    available = set(paths)
    if include_all:
        return sorted(available)

    selected = {path for path in files if path in available}
    selected.update(path for path in extra if path in available)
    for path in available:
        if path.startswith(prefixes):
            selected.add(path)
    return sorted(selected)


def target_relative(path: str, snapshot: str) -> Path | None:
    """Map a dataset path to the layout the pipeline reads.

    ``chunks/`` and the corpus parquet live under
    ``artifacts/corpus/<snapshot>/``; indexes are extracted separately. Files
    with no mapping stay in ``artifacts/raw`` for provenance.
    """
    for prefix, template in PREFIX_LAYOUT.items():
        if path.startswith(prefix):
            return Path(template.format(snapshot=snapshot)) / path[len(prefix) :]
    if path in SNAPSHOT_FILES:
        return Path("corpus") / snapshot / path
    return None


def organize(
    staging: Path, dest: Path, snapshot: str, paths: Iterable[str]
) -> list[Path]:
    """Move verified files from staging into the pipeline layout."""
    moved: list[Path] = []
    for path in paths:
        relative = target_relative(path, snapshot)
        if relative is None:
            continue
        source = staging / path
        if not source.is_file():
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.move(str(source), target)
        moved.append(target)
    return moved


def download_file(
    *,
    dataset: str,
    path: str,
    dest_dir: Path,
    expected_sha256: str | None = None,
    revision: str = DEFAULT_REVISION,
    force: bool = False,
) -> Path:
    """Download one dataset file and verify its digest before returning it."""
    target = Path(
        hf_hub_download(
            repo_id=dataset,
            filename=path,
            repo_type="dataset",
            revision=revision,
            local_dir=str(dest_dir),
            force_download=force,
        )
    )
    if expected_sha256 is not None:
        actual = sha256_file(target)
        if actual != expected_sha256:
            target.unlink(missing_ok=True)
            raise ValueError(
                f"digest mismatch for {path}: expected {expected_sha256}, got {actual}"
            )
    return target


def extract_indexes(archive: Path, dest: Path, *, force: bool = False) -> bool:
    """Extract ``indexes/`` from the archive. Returns False when already present."""
    marker = dest / INDEX_MARKER
    if marker.is_file() and not force:
        LOGGER.info("index sudah ada di %s, lewati ekstraksi", marker)
        return False
    if force:
        shutil.rmtree(dest / "indexes", ignore_errors=True)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:  # Python < 3.11.4 has no extraction filter
            tar.extractall(dest)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify the frozen PokeRAG artifacts"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("artifacts"),
        help="destination root, the archive extracts into <dest>/indexes",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="download every dataset file, including raw HTML and the embedding cache",
    )
    parser.add_argument(
        "--extract", action="store_true", help="extract indexes.tar.gz after verifying"
    )
    parser.add_argument(
        "--snapshot",
        default="pokerag-corpus-v2",
        help="snapshot directory under artifacts/corpus",
    )
    parser.add_argument(
        "--no-organize",
        action="store_true",
        help="keep downloads in artifacts/raw instead of the pipeline layout",
    )
    parser.add_argument("--force", action="store_true", help="redownload and re-extract")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dest: Path = args.dest
    raw = dest / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    integrity_path = download_file(
        dataset=args.dataset,
        path=INTEGRITY_NAME,
        dest_dir=raw,
        revision=args.revision,
        force=args.force,
    )
    entries = parse_integrity(integrity_path.read_text(encoding="utf-8"))
    if not entries:
        raise SystemExit(f"{INTEGRITY_NAME} parsed to zero entries")

    paths = select_paths(entries, include_all=args.all)
    LOGGER.info("%d dari %d file dataset dipilih", len(paths), len(entries))

    for path in paths:
        download_file(
            dataset=args.dataset,
            path=path,
            dest_dir=raw,
            expected_sha256=entries[path].sha256,
            revision=args.revision,
            force=args.force,
        )
        LOGGER.info("ok %s", path)

    if not args.no_organize:
        moved = organize(raw, dest, args.snapshot, paths)
        LOGGER.info("%d file ditempatkan di layout pipeline", len(moved))

    if args.extract:
        archive = raw / "indexes.tar.gz"
        if not archive.is_file():
            raise SystemExit(f"missing archive: {archive}")
        extract_indexes(archive, dest, force=args.force)
        LOGGER.info("index diekstrak ke %s", dest / "indexes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
