"""SHA-256 integrity checks for downloaded artifacts.

The corpus dataset ships an ``INTEGRITY.sha256`` file in the same format as
``sha256sum``: one ``<digest>  <path>`` pair per line. Every artifact is checked
before it is used, so a truncated download cannot silently poison a run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

CHUNK_SIZE = 1 << 20


@dataclass(frozen=True, slots=True)
class IntegrityEntry:
    sha256: str
    path: str


def parse_integrity(text: str) -> dict[str, IntegrityEntry]:
    """Parse an ``INTEGRITY.sha256`` document into a path-keyed mapping."""

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


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, entry: IntegrityEntry) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size == 0:
        return False
    return sha256_file(path) == entry.sha256


def verify_all(
    root: Path, entries: dict[str, IntegrityEntry], paths: Iterable[str]
) -> dict[str, bool]:
    return {path: verify_file(root / path, entries[path]) for path in paths}
