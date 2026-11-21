"""Hashing helpers.

All identifiers in the pipeline are content-addressed so that a snapshot can be
verified after the fact: the same bytes always produce the same id.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` deterministically (sorted keys, no extra spaces)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    """Return the SHA-256 of the canonical JSON encoding of ``obj``."""
    return sha256_text(canonical_json(obj))


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, read in chunks to bound memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
