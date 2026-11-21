"""Hashing helpers.

All identifiers in the pipeline are content-addressed so that a snapshot can be
verified after the fact: the same bytes always produce the same id.
"""

from __future__ import annotations

import hashlib
import json
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
