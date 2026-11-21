"""Persistent embedding cache.

Embeddings are expensive and (for hosted models) billable, so they are stored
keyed by ``chunk_id`` plus a hash of the exact text. Re-running the index build
with unchanged chunks therefore makes **zero** API calls, and a chunk whose text
changed is re-embedded instead of silently keeping a stale vector.

The file name carries a hash of the embedding function (model, revision,
dimension, dtype, ``max_seq_length``, prefix). Changing any of them changes
*which vectors are valid*, so it must change the file: a 512-token vector and a
1024-token vector of the same text are different vectors.

Saving once per API batch used to rewrite the whole parquet for every batch,
which is O(n^2) bytes written (tens of GB for a full corpus). Saving every
``save_every`` chunks (or ``save_interval_seconds``) keeps the interruption
window small without starving the GPU on disk I/O.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from pokerag.core.hashing import sha256_text
from pokerag.core.logging import get_logger

logger = get_logger("pokerag.index.cache")


class EmbeddingCache:
    """Parquet-backed vector store keyed by chunk id and text hash."""

    def __init__(
        self,
        path: Path,
        dimension: int,
        save_every: int = 2000,
        save_interval_seconds: float = 60.0,
    ) -> None:
        self.path = path
        self.dimension = dimension
        self.save_every = save_every
        self.save_interval_seconds = save_interval_seconds
        self.saves = 0
        self._pending = 0
        self._last_save = time.monotonic()
        self._vectors: dict[str, list[float]] = {}
        self._hashes: dict[str, str] = {}
        if path.exists():
            frame = pd.read_parquet(path, columns=["chunk_id", "text_sha256", "vector"])
            self._vectors = dict(zip(frame["chunk_id"], frame["vector"]))
            self._hashes = dict(zip(frame["chunk_id"], frame["text_sha256"]))
            logger.info("cache: %s vektor dimuat dari %s", len(self._vectors), path.name)

    def get(self, chunk_id: str, text: str) -> list[float] | None:
        """Return a cached vector only when the text is unchanged."""
        if self._hashes.get(chunk_id) != sha256_text(text):
            return None
        return self._vectors.get(chunk_id)

    def put(self, chunk_id: str, text: str, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"dimensi vektor {len(vector)} != {self.dimension}")
        self._vectors[chunk_id] = vector
        self._hashes[chunk_id] = sha256_text(text)
        self._pending += 1

    def maybe_save(self, force: bool = False) -> bool:
        """Flush when enough new vectors accumulated; returns True when saved."""
        if self._pending == 0:
            return False
        due = (
            self._pending >= self.save_every
            or (time.monotonic() - self._last_save) >= self.save_interval_seconds
        )
        if not (force or due):
            return False
        self.save()
        return True

    def save(self) -> None:
        if self._pending == 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "chunk_id": chunk_id,
                "text_sha256": self._hashes[chunk_id],
                "vector": np.asarray(self._vectors[chunk_id], dtype=np.float32),
            }
            for chunk_id in sorted(self._vectors)
        ]
        pd.DataFrame(rows).to_parquet(self.path, index=False)
        self.saves += 1
        self._pending = 0
        self._last_save = time.monotonic()
        logger.info("cache disimpan: %s vektor (%s)", len(self._vectors), self.path.name)

    def __len__(self) -> int:
        return len(self._vectors)
