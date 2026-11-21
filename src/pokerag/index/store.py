"""Vector store access over the frozen Qdrant local-mode database."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk_id: str
    score: float


class VectorStore(Protocol):
    def search(
        self, collection: str, vector: Sequence[float], limit: int
    ) -> list[ScoredChunk]: ...

    def close(self) -> None: ...


class LocalQdrantStore:
    """Qdrant local mode, the format the artifacts were built in.

    Exact search over 68k vectors is fast enough and keeps the experiment
    reproducible without running a server. The database is locked by the
    process that opens it, so one process at a time.
    """

    def __init__(self, path: Path) -> None:
        import warnings

        from qdrant_client import QdrantClient

        warnings.filterwarnings("ignore", message="Local mode is not recommended.*")
        self._client = QdrantClient(path=str(path))

    def search(
        self, collection: str, vector: Sequence[float], limit: int
    ) -> list[ScoredChunk]:
        response = self._client.query_points(
            collection_name=collection,
            query=list(vector),
            limit=limit,
            with_payload=True,
        )
        hits: list[ScoredChunk] = []
        for point in response.points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id") or str(point.id)
            hits.append(ScoredChunk(str(chunk_id), float(point.score)))
        return hits

    def close(self) -> None:
        self._client.close()
