"""Qdrant collection management.

Local (on-disk) mode is used so the index runs on a laptop without a container;
the same code accepts a server URL later because only the client constructor
differs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from pokerag.core.logging import get_logger
from pokerag.index.config import QdrantConfig

logger = get_logger("pokerag.index.qdrant")

DISTANCES = {"cosine": Distance.COSINE, "dot": Distance.DOT, "euclid": Distance.EUCLID}


def make_client(cfg: QdrantConfig, path: Path) -> QdrantClient:
    """Open an on-disk client or connect to a server."""
    if cfg.url:
        return QdrantClient(url=cfg.url)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def create_collection(
    client: QdrantClient, name: str, dimension: int, distance: str, recreate: bool = True
) -> None:
    """Create a collection, dropping any previous one with the same name."""
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dimension, distance=DISTANCES[distance]),
    )


def upsert_points(
    client: QdrantClient,
    name: str,
    payloads: Sequence[dict],
    vectors: Sequence[Sequence[float]],
    batch_size: int = 256,
) -> None:
    """Upload points in batches; the point id is the row index, chunk_id is payload."""
    for start in range(0, len(payloads), batch_size):
        chunk = [
            PointStruct(id=index, vector=list(vector), payload=payload)
            for index, (payload, vector) in enumerate(
                zip(payloads[start : start + batch_size], vectors[start : start + batch_size]),
                start=start,
            )
        ]
        client.upsert(collection_name=name, points=chunk)


def search_dense(
    client: QdrantClient,
    name: str,
    vector: Sequence[float],
    top_k: int = 10,
    query_filter=None,
) -> list[ScoredPoint]:
    """Nearest neighbours for one query vector, highest score first."""
    if top_k < 1:
        raise ValueError("top_k harus >= 1")
    return list(
        client.query_points(
            collection_name=name,
            query=list(vector),
            limit=top_k,
            query_filter=query_filter,
        ).points
    )


def collection_count(client: QdrantClient, name: str) -> int:
    """Number of stored points."""
    return client.count(collection_name=name, exact=True).count


def list_collections(client: QdrantClient) -> list[str]:
    """Names of every collection in this database."""
    return sorted(collection.name for collection in client.get_collections().collections)
