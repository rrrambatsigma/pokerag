"""Dense retrieval over one Qdrant collection.

The embedder and the client are created lazily and reused, because loading
bge-m3 costs more than the query itself. A caller can inject both, which is how
the tests run without a GPU or network.
"""

from __future__ import annotations

from typing import Sequence

from qdrant_client import QdrantClient

from pokerag.index.config import IndexConfig
from pokerag.index.embedders import Embedder, make_embedder
from pokerag.index.qdrant_store import make_client, search_dense
from pokerag.retrieval.types import RawHit


class DenseRetriever:
    """Embed a query and return the nearest chunks from ``chunks_<strategy>``."""

    def __init__(
        self,
        cfg: IndexConfig,
        strategy: str,
        embedder: Embedder | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self._cfg = cfg
        self._strategy = strategy
        self._embedder = embedder
        self._client = client
        self._owns_client = client is None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = make_embedder(self._cfg.embedder)
        return self._embedder

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = make_client(self._cfg.qdrant, self._cfg.qdrant_dir)
        return self._client

    def embed_query(self, query: str) -> list[float]:
        return list(self.embedder.embed_query(query))

    def search(self, query: str, top_k: int, query_filter=None) -> list[RawHit]:
        return self.search_vector(self.embed_query(query), top_k, query_filter=query_filter)

    def search_vector(
        self, vector: Sequence[float], top_k: int, query_filter=None
    ) -> list[RawHit]:
        points = search_dense(
            self.client,
            self._cfg.collection_name(self._strategy),
            vector,
            top_k=top_k,
            query_filter=query_filter,
        )
        hits: list[RawHit] = []
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if not chunk_id:
                raise RuntimeError(f"titik {point.id} di koleksi tidak punya chunk_id")
            hits.append(RawHit(chunk_id=str(chunk_id), score=float(point.score), payload=payload))
        return hits

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None
