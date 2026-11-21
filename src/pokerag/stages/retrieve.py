"""Retrieval orchestration: dense, lexical, or hybrid with RRF."""

from __future__ import annotations

from dataclasses import dataclass, field

from pokerag.core.config import settings
from pokerag.index.bm25 import BM25Store
from pokerag.index.catalog import Catalog, Chunk
from pokerag.index.encoder import BgeM3Encoder, QueryEncoder
from pokerag.index.fusion import reciprocal_rank_fusion
from pokerag.index.store import LocalQdrantStore, VectorStore

MODES = ("dense", "bm25", "hybrid")
STRATEGIES = ("fixed", "heading", "hierarchical")


@dataclass(slots=True)
class RetrievalConfig:
    strategy: str = "fixed"
    mode: str = "hybrid"
    top_k: int = 10
    candidates: int = 50
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy: {self.strategy}")
        if self.mode not in MODES:
            raise ValueError(f"unknown mode: {self.mode}")
        if self.top_k <= 0 or self.candidates < self.top_k:
            raise ValueError("require 0 < top_k <= candidates")


@dataclass(frozen=True, slots=True)
class Retrieved:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        config: RetrievalConfig,
        store: VectorStore,
        bm25: BM25Store,
        catalog: Catalog,
        encoder: QueryEncoder | None = None,
    ) -> None:
        self.config = config
        self._store = store
        self._bm25 = bm25
        self._catalog = catalog
        self._encoder = encoder

    @classmethod
    def build(
        cls,
        config: RetrievalConfig,
        *,
        device: str | None = None,
        encoder: QueryEncoder | None = None,
    ) -> Retriever:
        catalog = Catalog.load(config.strategy)
        store = LocalQdrantStore(settings.qdrant_dir)
        bm25 = BM25Store(settings.indexes_dir / "bm25" / config.strategy)
        if encoder is None and config.mode in ("dense", "hybrid"):
            encoder = BgeM3Encoder(device=device)
        return cls(config, store, bm25, catalog, encoder)

    def search(self, query: str) -> list[Retrieved]:
        config = self.config
        rankings: dict[str, list[str]] = {}
        scores: dict[str, float] = {}

        if config.mode in ("dense", "hybrid"):
            if self._encoder is None:
                raise RuntimeError("dense retrieval needs a query encoder")
            vector = self._encoder.encode(query)
            dense_hits = self._store.search(
                f"chunks_{config.strategy}", vector, config.candidates
            )
            rankings["dense"] = [hit.chunk_id for hit in dense_hits]
            scores.update({hit.chunk_id: hit.score for hit in dense_hits})

        if config.mode in ("bm25", "hybrid"):
            bm25_hits = self._bm25.search(query, config.candidates)
            rankings["bm25"] = [hit.chunk_id for hit in bm25_hits]
            scores.update({hit.chunk_id: hit.score for hit in bm25_hits})

        if config.mode == "hybrid":
            fused = reciprocal_rank_fusion(rankings, k=config.rrf_k)
            chosen = fused[: config.top_k]
            results: list[Retrieved] = []
            for hit in chosen:
                chunk = self._catalog.get(hit.chunk_id)
                if chunk is not None:
                    results.append(Retrieved(chunk, hit.score, hit.ranks))
            return results

        ordered = rankings[config.mode][: config.top_k]
        results = []
        for rank, chunk_id in enumerate(ordered, start=1):
            chunk = self._catalog.get(chunk_id)
            if chunk is not None:
                results.append(Retrieved(chunk, scores[chunk_id], {config.mode: rank}))
        return results

    def title(self, doc_id: str) -> str:
        return self._catalog.title(doc_id)

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> Retriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
