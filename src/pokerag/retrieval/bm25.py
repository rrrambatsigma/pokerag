"""BM25 retrieval over the ``bm25s`` index built by the index stage.

Nothing is rebuilt here: ``load_bm25`` reads the same files a previous build
wrote. The wrapper only clamps ``k`` to the corpus size and turns
``(chunk_id, score)`` pairs into :class:`RawHit`.
"""

from __future__ import annotations

from pokerag.index.bm25_index import load_bm25, search_bm25
from pokerag.index.config import IndexConfig
from pokerag.retrieval.types import RawHit


class Bm25Retriever:
    """Load one strategy's BM25 index once, query it many times."""

    def __init__(self, cfg: IndexConfig, strategy: str) -> None:
        self._cfg = cfg
        self._strategy = strategy
        self._retriever, self._chunk_ids = load_bm25(cfg.bm25_dir(strategy))

    @property
    def chunk_ids(self) -> list[str]:
        return list(self._chunk_ids)

    def search(self, query: str, top_k: int) -> list[RawHit]:
        if top_k < 1:
            raise ValueError("top_k harus >= 1")
        k = min(top_k, len(self._chunk_ids))
        pairs = search_bm25(self._retriever, self._chunk_ids, query, k=k)
        return [
            RawHit(chunk_id=str(chunk_id), score=float(score), payload={})
            for chunk_id, score in pairs
        ]

    def close(self) -> None:  # symmetry with DenseRetriever
        return None
