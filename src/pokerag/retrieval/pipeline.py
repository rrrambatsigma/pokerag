"""One query in, one ranked list out.

The pipeline runs dense and/or BM25 for the same query, fuses the rankings with
RRF, then joins every hit to the chunk text so the result can be cited. It
never writes anything: the Qdrant collections and the BM25 files are read-only
inputs produced by the index stage.
"""

from __future__ import annotations

from typing import Sequence

from qdrant_client import QdrantClient

from pokerag.core.logging import get_logger
from pokerag.index.config import STRATEGIES, IndexConfig
from pokerag.retrieval.bm25 import Bm25Retriever
from pokerag.retrieval.checks import verify_bm25_alignment, verify_manifest, verify_points
from pokerag.retrieval.config import RetrievalConfig
from pokerag.retrieval.corpus import load_chunks
from pokerag.retrieval.dense import DenseRetriever
from pokerag.retrieval.fuse import rrf
from pokerag.retrieval.types import ChunkText, Hit, RawHit, MODES

logger = get_logger("pokerag.retrieval.pipeline")


class Retriever:
    """Query-ready view over one strategy's dense and BM25 indexes."""

    def __init__(
        self,
        cfg: RetrievalConfig,
        index_cfg: IndexConfig | None = None,
        strategy: str | None = None,
        embedder=None,
        client: QdrantClient | None = None,
        table: dict[str, ChunkText] | None = None,
    ) -> None:
        self.cfg = cfg
        self.index = index_cfg or cfg.index_config()
        self.strategy = strategy or cfg.strategy
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategi tidak dikenal: {self.strategy}; pilihan: {list(STRATEGIES)}")

        self.manifest = verify_manifest(self.index, self.strategy)
        self.table = table if table is not None else load_chunks(self.index, self.strategy)
        self._dense = DenseRetriever(self.index, self.strategy, embedder=embedder, client=client)
        self._bm25: Bm25Retriever | None = None
        self._dense_checked = False
        self._bm25_checked = False

    # -- lazily opened retrievers -------------------------------------------

    def _dense_retriever(self) -> DenseRetriever:
        if not self._dense_checked:
            verify_points(self._dense.client, self.index, self.strategy, len(self.table))
            self._dense_checked = True
        return self._dense

    def _bm25_retriever(self) -> Bm25Retriever:
        if self._bm25 is None:
            verify_bm25_alignment(self.index, self.strategy, list(self.table))
            self._bm25 = Bm25Retriever(self.index, self.strategy)
            if self._bm25.chunk_ids != list(self.table):
                # Belt and braces: the loader and the guard must agree.
                raise RuntimeError("urutan chunk id BM25 berubah setelah dimuat")
            self._bm25_checked = True
        return self._bm25

    # -- query ---------------------------------------------------------------

    def search(
        self, query: str, mode: str = "hybrid", top_k: int | None = None
    ) -> list[Hit]:
        """Rank chunks for ``query``; ``mode`` is dense, bm25 or hybrid."""
        if not query.strip():
            raise ValueError("query kosong")
        if mode not in MODES:
            raise ValueError(f"mode tidak dikenal: {mode}; pilihan: {list(MODES)}")
        k = self.cfg.top_k if top_k is None else top_k
        if k < 1:
            raise ValueError("top_k harus >= 1")

        dense_hits: list[RawHit] = []
        bm25_hits: list[RawHit] = []
        if mode in ("dense", "hybrid"):
            dense_hits = self._dense_retriever().search(query, self.cfg.top_dense)
        if mode in ("bm25", "hybrid"):
            bm25_hits = self._bm25_retriever().search(query, self.cfg.top_bm25)

        if mode == "hybrid":
            return self._from_fusion(dense_hits, bm25_hits, k)
        if mode == "dense":
            return self._from_single(dense_hits, "dense", k)
        return self._from_single(bm25_hits, "bm25", k)

    def _from_single(self, hits: Sequence[RawHit], source: str, top_k: int) -> list[Hit]:
        return [
            self._hit(
                hit.chunk_id,
                score=hit.score,
                rank=rank,
                source=source,
                ranks={source: rank},
            )
            for rank, hit in enumerate(hits[:top_k], start=1)
        ]

    def _from_fusion(
        self, dense_hits: Sequence[RawHit], bm25_hits: Sequence[RawHit], top_k: int
    ) -> list[Hit]:
        fused = rrf(
            [[hit.chunk_id for hit in dense_hits], [hit.chunk_id for hit in bm25_hits]],
            k=self.cfg.rrf_k,
            weights=self.cfg.weights,
            sources=["dense", "bm25"],
        )
        return [
            self._hit(item.chunk_id, score=item.score, rank=rank, source="hybrid", ranks=item.ranks)
            for rank, item in enumerate(fused[:top_k], start=1)
        ]

    def _hit(
        self,
        chunk_id: str,
        score: float,
        rank: int,
        source: str,
        ranks: dict[str, int],
    ) -> Hit:
        chunk = self.table.get(chunk_id)
        if chunk is None:
            raise RuntimeError(
                f"hasil retrieval memuat chunk_id {chunk_id} yang tidak ada di parquet "
                f"`{self.strategy}`; index dan snapshot tidak sinkron"
            )
        return Hit(
            chunk_id=chunk.chunk_id,
            score=score,
            rank=rank,
            source=source,
            doc_id=chunk.doc_id,
            text=chunk.text,
            heading_path=chunk.heading_path,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            url=chunk.url,
            ranks=dict(ranks),
        )

    # -- lifecycle -----------------------------------------------------------

    def check(self) -> dict:
        """Verify manifest, point count and BM25 alignment without embedding."""
        points = verify_points(self._dense.client, self.index, self.strategy, len(self.table))
        bm25 = self._bm25_retriever()
        return {
            "collection": self.collection,
            "index_version": self.index_version,
            "chunks": len(self.table),
            "points": points,
            "bm25_chunks": len(bm25.chunk_ids),
            "ok": True,
        }

    @property
    def collection(self) -> str:
        return self.index.collection_name(self.strategy)

    @property
    def index_version(self) -> str:
        return str(self.manifest.get("index_version", ""))

    def close(self) -> None:
        self._dense.close()

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
