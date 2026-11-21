"""FastAPI application for the PokeRAG chat UI.

The app is a thin layer over the existing retrieval pipeline and a generator
interface. Until the Qwen backend exists, ``StubGenerator`` keeps the contract
real so the UI can be wired without a GPU.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Query

from pokerag.api.models import (
    MODES,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Citation,
    HealthResponse,
    SearchResponse,
    mode_for,
)
from pokerag.llm.base import Generator
from pokerag.llm.stub import StubGenerator
from pokerag.retrieval.config import load_retrieval_config
from pokerag.retrieval.document_meta import DocMeta, load_doc_meta
from pokerag.retrieval.pipeline import Retriever
from pokerag.retrieval.types import Hit

DEFAULT_CONFIG = "configs/retrieval.v2.yaml"


class RetrieverLike(Protocol):
    strategy: str
    collection: str
    index_version: str

    def search(
        self, query: str, mode: str = "hybrid", top_k: int = 5
    ) -> list[Hit]: ...

    def close(self) -> None: ...


class RetrieverPool:
    """One retriever per strategy, built on first use and closed on shutdown."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._cache: dict[str, RetrieverLike] = {}

    def get(self, strategy: str) -> RetrieverLike:
        if strategy not in self._cache:
            self._cache[strategy] = Retriever(self._cfg, strategy=strategy)
        return self._cache[strategy]

    def close(self) -> None:
        for retriever in self._cache.values():
            retriever.close()
        self._cache.clear()


class MappingPool:
    """Test double: a fixed mapping of strategy to retriever."""

    def __init__(self, retrievers: Mapping[str, RetrieverLike]) -> None:
        self._retrievers = dict(retrievers)

    def get(self, strategy: str) -> RetrieverLike:
        retriever = self._retrievers.get(strategy)
        if retriever is None:
            raise HTTPException(
                status_code=400,
                detail=f"strategy '{strategy}' tidak tersedia; pilihan: {sorted(self._retrievers)}",
            )
        return retriever

    def close(self) -> None:
        for retriever in self._retrievers.values():
            retriever.close()


def _citation(n: int, hit: Hit, meta: Mapping[str, DocMeta]) -> Citation:
    document = meta.get(hit.doc_id)
    return Citation(
        n=n,
        chunkId=hit.chunk_id,
        docId=hit.doc_id,
        title=document.title if document else hit.doc_id,
        dex=document.dex if document else None,
        types=list(document.types) if document else [],
        headingPath=list(hit.heading_path),
        excerpt=" ".join(hit.text.split())[:280],
        url=hit.url or (document.url if document else ""),
        score=round(float(hit.score), 6),
    )


def _citations(
    hits: Sequence[Hit], cited: Sequence[str], meta: Mapping[str, DocMeta]
) -> list[Citation]:
    by_id = {hit.chunk_id: hit for hit in hits}
    citations: list[Citation] = []
    for chunk_id in cited:
        hit = by_id.get(chunk_id)
        if hit is not None:
            citations.append(_citation(len(citations) + 1, hit, meta))
    return citations


def _check_mode(mode: str) -> str:
    if mode not in MODES:
        raise HTTPException(status_code=400, detail=f"mode tidak dikenal: {mode}")
    return mode


def create_app(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    retrievers: Mapping[str, RetrieverLike] | None = None,
    doc_meta: Mapping[str, DocMeta] | None = None,
    generator: Generator | None = None,
) -> FastAPI:
    cfg = load_retrieval_config(Path(config_path))
    pool: RetrieverPool | MappingPool
    pool = MappingPool(retrievers) if retrievers is not None else RetrieverPool(cfg)
    meta: Mapping[str, DocMeta] | None = doc_meta
    gen = generator or StubGenerator()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            pool.close()

    app = FastAPI(title="PokeRAG API", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.pool = pool
    app.state.generator = gen
    app.state.doc_meta = meta

    def document_meta() -> Mapping[str, DocMeta]:
        if app.state.doc_meta is None:
            app.state.doc_meta = load_doc_meta(cfg.index_config())
        return app.state.doc_meta

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        retriever = pool.get(cfg.strategy)
        return HealthResponse(
            status="ok",
            snapshot=cfg.index_config().snapshot,
            index_version=retriever.index_version,
            strategy=retriever.strategy,
            collection=retriever.collection,
            generator=gen.name,
            modes=list(MODES),
        )

    @app.get("/api/search", response_model=SearchResponse)
    def search(
        q: str = Query(..., min_length=1, max_length=2000),
        mode: str = "hybrid",
        strategy: str | None = None,
        k: int | None = Query(default=None, ge=1, le=50),
    ) -> SearchResponse:
        _check_mode(mode)
        chosen = strategy or cfg.strategy
        retriever = pool.get(chosen)
        if mode == "off":
            hits: list[Hit] = []
        else:
            hits = retriever.search(q, mode=mode, top_k=k or cfg.top_k)
        citations = [
            _citation(index, hit, document_meta())
            for index, hit in enumerate(hits, start=1)
        ]
        return SearchResponse(
            query=q, mode=mode, strategy=chosen, citations=citations
        )

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        mode = mode_for(request.settings)
        retriever = pool.get(request.settings.strategy)
        hits: list[Hit] = []
        if mode != "off":
            hits = retriever.search(
                request.message, mode=mode, top_k=request.settings.topK
            )

        answer = gen.generate(request.message, hits)
        citations = (
            []
            if answer.abstained
            else _citations(hits, answer.cited_chunk_ids, document_meta())
        )
        message = ChatMessage(
            id=uuid.uuid4().hex,
            text=answer.text,
            createdAt=int(time.time() * 1000),
            citations=citations,
            settings=request.settings,
            pending=False,
        )
        return ChatResponse(message=message)

    return app
