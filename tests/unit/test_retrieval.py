import pandas as pd
import pytest

from pokerag.index.catalog import Catalog
from pokerag.index.store import ScoredChunk
from pokerag.stages.retrieve import RetrievalConfig, Retriever


class FakeStore:
    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits
        self.closed = False

    def search(self, collection: str, vector: list[float], limit: int) -> list[ScoredChunk]:
        return self.hits[:limit]

    def close(self) -> None:
        self.closed = True


class FakeBM25:
    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits

    def search(self, query: str, limit: int) -> list[ScoredChunk]:
        return self.hits[:limit]


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, query: str) -> list[float]:
        self.calls += 1
        return [0.1, 0.2]


def make_catalog() -> Catalog:
    chunks = pd.DataFrame(
        {
            "chunk_id": ["a", "b", "c"],
            "doc_id": ["d1", "d1", "d2"],
            "strategy": ["fixed"] * 3,
            "role": ["child"] * 3,
            "parent_id": [None] * 3,
            "heading_path": ["[]"] * 3,
            "char_start": [0, 10, 20],
            "char_end": [10, 20, 30],
            "text": ["alpha beta", "beta gamma", "gamma delta"],
            "n_tokens": [2, 2, 2],
        }
    )
    documents = pd.DataFrame({"doc_id": ["d1", "d2"], "title": ["Doc One", "Doc Two"]})
    return Catalog.from_frames(chunks, documents)


def make_retriever(mode: str, encoder: FakeEncoder | None = None) -> Retriever:
    config = RetrievalConfig(
        strategy="fixed", mode=mode, top_k=3, candidates=10, rrf_k=60
    )
    return Retriever(
        config=config,
        store=FakeStore([ScoredChunk("a", 0.9), ScoredChunk("b", 0.8)]),
        bm25=FakeBM25([ScoredChunk("b", 7.0), ScoredChunk("c", 5.0)]),
        catalog=make_catalog(),
        encoder=encoder,
    )


def test_hybrid_fuses_rankings() -> None:
    retriever = make_retriever("hybrid", FakeEncoder())

    results = retriever.search("beta")

    assert [item.chunk.chunk_id for item in results] == ["b", "a", "c"]
    assert results[0].ranks == {"dense": 2, "bm25": 1}


def test_dense_mode_preserves_scores() -> None:
    retriever = make_retriever("dense", FakeEncoder())

    results = retriever.search("beta")

    assert [item.chunk.chunk_id for item in results] == ["a", "b"]
    assert results[0].score == pytest.approx(0.9)
    assert results[0].ranks == {"dense": 1}


def test_bm25_mode_works_without_encoder() -> None:
    retriever = make_retriever("bm25", None)

    results = retriever.search("gamma")

    assert [item.chunk.chunk_id for item in results] == ["b", "c"]
    assert results[0].score == pytest.approx(7.0)


def test_dense_mode_without_encoder_fails() -> None:
    retriever = make_retriever("dense", None)

    with pytest.raises(RuntimeError):
        retriever.search("beta")


def test_top_k_limits_results() -> None:
    retriever = make_retriever("hybrid", FakeEncoder())
    retriever.config.top_k = 1

    assert len(retriever.search("beta")) == 1


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        RetrievalConfig(top_k=20, candidates=10)
    with pytest.raises(ValueError):
        RetrievalConfig(strategy="nope")
    with pytest.raises(ValueError):
        RetrievalConfig(mode="nope")
