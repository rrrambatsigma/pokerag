"""HTTP contract tests with a fake retriever and the stub generator."""

from __future__ import annotations

from fastapi.testclient import TestClient

from pokerag.api.app import create_app
from pokerag.llm.base import Answer
from pokerag.retrieval.document_meta import DocMeta
from pokerag.retrieval.types import Hit

META = {
    "d1": DocMeta(
        doc_id="d1",
        title="Cloyster",
        dex=91,
        types=("water", "ice"),
        url="https://bulbapedia.bulbagarden.net/wiki/Cloyster",
    )
}


def hit(rank: int = 1, chunk_id: str = "c1", doc_id: str = "d1") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        score=0.87,
        rank=rank,
        source="hybrid",
        doc_id=doc_id,
        text="Cloyster is a  Water / Ice  Pokémon.",
        heading_path=["Biology"],
        char_start=100,
        char_end=160,
        url="https://bulbapedia.bulbagarden.net/wiki/Cloyster",
    )


class FakeRetriever:
    strategy = "heading"
    collection = "chunks_heading"
    index_version = "abc123"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, str, int]] = []
        self.closed = False

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> list[Hit]:
        self.calls.append((query, mode, top_k))
        return self.hits[:top_k]

    def close(self) -> None:
        self.closed = True


def client(hits: list[Hit] | None = None) -> tuple[TestClient, FakeRetriever]:
    retriever = FakeRetriever(hits if hits is not None else [hit()])
    app = create_app(
        retrievers={"heading": retriever},
        doc_meta=META,
    )
    return TestClient(app), retriever


def test_health_reports_index_and_generator() -> None:
    with client()[0] as http:
        response = http.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshot"] == "pokerag-corpus-v2"
    assert payload["index_version"] == "abc123"
    assert payload["generator"] == "stub"
    assert "off" in payload["modes"]


def test_search_returns_citations_with_metadata() -> None:
    with client()[0] as http:
        response = http.get("/api/search", params={"q": "type of Cloyster", "k": 1})

    assert response.status_code == 200
    citation = response.json()["citations"][0]
    assert citation["title"] == "Cloyster"
    assert citation["dex"] == 91
    assert citation["types"] == ["water", "ice"]
    assert citation["headingPath"] == ["Biology"]
    assert citation["excerpt"] == "Cloyster is a Water / Ice Pokémon."
    assert citation["score"] == 0.87


def test_search_rejects_unknown_mode_and_strategy() -> None:
    with client()[0] as http:
        assert http.get("/api/search", params={"q": "x", "mode": "nope"}).status_code == 400
        assert http.get("/api/search", params={"q": "x", "strategy": "fixed"}).status_code == 400


def test_chat_maps_settings_to_retrieval_mode() -> None:
    http, retriever = client()
    with http:
        response = http.post(
            "/api/chat",
            json={
                "message": "type of Cloyster",
                "settings": {"dense": True, "bm25": False, "strategy": "heading", "topK": 3},
            },
        )

    assert response.status_code == 200
    assert retriever.calls[-1] == ("type of Cloyster", "dense", 3)
    message = response.json()["message"]
    assert message["role"] == "assistant"
    assert message["settings"]["topK"] == 3
    assert len(message["citations"]) == 1


def test_chat_off_abstains_without_retrieval() -> None:
    http, retriever = client()
    with http:
        response = http.post(
            "/api/chat",
            json={
                "message": "type of Garchomp",
                "settings": {"dense": False, "bm25": False, "strategy": "heading", "topK": 5},
            },
        )

    assert response.status_code == 200
    assert retriever.calls == []
    message = response.json()["message"]
    assert message["citations"] == []
    assert "Not enough evidence" in message["text"]


def test_chat_abstains_when_retrieval_is_empty() -> None:
    with client(hits=[])[0] as http:
        response = http.post("/api/chat", json={"message": "unknown"})

    assert response.status_code == 200
    assert response.json()["message"]["citations"] == []


def test_chat_validates_top_k() -> None:
    with client()[0] as http:
        response = http.post(
            "/api/chat",
            json={
                "message": "x",
                "settings": {"dense": True, "bm25": True, "strategy": "heading", "topK": 0},
            },
        )

    assert response.status_code == 422


class SilentGenerator:
    name = "silent"

    def generate(self, question, evidence):
        return Answer(text="Catch rate 60.", cited_chunk_ids=(), abstained=False)


def test_chat_falls_back_to_top_source_when_model_omits_citations() -> None:
    app = create_app(
        retrievers={"heading": FakeRetriever([hit()])},
        doc_meta=META,
        generator=SilentGenerator(),
    )
    with TestClient(app) as http:
        response = http.post("/api/chat", json={"message": "catch rate?"})

    message = response.json()["message"]
    assert message["citations"][0]["chunkId"] == "c1"
