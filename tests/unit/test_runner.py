"""Runner wiring with fake retrieval and generation (no GPU, no index)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokerag.llm.base import Answer
from pokerag.llm.runner import run_variant
from pokerag.retrieval.types import Hit

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "configs" / "ladder.yaml"


def query_file(tmp_path: Path) -> Path:
    path = tmp_path / "queries.csv"
    path.write_text(
        "query_id,text,slice\nq1,What type is Cloyster?,single_hop\n"
        "q2,What is the catch rate of Pikachu?,single_hop\n",
        encoding="utf-8",
    )
    return path


def hit() -> Hit:
    return Hit(
        chunk_id="c1",
        score=0.9,
        rank=1,
        source="hybrid",
        doc_id="bulba:0000091",
        text="Cloyster is Water/Ice.",
        heading_path=["Biology"],
        char_start=100,
        char_end=160,
        url="https://example.invalid",
    )


class FakeRetriever:
    index_version = "idx123"
    strategy = "fixed"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []
        self.closed = False

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> list[Hit]:
        self.calls.append((query, mode, top_k))
        return [hit()]

    def close(self) -> None:
        self.closed = True


class FakeGenerator:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, question: str, evidence) -> Answer:
        self.calls.append(question)
        return Answer(
            text="Cloyster is Water/Ice.",
            cited_chunk_ids=(evidence[0].chunk_id,) if evidence else (),
            abstained=False,
            prompt_tokens=100,
            completion_tokens=20,
        )


def test_runner_writes_predictions_and_manifest(tmp_path: Path) -> None:
    retriever = FakeRetriever()
    generator = FakeGenerator()

    run_dir = run_variant(
        ladder_path=LADDER,
        variant_name="B2",
        queries_path=query_file(tmp_path),
        run_index=0,
        limit=1,
        out_root=tmp_path / "runs",
        generator=generator,
        retriever=retriever,
    )

    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert run_dir.name == "B2-r00"
    assert retriever.calls[0][0] == "warmup"
    assert retriever.calls[-1] == ("What type is Cloyster?", "dense", 5)
    assert predictions[0]["query_id"] == "q1"
    assert predictions[0]["citations"][0] == {
        "chunkId": "c1",
        "docId": "bulba:0000091",
        "charStart": 100,
        "charEnd": 160,
    }
    assert manifest["mode"] == "dense"
    assert manifest["index_version"] == "idx123"
    assert manifest["summary"]["requests"] == 1
    assert manifest["summary"]["tokens"]["input_total"] == 100


def test_off_variant_skips_retrieval_and_marks_index_off(tmp_path: Path) -> None:
    run_dir = run_variant(
        ladder_path=LADDER,
        variant_name="B0",
        queries_path=query_file(tmp_path),
        limit=1,
        out_root=tmp_path / "runs",
        generator=FakeGenerator(),
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["mode"] == "off"
    assert manifest["index_version"] == "off"
    assert manifest["extra"]["variant"] == "B0"


def test_run_index_must_fit_the_seed_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed"):
        run_variant(
            ladder_path=LADDER,
            variant_name="B0",
            queries_path=query_file(tmp_path),
            run_index=9,
            out_root=tmp_path / "runs",
            generator=FakeGenerator(),
        )


def test_unknown_variant_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="variant"):
        run_variant(
            ladder_path=LADDER,
            variant_name="B7",
            queries_path=query_file(tmp_path),
            out_root=tmp_path / "runs",
            generator=FakeGenerator(),
        )


class FakeReranker:
    name = "fake-rerank"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def rerank(self, query: str, hits, top_k: int):
        self.calls.append((query, len(hits), top_k))
        return list(reversed(hits))[:top_k]


def hits_three() -> list[Hit]:
    base = hit()
    return [
        base,
        Hit(
            chunk_id="c2",
            score=0.8,
            rank=2,
            source="hybrid",
            doc_id="bulba:0000092",
            text="Minun is Electric.",
            heading_path=["Biology"],
            char_start=10,
            char_end=20,
            url="",
        ),
        Hit(
            chunk_id="c3",
            score=0.7,
            rank=3,
            source="hybrid",
            doc_id="bulba:0000093",
            text="Plusle is Electric.",
            heading_path=["Biology"],
            char_start=30,
            char_end=40,
            url="",
        ),
    ]


class FakeRetrieverMany(FakeRetriever):
    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> list[Hit]:
        self.calls.append((query, mode, top_k))
        return hits_three()[:top_k]


def test_b3_uses_candidates_and_reranks(tmp_path: Path) -> None:
    retriever = FakeRetrieverMany()
    reranker = FakeReranker()

    run_dir = run_variant(
        ladder_path=LADDER,
        variant_name="B3",
        queries_path=query_file(tmp_path),
        limit=1,
        out_root=tmp_path / "runs",
        generator=FakeGenerator(),
        retriever=retriever,
        reranker=reranker,
    )

    predictions = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert retriever.calls[-1] == ("What type is Cloyster?", "hybrid", 20)
    assert reranker.calls == [("What type is Cloyster?", 3, 5)]
    assert predictions[0]["citations"][0]["chunkId"] == "c3"
    assert manifest["extra"]["rerank"] is True
    assert manifest["extra"]["reranker_model"] == "BAAI/bge-reranker-v2-m3"


def test_no_rerank_override_skips_reranker(tmp_path: Path) -> None:
    retriever = FakeRetrieverMany()
    reranker = FakeReranker()

    run_dir = run_variant(
        ladder_path=LADDER,
        variant_name="B3",
        queries_path=query_file(tmp_path),
        limit=1,
        out_root=tmp_path / "runs",
        generator=FakeGenerator(),
        retriever=retriever,
        reranker=reranker,
        rerank=False,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert retriever.calls[-1] == ("What type is Cloyster?", "hybrid", 5)
    assert reranker.calls == []
    assert manifest["extra"]["rerank"] is False


def test_mode_override_changes_retrieval_and_manifest(tmp_path: Path) -> None:
    retriever = FakeRetriever()

    run_dir = run_variant(
        ladder_path=LADDER,
        variant_name="B2",
        queries_path=query_file(tmp_path),
        limit=1,
        out_root=tmp_path / "runs",
        generator=FakeGenerator(),
        retriever=retriever,
        mode="bm25",
    )

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert retriever.calls[-1] == ("What type is Cloyster?", "bm25", 5)
    assert manifest["mode"] == "bm25"
    assert manifest["extra"]["mode_override"] == "bm25"
