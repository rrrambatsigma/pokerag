"""Pooling for A2: dedup, blindness, coverage, determinism."""

from __future__ import annotations

import pytest

from pokerag.eval.pool import (
    METHODS,
    PoolError,
    build_pool,
    build_query_pool,
    validate,
)
from pokerag.retrieval.types import Hit


def hit(chunk_id: str, doc_id: str = "d1", rank: int = 1, text: str | None = None) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        score=1.0 / rank,
        rank=rank,
        source="test",
        doc_id=doc_id,
        text=text if text is not None else f"passage {chunk_id}",
        heading_path=["Biology"],
        char_start=rank * 100,
        char_end=rank * 100 + 90,
        url="",
    )


REVIDS = {"d1": 111, "d2": 222}


def test_dedups_a_chunk_found_by_every_method():
    shared = hit("c1")
    record = build_query_pool(
        "q1",
        "question",
        {
            "dense": [shared],
            "bm25": [shared],
            "hybrid": [shared],
            "rerank": [shared],
        },
        depth=10,
        seed=7,
        revid_by_doc=REVIDS,
    )
    assert len(record.entries) == 1
    assert record.entries[0].sources == tuple(sorted(METHODS))
    assert record.entries[0].ranks == {"dense": 1, "bm25": 1, "hybrid": 1, "rerank": 1}


def test_keeps_the_top_depth_of_every_method():
    lists = {
        "dense": [hit("d1"), hit("d2")],
        "bm25": [hit("b1"), hit("b2")],
        "hybrid": [hit("h1"), hit("h2")],
        "rerank": [hit("r1"), hit("r2")],
    }
    record = build_query_pool(
        "q1", "question", lists, depth=2, seed=7, revid_by_doc=REVIDS
    )
    assert {entry.chunk_id for entry in record.entries} == {
        "d1",
        "d2",
        "b1",
        "b2",
        "h1",
        "h2",
        "r1",
        "r2",
    }


def test_depth_cutoff_drops_lower_ranks():
    lists = {method: [hit(f"{method}-{index}") for index in range(1, 4)] for method in METHODS}
    record = build_query_pool(
        "q1", "question", lists, depth=1, seed=7, revid_by_doc=REVIDS
    )
    assert {entry.chunk_id for entry in record.entries} == {
        f"{method}-1" for method in METHODS
    }


def test_order_is_deterministic_and_seeded():
    lists = {method: [hit(f"c{index}") for index in range(1, 9)] for method in METHODS}
    first = build_query_pool("q1", "q", lists, depth=8, seed=7, revid_by_doc=REVIDS)
    second = build_query_pool("q1", "q", lists, depth=8, seed=7, revid_by_doc=REVIDS)
    assert first.order == second.order
    assert sorted(first.order) == sorted({entry.chunk_id for entry in first.entries})
    orders = {
        build_query_pool("q1", "q", lists, depth=8, seed=seed, revid_by_doc=REVIDS).order
        for seed in (7, 17, 27)
    }
    assert len(orders) >= 2


def test_ui_payload_never_leaks_the_retriever():
    lists = {method: [hit(f"c{index}") for index in range(1, 4)] for method in METHODS}
    record = build_query_pool("q1", "q", lists, depth=3, seed=7, revid_by_doc=REVIDS)
    ui = record.ui_dict()
    assert "sources" not in str(ui)
    assert set(ui) == {"query_id", "query", "chunks"}
    assert set(ui["chunks"][0]) == {
        "chunk_id",
        "doc_id",
        "revid",
        "char_start",
        "char_end",
        "heading",
        "text",
    }
    assert "sources" in record.sources_dict()


def test_missing_method_is_an_error():
    with pytest.raises(PoolError):
        build_query_pool(
            "q1",
            "q",
            {"dense": [hit("c1")]},
            depth=1,
            seed=7,
            revid_by_doc=REVIDS,
        )


def test_unknown_revid_fails_validation():
    lists = {method: [hit("c1", doc_id="d9")] for method in METHODS}
    record = build_query_pool("q1", "q", lists, depth=1, seed=7, revid_by_doc=REVIDS)
    report = validate([record])
    assert report["ok"] is False
    assert any("revid" in problem for problem in report["problems"])


def test_build_pool_runs_four_methods_per_query():
    calls: list[tuple[str, str, int]] = []

    def search(query: str, mode: str, top_k: int):
        calls.append((query, mode, top_k))
        return [hit(f"{mode}-{query}-{index}") for index in range(1, top_k + 1)]

    def rerank(query: str, hits, top_k: int):
        return list(hits)[:top_k]

    records = build_pool(
        [("q1", "a"), ("q2", "b")],
        search,
        rerank,
        depth=2,
        candidates=5,
        seed=7,
        revid_by_doc=REVIDS,
    )
    assert len(records) == 2
    assert [mode for _, mode, _ in calls].count("hybrid") == 4  # two per query
    assert all(len(record.entries) > 0 for record in records)
