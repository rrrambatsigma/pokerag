"""Retrieval stage tests: fusion, guards, modes and metrics.

Everything runs offline. Qdrant local mode allows one client per path, so each
test uses its own ``tmp_path`` and closes the retriever before touching files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from pokerag.core.jsonl import write_jsonl
from pokerag.index.build import build_all
from pokerag.retrieval.checks import verify_manifest
from pokerag.retrieval.config import RetrievalConfig
from pokerag.retrieval.corpus import load_chunks
from pokerag.retrieval.eval import Qrel, evaluate, load_qrels, score_hits
from pokerag.retrieval.fuse import rrf
from pokerag.retrieval.pipeline import Retriever
from pokerag.retrieval.types import Hit

from index_fixtures import FakeEmbedder, config, make_snapshot


def built_index(root: Path, n: int = 3) -> tuple:
    """Build a tiny heading index and return (index config, embedder)."""
    make_snapshot(root, n=n)
    cfg = config(root)
    embedder = FakeEmbedder()
    build_all(cfg, embedder=embedder)
    return cfg, embedder


def retrieval(root: Path, **overrides) -> RetrievalConfig:
    values = {
        "index": str(root / "configs" / "index.yaml"),
        "strategy": "heading",
        "top_dense": 10,
        "top_bm25": 10,
        "top_k": 3,
    }
    values.update(overrides)
    return RetrievalConfig(**values)


def hit(rank: int, doc_id: str, text: str = "text", chunk_id: str | None = None) -> Hit:
    return Hit(
        chunk_id=chunk_id or f"c{rank}",
        score=1.0 / rank,
        rank=rank,
        source="dense",
        doc_id=doc_id,
        text=text,
        heading_path=["Biology"],
        char_start=0,
        char_end=len(text),
        url=f"https://example.invalid/{doc_id}",
    )


# -- fusion ------------------------------------------------------------------


def test_rrf_orders_by_combined_rank_and_dedupes():
    fused = rrf([["a", "b", "c"], ["b", "d"]], k=60, sources=["dense", "bm25"])
    assert [item.chunk_id for item in fused] == ["b", "a", "d", "c"]
    assert len({item.chunk_id for item in fused}) == len(fused)
    assert fused[0].ranks == {"dense": 2, "bm25": 1}


def test_rrf_weights_change_the_winner():
    rankings = [["a"], ["b"]]
    sources = ["dense", "bm25"]
    equal = rrf(rankings, k=60, sources=sources)
    assert equal[0].chunk_id == "a", "ties break by chunk_id, not by source"

    dense_only = rrf(rankings, k=60, weights={"dense": 2.0, "bm25": 0.0}, sources=sources)
    assert dense_only[0].chunk_id == "a"
    bm25_only = rrf(rankings, k=60, weights={"dense": 0.0, "bm25": 2.0}, sources=sources)
    assert bm25_only[0].chunk_id == "b"


def test_rrf_rejects_duplicate_ids_and_bad_labels():
    with pytest.raises(ValueError, match="duplikat"):
        rrf([["a", "a"]])
    with pytest.raises(ValueError, match="sama panjang"):
        rrf([["a"], ["b"]], sources=["dense"])


# -- pipeline ----------------------------------------------------------------


def test_hybrid_merges_both_sources(tmp_path: Path):
    index_cfg, embedder = built_index(tmp_path)
    with Retriever(retrieval(tmp_path), index_cfg=index_cfg, embedder=embedder) as retriever:
        hits = retriever.search("grass poison", mode="hybrid")
        assert hits, "hybrid harus mengembalikan hasil"
        assert all(item.source == "hybrid" for item in hits)
        assert all(item.ranks for item in hits)
        assert all("dense" in item.ranks or "bm25" in item.ranks for item in hits)
        assert hits[0].url.startswith("https://example.invalid/")
        assert hits[0].text


def test_dense_mode_does_not_need_bm25_files(tmp_path: Path):
    index_cfg, embedder = built_index(tmp_path)
    bm25_dir = index_cfg.bm25_dir("heading")
    for file in bm25_dir.iterdir():
        file.unlink()
    bm25_dir.rmdir()

    with Retriever(retrieval(tmp_path), index_cfg=index_cfg, embedder=embedder) as retriever:
        hits = retriever.search("grass", mode="dense")
    assert hits and all(item.source == "dense" for item in hits)


def test_bm25_mode_does_not_call_the_embedder(tmp_path: Path):
    index_cfg, embedder = built_index(tmp_path)
    calls_after_build = embedder.calls
    with Retriever(retrieval(tmp_path), index_cfg=index_cfg, embedder=embedder) as retriever:
        hits = retriever.search("grass poison", mode="bm25")
    assert hits and all(item.source == "bm25" for item in hits)
    assert embedder.calls == calls_after_build, "mode bm25 tidak boleh memanggil embedder"


def test_top_k_limits_the_result(tmp_path: Path):
    index_cfg, embedder = built_index(tmp_path, n=5)
    with Retriever(retrieval(tmp_path), index_cfg=index_cfg, embedder=embedder) as retriever:
        hits = retriever.search("grass", mode="hybrid", top_k=2)
    assert len(hits) == 2
    assert [item.rank for item in hits] == [1, 2]


# -- guards ------------------------------------------------------------------


def test_chunk_table_matches_index_order_and_drops_parents(tmp_path: Path):
    make_snapshot(tmp_path, n=3)
    cfg = config(tmp_path)
    path = cfg.chunks_dir / "heading.parquet"
    frame = pd.read_parquet(path)
    parent = frame.iloc[[0]].copy()
    parent["chunk_id"] = "parent-0"
    parent["role"] = "parent"
    parent["order"] = -1
    frame = pd.concat([parent, frame.iloc[::-1]], ignore_index=True)
    frame.to_parquet(path, index=False)

    table = load_chunks(cfg, "heading")
    assert "parent-0" not in table, "parent bukan unit retrieval"
    assert list(table) == ["h0", "h1", "h2"], "harus sama dengan urutan index stage"


def test_guard_rejects_stale_chunks(tmp_path: Path):
    index_cfg, _ = built_index(tmp_path)
    path = index_cfg.chunks_dir / "heading.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "text"] = "changed after the build"
    frame.to_parquet(path, index=False)

    with pytest.raises(RuntimeError, match="basi"):
        verify_manifest(index_cfg, "heading")


def test_guard_rejects_misaligned_bm25_order(tmp_path: Path):
    index_cfg, embedder = built_index(tmp_path)
    ids_path = index_cfg.bm25_dir("heading") / "chunk_ids.json"
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    ids_path.write_text(json.dumps(list(reversed(ids))), encoding="utf-8")

    with Retriever(retrieval(tmp_path), index_cfg=index_cfg, embedder=embedder) as retriever:
        with pytest.raises(RuntimeError, match="urutan chunk id"):
            retriever.search("grass", mode="bm25")


def test_guard_rejects_missing_manifest(tmp_path: Path):
    index_cfg, _ = built_index(tmp_path)
    index_cfg.manifest_path("heading").unlink()
    with pytest.raises(RuntimeError, match="belum dibangun"):
        verify_manifest(index_cfg, "heading")


# -- evaluation --------------------------------------------------------------


def test_score_hits_document_and_span_levels():
    qrel = Qrel(query_id="q1", query="needle", doc_id="d2", evidence="a needle here")
    hits = [hit(1, "d1"), hit(2, "d2", "no match"), hit(3, "d2", "a needle here")]
    scores = score_hits(hits, qrel, [1, 3])

    assert scores["recall@1"] == 0.0
    assert scores["recall@3"] == 1.0
    assert scores["mrr@3"] == pytest.approx(0.5)
    assert scores["ndcg@3"] == pytest.approx(1 / math.log2(3))  # rank 2 -> 1/log2(2+1)
    assert scores["span_hit@3"] == 1.0
    assert scores["span_mrr@3"] == pytest.approx(1 / 3)


def test_evaluate_runs_every_mode_and_aggregates():
    class Stub:
        def search(self, query, mode="hybrid", top_k=5):
            if mode == "dense":
                return [hit(1, "d1"), hit(2, "d2")]
            return [hit(1, "d2")]

    qrel = Qrel(query_id="q1", query="q", doc_id="d2")
    result = evaluate(Stub(), [qrel], modes=["dense", "bm25"], ks=[1, 2])

    assert result["queries"] == 1
    assert set(result["modes"]) == {"dense", "bm25"}
    assert result["modes"]["dense"]["summary"]["recall@2"] == 1.0
    assert result["modes"]["dense"]["summary"]["mrr@1"] == 0.0
    assert result["modes"]["bm25"]["summary"]["mrr@1"] == 1.0


def test_load_qrels_keeps_order_and_rejects_duplicates(tmp_path: Path):
    path = tmp_path / "qrels.jsonl"
    write_jsonl(
        path,
        [
            {"query_id": "q1", "query": "first", "doc_id": "d1"},
            {"query_id": "q2", "query": "second", "doc_id": "d2", "evidence": "x"},
        ],
    )
    qrels = load_qrels(path)
    assert [qrel.query_id for qrel in qrels] == ["q1", "q2"]
    assert qrels[1].evidence == "x"

    write_jsonl(
        path,
        [
            {"query_id": "q1", "query": "first", "doc_id": "d1"},
            {"query_id": "q1", "query": "again", "doc_id": "d2"},
        ],
    )
    with pytest.raises(ValueError, match="duplikat"):
        load_qrels(path)
