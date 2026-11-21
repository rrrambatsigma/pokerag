"""Index identity and idempotency, with a fake embedder (no network).

Covers the A1 layout: three strategies share one database, one embedding cache
and one embedder, but each gets its own collection and manifest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pokerag.core.jsonl import read_json
from pokerag.index.build import build_all, cache_path, compute_index_version, identity_hash
from pokerag.index.qdrant_store import make_client

from index_fixtures import CountingTokens, FakeEmbedder, config, make_snapshot


def test_build_stores_points_and_reuses_cache(tmp_path: Path):
    make_snapshot(tmp_path)
    cfg = config(tmp_path)
    embedder = FakeEmbedder()

    first = build_all(cfg, embedder=embedder)
    manifest = first["manifests"]["heading"]
    assert manifest["counts"]["points"] == 3
    assert first["run"]["heading"]["embedded_now"] == 3
    assert embedder.texts == 3
    assert read_json(cfg.manifest_path("heading"))["index_version"] == manifest["index_version"]

    second = build_all(cfg, embedder=embedder)
    assert second["run"]["heading"]["cached"] is True
    assert second["run"]["heading"]["embedded_now"] == 0
    assert embedder.texts == 3, "second build must not call the embedder"
    assert read_json(cfg.manifest_path("heading")) == manifest, "manifest must stay identical"


def test_three_strategies_share_one_database(tmp_path: Path):
    strategies = ("fixed", "heading", "hierarchical")
    make_snapshot(tmp_path, strategies=strategies)
    cfg = config(tmp_path, strategies=list(strategies))
    embedder = FakeEmbedder()

    result = build_all(cfg, embedder=embedder)
    assert set(result["manifests"]) == set(strategies)
    assert embedder.texts == 9, "every strategy is embedded (different texts)"

    client = make_client(cfg.qdrant, cfg.qdrant_dir)
    try:
        names = sorted(collection.name for collection in client.get_collections().collections)
        counts = {
            name: client.count(collection_name=name, exact=True).count for name in names
        }
    finally:
        client.close()
    assert names == ["chunks_fixed", "chunks_heading", "chunks_hierarchical"]
    assert counts == {name: 3 for name in names}

    for strategy in strategies:
        manifest = read_json(cfg.manifest_path(strategy))
        assert manifest["collection"] == f"chunks_{strategy}"
        assert manifest["embedder_identity_sha256"] == identity_hash(cfg)
    assert (tmp_path / "indexes" / "index_report.md").exists()
    assert (tmp_path / "docs" / "experiments" / "A1-index.md").exists()
    assert (tmp_path / "docs" / "models" / "embedding-fake-embed.md").exists()


def test_index_version_tracks_inputs(tmp_path: Path):
    cfg = config(tmp_path)
    base = compute_index_version(cfg, "heading", "chunks-sha", "snap-1", None)
    assert base == compute_index_version(cfg, "heading", "chunks-sha", "snap-1", None)

    assert compute_index_version(cfg, "fixed", "chunks-sha", "snap-1", None) != base
    other_embedder = cfg.model_copy(
        update={"embedder": cfg.embedder.model_copy(update={"dimension": 16})}
    )
    assert compute_index_version(other_embedder, "heading", "chunks-sha", "snap-1", None) != base
    assert compute_index_version(cfg, "heading", "chunks-sha", "snap-1", 20) != base


def test_cache_file_is_bound_to_the_embedding_function(tmp_path: Path):
    base = config(tmp_path)
    local = base.embedder.model_copy(
        update={"provider": "st", "dtype": "float16", "max_seq_length": 1024}
    )
    base = base.model_copy(update={"embedder": local})
    wider = base.model_copy(
        update={"embedder": local.model_copy(update={"max_seq_length": 2048})}
    )
    assert cache_path(base) != cache_path(wider)
    assert identity_hash(base)[:8] in cache_path(base).name
    assert cache_path(base).suffix == ".parquet"


def test_cache_reembeds_changed_text(tmp_path: Path):
    make_snapshot(tmp_path)
    cfg = config(tmp_path)
    embedder = FakeEmbedder()
    build_all(cfg, embedder=embedder)

    chunks_path = tmp_path / "corpus" / "tiny" / "chunks" / "heading.parquet"
    chunks = pd.read_parquet(chunks_path)
    chunks.loc[0, "text"] = "Bulbasaur fact number 0 changed"
    chunks.to_parquet(chunks_path, index=False)

    rebuilt = build_all(cfg, embedder=embedder, force=True)
    assert rebuilt["run"]["heading"]["embedded_now"] == 1
    assert embedder.texts == 4


def test_different_embedder_requires_force(tmp_path: Path):
    make_snapshot(tmp_path)
    cfg = config(tmp_path)
    build_all(cfg, embedder=FakeEmbedder())

    wider = config(tmp_path, dimension=16)
    with pytest.raises(RuntimeError, match="embedder lain"):
        build_all(wider, embedder=FakeEmbedder(dimension=16))

    result = build_all(wider, embedder=FakeEmbedder(dimension=16), force=True)
    assert result["manifests"]["heading"]["embedder"]["dimension"] == 16


def test_truncation_fails_before_embedding(tmp_path: Path):
    make_snapshot(tmp_path)
    cfg = config(tmp_path)
    embedder = FakeEmbedder()
    with pytest.raises(RuntimeError, match="dipotong"):
        build_all(cfg, embedder=embedder, token_counter=CountingTokens())
    assert embedder.texts == 0, "must fail before paying for embeddings"

    allowed = cfg.model_copy(update={"allow_truncation": True})
    result = build_all(allowed, embedder=embedder, token_counter=CountingTokens())
    assert result["manifests"]["heading"]["token_stats"]["chunks_over_window"] == 3


def test_limit_refuses_to_overwrite_a_full_index(tmp_path: Path):
    make_snapshot(tmp_path)
    cfg = config(tmp_path)
    build_all(cfg, embedder=FakeEmbedder())

    with pytest.raises(RuntimeError, match="index penuh"):
        build_all(cfg, embedder=FakeEmbedder(), limit=1)


def test_partial_build_keeps_identity_counts_for_all_strategies(tmp_path: Path):
    strategies = ("fixed", "heading")
    make_snapshot(tmp_path, strategies=strategies)
    cfg = config(tmp_path, strategies=list(strategies))
    counter = CountingTokens(tokens=100)  # below the window, so it never aborts
    build_all(cfg, embedder=FakeEmbedder(), token_counter=counter)
    assert read_json(cfg.identity_path)["truncation"]["chunks"] == 6

    build_all(cfg, embedder=FakeEmbedder(), strategies=["heading"], token_counter=counter)
    identity = read_json(cfg.identity_path)
    assert identity["truncation"]["chunks"] == 6, "a partial build must not shrink the record"


def test_provenance_keeps_history_and_real_timings(tmp_path: Path):
    make_snapshot(tmp_path)
    cfg = config(tmp_path)
    build_all(cfg, embedder=FakeEmbedder())
    build_all(cfg, embedder=FakeEmbedder())  # cached re-run

    provenance = read_json(cfg.provenance_path)
    history = provenance["history"]
    assert any(entry["cached"] is False for entry in history)
    assert provenance["runs"]["heading"]["cached"] is True
