"""Index builder: chunks -> dense vectors (GPU or hosted) + BM25 -> Qdrant.

Identity of an index is derived from everything that can change retrieval
behaviour: the corpus snapshot, the chunk file, the strategy, the embedder
identity (model, revision, dimension, dtype, ``max_seq_length``), the distance
metric and the BM25 parameters. Two different configurations can never share an
``index_version``, so a result can always be traced to an index.

Layout (ablation A1 shares one database, one cache and one embedder):

    artifacts/indexes/qdrant/                    one on-disk Qdrant database
        collection/chunks_fixed|heading|hierarchical
    artifacts/indexes/bm25/<strategy>/
    artifacts/indexes/<strategy>/manifest.json   deterministic per collection
    artifacts/indexes/provenance.json            run facts (time, calls) only
    artifacts/indexes/index_report.md            A1 table (artefact copy)
    docs/experiments/A1-index.md                 A1 table (committed copy)

Re-running with unchanged inputs makes zero embedding calls and produces the
same manifest bytes; run-dependent numbers live in ``provenance.json`` so they
cannot move the hashes.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pokerag.chunking.tokens import TokenCounter
from pokerag.core.hashing import sha256_bytes, sha256_json
from pokerag.core.jsonl import atomic_write_text, read_json, write_json
from pokerag.core.logging import get_logger
from pokerag.index.bm25_index import build_bm25
from pokerag.index.cache import EmbeddingCache
from pokerag.index.config import STRATEGIES, IndexConfig
from pokerag.index.embedders import Embedder, make_embedder
from pokerag.index.identity import (
    build_identity,
    identity_sha256,
    load_identity,
    save_identity,
)
from pokerag.index.qdrant_store import (
    collection_count,
    create_collection,
    list_collections,
    make_client,
    upsert_points,
)

logger = get_logger("pokerag.index.build")

FORMAT = 2


def embedder_slug(cfg: IndexConfig) -> str:
    """Filesystem-safe identifier for the embedding backend."""
    name = cfg.embedder.model.replace("/", "-")
    return f"{cfg.embedder.provider}-{name}-{cfg.embedder.dimension}"


def embedder_identity(cfg: IndexConfig) -> dict:
    """Deterministic identity of the embedding function.

    Everything here changes the vectors, so it is hashed into ``index_version``
    and into the cache file name. Runtime facts (torch version, GPU name, timed
    runs) are recorded separately, because they must not move the hash.
    """
    embedder = cfg.embedder
    local = embedder.provider in ("st", "local")
    return {
        "provider": embedder.provider,
        "model": embedder.model,
        "revision": embedder.revision,
        "dimension": embedder.dimension,
        "dtype": embedder.dtype if embedder.provider == "st" else None,
        "max_seq_length": embedder.max_seq_length if local else None,
        "query_prefix": embedder.query_prefix,
        "task_document": embedder.task_document if embedder.provider == "gemini" else "none",
        "task_query": embedder.task_query if embedder.provider == "gemini" else "none",
    }


def identity_hash(cfg: IndexConfig) -> str:
    """Hash of the embedding function, used for the cache file and the guard."""
    return sha256_json(embedder_identity(cfg))


def cache_path(cfg: IndexConfig) -> Path:
    """One cache file per embedding function; 512-token vectors never leak in."""
    return (
        Path(cfg.artifacts_root)
        / "embeddings"
        / f"{embedder_slug(cfg)}-{identity_hash(cfg)[:8]}.parquet"
    )


def compute_index_version(
    cfg: IndexConfig, strategy: str, chunks_sha: str, snapshot_id: str, limit: int | None
) -> str:
    """Hash of every input that can change retrieval results."""
    return sha256_json(
        {
            "snapshot_id": snapshot_id,
            "chunks_sha256": chunks_sha,
            "strategy": strategy,
            "collection": cfg.collection_name(strategy),
            "embedder": embedder_identity(cfg),
            "qdrant": {"distance": cfg.qdrant.distance, "on_disk": cfg.qdrant.on_disk},
            "bm25": cfg.bm25.model_dump(),
            "limit": limit,
            "format": FORMAT,
        }
    )


def _load_chunks(cfg: IndexConfig, strategy: str) -> pd.DataFrame:
    path = cfg.chunks_dir / f"{strategy}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"chunk file tidak ada: {path}. Jalankan pokerag-chunk dulu.")
    frame = pd.read_parquet(path)
    frame = frame[frame["role"] != "parent"].copy()
    return frame.sort_values(["doc_id", "order"], kind="stable").reset_index(drop=True)


def _payloads(frame: pd.DataFrame, documents: pd.DataFrame, strategy: str, collection: str) -> list[dict]:
    meta = {
        row.doc_id: row
        for row in documents[["doc_id", "url", "revid", "license_name", "attribution"]].itertuples()
    }
    payloads = []
    for row in frame.itertuples():
        document = meta.get(row.doc_id)
        payloads.append(
            {
                "chunk_id": row.chunk_id,
                "doc_id": row.doc_id,
                "revid": int(document.revid) if document is not None else int(row.revid),
                "url": document.url if document is not None else "",
                "heading_path": list(json.loads(row.heading_path)),
                "char_start": int(row.char_start),
                "char_end": int(row.char_end),
                "n_chars": int(row.n_chars),
                "n_tokens": int(row.n_tokens),
                "strategy": strategy,
                "collection": collection,
                "parent_id": row.parent_id if isinstance(row.parent_id, str) else None,
                "license_name": document.license_name if document is not None else "",
                "attribution": document.attribution if document is not None else "",
            }
        )
    return payloads


def _embed_missing(
    frame: pd.DataFrame, cache: EmbeddingCache, embedder: Embedder, batch_size: int = 100
) -> int:
    """Embed only chunks whose text is not cached; returns the number embedded.

    The cache flushes on a size/time threshold (see :mod:`pokerag.index.cache`),
    so an interrupted run loses at most a couple of minutes of work while the GPU
    is not stalled by rewriting a growing parquet for every batch.
    """
    pending = [
        index
        for index, row in enumerate(frame.itertuples())
        if cache.get(row.chunk_id, row.text) is None
    ]
    if not pending:
        logger.info("cache lengkap: 0 chunk perlu di-embed")
        return 0
    logger.info("%s chunk perlu di-embed (%s dari cache)", len(pending), len(frame) - len(pending))
    size = max(1, batch_size)
    start_time = time.perf_counter()
    for start in range(0, len(pending), size):
        indexes = pending[start : start + size]
        texts = [frame.iloc[index].text for index in indexes]
        vectors = embedder.embed_documents(texts)
        for index, vector in zip(indexes, vectors):
            row = frame.iloc[index]
            cache.put(row.chunk_id, row.text, vector)
        cache.maybe_save()
        done = min(start + size, len(pending))
        if done % (size * 50) < size or done == len(pending):
            rate = done / max(1e-9, time.perf_counter() - start_time)
            logger.info("embedded %s/%s (%.1f chunk/s, cache %s)", done, len(pending), rate, len(cache))
    cache.maybe_save(force=True)
    return len(pending)


def measure_tokens(
    cfg: IndexConfig,
    frames: dict[str, pd.DataFrame],
    token_counter: TokenCounter | None = None,
) -> dict[str, dict]:
    """Real token counts per strategy (the embedder's own tokenizer).

    ``chars/4`` underestimates Bulbapedia text by ~21% (measured), which is why a
    2048-character window looked like "512 tokens" while it was actually ~600-1000.
    """
    counter = token_counter
    if counter is None and cfg.embedder.provider in ("st", "local"):
        counter = TokenCounter(f"hf:{cfg.embedder.model}")
    stats: dict[str, dict] = {}
    for strategy, frame in frames.items():
        texts = frame["text"].tolist()
        if counter is None:
            counts = [max(1, len(text) // 4) for text in texts]
            measured = False
        else:
            counts = counter.count_many(texts)
            measured = True
        values = np.asarray(counts, dtype=np.int64)
        stats[strategy] = {
            "chunks": int(values.size),
            "tokens_p50": int(np.percentile(values, 50)),
            "tokens_p95": int(np.percentile(values, 95)),
            "tokens_max": int(values.max()),
            "chunks_over_window": int((values > cfg.embedder.max_seq_length).sum()),
            "max_seq_length": cfg.embedder.max_seq_length,
            "measured": measured,
        }
    return stats


def _truncation_from_stats(token_stats: dict[str, dict], cfg: IndexConfig) -> dict:
    """Aggregate the truncation gate across every strategy that was measured."""
    return {
        "chunks": sum(entry["chunks"] for entry in token_stats.values()),
        "chunks_over_window": sum(entry["chunks_over_window"] for entry in token_stats.values()),
        "max_tokens": max((entry["tokens_max"] for entry in token_stats.values()), default=0),
        "max_seq_length": cfg.embedder.max_seq_length,
    }


def truncation_summary(cfg: IndexConfig, token_counter: TokenCounter | None = None) -> dict:
    """Truncation numbers for **all** configured strategies.

    The identity document describes the corpus, not the subset that one command
    happened to build, so a partial rebuild must not shrink its counts.
    """
    frames: dict[str, pd.DataFrame] = {}
    for strategy in cfg.strategies:
        path = cfg.chunks_dir / f"{strategy}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["text", "role"])
        frames[strategy] = frame[frame["role"] != "parent"].reset_index(drop=True)
    return _truncation_from_stats(measure_tokens(cfg, frames, token_counter), cfg)


def preflight_truncation(cfg: IndexConfig, token_stats: dict[str, dict]) -> None:
    """Fail before spending an hour on GPU time if a chunk cannot fit."""
    over = {name: entry["chunks_over_window"] for name, entry in token_stats.items() if entry["chunks_over_window"]}
    if not over:
        return
    detail = ", ".join(
        f"{name}: {count} chunk > {token_stats[name]['max_seq_length']} token "
        f"(maks {token_stats[name]['tokens_max']})"
        for name, count in over.items()
    )
    message = (
        "chunk melewati jendela embedder dan akan dipotong sebelum di-embed: "
        f"{detail}. Naikkan embedder.max_seq_length atau pakai --allow-truncation "
        "kalau pemotongan itu memang disengaja dan akan dilaporkan."
    )
    if cfg.allow_truncation:
        logger.warning(message)
        return
    raise RuntimeError(message)


def _unit_vectors(frame: pd.DataFrame, cache: EmbeddingCache, dimension: int) -> np.ndarray:
    vectors = np.asarray(
        [cache.get(row.chunk_id, row.text) for row in frame.itertuples()], dtype=np.float32
    )
    if vectors.shape != (len(frame), dimension):
        raise RuntimeError(f"bentuk vektor {vectors.shape} tidak sesuai ({len(frame)}, {dimension})")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build_all(
    cfg: IndexConfig,
    embedder: Embedder | None = None,
    strategies: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
    force: bool = False,
    token_counter: TokenCounter | None = None,
) -> dict:
    """Build every requested strategy into one Qdrant database.

    Returns ``{"manifests": {strategy: manifest}, "run": {strategy: facts}}``.
    The manifest is deterministic; timing and call counts live under ``run`` and
    in ``provenance.json``.
    """
    started = datetime.now(timezone.utc)
    snap = cfg.snapshot_dir
    snapshot = read_json(snap / "snapshot.json")
    selected = tuple(strategies or cfg.strategies)
    unknown = [name for name in selected if name not in STRATEGIES]
    if unknown:
        raise ValueError(f"strategi tidak dikenal: {unknown}")

    if limit is not None:
        for strategy in selected:
            manifest_path = cfg.manifest_path(strategy)
            if manifest_path.exists() and read_json(manifest_path).get("limit") is None:
                raise RuntimeError(
                    f"koleksi {cfg.collection_name(strategy)} berisi index penuh; build --limit "
                    "akan menghapusnya. Untuk smoke test, pakai artifacts_root terpisah "
                    "(config lain), bukan menimpa index produksi."
                )

    frames: dict[str, pd.DataFrame] = {}
    chunks_sha: dict[str, str] = {}
    for strategy in selected:
        frame = _load_chunks(cfg, strategy)
        if limit is not None:
            frame = frame.head(limit).copy()
        frames[strategy] = frame
        chunks_sha[strategy] = sha256_bytes((cfg.chunks_dir / f"{strategy}.parquet").read_bytes())

    current_identity = identity_hash(cfg)
    existing = {
        strategy: read_json(cfg.manifest_path(strategy))
        for strategy in selected
        if cfg.manifest_path(strategy).exists()
    }
    for strategy, manifest in existing.items():
        if manifest.get("embedder_identity_sha256") != current_identity and not force:
            raise RuntimeError(
                f"koleksi {cfg.collection_name(strategy)} dibangun dengan embedder lain "
                f"({manifest.get('embedder', {}).get('model')} @ {manifest.get('embedder', {}).get('max_seq_length')} token). "
                "Bangun ulang dengan --force kalau memang ingin menggantinya."
            )

    documents = pd.read_parquet(
        snap / "documents.parquet", columns=["doc_id", "url", "revid", "license_name", "attribution"]
    )
    cache = EmbeddingCache(cache_path(cfg), cfg.embedder.dimension)
    pending = {
        strategy: sum(1 for row in frame.itertuples() if cache.get(row.chunk_id, row.text) is None)
        for strategy, frame in frames.items()
    }
    token_stats = measure_tokens(cfg, frames, token_counter)
    preflight_truncation(cfg, token_stats)

    truncation = _truncation_from_stats(token_stats, cfg)
    if set(selected) != set(cfg.strategies):
        # A partial build must still record what the whole configured corpus
        # looks like, otherwise the identity document silently describes only
        # the strategies that happened to run.
        truncation = truncation_summary(cfg, token_counter)

    if embedder is None and any(pending.values()):
        logger.info("memuat embedder %s (%s)", cfg.embedder.model, cfg.embedder.device)
        embedder = make_embedder(cfg.embedder)
    identity = None
    if embedder is not None:
        identity = build_identity(cfg, embedder.describe(), truncation)
    else:
        identity = load_identity(cfg)
        if identity is not None:
            # Keep the measured runtime/digest from the real load, refresh the
            # truncation numbers (which depend on chunk files, not the model).
            identity = dict(identity, truncation=truncation)
    if identity is not None:
        save_identity(cfg, identity, cfg.models_docs_dir)

    manifests: dict[str, dict] = {}
    run: dict[str, dict] = {}
    versions = {
        strategy: compute_index_version(
            cfg, strategy, chunks_sha[strategy], snapshot["snapshot_id"], limit
        )
        for strategy in selected
    }

    # Phase 1: decide what can be reused, and drop every collection that must be
    # rebuilt. Recreating inside the same client instance leaves a stale
    # in-memory collection in Qdrant local mode, so the client is reopened.
    cached: dict[str, bool] = {}
    stale: list[str] = []
    client = make_client(cfg.qdrant, cfg.qdrant_dir)
    try:
        for strategy in selected:
            collection = cfg.collection_name(strategy)
            exists = client.collection_exists(collection)
            reusable = (
                strategy in existing
                and not force
                and existing[strategy].get("index_version") == versions[strategy]
                and exists
                and collection_count(client, collection) == existing[strategy]["counts"]["points"]
            )
            cached[strategy] = reusable
            if reusable:
                logger.info("%s sudah lengkap (%s titik), 0 embed baru", collection, existing[strategy]["counts"]["points"])
            elif exists:
                client.delete_collection(collection)
                stale.append(collection)
    finally:
        client.close()
    # Qdrant local leaves the collection directory behind after a delete, and on
    # Windows the sqlite handle is only released once the client is collected.
    # A stale directory keeps the old vector size, so it must really be gone.
    del client
    gc.collect()
    for name in stale:
        shutil.rmtree(cfg.qdrant_dir / "collection" / name, ignore_errors=True)

    # Phase 2: embed what is missing, then write each collection.
    client = make_client(cfg.qdrant, cfg.qdrant_dir)
    try:
        for strategy in selected:
            collection = cfg.collection_name(strategy)
            if cached[strategy]:
                manifests[strategy] = dict(existing[strategy])
                run[strategy] = {
                    "cached": True,
                    "embedded_now": 0,
                    "build_seconds": 0.0,
                    "cache_saves": cache.saves,
                    "collection": collection,
                }
                continue

            frame = frames[strategy]
            started_strategy = time.perf_counter()
            embedded_now = (
                _embed_missing(frame, cache, embedder, batch_size=cfg.embedder.batch_size)
                if embedder is not None
                else 0
            )
            vectors = _unit_vectors(frame, cache, cfg.embedder.dimension)
            create_collection(client, collection, cfg.embedder.dimension, cfg.qdrant.distance, recreate=False)
            upsert_points(client, collection, _payloads(frame, documents, strategy, collection), vectors)
            stored = collection_count(client, collection)
            build_bm25(frame["text"].tolist(), cfg.bm25, cfg.bm25_dir(strategy), chunk_ids=frame["chunk_id"].tolist())
            build_seconds = round(time.perf_counter() - started_strategy, 1)

            manifest = {
                "index_version": versions[strategy],
                "snapshot": cfg.snapshot,
                "snapshot_id": snapshot["snapshot_id"],
                "strategy": strategy,
                "collection": collection,
                "chunks_sha256": chunks_sha[strategy],
                "embedder": cfg.embedder.model_dump(),
                "embedder_identity": embedder_identity(cfg),
                "embedder_identity_sha256": current_identity,
                "identity_doc_sha256": identity_sha256(identity) if identity else None,
                "qdrant": {"distance": cfg.qdrant.distance, "on_disk": cfg.qdrant.on_disk},
                "bm25": cfg.bm25.model_dump(),
                "limit": limit,
                "counts": {"points": stored, "chunks": int(len(frame))},
                "token_stats": token_stats[strategy],
                "vectors_sha256": sha256_bytes(vectors.tobytes()),
                "files": {
                    "embeddings_cache": cache.path.name,
                    "bm25": f"bm25/{strategy}",
                },
            }
            manifests[strategy] = manifest
            run[strategy] = {
                "cached": False,
                "embedded_now": embedded_now,
                "build_seconds": build_seconds,
                "cache_saves": cache.saves,
                "collection": collection,
            }
            logger.info(
                "%s: %s titik, %s chunk di-embed sekarang (%.1f s)",
                collection,
                stored,
                embedded_now,
                build_seconds,
            )
        collections = list_collections(client)
    finally:
        client.close()

    # Sizes need the database closed, and are run facts rather than hashed inputs.
    for strategy, manifest in manifests.items():
        collection_dir = cfg.qdrant_dir / "collection" / cfg.collection_name(strategy)
        run[strategy]["index_size_bytes"] = _dir_bytes(collection_dir) + _dir_bytes(cfg.bm25_dir(strategy))
        # The identity document can be rewritten without touching the vectors
        # (e.g. a refreshed truncation count), so re-record its hash here.
        manifest["identity_doc_sha256"] = identity_sha256(identity) if identity else None
        write_json(cfg.manifest_path(strategy), manifest)

    finished = datetime.now(timezone.utc)
    history = _load_history(cfg, snapshot["snapshot_id"])
    for strategy, facts in run.items():
        history.append(
            {
                "started_at": started.isoformat(timespec="seconds"),
                "finished_at": finished.isoformat(timespec="seconds"),
                "strategy": strategy,
                "index_version": manifests[strategy]["index_version"],
                "cached": facts["cached"],
                "embedded_now": facts["embedded_now"],
                "build_seconds": facts["build_seconds"],
                "index_size_bytes": facts["index_size_bytes"],
            }
        )
    provenance = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "snapshot": cfg.snapshot,
        "snapshot_id": snapshot["snapshot_id"],
        "database": str(cfg.qdrant_dir),
        "collections": collections,
        "embedder": cfg.embedder.model_dump(),
        "embedder_identity_sha256": current_identity,
        "cache_file": cache.path.name,
        "cache_size": len(cache),
        "runs": run,
        "history": history,
    }
    write_json(cfg.provenance_path, provenance)
    _write_report(cfg, manifests, provenance)
    missing = [name for name in cfg.strategies if name not in manifests]
    if not missing:
        _write_a1_doc(cfg, manifests, provenance)
    else:
        logger.warning("dokumen A1 tidak ditulis: strategi belum lengkap (%s)", missing)
    return {"manifests": manifests, "run": run, "provenance": provenance}


def _load_history(cfg: IndexConfig, snapshot_id: str) -> list[dict]:
    """Append-only build history: a cached re-run must not erase real timings."""
    if not cfg.provenance_path.exists():
        return []
    try:
        previous = read_json(cfg.provenance_path)
    except Exception:  # pragma: no cover - corrupt provenance should not block a build
        return []
    if previous.get("snapshot_id") != snapshot_id:
        return []
    history = previous.get("history")
    return list(history) if isinstance(history, list) else []


def build_timings(provenance: dict) -> dict[str, dict]:
    """Last *real* build per strategy, so reports quote measured work, not cache hits."""
    timings: dict[str, dict] = {}
    for entry in provenance.get("history", []):
        if entry.get("cached"):
            continue
        timings[entry["strategy"]] = entry
    if not timings:  # older provenance without history
        timings = {name: facts for name, facts in provenance.get("runs", {}).items()}
    return timings


def _write_report(cfg: IndexConfig, manifests: dict[str, dict], provenance: dict) -> None:
    lines = [
        "# Index report",
        "",
        f"- snapshot_id: `{provenance['snapshot_id']}`",
        f"- database: `{provenance['database']}`",
        f"- koleksi: {', '.join(provenance['collections']) or 'kosong'}",
        f"- embedder: `{cfg.embedder.model}` @ `{cfg.embedder.revision or 'unpinned'}` "
        f"({cfg.embedder.dimension} dim, {cfg.embedder.dtype}, max {cfg.embedder.max_seq_length} token)",
        "",
        "| Koleksi | Titik | Chunk | p50 token | p95 token | max | > jendela | ctx@k=5 (p50) | index size | embed baru | build |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for strategy in cfg.strategies:
        manifest = manifests.get(strategy)
        if manifest is None:
            lines.append(f"| `{cfg.collection_name(strategy)}` | belum dibangun | | | | | | | | | |")
            continue
        tokens = manifest["token_stats"]
        facts = build_timings(provenance).get(strategy, {})
        size_mb = facts.get("index_size_bytes", 0) / 1024 / 1024
        lines.append(
            f"| `{manifest['collection']}` | {manifest['counts']['points']} | {manifest['counts']['chunks']} | "
            f"{tokens['tokens_p50']} | {tokens['tokens_p95']} | {tokens['tokens_max']} | "
            f"{tokens['chunks_over_window']} | {tokens['tokens_p50'] * 5} | {size_mb:.1f} MB | "
            f"{facts.get('embedded_now', '-')} | {facts.get('build_seconds', '-')} s |"
        )
    lines += [
        "",
        "`ctx@k=5` adalah perkiraan token konteks yang masuk generator bila 5 chunk teratas",
        "digabung, memakai token median. `> jendela` harus 0: chunk di atas `max_seq_length`",
        "akan dipotong sebelum di-embed, sehingga vektornya tidak mewakili teks utuh.",
        "Waktu build dan jumlah embed baru diambil dari build nyata terakhir per koleksi",
        "(riwayat di `provenance.json`), bukan dari run cache yang hanya membaca ulang.",
        "",
    ]
    atomic_write_text(cfg.report_path, "\n".join(lines))


def _write_a1_doc(cfg: IndexConfig, manifests: dict[str, dict], provenance: dict) -> None:
    """Committed copy of the A1 table: artefacts are git-ignored, this is not."""
    chunks_manifest_path = cfg.snapshot_dir / "chunks_manifest.json"
    chunks_manifest = read_json(chunks_manifest_path) if chunks_manifest_path.exists() else {"strategies": {}}
    rows = []
    parent_notes = []
    for strategy in cfg.strategies:
        manifest = manifests[strategy]
        entry = chunks_manifest.get("strategies", {}).get(strategy, {})
        tokens = manifest["token_stats"]
        facts = build_timings(provenance).get(strategy, {})
        units = manifest["counts"]["chunks"]
        parents = int(entry.get("n_chunks", units)) - units
        if parents > 0:
            parent_notes.append(f"`{strategy}` juga menyimpan {parents:,} baris parent untuk context expansion (tidak di-embed).")
        rows.append(
            f"| `{strategy}` | {units} | "
            f"{manifest['counts']['points']} | {tokens['tokens_p50']} / {tokens['tokens_p95']} | "
            f"{tokens['tokens_max']} | {tokens['tokens_p50'] * 5} | "
            f"{facts.get('index_size_bytes', 0) / 1024 / 1024:.1f} MB | "
            f"{facts.get('build_seconds', '-')} s | "
            f"{_coverage(entry)} | {manifest['index_version'][:12]} |"
        )
    identity = read_json(cfg.identity_path) if cfg.identity_path.exists() else None
    identity_line = (
        f"`{identity['model']['id']}` @ `{identity['model']['revision_resolved']}`, "
        f"{identity['dimension']} dim, {identity['quantization']['inference']}, "
        f"jendela {identity['context']['used_max_tokens']} token"
        if identity
        else "belum dicatat"
    )
    doc = "\n".join(
        [
            "# A1 — perbandingan strategi chunking (sisi index)",
            "",
            "> Dibuat otomatis oleh `pokerag-index build`; angka berasal dari manifest",
            "> yang hash-nya tercatat di tabel. Artefak besar ada di HF (v2), bukan di git.",
            "",
            f"- snapshot: `{cfg.snapshot}` — `{provenance['snapshot_id'][:16]}`",
            f"- embedder: {identity_line}",
            f"- database: satu Qdrant, {len(cfg.strategies)} koleksi "
            f"({', '.join(cfg.collection_name(name) for name in cfg.strategies)})",
            "- vektor: L2-normalised, cosine, BM25 terpisah per koleksi",
            "",
            "## Hasil",
            "",
            "| Strategi | Chunk | Titik | token p50 / p95 | max | ctx@k=5 | index size | build | cakupan teks | index_version |",
            "|---|---|---|---|---|---|---|---|---|---|",
            *rows,
            "",
            "## Definisi",
            "",
            "| Metrik | Definisi |",
            "|---|---|",
            "| Chunk | unit retrieval yang di-embed (baris di `chunks/<strategi>.parquet` dengan `role != parent`) |",
            "| Titik | vektor di koleksi Qdrant (satu per unit retrieval) |",
            "| token p50/p95 | tokenizer model embedding, `add_special_tokens=false` |",
            "| max | chunk terpanjang; harus ≤ jendela embedder |",
            "| ctx@k=5 | 5 × token p50 — perkiraan konteks yang masuk generator |",
            "| index size | byte di disk: direktori koleksi Qdrant + BM25 |",
            "| build | waktu build nyata terakhir per koleksi, dari riwayat `provenance.json` (run cache tidak dihitung) |",
            "| cakupan teks | proporsi dokumen yang tercakup minimal satu chunk (gabungan span) |",
            "| index_version | hash snapshot, chunk, strategi, identitas embedder, metrik jarak, BM25 |",
            "",
            *parent_notes,
            "",
            "## Tertunda (belum diukur, bukan nol)",
            "",
            "| Metrik | Kenapa belum ada | Cara mengukurnya nanti |",
            "|---|---|---|",
            "| passage recall@k | butuh qrels; belum dibuat | qrels menyimpan `doc_id` + rentang `char_start/char_end`; hit dihitung bila chunk top-k berasal dari dokumen yang sama dan tumpang-tindih dengan rentang gold (dua varian: ≥1 char, gold tertutup penuh) |",
            "| groundedness | butuh generator (B0–B6) | jawaban dinilai terhadap chunk yang dikutip, dengan rubrik yang dicatat di laporan tahap generator |",
            "",
            "Kedua metrik di atas **tidak** boleh dilaporkan sebagai 0 atau \"tidak ada\";",
            "selama belum diukur, selnya tetap kosong dengan alasan ini.",
            "",
            "## Reproduksi",
            "",
            "```bash",
            "pokerag-chunk build",
            "pokerag-index build",
            "pokerag-index stats",
            "```",
            "",
        ]
    )
    atomic_write_text(cfg.experiments_docs_dir / "A1-index.md", doc)


def _coverage(entry: dict) -> str:
    ratio = entry.get("coverage_ratio")
    return f"{ratio:.1%}" if isinstance(ratio, (int, float)) else "-"
