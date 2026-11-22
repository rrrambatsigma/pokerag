"""Build the pooled annotation set for A2.

A2 compares dense, BM25, hybrid, and hybrid+reranking on the same qrels, and
reports Recall@5/10, nDCG@10, and MRR@10. That only works if every scored method
contributes its top-``depth`` chunks to the pool, otherwise a method that finds
evidence at rank 8 is punished for a judgment nobody made.

Two files come out on purpose:

* ``pool.jsonl`` is what the annotation tool loads. It carries no method names,
  so the annotator judges the passage and not the retriever that found it.
* ``pool_sources.jsonl`` keeps which methods retrieved each chunk and at what
  rank. It is for analysis and adjudication, never for the annotator UI.

Order is shuffled with a seed derived from ``(seed, query_id)``, so the same
command produces byte-identical files.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from pokerag.core.hashing import sha256_json, sha256_text
from pokerag.core.jsonl import write_json, write_jsonl
from pokerag.retrieval.types import Hit

METHODS: tuple[str, ...] = ("dense", "bm25", "hybrid", "rerank")
"""A2: dense, BM25, hybrid, hybrid+reranking."""

FORMAT = "pool-v1"


class PoolError(RuntimeError):
    """Raised when a pool would be invalid for the annotation task."""


@dataclass(frozen=True)
class PoolEntry:
    chunk_id: str
    doc_id: str
    revid: int
    char_start: int
    char_end: int
    heading_path: tuple[str, ...]
    text: str
    sources: tuple[str, ...]
    ranks: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PoolQuery:
    query_id: str
    query: str
    order: tuple[str, ...]
    entries: tuple[PoolEntry, ...]

    def by_id(self) -> dict[str, PoolEntry]:
        return {entry.chunk_id: entry for entry in self.entries}

    def ui_dict(self) -> dict:
        """What the annotation tool sees: passages, no method labels."""
        entries = self.by_id()
        return {
            "query_id": self.query_id,
            "query": self.query,
            "chunks": [
                {
                    "chunk_id": entries[chunk_id].chunk_id,
                    "doc_id": entries[chunk_id].doc_id,
                    "revid": entries[chunk_id].revid,
                    "char_start": entries[chunk_id].char_start,
                    "char_end": entries[chunk_id].char_end,
                    "heading": list(entries[chunk_id].heading_path),
                    "text": entries[chunk_id].text,
                }
                for chunk_id in self.order
            ],
        }

    def sources_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "sources": {
                entry.chunk_id: {"methods": list(entry.sources), "ranks": dict(entry.ranks)}
                for entry in self.entries
            },
        }


def _order(seed: int, query_id: str, chunk_ids: Iterable[str]) -> tuple[str, ...]:
    identifiers = sorted(chunk_ids)
    random.Random(int(sha256_text(f"{seed}|{query_id}")[:16], 16)).shuffle(identifiers)
    return tuple(identifiers)


def build_query_pool(
    query_id: str,
    query: str,
    hits_by_method: Mapping[str, Sequence[Hit]],
    *,
    depth: int,
    seed: int,
    revid_by_doc: Mapping[str, int],
) -> PoolQuery:
    """Merge one query's four ranked lists into a blind, deduplicated pool."""
    if depth < 1:
        raise PoolError("depth harus minimal 1")
    missing = [method for method in METHODS if method not in hits_by_method]
    if missing:
        raise PoolError(f"{query_id}: metode tidak dijalankan: {missing}")

    entries: dict[str, dict] = {}
    for method in METHODS:
        hits = list(hits_by_method[method])[:depth]
        if not hits:
            continue
        for rank, hit in enumerate(hits, start=1):
            current = entries.setdefault(
                hit.chunk_id,
                {
                    "hit": hit,
                    "sources": [],
                    "ranks": {},
                },
            )
            current["sources"].append(method)
            current["ranks"][method] = rank

    for method in METHODS:
        top = {hit.chunk_id for hit in list(hits_by_method[method])[:depth]}
        absent = top - set(entries)
        if absent:
            raise PoolError(f"{query_id}: top-{depth} {method} hilang dari pool: {sorted(absent)}")

    for chunk_id, payload in entries.items():
        if not payload["hit"].text.strip():
            raise PoolError(f"{query_id}: chunk {chunk_id} tanpa teks")

    order = _order(seed, query_id, entries)
    resolved = tuple(
        PoolEntry(
            chunk_id=chunk_id,
            doc_id=entries[chunk_id]["hit"].doc_id,
            revid=int(revid_by_doc.get(entries[chunk_id]["hit"].doc_id, 0)),
            char_start=entries[chunk_id]["hit"].char_start,
            char_end=entries[chunk_id]["hit"].char_end,
            heading_path=tuple(entries[chunk_id]["hit"].heading_path),
            text=entries[chunk_id]["hit"].text,
            sources=tuple(sorted(entries[chunk_id]["sources"])),
            ranks=dict(entries[chunk_id]["ranks"]),
        )
        for chunk_id in order
    )
    return PoolQuery(query_id=query_id, query=query, order=order, entries=resolved)


def build_pool(
    queries: Sequence[tuple[str, str]],
    search,
    rerank,
    *,
    depth: int,
    candidates: int,
    seed: int,
    revid_by_doc: Mapping[str, int],
) -> list[PoolQuery]:
    """Run the four methods for every query and merge the results.

    ``search(query, mode, top_k)`` and ``rerank(query, hits, top_k)`` are
    injected so the pooling logic can be tested without a GPU or an index.
    """
    records: list[PoolQuery] = []
    for query_id, text in queries:
        hits_by_method = {
            "dense": search(text, "dense", depth),
            "bm25": search(text, "bm25", depth),
            "hybrid": search(text, "hybrid", depth),
        }
        pool_hits = search(text, "hybrid", candidates)
        hits_by_method["rerank"] = rerank(text, pool_hits, depth)
        records.append(
            build_query_pool(
                query_id,
                text,
                hits_by_method,
                depth=depth,
                seed=seed,
                revid_by_doc=revid_by_doc,
            )
        )
    return records


def validate(records: Sequence[PoolQuery]) -> dict:
    """Structural checks that must pass before annotators see the pool."""
    problems: list[str] = []
    seen: set[str] = set()
    for record in records:
        if record.query_id in seen:
            problems.append(f"query_id duplikat: {record.query_id}")
        seen.add(record.query_id)

        identifiers = [entry.chunk_id for entry in record.entries]
        if len(set(identifiers)) != len(identifiers):
            problems.append(f"{record.query_id}: chunk duplikat di dalam pool")
        if tuple(identifiers) != record.order:
            problems.append(f"{record.query_id}: urutan tidak cocok dengan daftar chunk")
        if not record.entries:
            problems.append(f"{record.query_id}: pool kosong")
        for entry in record.entries:
            if not entry.sources:
                problems.append(f"{record.query_id}:{entry.chunk_id}: tanpa metode asal")
            if entry.revid <= 0:
                problems.append(f"{record.query_id}:{entry.chunk_id}: revid tidak diketahui")
    return {"ok": not problems, "problems": problems, "queries": len(records)}


def _method_stats(records: Sequence[PoolQuery]) -> dict:
    exclusive: Counter = Counter()
    hits: Counter = Counter()
    overlap = 0
    for record in records:
        for entry in record.entries:
            hits.update(entry.sources)
            if len(entry.sources) > 1:
                overlap += 1
            if len(entry.sources) == 1:
                exclusive[entry.sources[0]] += 1
    return {
        "retrieved_by_method": {method: hits.get(method, 0) for method in METHODS},
        "exclusive_by_method": {method: exclusive.get(method, 0) for method in METHODS},
        "found_by_two_or_more": overlap,
    }


def manifest(
    records: Sequence[PoolQuery],
    *,
    depth: int,
    seed: int,
    meta: Mapping[str, object],
    pool_sha256: str,
    sources_sha256: str,
) -> dict:
    sizes = [len(record.entries) for record in records]
    stats = _method_stats(records)
    return {
        "format": FORMAT,
        "depth": depth,
        "methods": list(METHODS),
        "seed": seed,
        "n_queries": len(records),
        "n_passages": sum(sizes),
        "passages_per_query": {
            "min": min(sizes) if sizes else 0,
            "mean": round(sum(sizes) / len(sizes), 2) if sizes else 0,
            "max": max(sizes) if sizes else 0,
        },
        **stats,
        "pool_sha256": pool_sha256,
        "sources_sha256": sources_sha256,
        **dict(meta),
    }


def write(
    out_dir: Path,
    records: Sequence[PoolQuery],
    *,
    depth: int,
    seed: int,
    meta: Mapping[str, object],
) -> dict:
    """Write the blind pool, the sources file, and the manifest."""
    report = validate(records)
    if not report["ok"]:
        raise PoolError("pool tidak valid: " + "; ".join(report["problems"][:5]))

    pool_path = out_dir / "pool.jsonl"
    sources_path = out_dir / "pool_sources.jsonl"
    write_jsonl(pool_path, [record.ui_dict() for record in records])
    write_jsonl(sources_path, [record.sources_dict() for record in records])

    payload = manifest(
        records,
        depth=depth,
        seed=seed,
        meta=meta,
        pool_sha256=sha256_json([record.ui_dict() for record in records]),
        sources_sha256=sha256_json([record.sources_dict() for record in records]),
    )
    write_json(out_dir / "manifest.json", payload)
    return payload
