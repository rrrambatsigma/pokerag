"""Build chunk files for every strategy and report A1 diagnostics.

Outputs per strategy:

* ``chunks/<strategy>.parquet`` — the retrieval units;
* ``chunks_manifest.json`` — counts, span checks, config and snapshot identity;
* ``chunks_report.md`` — the table used in the ablation write-up.

The build fails loudly if any chunk's text does not equal the document slice it
claims to cover, because that would silently break citations and qrels.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from pokerag.chunking.config import ChunkingConfig
from pokerag.chunking.models import Chunk, StrategyStats
from pokerag.chunking.split import merge_spans
from pokerag.chunking.strategies import fixed_chunks, heading_chunks, hierarchical_chunks
from pokerag.chunking.tokens import TokenCounter
from pokerag.core.hashing import sha256_bytes, sha256_json
from pokerag.core.jsonl import read_json, write_json
from pokerag.core.logging import get_logger

logger = get_logger("pokerag.chunking.build")

STRATEGIES = ("fixed", "heading", "hierarchical")


class ChunkInvariantError(RuntimeError):
    """Raised when a chunk does not match its document span."""


def parse_sections(raw: object) -> list[dict]:
    """Decode ``sections_json``, tolerating the pandas ``{"0": [...]}`` wrapper."""
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, dict):
        values = list(data.values())
        data = values[0] if values else []
    return list(data)


def _chunk_rows(chunks: list[Chunk]) -> list[dict]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "revid": chunk.revid,
            "strategy": chunk.strategy,
            "parent_id": chunk.parent_id,
            "role": chunk.role,
            "heading_path": json.dumps(chunk.heading_path, ensure_ascii=False),
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "text": chunk.text,
            "n_chars": chunk.n_chars,
            "n_tokens": chunk.n_tokens,
            "order": chunk.order,
        }
        for chunk in chunks
    ]


def _stats(
    strategy: str,
    chunks: list[Chunk],
    documents: dict[str, int],
    budget_tokens: int,
    min_chars: int,
    dropped: int = 0,
) -> StrategyStats:
    lengths = sorted(chunk.n_chars for chunk in chunks)
    by_doc: dict[str, list[tuple[int, int]]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append((chunk.char_start, chunk.char_end))
    total_covered = 0
    for doc_id, doc_length in documents.items():
        for start, end in merge_spans(by_doc.get(doc_id, [])):
            left, right = max(0, min(start, doc_length)), max(0, min(end, doc_length))
            if right > left:
                total_covered += right - left
    embedded = [chunk for chunk in chunks if chunk.role != "parent"]
    return StrategyStats(
        strategy=strategy,
        n_chunks=len(chunks),
        p50_chars=lengths[len(lengths) // 2] if lengths else 0,
        p95_chars=lengths[int(0.95 * (len(lengths) - 1))] if lengths else 0,
        max_chars=lengths[-1] if lengths else 0,
        chunks_over_budget=sum(1 for chunk in chunks if chunk.n_tokens > budget_tokens),
        chunks_below_min=sum(1 for chunk in chunks if chunk.n_chars < min_chars),
        total_tokens=sum(chunk.n_tokens for chunk in embedded),
        documents_covered=len(by_doc),
        coverage_ratio=round(total_covered / max(1, sum(documents.values())), 4),
        chunks_dropped=dropped,
    )


def build(cfg: ChunkingConfig, only: str = "all") -> dict:
    """Build the requested strategies and write manifest + report."""
    snap = cfg.snapshot_dir
    documents = pd.read_parquet(snap / "documents.parquet", columns=["doc_id", "revid", "text", "sections_json"])
    snapshot = read_json(snap / "snapshot.json")
    counter = TokenCounter(cfg.tokenizer)
    cfg.chunks_dir.mkdir(parents=True, exist_ok=True)

    lengths = {row.doc_id: len(row.text) for row in documents.itertuples()}
    selected = STRATEGIES if only == "all" else (only,)
    stats: dict[str, dict] = {}

    for strategy in selected:
        chunks: list[Chunk] = []
        dropped = 0
        for row in documents.itertuples():
            sections = parse_sections(row.sections_json) if strategy != "fixed" else []
            if strategy == "fixed":
                produced = fixed_chunks(
                    doc_id=row.doc_id, revid=int(row.revid), text=row.text, cfg=cfg, counter=counter
                )
            elif strategy == "heading":
                produced = heading_chunks(
                    doc_id=row.doc_id,
                    revid=int(row.revid),
                    text=row.text,
                    sections=sections,
                    cfg=cfg,
                    counter=counter,
                )
            elif strategy == "hierarchical":
                produced = hierarchical_chunks(
                    doc_id=row.doc_id,
                    revid=int(row.revid),
                    text=row.text,
                    sections=sections,
                    cfg=cfg,
                    counter=counter,
                )
            else:
                raise ValueError(f"unknown strategy: {strategy}")

            for chunk in produced:
                if row.text[chunk.char_start : chunk.char_end] != chunk.text:
                    raise ChunkInvariantError(
                        f"{chunk.chunk_id}: span tidak cocok untuk {row.doc_id} "
                        f"[{chunk.char_start}:{chunk.char_end}]"
                    )
            kept = [chunk for chunk in produced if chunk.n_chars >= cfg.min_chars_keep]
            dropped += len(produced) - len(kept)
            chunks.extend(kept)

        output = cfg.chunks_dir / f"{strategy}.parquet"
        pd.DataFrame(_chunk_rows(chunks)).to_parquet(output, index=False)

        min_chars = 0
        if strategy == "heading":
            min_chars = cfg.heading.min_chars
        elif strategy == "hierarchical":
            min_chars = cfg.hierarchical.min_chars
        entry = _stats(strategy, chunks, lengths, cfg.budget_tokens, min_chars, dropped=dropped).model_dump()
        entry["file_sha256"] = sha256_bytes(output.read_bytes())
        entry["file_bytes"] = output.stat().st_size
        entry["estimated_vectors_mb"] = round(
            sum(1 for chunk in chunks if chunk.role != "parent")
            * cfg.embedding_dimension
            * 4
            / 1024
            / 1024,
            1,
        )
        stats[strategy] = entry
        logger.info("%s: %s chunks (%s)", strategy, len(chunks), output.name)

    manifest = {
        "snapshot": cfg.snapshot,
        "snapshot_id": snapshot["snapshot_id"],
        "documents_parquet_sha256": sha256_bytes((snap / "documents.parquet").read_bytes()),
        "chunking_config_sha256": sha256_json(cfg.model_dump()),
        "tokenizer": cfg.tokenizer,
        "budget_tokens": cfg.budget_tokens,
        "strategies": stats,
    }
    previous = snap / "chunks_manifest.json"
    if previous.exists():
        merged = read_json(previous)
        if merged.get("snapshot_id") == manifest["snapshot_id"]:
            merged["strategies"].update(stats)
            manifest["strategies"] = merged["strategies"]
    write_json(previous, manifest)
    _write_report(snap / "chunks_report.md", manifest)
    return manifest


def _write_report(path: Path, manifest: dict) -> None:
    lines = [
        f"# Chunking report - {manifest['snapshot']}",
        "",
        f"Snapshot: `{manifest['snapshot_id'][:16]}` - tokenizer: `{manifest['tokenizer']}` "
        f"- budget: {manifest['budget_tokens']} token",
        "",
        "| Strategi | Chunk | p50 char | p95 char | max | > budget | < min | dibuang | total token | cakupan | vektor (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in manifest["strategies"].items():
        lines.append(
            f"| {name} | {entry['n_chunks']} | {entry['p50_chars']} | {entry['p95_chars']} | "
            f"{entry['max_chars']} | {entry['chunks_over_budget']} | {entry['chunks_below_min']} | "
            f"{entry.get('chunks_dropped', 0)} | "
            f"{entry['total_tokens']} | {entry['coverage_ratio']:.1%} | {entry['estimated_vectors_mb']} |"
        )
    lines += [
        "",
        "`> budget` dihitung dengan tokenizer yang tercatat di atas; `cakupan` adalah proporsi "
        "teks dokumen yang tercakup minimal satu chunk (gabungan span). `dibuang` adalah chunk di "
        "bawah min_chars_keep yang tidak membawa kalimat utuh (mis. tautan 'Main article').",
        "Pada `hierarchical`, parent tidak di-embed: `total token` dan `vektor` hanya menghitung child.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
