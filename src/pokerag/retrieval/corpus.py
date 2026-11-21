"""Chunk text and citation metadata, read from the frozen snapshot.

Qdrant stores vectors and payloads, but the payload deliberately has no ``text``
field: sending 58k chunks of text into the vector database would duplicate the
snapshot. Retrieval therefore joins hits back to ``chunks/<strategy>.parquet``,
which also carries the character span needed for a citation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pokerag.core.logging import get_logger
from pokerag.index.config import IndexConfig
from pokerag.retrieval.types import ChunkText

logger = get_logger("pokerag.retrieval.corpus")

CHUNK_COLUMNS = ("chunk_id", "doc_id", "role", "order", "heading_path", "char_start", "char_end", "text")


def chunks_path(cfg: IndexConfig, strategy: str) -> Path:
    return cfg.chunks_dir / f"{strategy}.parquet"


def _heading_path(value) -> list[str]:
    if isinstance(value, str):
        return list(json.loads(value))
    if value is None:
        return []
    return list(value)


def load_urls(cfg: IndexConfig) -> dict[str, str]:
    """``doc_id -> url`` from the snapshot's documents table (best effort)."""
    path = cfg.snapshot_dir / "documents.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path, columns=["doc_id", "url"])
    return {str(row.doc_id): str(row.url) for row in frame.itertuples()}


def load_chunks(cfg: IndexConfig, strategy: str) -> dict[str, ChunkText]:
    """Read the retrieval units of one strategy, keyed by ``chunk_id``.

    The rows are filtered and ordered exactly like the index stage does
    (``role != "parent"``, then ``doc_id``/``order``) so that BM25's
    ``chunk_ids.json`` maps row numbers to the same chunks. ``hierarchical``
    stores parent sections in the same parquet for context expansion; those are
    not retrieval units and must not appear here.
    """
    path = chunks_path(cfg, strategy)
    if not path.exists():
        raise FileNotFoundError(
            f"chunk `{strategy}` tidak ada di {path}; jalankan `pokerag-chunk build --strategy {strategy}`"
        )
    frame = pd.read_parquet(path, columns=list(CHUNK_COLUMNS))
    frame = frame[frame["role"] != "parent"].copy()
    frame = frame.sort_values(["doc_id", "order"], kind="stable").reset_index(drop=True)
    urls = load_urls(cfg)
    table: dict[str, ChunkText] = {}
    for row in frame.itertuples():
        chunk_id = str(row.chunk_id)
        if chunk_id in table:
            raise RuntimeError(f"chunk_id duplikat di {path.name}: {chunk_id}")
        table[chunk_id] = ChunkText(
            chunk_id=chunk_id,
            doc_id=str(row.doc_id),
            text=str(row.text),
            heading_path=_heading_path(row.heading_path),
            char_start=int(row.char_start),
            char_end=int(row.char_end),
            url=urls.get(str(row.doc_id), ""),
        )
    logger.info("chunks: %s baris dari %s", len(table), path.name)
    return table
