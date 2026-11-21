"""Chunk metadata and document titles, loaded from the frozen parquet files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pokerag.core.config import settings

CHUNK_COLUMNS = (
    "chunk_id",
    "doc_id",
    "strategy",
    "role",
    "parent_id",
    "heading_path",
    "char_start",
    "char_end",
    "text",
    "n_tokens",
)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: str
    role: str
    parent_id: str | None
    heading_path: tuple[str, ...]
    char_start: int
    char_end: int
    text: str
    n_tokens: int

    @property
    def label(self) -> str:
        if not self.heading_path:
            return "(tanpa heading)"
        return " > ".join(self.heading_path)


def parse_heading_path(raw: object) -> tuple[str, ...]:
    """The parquet stores heading paths as JSON strings, sometimes as lists."""

    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text in ("", "[]", "null"):
            return ()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return ()
        if isinstance(value, list):
            return tuple(str(item) for item in value)
    return ()


class Catalog:
    """Lookup table from ``chunk_id`` to chunk metadata and document title."""

    def __init__(self, chunks: dict[str, Chunk], titles: dict[str, str]) -> None:
        self._chunks = chunks
        self._titles = titles

    @classmethod
    def load(cls, strategy: str) -> Catalog:
        chunks_path = settings.raw_dir / "chunks" / f"{strategy}.parquet"
        documents_path = settings.raw_dir / "documents.parquet"
        if not chunks_path.is_file():
            raise FileNotFoundError(
                f"missing {chunks_path}; run scripts/fetch_artifacts.py first"
            )
        chunks = pd.read_parquet(chunks_path, columns=list(CHUNK_COLUMNS))
        documents = pd.read_parquet(documents_path, columns=["doc_id", "title"])
        return cls.from_frames(chunks, documents)

    @classmethod
    def from_frames(cls, chunks: pd.DataFrame, documents: pd.DataFrame) -> Catalog:
        by_id: dict[str, Chunk] = {}
        for row in chunks.itertuples(index=False):
            parent = None if pd.isna(row.parent_id) else str(row.parent_id)
            by_id[str(row.chunk_id)] = Chunk(
                chunk_id=str(row.chunk_id),
                doc_id=str(row.doc_id),
                strategy=str(row.strategy),
                role=str(row.role),
                parent_id=parent,
                heading_path=parse_heading_path(row.heading_path),
                char_start=int(row.char_start),
                char_end=int(row.char_end),
                text=str(row.text),
                n_tokens=int(row.n_tokens),
            )
        titles = {
            str(row.doc_id): str(row.title) for row in documents.itertuples(index=False)
        }
        return cls(by_id, titles)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def title(self, doc_id: str) -> str:
        return self._titles.get(doc_id, doc_id)

    def __len__(self) -> int:
        return len(self._chunks)


def strategy_path(strategy: str) -> Path:
    return settings.raw_dir / "chunks" / f"{strategy}.parquet"
