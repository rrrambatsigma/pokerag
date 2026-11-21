"""Types shared by the retrieval stage.

``RawHit`` is what a retriever returns (id + score); ``Hit`` is what the
pipeline returns (id + score + the chunk text, heading path and citation span).
Keeping them separate means fusion only ever sees ids and ranks, which is what
reciprocal-rank fusion needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A retrieval mode decides which retrievers run; "hybrid" runs both and fuses.
MODES = ("dense", "bm25", "hybrid")


@dataclass(frozen=True)
class ChunkText:
    """The fields a citation needs, read from ``chunks/<strategy>.parquet``."""

    chunk_id: str
    doc_id: str
    text: str
    heading_path: list[str]
    char_start: int
    char_end: int
    url: str = ""


@dataclass(frozen=True)
class RawHit:
    """One retriever's opinion: which chunk, how strongly, and its payload."""

    chunk_id: str
    score: float
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Fused:
    """One fused id with the rank it held in each source ranking."""

    chunk_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Hit:
    """A ranked, citable chunk."""

    chunk_id: str
    score: float
    rank: int
    source: str  # "dense", "bm25" or "hybrid"
    doc_id: str
    text: str
    heading_path: list[str]
    char_start: int
    char_end: int
    url: str
    ranks: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "source": self.source,
            "ranks": dict(self.ranks),
            "doc_id": self.doc_id,
            "heading_path": list(self.heading_path),
            "char_start": self.char_start,
            "char_end": self.char_end,
            "url": self.url,
            "text": self.text,
        }
