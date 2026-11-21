"""Chunking stage: documents -> retrieval units with exact character spans.

The invariant is the same as in cleaning: ``chunk.text == doc.text[char_start:char_end]``.
Without it, a retrieved chunk cannot be mapped back to a citation, and qrels
built on document spans cannot be reused across chunking strategies (A1).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Strategy = Literal["fixed", "heading", "hierarchical"]


class Chunk(BaseModel):
    """One retrieval unit."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    revid: int
    strategy: Strategy
    parent_id: str | None = None
    role: Literal["unit", "parent", "child"] = "unit"
    heading_path: list[str] = Field(default_factory=list)
    char_start: int
    char_end: int
    text: str
    n_chars: int
    n_tokens: int
    order: int


class StrategyStats(BaseModel):
    """Diagnostics for one strategy, used by ablation A1."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    n_chunks: int
    p50_chars: int
    p95_chars: int
    max_chars: int
    chunks_over_budget: int
    chunks_below_min: int
    total_tokens: int
    documents_covered: int
    coverage_ratio: float
    chunks_dropped: int = 0
