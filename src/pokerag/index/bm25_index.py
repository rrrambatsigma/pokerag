"""BM25 index over chunk text.

Kept local and separate from the dense index so hybrid retrieval (B3) can fuse
two genuinely different rankings, and so ablation A2 can isolate BM25.

Row order is part of the contract: chunk ids are stored in a sidecar file in the
same order as the indexed rows, so a BM25 hit can always be mapped back to the
exact chunk (and therefore to a document span).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import bm25s

from pokerag.core.logging import get_logger
from pokerag.index.config import Bm25Config

logger = get_logger("pokerag.index.bm25")

IDS_FILE = "chunk_ids.json"


def build_bm25(
    texts: list[str], cfg: Bm25Config, path: Path, chunk_ids: Sequence[str]
) -> bm25s.BM25:
    """Tokenise, index, and persist a BM25 retriever plus its id mapping."""
    if len(texts) != len(chunk_ids):
        raise ValueError("jumlah teks dan chunk id harus sama")
    tokens = bm25s.tokenize(texts, stopwords=None, stemmer=None, show_progress=False)
    retriever = bm25s.BM25(k1=cfg.k1, b=cfg.b)
    retriever.index(tokens, show_progress=False)
    path.mkdir(parents=True, exist_ok=True)
    retriever.save(str(path))
    (path / IDS_FILE).write_text(
        json.dumps(list(chunk_ids), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("bm25: %s dokumen diindeks", len(texts))
    return retriever


def load_bm25(path: Path) -> tuple[bm25s.BM25, list[str]]:
    """Load the retriever and its chunk id mapping."""
    retriever = bm25s.BM25.load(str(path), load_corpus=False)
    chunk_ids = json.loads((path / IDS_FILE).read_text(encoding="utf-8"))
    return retriever, chunk_ids


def search_bm25(
    retriever: bm25s.BM25, chunk_ids: Sequence[str], query: str, k: int = 10
) -> list[tuple[str, float]]:
    """Return ``(chunk_id, score)`` pairs ranked by BM25."""
    tokens = bm25s.tokenize([query], stopwords=None, stemmer=None, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, show_progress=False)
    pairs: list[tuple[str, float]] = []
    for position, score in zip(results[0], scores[0]):
        index = int(position["id"]) if isinstance(position, dict) else int(position)
        pairs.append((chunk_ids[index], float(score)))
    return pairs
