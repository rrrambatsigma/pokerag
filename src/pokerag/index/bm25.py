"""BM25 access over the bm25s sidecars shipped with the corpus."""

from __future__ import annotations

import json
from pathlib import Path

from pokerag.index.store import ScoredChunk


class BM25Store:
    """Lexical retrieval that must match the index build exactly.

    The shipped vocabulary keeps stopwords and is not stemmed, so queries are
    tokenized with ``stopwords=None`` and ``stemmer=None`` and then mapped
    through the retriever's own vocabulary. Using bm25s defaults would silently
    drop "the" and shift every token id.
    """

    def __init__(self, directory: Path) -> None:
        import bm25s

        self._bm25s = bm25s
        self._retriever = bm25s.BM25.load(
            str(directory),
            load_corpus=False,
            show_progress=False,
            verbosity=0,
        )
        self._chunk_ids = json.loads(
            (directory / "chunk_ids.json").read_text(encoding="utf-8")
        )
        self.strategy = directory.name

    def search(self, query: str, limit: int) -> list[ScoredChunk]:
        if not query.strip():
            return []
        tokenized = self._bm25s.tokenize(
            [query],
            stopwords=None,
            stemmer=None,
            return_ids=False,
            show_progress=False,
        )
        vocab = self._retriever.vocab_dict
        query_ids = [
            [vocab[token] for token in tokens if token in vocab] for tokens in tokenized
        ]
        if not query_ids or not query_ids[0]:
            return []
        k = min(limit, len(self._chunk_ids))
        results, scores = self._retriever.retrieve(
            query_ids, k=k, show_progress=False
        )
        return [
            ScoredChunk(self._chunk_ids[int(index)], float(score))
            for index, score in zip(results[0], scores[0], strict=True)
        ]
