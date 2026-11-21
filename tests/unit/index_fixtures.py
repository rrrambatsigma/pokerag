"""Shared builders for index and retrieval tests.

``FakeEmbedder`` is deterministic and offline, so a test can build a real
Qdrant collection in a temporary directory without a GPU or network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pokerag.core.jsonl import write_json
from pokerag.index.config import IndexConfig


class FakeEmbedder:
    """Deterministic stand-in for a real model: same text -> same vector, no calls."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls = 0
        self.texts = 0

    @property
    def identity(self) -> dict:
        return {"provider": "fake", "model": "fake-embed", "dimension": self.dimension}

    def describe(self) -> dict:
        return {
            "provider": "fake",
            "device": "cpu",
            "architecture": "FakeModel",
            "dimension": self.dimension,
            "parameters": 1234,
            "dtype_inference": "float32",
            "max_seq_length": 1024,
            "model_max_seq_length": 8192,
            "pooling": "cls",
            "normalize": True,
            "library": {"pytest": "local"},
        }

    def _vector(self, text: str) -> list[float]:
        codes = [ord(char) for char in text[: self.dimension]]
        padded = codes + [0] * (self.dimension - len(codes))
        norm = sum(value * value for value in padded) ** 0.5 or 1.0
        return [value / norm for value in padded]

    def embed_documents(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.calls += 1
        return self._vector(text)

    def count_tokens(self, text):
        return max(1, len(text) // 4)


class CountingTokens:
    """Pretends every chunk is over the window, to exercise the pre-flight gate."""

    def __init__(self, tokens: int = 5000) -> None:
        self.tokens = tokens

    def count(self, text: str) -> int:
        return self.tokens

    def count_many(self, texts: list[str], batch_size: int = 512) -> list[int]:
        return [self.tokens for _ in texts]


def make_snapshot(root: Path, n: int = 3, strategies: tuple[str, ...] = ("heading",)) -> None:
    snap = root / "corpus" / "tiny"
    (snap / "chunks").mkdir(parents=True)
    write_json(snap / "snapshot.json", {"snapshot_id": "snap-1", "name": "tiny"})
    docs = pd.DataFrame(
        [
            {
                "doc_id": f"d{i}",
                "url": f"https://example.invalid/{i}",
                "revid": 100 + i,
                "license_name": "CC-BY-NC-SA-2.5",
                "attribution": f"Bulbapedia, rev {100 + i}",
                "text": "x" * 100,
            }
            for i in range(n)
        ]
    )
    docs.to_parquet(snap / "documents.parquet", index=False)
    for strategy in strategies:
        prefix = strategy[0]
        chunks = pd.DataFrame(
            [
                {
                    "chunk_id": f"{prefix}{i}",
                    "doc_id": f"d{i}",
                    "revid": 100 + i,
                    "strategy": strategy,
                    "parent_id": None,
                    "role": "unit",
                    "heading_path": json.dumps(["Biology"]),
                    "char_start": 0,
                    "char_end": 40,
                    "text": f"Bulbasaur {strategy} fact number {i} about grass poison",
                    "n_chars": 40,
                    "n_tokens": 10,
                    "order": 0,
                }
                for i in range(n)
            ]
        )
        chunks.to_parquet(snap / "chunks" / f"{strategy}.parquet", index=False)


def config(root: Path, strategies: list[str] | None = None, **embedder) -> IndexConfig:
    return IndexConfig(
        snapshot="tiny",
        strategies=strategies or ["heading"],
        artifacts_root=str(root),
        docs_root=str(root / "docs"),
        embedder={
            "provider": "gemini",
            "model": "fake-embed",
            "dimension": 8,
            **embedder,
        },
    )
