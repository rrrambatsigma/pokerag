"""Typed configuration for the retrieval stage.

One file describes one query configuration: which index config to use, which
strategy to query, how many candidates each retriever nominates, and how they
are fused. Retrieval never writes an index, so this config is safe to edit
between experiments; only the index config changes what has to be rebuilt.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pokerag.index.config import STRATEGIES, IndexConfig, load_index_config
from pokerag.retrieval.types import MODES


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalConfig(StrictConfig):
    # Path to the index config that built the collections being queried. It is
    # reused on purpose: the embedder identity check compares against it, so a
    # query can never be embedded with a different model than the documents.
    index: str = "configs/index.gpu.yaml"
    strategy: str = "heading"
    top_dense: int = 20
    top_bm25: int = 20
    top_k: int = 5
    # Reciprocal-rank fusion constant: score = sum(weight / (k + rank)).
    rrf_k: int = 60
    weights: dict[str, float] = Field(default_factory=lambda: {"dense": 1.0, "bm25": 1.0})
    # Defaults for `pokerag-retrieve eval`.
    modes: list[str] = Field(default_factory=lambda: ["dense", "bm25", "hybrid"])
    ks: list[int] = Field(default_factory=lambda: [5, 10])

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        if value not in STRATEGIES:
            raise ValueError(f"strategi tidak dikenal: {value}; pilihan: {list(STRATEGIES)}")
        return value

    @field_validator("modes")
    @classmethod
    def _known_modes(cls, value: list[str]) -> list[str]:
        unknown = [mode for mode in value if mode not in MODES]
        if unknown:
            raise ValueError(f"mode tidak dikenal: {unknown}; pilihan: {list(MODES)}")
        if not value:
            raise ValueError("modes tidak boleh kosong")
        return value

    @field_validator("top_dense", "top_bm25", "top_k")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("nilai harus >= 1")
        return value

    @field_validator("rrf_k")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("rrf_k harus >= 0")
        return value

    @field_validator("ks")
    @classmethod
    def _ks_sorted(cls, value: list[int]) -> list[int]:
        if not value or any(k < 1 for k in value):
            raise ValueError("ks harus berisi angka >= 1")
        return sorted(set(value))

    @field_validator("weights")
    @classmethod
    def _known_weights(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = [name for name in value if name not in ("dense", "bm25")]
        if unknown:
            raise ValueError(f"bobot tidak dikenal: {unknown}")
        if any(weight < 0 for weight in value.values()):
            raise ValueError("bobot tidak boleh negatif")
        return value

    def index_config(self) -> IndexConfig:
        """Load the index config this retrieval config points at."""
        return load_index_config(Path(self.index))

    def weight(self, source: str) -> float:
        return float(self.weights.get(source, 1.0))


def load_retrieval_config(path: Path) -> RetrievalConfig:
    """Load and validate a YAML retrieval config."""
    return RetrievalConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
