"""Typed configuration for the index stage.

One configuration describes one *embedding function* and every chunking strategy
that is indexed with it. All strategies share a single Qdrant database and a
single embedding cache, but each strategy gets its own collection
(``chunks_<strategy>``), its own BM25 store and its own manifest. That layout is
what keeps ablation A1 comparable: same documents, same embedder, only the
retrieval unit changes.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

STRATEGIES = ("fixed", "heading", "hierarchical")


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbedderConfig(StrictConfig):
    provider: str = "gemini"          # "gemini", "local" (fastembed/ONNX) or "st" (torch/CUDA)
    model: str = "gemini-embedding-001"
    # Hugging Face revision (commit sha) for local models. Pinning it is what
    # makes "the same model" a checkable claim instead of an assumption: a
    # re-download can never silently swap the weights.
    revision: str | None = None
    dimension: int = 768
    task_document: str = "RETRIEVAL_DOCUMENT"
    task_query: str = "RETRIEVAL_QUERY"
    batch_size: int = 100             # Gemini limit for batchEmbedContents
    items_per_minute: int = 90        # free tier allows 100 items/minute; keep margin
    max_retries: int = 5
    timeout_seconds: int = 120
    query_prefix: str = ""            # local bge models want an instruction prefix
    device: str = "cuda"              # for provider "st"
    dtype: str = "float16"            # fp16 halves VRAM; 4 GB card
    max_seq_length: int = 512         # hard truncation point; must cover the longest chunk


class QdrantConfig(StrictConfig):
    mode: str = "local"               # "local" (on-disk) or a server URL
    url: str | None = None
    path: str = "indexes/qdrant"      # relative to artifacts_root; shared by all strategies
    distance: str = "cosine"
    on_disk: bool = True


class Bm25Config(StrictConfig):
    k1: float = 1.2
    b: float = 0.75


class IndexConfig(StrictConfig):
    snapshot: str
    strategies: list[str] = Field(default_factory=lambda: ["heading"])
    artifacts_root: str = "artifacts"
    # Committed documentation is written here (artefacts are git-ignored, so the
    # A1 table and the model identity must live in the repository).
    docs_root: str = "docs"
    # Fail before spending an hour of GPU time when a chunk exceeds the
    # embedder window; the build would otherwise silently truncate it.
    allow_truncation: bool = False
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    bm25: Bm25Config = Field(default_factory=Bm25Config)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_strategy(cls, data):
        """Accept ``strategy: heading`` from the previous single-collection config."""
        if isinstance(data, dict) and "strategy" in data and "strategies" not in data:
            data = dict(data)
            data["strategies"] = [data.pop("strategy")]
        return data

    @field_validator("strategies")
    @classmethod
    def _validate_strategies(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("strategies tidak boleh kosong")
        unknown = [name for name in value if name not in STRATEGIES]
        if unknown:
            raise ValueError(f"strategi tidak dikenal: {unknown}; pilihan: {list(STRATEGIES)}")
        if len(set(value)) != len(value):
            raise ValueError("strategi duplikat")
        return value

    @property
    def snapshot_dir(self) -> Path:
        return Path(self.artifacts_root) / "corpus" / self.snapshot

    @property
    def chunks_dir(self) -> Path:
        return self.snapshot_dir / "chunks"

    @property
    def indexes_root(self) -> Path:
        return Path(self.artifacts_root) / "indexes"

    @property
    def qdrant_dir(self) -> Path:
        """One on-disk database for every collection, so A1 shares one store."""
        return Path(self.artifacts_root) / self.qdrant.path

    def collection_name(self, strategy: str) -> str:
        return f"chunks_{strategy}"

    def manifest_path(self, strategy: str) -> Path:
        return self.indexes_root / strategy / "manifest.json"

    def bm25_dir(self, strategy: str) -> Path:
        return self.indexes_root / "bm25" / strategy

    @property
    def report_path(self) -> Path:
        return self.indexes_root / "index_report.md"

    @property
    def provenance_path(self) -> Path:
        return self.indexes_root / "provenance.json"

    @property
    def identity_path(self) -> Path:
        return Path(self.artifacts_root) / "models" / "embedder_identity.json"

    @property
    def models_docs_dir(self) -> Path:
        return Path(self.docs_root) / "models"

    @property
    def experiments_docs_dir(self) -> Path:
        return Path(self.docs_root) / "experiments"


def load_index_config(path: Path) -> IndexConfig:
    """Load and validate a YAML index config."""
    return IndexConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
