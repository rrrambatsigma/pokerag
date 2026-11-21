"""Typed configuration for the chunking stage."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FixedConfig(StrictConfig):
    """Sliding window over the whole document (B2 baseline)."""

    window_chars: int = 2048
    overlap_chars: int = 256


class HeadingConfig(StrictConfig):
    """Heading-aware splitting with padding and minimum size."""

    max_chars: int = 1200
    overlap_chars: int = 120
    min_chars: int = 240


class HierarchicalConfig(StrictConfig):
    """Parent sections plus child windows for context expansion."""

    parent_chars: int = 1500
    parent_overlap_chars: int = 0
    child_chars: int = 1000
    child_overlap_chars: int = 100
    min_chars: int = 200


class ChunkingConfig(StrictConfig):
    snapshot: str
    artifacts_root: str = "artifacts"
    tokenizer: str = "estimate"  # "estimate" (chars/4) or "hf:<model id>"
    budget_tokens: int = 2048
    # Only used for the "estimated vectors (MB)" column in the report; chunking
    # itself never depends on the embedding model (ablation A1 must change one
    # thing at a time).
    embedding_dimension: int = 1024
    # Chunks shorter than this carry no sentence worth citing (e.g. a bare
    # "- Main article: X (TCG)" link); they are dropped rather than embedded.
    min_chars_keep: int = 80
    fixed: FixedConfig = Field(default_factory=FixedConfig)
    heading: HeadingConfig = Field(default_factory=HeadingConfig)
    hierarchical: HierarchicalConfig = Field(default_factory=HierarchicalConfig)

    @property
    def snapshot_dir(self) -> Path:
        return Path(self.artifacts_root) / "corpus" / self.snapshot

    @property
    def chunks_dir(self) -> Path:
        return self.snapshot_dir / "chunks"


def load_chunking_config(path: Path) -> ChunkingConfig:
    """Load and validate a YAML chunking config."""
    return ChunkingConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
