"""Project settings, loaded from environment variables prefixed with ``POKERAG_``."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration shared by scripts and the library."""

    model_config = SettingsConfigDict(
        env_prefix="POKERAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hf_dataset: str = "meivawijaya/pokerag-corpus-v3"
    hf_revision: str = "main"
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts")

    @property
    def raw_dir(self) -> Path:
        return self.artifacts_dir / "raw"

    @property
    def indexes_dir(self) -> Path:
        return self.artifacts_dir / "indexes"

    @property
    def qdrant_dir(self) -> Path:
        return self.indexes_dir / "qdrant"

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
