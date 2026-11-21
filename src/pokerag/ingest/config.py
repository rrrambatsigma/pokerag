"""Typed configuration for the corpus snapshot."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from pokerag.ingest.models import LicenseInfo


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectionConfig(StrictConfig):
    method: str
    list_page: str
    from_dex: int
    to_dex: int
    namespace: int = 0


class FetchConfig(StrictConfig):
    api: str
    impersonate: str = "chrome"
    user_agent: str
    delay_seconds: float = 5.0
    batch_titles: int = 5
    timeout_seconds: int = 30
    max_retries: int = 3
    backoff_seconds: list[int] = Field(default_factory=lambda: [10, 30, 60])


class LicenseConfig(StrictConfig):
    name: str
    url: str
    allow_retrieval: bool = True
    allow_redistribute: bool = False
    allow_ft_training: bool = False
    attribution_template: str


class CleaningConfig(StrictConfig):
    min_chars: int = 800
    min_sections: int = 3
    drop_selectors: list[str] = Field(default_factory=list)
    drop_section_titles: list[str] = Field(default_factory=list)


class DedupConfig(StrictConfig):
    shingle_size: int = 5
    num_perm: int = 128
    threshold: float = 0.85


class ValidateConfig(StrictConfig):
    min_docs: int = 300
    min_chars: int = 800
    min_sections: int = 3


class CorpusConfig(StrictConfig):
    name: str
    source_site: str
    artifacts_root: str = "artifacts"
    selection: SelectionConfig
    fetch: FetchConfig
    license: LicenseConfig
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    validation: ValidateConfig = Field(default_factory=ValidateConfig)

    @property
    def snapshot_dir(self) -> Path:
        """Directory holding every artefact of this snapshot."""
        root = os.environ.get("POKERAG_ARTIFACTS_ROOT", self.artifacts_root)
        return Path(root) / "corpus" / self.name

    def license_for(self, revid: int) -> LicenseInfo:
        """Build the per-document licence record for a given revision."""
        return LicenseInfo(
            name=self.license.name,
            url=self.license.url,
            allow_retrieval=self.license.allow_retrieval,
            allow_redistribute=self.license.allow_redistribute,
            allow_ft_training=self.license.allow_ft_training,
            attribution=self.license.attribution_template.format(revid=revid),
        )


def load_config(path: Path) -> CorpusConfig:
    """Load and validate a YAML corpus config."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return CorpusConfig.model_validate(data)
