"""Data contracts for the ingestion stage.

These models are the only shape the rest of the project is allowed to depend
on. Character offsets in :class:`Section` are the anchor that keeps qrels valid
when the chunking strategy changes later (ablation A1).

Two rules keep artefacts reproducible:

* no wall-clock timestamps in canonical records (provenance lives in
  ``provenance.json``);
* every derived record carries the version of the code and config that produced
  it, so a change forces a rebuild instead of silently mixing generations.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects unknown fields, so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class LicenseInfo(StrictModel):
    """Per-document usage rights. Defaults are deliberately restrictive."""

    name: str
    url: str
    allow_retrieval: bool = True
    allow_redistribute: bool = False
    allow_ft_training: bool = False
    attribution: str


class SelectedPage(StrictModel):
    """One row of the deterministic selection frame (National Dex order)."""

    dex: int
    name: str
    title: str
    url: str


class PageRecord(StrictModel):
    """Revision identity of one fetched page.

    ``raw_sha256`` is the hash of the stored wikitext file, which lets resume
    decide "already fetched" by content instead of by file existence.
    """

    doc_id: str
    requested_title: str
    pageid: int
    revid: int
    title: str
    url: str
    rev_timestamp: str
    api_sha1: str
    api_size: int
    raw_sha256: str
    wikitext_path: str
    is_redirect: bool = False
    redirect_target: str | None = None


class Section(StrictModel):
    """A contiguous span of the cleaned document text.

    Invariant enforced by tests: ``text[char_start:char_end] == section_text``.
    """

    heading_path: list[str] = Field(default_factory=list)
    level: int = 0
    kind: Literal["infobox", "lead", "section", "table", "trivia"] = "section"
    char_start: int
    char_end: int


class CleanDoc(StrictModel):
    """A document after cleaning, still traceable to its revision."""

    doc_id: str
    pageid: int
    revid: int
    title: str
    url: str
    text: str
    content_sha256: str
    n_chars: int
    n_sections: int
    sections: list[Section]
    infobox: list[str] = Field(default_factory=list)
    is_redirect: bool = False
    redirect_target: str | None = None
    is_disambiguation: bool = False
    is_stub: bool = False
    dropped: dict[str, int] = Field(default_factory=dict)
    license: LicenseInfo
    cleaner_version: str
    clean_config_sha: str


class HtmlRecord(StrictModel):
    """Locator and content hash for one stored HTML page."""

    doc_id: str
    html_path: str
    html_sha256: str


class DuplicateGroup(StrictModel):
    """A cluster of documents judged identical or near-identical."""

    reason: Literal["exact", "near"]
    canonical: str
    members: list[str]
    max_similarity: float


class ValidationGate(StrictModel):
    """One pass/fail check with the evidence needed to justify it."""

    gate: str
    threshold: str
    observed: str
    passed: bool
