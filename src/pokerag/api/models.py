"""HTTP contract for the chat UI.

Field names are camelCase on purpose: they are copied from
``frontend/src/types.ts`` so the UI can consume the payload without a mapping
layer. ``off`` is a valid mode and means no retrieval (the B0 configuration).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

STRATEGIES = ("fixed", "heading", "hierarchical")
MODES = ("dense", "bm25", "hybrid", "off")


class SettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dense: bool = True
    bm25: bool = True
    strategy: Literal["fixed", "heading", "hierarchical"] = "heading"
    topK: int = Field(default=5, ge=1, le=50)


class Citation(BaseModel):
    n: int
    chunkId: str
    docId: str
    title: str
    dex: int | None = None
    types: list[str] = Field(default_factory=list)
    headingPath: list[str] = Field(default_factory=list)
    excerpt: str
    url: str
    score: float


class ChatMessage(BaseModel):
    id: str
    role: Literal["assistant"] = "assistant"
    text: str
    createdAt: int
    citations: list[Citation] = Field(default_factory=list)
    settings: SettingsModel
    pending: bool = False


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    settings: SettingsModel = Field(default_factory=SettingsModel)


class ChatResponse(BaseModel):
    message: ChatMessage


class SearchResponse(BaseModel):
    query: str
    mode: str
    strategy: str
    citations: list[Citation] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    snapshot: str
    index_version: str
    strategy: str
    collection: str
    generator: str
    variant: str
    modes: list[str]


def mode_for(settings: SettingsModel) -> str:
    """Map the UI toggles to one retrieval mode."""
    if settings.dense and settings.bm25:
        return "hybrid"
    if settings.dense:
        return "dense"
    if settings.bm25:
        return "bm25"
    return "off"
