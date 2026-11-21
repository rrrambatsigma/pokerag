"""Generator interface and prompt-facing answer shape."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pokerag.retrieval.types import Hit


@dataclass(frozen=True)
class Answer:
    """What every generator must return, regardless of backend."""

    text: str
    cited_chunk_ids: tuple[str, ...] = ()
    abstained: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: str | None = None


class Generator(Protocol):
    name: str

    def generate(self, question: str, evidence: Sequence[Hit]) -> Answer: ...
