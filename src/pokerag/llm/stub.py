"""Deterministic generator used before the Qwen backend exists.

It exists so the API and the UI can be wired and tested without a GPU. It always
cites the top retrieved chunk, or abstains when retrieval returns nothing. The
API contract is identical to the real generator, so replacing it is a one-line
change in ``create_app``.
"""

from __future__ import annotations

from collections.abc import Sequence

from pokerag.llm.base import Answer
from pokerag.retrieval.types import Hit


class StubGenerator:
    name = "stub"

    def generate(self, question: str, evidence: Sequence[Hit]) -> Answer:
        if not evidence:
            return Answer(
                text="Not enough evidence in the corpus.",
                cited_chunk_ids=(),
                abstained=True,
            )
        top = evidence[0]
        excerpt = " ".join(top.text.split())[:240]
        return Answer(
            text=f"Stub answer from {top.doc_id}: {excerpt}",
            cited_chunk_ids=(top.chunk_id,),
        )
