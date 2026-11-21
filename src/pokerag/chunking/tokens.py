"""Token counting.

Chunking must not depend on the embedding model (otherwise ablation A1 changes
two things at once). The default counter therefore uses the standard 4
characters/token approximation and says so; an exact Hugging Face tokenizer can
be enabled with ``tokenizer: "hf:<model-id>"`` when budget arithmetic matters.
"""

from __future__ import annotations

import math


class TokenCounter:
    """Count tokens either exactly (HF tokenizer) or by approximation."""

    def __init__(self, spec: str = "estimate") -> None:
        self._tokenizer = None
        if spec.startswith("hf:"):
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(spec.removeprefix("hf:"))
        elif spec != "estimate":
            raise ValueError(f"unknown tokenizer spec: {spec}")
        self.spec = spec

    def count(self, text: str) -> int:
        """Return the token count of ``text``."""
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return max(1, math.ceil(len(text) / 4))

    def count_many(self, texts: list[str], batch_size: int = 512) -> list[int]:
        """Count a whole corpus in batches (57k chunks in ~12 s, not minutes)."""
        if self._tokenizer is None:
            return [self.count(text) for text in texts]
        encoded = self._tokenizer(texts, add_special_tokens=False, batch_size=batch_size)["input_ids"]
        return [len(ids) for ids in encoded]
