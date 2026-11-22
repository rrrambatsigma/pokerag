"""Cross-encoder reranking for B3-B6 and A2.

The retrieval stage proposes a wider candidate list, the reranker scores each
(query, passage) pair directly, and the runner keeps the top ``top_k``. The
reranker runs in the retrieval pass only, then is unloaded before the generator
loads; bge-m3, the reranker and a 1.5B generator do not fit together in 6 GB.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from pokerag.llm.config import RerankerRef
from pokerag.retrieval.types import Hit


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]: ...


class CrossEncoderReranker:
    def __init__(self, config: RerankerRef, *, device: str | None = None) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as error:  # pragma: no cover - depends on local install
            raise RuntimeError("reranker needs the gpu extra: uv sync --extra gpu") from error

        target = device or config.device
        model_kwargs: dict[str, object] = {}
        dtype = getattr(torch, config.dtype) if config.dtype else None
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        self.name = config.model_id
        self._model = CrossEncoder(
            config.model_id,
            revision=config.revision,
            device=target,
            model_kwargs=model_kwargs,
            max_length=1024,
        )

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._model.predict(
            [(query, hit.text) for hit in hits],
            batch_size=4,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        ranked = sorted(
            zip(hits, scores, strict=True),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )[:top_k]
        return [
            replace(
                hit,
                score=float(score),
                rank=rank,
                source="rerank",
                ranks={**hit.ranks, "rerank": rank},
            )
            for rank, (hit, score) in enumerate(ranked, start=1)
        ]
