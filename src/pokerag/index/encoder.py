"""Query encoder for BAAI/bge-m3, matching the identity pinned in the corpus."""

from __future__ import annotations

from typing import Protocol


class QueryEncoder(Protocol):
    def encode(self, query: str) -> list[float]: ...


class BgeM3Encoder:
    """Sentence-transformers wrapper with the settings used to build the index.

    CLS pooling, L2 normalization, 1024 tokens, fp16 on GPU, no instruction
    prefix for queries. Everything is pinned to the revision recorded in
    ``manifests/embedder_identity.json``.
    """

    MODEL_ID = "BAAI/bge-m3"
    REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
    MAX_SEQ_LENGTH = 1024

    def __init__(self, device: str | None = None, dtype: str | None = "float16") -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "query encoder needs the ml extra: uv sync --extra ml"
            ) from error

        model_kwargs: dict[str, object] = {}
        if dtype:
            model_kwargs["torch_dtype"] = getattr(torch, dtype)

        self._model = SentenceTransformer(
            self.MODEL_ID,
            revision=self.REVISION,
            device=device,
            model_kwargs=model_kwargs,
        )
        self._model.max_seq_length = self.MAX_SEQ_LENGTH

    def encode(self, query: str) -> list[float]:
        vectors = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(value) for value in vectors[0]]
