"""Rank fusion for hybrid retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FusedHit:
    chunk_id: str
    score: float
    ranks: dict[str, int]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], k: int = 60
) -> list[FusedHit]:
    """Fuse ranked id lists with RRF.

    ``score(d) = sum(1 / (k + rank_i(d)))`` across the lists that contain ``d``.
    RRF uses ranks only, so BM25 and cosine scores never need rescaling.
    """

    if k <= 0:
        raise ValueError("rrf k must be positive")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for source, ids in rankings.items():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(chunk_id, {})[source] = rank

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        FusedHit(chunk_id=chunk_id, score=score, ranks=ranks[chunk_id])
        for chunk_id, score in ordered
    ]
