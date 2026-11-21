"""Reciprocal-rank fusion.

Dense scores are cosine similarities in ``[0, 1]``; BM25 scores are unbounded
and corpus-dependent. Summing them would let whichever number happens to be
larger decide the ranking, so fusion uses ranks instead:

    score(chunk) = sum over sources of weight / (k + rank)      rank starts at 1

A chunk nominated by both retrievers collects two contributions and moves up;
a chunk nominated by one stays, but lower. No score normalisation is involved,
which is why the two scales can differ without special handling.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from pokerag.retrieval.types import Fused


def rrf(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    sources: Sequence[str] | None = None,
) -> list[Fused]:
    """Fuse ranked id lists. Input order is the ranking; duplicates are merged.

    ``weights`` and ``sources`` are parallel to ``rankings``. Ties are broken by
    ``chunk_id`` so the same inputs always produce the same output order.
    """
    if k < 0:
        raise ValueError("k harus >= 0")
    labels = list(sources) if sources is not None else [f"source{i}" for i in range(len(rankings))]
    if len(labels) != len(rankings):
        raise ValueError("sources dan rankings harus sama panjang")
    if len(set(labels)) != len(labels):
        raise ValueError(f"sources duplikat: {labels}")

    fused: dict[str, Fused] = {}
    for label, ranking in zip(labels, rankings):
        weight = float((weights or {}).get(label, 1.0))
        if weight < 0:
            raise ValueError(f"bobot negatif untuk {label}: {weight}")
        seen: set[str] = set()
        for position, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                raise ValueError(f"chunk_id duplikat di ranking {label}: {chunk_id}")
            seen.add(chunk_id)
            contribution = weight / (k + position)
            current = fused.get(chunk_id)
            if current is None:
                fused[chunk_id] = Fused(
                    chunk_id=chunk_id, score=contribution, ranks={label: position}
                )
            else:
                fused[chunk_id] = Fused(
                    chunk_id=chunk_id,
                    score=current.score + contribution,
                    ranks={**current.ranks, label: position},
                )

    return sorted(fused.values(), key=lambda item: (-item.score, item.chunk_id))
