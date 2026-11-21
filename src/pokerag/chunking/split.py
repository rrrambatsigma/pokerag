"""Span arithmetic shared by the chunking strategies.

Everything is expressed in absolute character offsets into the document text,
because that is the only representation that survives a change of strategy.
"""

from __future__ import annotations

from typing import Iterable

Span = tuple[int, int]


def windows(start: int, end: int, size: int, overlap: int) -> list[Span]:
    """Sliding windows of at most ``size`` characters over ``[start, end)``."""
    if end <= start:
        return []
    if end - start <= size:
        return [(start, end)]
    step = max(1, size - overlap)
    spans: list[Span] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + size, end)
        spans.append((cursor, stop))
        if stop >= end:
            break
        cursor += step
    return spans


def pad(start: int, end: int, min_chars: int, lower: int, upper: int) -> Span:
    """Grow ``[start, end)`` to ``min_chars`` without leaving ``[lower, upper)``.

    Padding keeps a chunk contiguous (so citations stay exact) instead of
    concatenating two distant sections, which would break the span invariant.
    """
    length = end - start
    if length >= min_chars:
        return start, end
    needed = min_chars - length
    left = min(needed // 2, start - lower)
    right = min(needed - left, upper - end)
    left = min(needed - right, start - lower)
    return start - left, end + right


def top_level_blocks(sections: Iterable[dict], tolerance: int = 4) -> list[tuple[str, int, int]]:
    """Group sections into top-level (``h2``) blocks: ``(key, start, end)``."""
    blocks: list[tuple[str, int, int]] = []
    for section in sections:
        path = section.get("heading_path") or []
        key = path[0] if path else ""
        start, end = int(section["char_start"]), int(section["char_end"])
        if blocks and blocks[-1][0] == key and start <= blocks[-1][2] + tolerance:
            blocks[-1] = (key, blocks[-1][1], end)
        else:
            blocks.append((key, start, end))
    return blocks


def block_of(blocks: list[tuple[str, int, int]], position: int) -> tuple[str, int, int]:
    """Return the top-level block containing ``position`` (or the nearest one)."""
    for key, start, end in blocks:
        if start <= position < end:
            return key, start, end
    if not blocks:
        return "", position, position
    return min(blocks, key=lambda block: abs(block[1] - position))


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    """Merge overlapping/adjacent spans, used to compute coverage."""
    ordered = sorted(spans)
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]
