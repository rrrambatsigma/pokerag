"""The three chunking strategies required by ablation A1.

All of them emit absolute spans into the document text, and none of them may
cross a top-level (``h2``) section boundary — except ``fixed``, which is the
deliberately naive baseline mandated for B2.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pokerag.chunking.config import ChunkingConfig
from pokerag.chunking.models import Chunk, Strategy
from pokerag.chunking.split import block_of, pad, top_level_blocks, windows
from pokerag.chunking.tokens import TokenCounter


def make_chunk_id(doc_id: str, strategy: str, start: int, end: int, suffix: str = "") -> str:
    """Content-addressed id: the same span always yields the same id."""
    raw = f"{doc_id}|{strategy}|{start}|{end}|{suffix}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _emit(
    *,
    doc_id: str,
    revid: int,
    text: str,
    strategy: Strategy,
    start: int,
    end: int,
    order: int,
    counter: TokenCounter,
    heading_path: list[str] | None = None,
    role: str = "unit",
    parent_id: str | None = None,
    suffix: str = "",
) -> Chunk:
    piece = text[start:end]
    return Chunk(
        chunk_id=make_chunk_id(doc_id, strategy, start, end, suffix),
        doc_id=doc_id,
        revid=revid,
        strategy=strategy,
        parent_id=parent_id,
        role=role,  # type: ignore[arg-type]
        heading_path=heading_path or [],
        char_start=start,
        char_end=end,
        text=piece,
        n_chars=len(piece),
        n_tokens=counter.count(piece),
        order=order,
    )


def fixed_chunks(
    *, doc_id: str, revid: int, text: str, cfg: ChunkingConfig, counter: TokenCounter
) -> list[Chunk]:
    """Sliding window over the raw document text (B2 baseline)."""
    spans = windows(0, len(text), cfg.fixed.window_chars, cfg.fixed.overlap_chars)
    return [
        _emit(
            doc_id=doc_id,
            revid=revid,
            text=text,
            strategy="fixed",
            start=start,
            end=end,
            order=index,
            counter=counter,
        )
        for index, (start, end) in enumerate(spans)
    ]


def _heading_at(sections: list[dict[str, Any]], position: int) -> list[str]:
    for section in sections:
        if int(section["char_start"]) <= position < int(section["char_end"]):
            return list(section.get("heading_path") or [])
    return []


def heading_chunks(
    *,
    doc_id: str,
    revid: int,
    text: str,
    sections: list[dict[str, Any]],
    cfg: ChunkingConfig,
    counter: TokenCounter,
) -> list[Chunk]:
    """Split at headings, split long sections, pad short ones inside the block."""
    blocks = top_level_blocks(sections)
    spans: list[tuple[int, int, list[str]]] = []
    for section in sections:
        start, end = int(section["char_start"]), int(section["char_end"])
        path = list(section.get("heading_path") or [])
        if end <= start:
            continue
        if end - start > cfg.heading.max_chars:
            for window_start, window_end in windows(
                start, end, cfg.heading.max_chars, cfg.heading.overlap_chars
            ):
                spans.append((window_start, window_end, path))
        else:
            _, lower, upper = block_of(blocks, start)
            padded = pad(start, end, cfg.heading.min_chars, lower, upper)
            spans.append((padded[0], padded[1], path))

    seen: set[tuple[int, int]] = set()
    chunks: list[Chunk] = []
    for start, end, path in sorted(spans):
        if (start, end) in seen:
            continue
        seen.add((start, end))
        chunks.append(
            _emit(
                doc_id=doc_id,
                revid=revid,
                text=text,
                strategy="heading",
                start=start,
                end=end,
                order=len(chunks),
                counter=counter,
                heading_path=path,
            )
        )
    return chunks


def _section_blocks(sections: list[dict[str, Any]], text_length: int) -> list[tuple[str, int, int]]:
    """Contiguous h2 blocks that cover the whole document.

    ``top_level_blocks`` gives the h2 groups but leaves whitespace gaps between
    them and can start after character 0. Parents must not cross an h2 boundary,
    and coverage must stay 100%, so each block is extended to the start of the
    next one (and the last one to the end of the text).
    """
    ordered = sorted(sections, key=lambda section: int(section["char_start"]))
    blocks = top_level_blocks(ordered)
    if not blocks:
        return [("", 0, text_length)]
    spans: list[tuple[str, int, int]] = []
    for index, (key, _start, end) in enumerate(blocks):
        start = spans[-1][2] if spans else 0
        stop = blocks[index + 1][1] if index + 1 < len(blocks) else text_length
        spans.append((key, start, max(stop, end)))
    return spans


def hierarchical_chunks(
    *,
    doc_id: str,
    revid: int,
    text: str,
    sections: list[dict[str, Any]],
    cfg: ChunkingConfig,
    counter: TokenCounter,
) -> list[Chunk]:
    """Parent sections plus child windows, so retrieval can expand to context.

    Parents are built **inside** h2 blocks: a parent (and therefore any child)
    never crosses a top-level heading. Building parents as sliding windows over
    the whole document (the previous behaviour) silently turned this strategy
    into "fixed with parents": 2.117 of 23.399 children crossed an h2 boundary,
    which is what the A1 audit caught.
    """
    parents: list[Chunk] = []
    for _key, block_start, block_end in _section_blocks(sections, len(text)):
        for start, end in windows(
            block_start, block_end, cfg.hierarchical.parent_chars, cfg.hierarchical.parent_overlap_chars
        ):
            parents.append(
                _emit(
                    doc_id=doc_id,
                    revid=revid,
                    text=text,
                    strategy="hierarchical",
                    start=start,
                    end=end,
                    order=len(parents),
                    counter=counter,
                    heading_path=_heading_at(sections, start),
                    role="parent",
                )
            )

    children: list[Chunk] = []
    for parent in parents:
        for start, end in windows(
            parent.char_start, parent.char_end, cfg.hierarchical.child_chars, cfg.hierarchical.child_overlap_chars
        ):
            padded = pad(
                start, end, cfg.hierarchical.min_chars, parent.char_start, parent.char_end
            )
            children.append(
                _emit(
                    doc_id=doc_id,
                    revid=revid,
                    text=text,
                    strategy="hierarchical",
                    start=padded[0],
                    end=padded[1],
                    order=len(children),
                    counter=counter,
                    heading_path=_heading_at(sections, padded[0]),
                    role="child",
                    parent_id=parent.chunk_id,
                    suffix=parent.chunk_id,
                )
            )
    return parents + children
