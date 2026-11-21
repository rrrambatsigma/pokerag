"""Chunking invariants: exact spans, no cross-block leaks, deterministic ids."""

from __future__ import annotations

from pokerag.chunking.config import ChunkingConfig
from pokerag.chunking.strategies import fixed_chunks, heading_chunks, hierarchical_chunks
from pokerag.chunking.tokens import TokenCounter

LEAD = "Bulbasaur is a Grass-type Pokemon. " * 4
BIOLOGY = "It has a bulb on its back. " * 10
GAMEDATA = "Vine Whip | Grass | 25 | 100% | 25 " * 120


def make_doc() -> tuple[str, list[dict]]:
    parts = [("", LEAD), ("Biology", BIOLOGY), ("Game data", GAMEDATA)]
    text = ""
    sections: list[dict] = []
    for index, (heading, body) in enumerate(parts):
        if index:
            text += "\n\n"
        start = len(text)
        text += body
        sections.append(
            {
                "heading_path": [heading] if heading else [],
                "level": 2 if heading else 0,
                "kind": "section" if heading else "lead",
                "char_start": start,
                "char_end": len(text),
            }
        )
    return text, sections


def config() -> ChunkingConfig:
    return ChunkingConfig(snapshot="test-snapshot", artifacts_root="artifacts")


def test_every_strategy_preserves_spans():
    text, sections = make_doc()
    cfg = config()
    counter = TokenCounter("estimate")
    chunks = (
        fixed_chunks(doc_id="d1", revid=1, text=text, cfg=cfg, counter=counter)
        + heading_chunks(doc_id="d1", revid=1, text=text, sections=sections, cfg=cfg, counter=counter)
        + hierarchical_chunks(doc_id="d1", revid=1, text=text, sections=sections, cfg=cfg, counter=counter)
    )
    assert chunks
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text


def test_fixed_covers_the_whole_document():
    text, _ = make_doc()
    chunks = fixed_chunks(doc_id="d1", revid=1, text=text, cfg=config(), counter=TokenCounter())
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.char_start < previous.char_end, "overlap must keep windows connected"


def test_heading_respects_blocks_and_minimum_size():
    text, sections = make_doc()
    cfg = config()
    chunks = heading_chunks(
        doc_id="d1", revid=1, text=text, sections=sections, cfg=cfg, counter=TokenCounter()
    )
    blocks = {tuple(section["heading_path"]): (section["char_start"], section["char_end"]) for section in sections}
    for chunk in chunks:
        lower, upper = blocks[tuple(chunk.heading_path)]
        assert lower <= chunk.char_start and chunk.char_end <= upper, "chunk crossed an h2 boundary"
        assert chunk.n_chars >= cfg.heading.min_chars or chunk.char_end - chunk.char_start == upper - lower
    assert all(chunk.n_chars <= cfg.heading.max_chars for chunk in chunks)
    assert sum(1 for chunk in chunks if chunk.heading_path == ["Game data"]) > 1


def test_hierarchical_children_point_to_real_parents():
    text, sections = make_doc()
    chunks = hierarchical_chunks(
        doc_id="d1", revid=1, text=text, sections=sections, cfg=config(), counter=TokenCounter()
    )
    parents = {chunk.chunk_id for chunk in chunks if chunk.role == "parent"}
    children = [chunk for chunk in chunks if chunk.role == "child"]
    assert parents and children
    for child in children:
        assert child.parent_id in parents
        assert child.char_start >= 0 and child.char_end <= len(text)


def test_hierarchical_never_crosses_an_h2_boundary():
    text, sections = make_doc()
    chunks = hierarchical_chunks(
        doc_id="d1", revid=1, text=text, sections=sections, cfg=config(), counter=TokenCounter()
    )
    h2_starts = [section["char_start"] for section in sections if section["level"] == 2]
    for chunk in chunks:
        crossed = [start for start in h2_starts if chunk.char_start < start < chunk.char_end]
        assert not crossed, f"{chunk.chunk_id} ({chunk.role}) melewati heading di {crossed}"


def test_hierarchical_parents_cover_the_whole_document():
    text, sections = make_doc()
    chunks = hierarchical_chunks(
        doc_id="d1", revid=1, text=text, sections=sections, cfg=config(), counter=TokenCounter()
    )
    parents = sorted(
        (chunk for chunk in chunks if chunk.role == "parent"), key=lambda chunk: chunk.char_start
    )
    assert parents[0].char_start == 0
    assert parents[-1].char_end == len(text)
    for previous, current in zip(parents, parents[1:]):
        assert current.char_start == previous.char_end, "parents must stay contiguous"


def test_ids_are_deterministic():
    text, sections = make_doc()
    cfg = config()
    first = heading_chunks(doc_id="d1", revid=1, text=text, sections=sections, cfg=cfg, counter=TokenCounter())
    second = heading_chunks(doc_id="d1", revid=1, text=text, sections=sections, cfg=cfg, counter=TokenCounter())
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]
