"""Chunking audit maths, on hand-built cases (no corpus, no GPU)."""

from __future__ import annotations

import json

import pandas as pd

from pokerag.chunking.audit import (
    boundary_integrity,
    ends_at_sentence,
    overlap_redundancy,
    parent_structure,
    section_fragmentation,
    size_buckets,
    splits_word,
    starts_at_sentence,
)


def chunks_frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "chunk_id": "c",
        "doc_id": "d1",
        "role": "unit",
        "parent_id": None,
        "char_start": 0,
        "char_end": 1,
        "n_chars": 1,
        "n_tokens": 1,
        "text": "x",
        "heading_path": "[]",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def documents_frame(text: str, sections: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"doc_id": "d1", "text": text, "sections_json": json.dumps(sections)}]
    )


def test_sentence_boundaries():
    text = "Hello world. Second sentence. Third."
    assert starts_at_sentence(text, 0)
    assert starts_at_sentence(text, 13), "after a full stop is a clean start"
    assert not starts_at_sentence(text, 6), "mid-sentence start"
    assert ends_at_sentence(text, 12), "ends on the full stop"
    assert not ends_at_sentence(text, 6)
    assert ends_at_sentence(text, len(text))


def test_splits_word_detects_mid_token_cuts():
    text = "Charizard is a Fire-type Pokemon."
    assert splits_word(text, 9) is False, "boundary at the space is clean"
    assert splits_word(text, 4) is True, "inside 'Charizard'"
    assert splits_word(text, len(text)) is False, "end of text is not a cut"


def test_overlap_redundancy_counts_duplicated_characters():
    chunks = chunks_frame(
        [
            {"chunk_id": "a", "char_start": 0, "char_end": 10},
            {"chunk_id": "b", "char_start": 5, "char_end": 15},
        ]
    )
    documents = documents_frame("x" * 15, [])
    stats = overlap_redundancy(documents, chunks)
    assert stats["chunk_chars"] == 20
    assert stats["unique_chars"] == 15
    assert stats["duplicated_chars"] == 5
    assert stats["redundancy"] == 0.25


def test_section_fragmentation_counts_intact_and_split():
    text = "x" * 100
    sections = [
        {"char_start": 0, "char_end": 40, "level": 2, "heading_path": ["A"], "kind": "section"},
        {"char_start": 40, "char_end": 100, "level": 2, "heading_path": ["B"], "kind": "section"},
    ]
    chunks = chunks_frame(
        [
            {"chunk_id": "a", "char_start": 0, "char_end": 40},
            {"chunk_id": "b", "char_start": 40, "char_end": 70},
            {"chunk_id": "c", "char_start": 70, "char_end": 100},
        ]
    )
    stats = section_fragmentation(documents_frame(text, sections), chunks)
    assert stats["sections"] == 2
    assert stats["intact"] == 0.5, "one section sits in exactly one chunk"
    assert stats["split_three_or_more"] == 0.0
    assert stats["max"] == 2


def test_boundary_integrity_flags_crossing_h2():
    text = "First part. Second part starts here."
    sections = [
        {"char_start": 0, "char_end": 12, "level": 2, "heading_path": ["A"], "kind": "section"},
        {"char_start": 12, "char_end": len(text), "level": 2, "heading_path": ["B"], "kind": "section"},
    ]
    chunks = chunks_frame(
        [
            {"chunk_id": "a", "char_start": 0, "char_end": 12, "text": text[:12]},
            {"chunk_id": "b", "char_start": 12, "char_end": len(text), "text": text[12:]},
        ]
    )
    stats = boundary_integrity(documents_frame(text, sections), chunks)
    assert stats["crosses_h2"] == 0
    assert stats["starts_at_heading"] == 1.0

    crossing = chunks_frame(
        [{"chunk_id": "c", "char_start": 0, "char_end": len(text), "text": text}]
    )
    assert boundary_integrity(documents_frame(text, sections), crossing)["crosses_h2"] == 1


def test_parent_structure_detects_containment_violation():
    chunks = chunks_frame(
        [
            {"chunk_id": "p", "role": "parent", "char_start": 0, "char_end": 100, "n_tokens": 50},
            {"chunk_id": "c1", "role": "unit", "parent_id": "p", "char_start": 0, "char_end": 50},
            {"chunk_id": "c2", "role": "unit", "parent_id": "p", "char_start": 60, "char_end": 120},
        ]
    )
    stats = parent_structure(chunks)
    assert stats["parents"] == 1
    assert stats["children"] == 2
    assert stats["children_per_parent_max"] == 2
    assert stats["containment_violations"] == 1, "c2 sticks out of its parent"
    assert stats["parent_tokens"] == 50


def test_size_buckets_partition_every_chunk():
    chunks = chunks_frame(
        [{"n_tokens": tokens} for tokens in (100, 250, 500, 900, 1100)]
    )
    buckets = size_buckets(chunks)
    assert buckets == {"<200": 1, "200-400": 1, "400-600": 1, "600-800": 0, "800-1024": 1, ">1024": 1}
