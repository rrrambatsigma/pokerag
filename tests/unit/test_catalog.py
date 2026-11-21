import pandas as pd

from pokerag.index.catalog import Catalog, parse_heading_path


def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    chunks = pd.DataFrame(
        {
            "chunk_id": ["c1", "c2"],
            "doc_id": ["d1", "d1"],
            "strategy": ["heading", "heading"],
            "role": ["child", "child"],
            "parent_id": [None, "p1"],
            "heading_path": ['["Biology", "Evolution"]', "[]"],
            "char_start": [0, 100],
            "char_end": [100, 200],
            "text": ["alpha", "beta"],
            "n_tokens": [10, 20],
        }
    )
    documents = pd.DataFrame({"doc_id": ["d1"], "title": ["Pikachu (Pokémon)"]})
    return chunks, documents


def test_catalog_lookup_and_title() -> None:
    catalog = Catalog.from_frames(*frames())

    chunk = catalog.get("c1")
    assert chunk is not None
    assert chunk.heading_path == ("Biology", "Evolution")
    assert chunk.label == "Biology > Evolution"
    assert chunk.parent_id is None
    assert catalog.title("d1") == "Pikachu (Pokémon)"
    assert catalog.title("missing") == "missing"
    assert len(catalog) == 2


def test_catalog_marks_missing_chunk_as_none() -> None:
    catalog = Catalog.from_frames(*frames())

    assert catalog.get("nope") is None


def test_parse_heading_path_variants() -> None:
    assert parse_heading_path('["A", "B"]') == ("A", "B")
    assert parse_heading_path(["A"]) == ("A",)
    assert parse_heading_path("[]") == ()
    assert parse_heading_path(None) == ()
    assert parse_heading_path("not json") == ()
