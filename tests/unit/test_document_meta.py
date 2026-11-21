"""Citation metadata parsed from the frozen documents.parquet infobox."""

from __future__ import annotations

import json

import pandas as pd

from pokerag.retrieval.document_meta import build_doc_meta, parse_infobox


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "doc_id": "d1",
                "title": "Cloyster (Pokémon)",
                "url": "https://bulbapedia.bulbagarden.net/wiki/Cloyster",
                "infobox_json": json.dumps(
                    {
                        "0": [
                            "Cloyster Bivalve Pokémon パルシェン Parshen | #0091",
                            "Cloyster Bivalve Pokémon | パルシェン Parshen",
                            "Water | Ice",
                        ]
                    }
                ),
            },
            {
                "doc_id": "d2",
                "title": "Missingno (Pokémon)",
                "url": "https://example.invalid",
                "infobox_json": "{}",
            },
            {
                "doc_id": "d3",
                "title": "Minun (Pokémon)",
                "url": "https://example.invalid/minun",
                "infobox_json": json.dumps(
                    {"0": ["Minun Cheering Pokémon | #0312", "Minun", "Electric | Unknown"]}
                ),
            },
        ]
    )


def test_parse_infobox_extracts_dex_and_types() -> None:
    dex, types = parse_infobox(frame().iloc[0]["infobox_json"])

    assert dex == 91
    assert types == ("water", "ice")


def test_parse_infobox_is_tolerant() -> None:
    assert parse_infobox("{}") == (None, ())
    assert parse_infobox("not json") == (None, ())
    assert parse_infobox(None) == (None, ())


def test_build_doc_meta_strips_title_suffix() -> None:
    meta = build_doc_meta(frame())

    assert meta["d1"].title == "Cloyster"
    assert meta["d1"].dex == 91
    assert meta["d1"].types == ("water", "ice")
    assert meta["d2"].dex is None
    assert meta["d2"].types == ()
    assert meta["d3"].dex == 312
    assert meta["d3"].types == ("electric",)
