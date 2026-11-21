"""Selection must be deterministic and dex-ordered."""

from __future__ import annotations

from pokerag.ingest.select import parse_ndex, title_for

WIKITEXT = """
{{ndexh|Red|Green}}
{{ndex|0001|Bulbasaur|Grass|Poison}}
{{ndex|0002|Ivysaur|Grass|Poison}}
{{ndex|?? ??|Browt|Grass}}
{{ndex|0025|Pikachu|Electric}}
"""


def test_parse_ndex_keeps_order_and_skips_unreleased_species():
    rows = parse_ndex(WIKITEXT)
    assert rows == [(1, "Bulbasaur"), (2, "Ivysaur"), (25, "Pikachu")]


def test_title_uses_species_namespace():
    assert title_for("Pikachu") == "Pikachu (Pokémon)"
