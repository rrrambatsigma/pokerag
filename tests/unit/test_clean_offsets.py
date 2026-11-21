"""The span invariant is the contract the whole evaluation depends on."""

from __future__ import annotations

from pokerag.ingest.clean import clean_html

HTML = """
<div class="mw-parser-output">
  <table>
    <tr><td>For Pokémon GO information on this species, see Bulbasaur (GO). Pokémon | #0002: Ivysaur -> This article is about the species.</td></tr>
  </table>
  <table class="infobox">
    <tr><th>Type</th><td>Grass</td></tr>
    <tr><th>Ability</th><td>Overgrow</td></tr>
  </table>
  <p>Bulbasaur is a Grass-type Pokémon introduced in Generation I.</p>
  <h2>Biology</h2>
  <p>It has a bulb on its back that grows with it.</p>
  <table>
    <tr><th>Stat</th><th>Value</th></tr>
    <tr><td>HP</td><td>45</td></tr>
  </table>
  <h2>Game data</h2>
  <h3>Learnset</h3>
  <p>It learns Vine Whip at level 9<sup class="reference">[1]</sup>.</p>
  <div class="navbox">Navigation box that must be dropped</div>
  <table>
    <tr><td>This Pokémon article is part of Project Pokédex, a Bulbapedia project.</td></tr>
  </table>
</div>
"""


def test_sections_tile_the_text_without_gaps(cfg):
    doc = clean_html(
        HTML,
        doc_id="bulba:0000001",
        pageid=1,
        revid=10,
        title="Bulbasaur (Pokémon)",
        url="https://example.invalid/Bulbasaur",
        license=cfg.license_for(10),
        cfg=cfg,
    )

    assert doc.n_chars > 0
    assert doc.sections[0].char_start == 0
    for previous, current in zip(doc.sections, doc.sections[1:]):
        assert current.char_start == previous.char_end + 2, "sections must be separated by exactly one blank line"
    assert doc.sections[-1].char_end == len(doc.text)
    for section in doc.sections:
        assert section.char_end > section.char_start
        assert doc.text[section.char_start : section.char_end].strip()


def test_infobox_is_promoted_to_text_and_chrome_is_dropped(cfg):
    doc = clean_html(
        HTML,
        doc_id="bulba:0000001",
        pageid=1,
        revid=10,
        title="Bulbasaur (Pokémon)",
        url="https://example.invalid/Bulbasaur",
        license=cfg.license_for(10),
        cfg=cfg,
    )

    assert doc.infobox, "infobox facts should be preserved"
    assert "Type | Grass" in doc.infobox
    assert doc.text.startswith("Type | Grass")
    assert "Navigation box" not in doc.text
    assert "Project Pokédex" not in doc.text
    assert "For Pokémon GO information" not in doc.text
    assert "[1]" not in doc.text
    assert any("Vine Whip" in doc.text[section.char_start : section.char_end] for section in doc.sections)


def test_heading_paths_are_nested(cfg):
    doc = clean_html(
        HTML,
        doc_id="bulba:0000001",
        pageid=1,
        revid=10,
        title="Bulbasaur (Pokémon)",
        url="https://example.invalid/Bulbasaur",
        license=cfg.license_for(10),
        cfg=cfg,
    )
    paths = [section.heading_path for section in doc.sections]
    assert ["Game data", "Learnset"] in paths


def test_wikitext_flags_detect_redirect_and_disambiguation():
    from pokerag.ingest.clean import parse_wikitext_flags

    assert parse_wikitext_flags("#REDIRECT [[Bulbasaur (Pokémon)]]")[0] is True
    assert parse_wikitext_flags("#REDIRECT [[Bulbasaur (Pokémon)]]")[1] == "Bulbasaur (Pokémon)"
    assert parse_wikitext_flags("{{disambig}}\nSome page")[2] is True
