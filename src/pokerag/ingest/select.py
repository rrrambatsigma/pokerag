"""Deterministic corpus selection (National Pokédex order, not alphabetical).

The list page renders each entry through the ``{{ndex|dex|name|type...}}``
template, so the wikitext is the stable and cheap source of truth. Titles are
then requested as ``"<Name> (Pokémon)"``, the article namespace used by
Bulbapedia for species.
"""

from __future__ import annotations

from urllib.parse import quote

import mwparserfromhell

from pokerag.core.logging import get_logger
from pokerag.ingest.config import CorpusConfig
from pokerag.ingest.models import SelectedPage
from pokerag.ingest.transport import MediaWikiTransport

logger = get_logger("pokerag.ingest.select")

BASE_URL = "https://bulbapedia.bulbagarden.net/wiki/"


def parse_ndex(wikitext: str) -> list[tuple[int, str]]:
    """Extract ``(dex, display_name)`` pairs from a National Dex list page.

    Entries whose dex number is not numeric (unreleased species) are skipped.
    """
    parsed = mwparserfromhell.parse(wikitext)
    rows: list[tuple[int, str]] = []
    for template in parsed.filter_templates():
        if template.name.strip().lower() != "ndex":
            continue
        params = [str(p.value).strip() for p in template.params]
        if len(params) < 2:
            continue
        dex_text, name = params[0], params[1]
        if not dex_text.isdigit():
            continue
        rows.append((int(dex_text), name))
    return rows


def title_for(name: str) -> str:
    """Article title of a species page."""
    return f"{name} (Pokémon)"


def url_for(title: str) -> str:
    """Canonical wiki URL for a title."""
    return BASE_URL + quote(title.replace(" ", "_"))


def build_selection(cfg: CorpusConfig, transport: MediaWikiTransport) -> list[SelectedPage]:
    """Fetch the list page and return the configured dex range."""
    payload = transport.get_json(
        {
            "action": "parse",
            "page": cfg.selection.list_page,
            "prop": "wikitext",
            "format": "json",
            "formatversion": "2",
        }
    )
    rows = parse_ndex(payload["parse"]["wikitext"])
    chosen = [
        SelectedPage(dex=dex, name=name, title=title_for(name), url=url_for(title_for(name)))
        for dex, name in rows
        if cfg.selection.from_dex <= dex <= cfg.selection.to_dex
    ]
    logger.info("selected %s species (dex %s-%s)", len(chosen), cfg.selection.from_dex, cfg.selection.to_dex)
    if not chosen:
        raise RuntimeError("selection is empty; check the list page or dex range")
    return chosen
