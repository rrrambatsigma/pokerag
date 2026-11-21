"""Shared fixtures for ingestion tests."""

from __future__ import annotations

import pytest

from pokerag.ingest.config import CorpusConfig

BASE_CONFIG = {
    "name": "test-corpus",
    "source_site": "bulbapedia",
    "artifacts_root": "artifacts",
    "selection": {
        "method": "national_dex_range",
        "list_page": "List of Pokémon by National Pokédex number",
        "from_dex": 1,
        "to_dex": 3,
        "namespace": 0,
    },
    "fetch": {
        "api": "https://example.invalid/w/api.php",
        "user_agent": "PokeRAG-test/0.1",
        "delay_seconds": 0.0,
        "batch_titles": 2,
        "timeout_seconds": 5,
        "max_retries": 2,
        "backoff_seconds": [0, 0],
    },
    "license": {
        "name": "CC-BY-NC-SA-2.5",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.5/",
        "allow_retrieval": True,
        "allow_redistribute": False,
        "allow_ft_training": False,
        "attribution_template": "Bulbapedia, CC BY-NC-SA 2.5, revision {revid}",
    },
    "cleaning": {
        "min_chars": 80,
        "min_sections": 2,
        "drop_selectors": [
            ".mw-editsection",
            "sup.reference",
            "ol.references",
            ".toc",
            "figure",
            "style",
            "script",
            ".navbox",
            ".infobox",
            ".noprint",
        ],
        "drop_section_titles": ["Contents"],
    },
    "dedup": {"shingle_size": 3, "num_perm": 32, "threshold": 0.8},
    "validation": {"min_docs": 1, "min_chars": 80, "min_sections": 2},
}


@pytest.fixture
def cfg() -> CorpusConfig:
    return CorpusConfig.model_validate(BASE_CONFIG)
