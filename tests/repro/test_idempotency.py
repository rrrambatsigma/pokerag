"""Idempotency is claimed in the README, so it is tested here.

Every stage must satisfy: running it twice leaves byte-identical artefacts and
performs no new work. The fake transport counts requests, which is the only
honest way to prove "resume actually skipped everything".
"""

from __future__ import annotations

from pathlib import Path

from pokerag.core.hashing import sha256_bytes
from pokerag.core.jsonl import read_jsonl, write_jsonl
from pokerag.ingest.clean import run_clean
from pokerag.ingest.dedup import run_dedup
from pokerag.ingest.fetch import fetch_html, fetch_wikitext
from pokerag.ingest.freeze import run_freeze
from pokerag.ingest.models import SelectedPage
from pokerag.ingest.validate import run_validation

WIKITEXT = "'''Bulbasaur''' is a Grass-type Pokémon. It evolves into Ivysaur."
HTML = """
<div class="mw-parser-output">
  <p>Bulbasaur is a dual-type Grass and Poison Pokémon introduced in Generation I.
     It has a bulb on its back that grows as it ages and stores energy from sunlight.</p>
  <h2>Biology</h2>
  <p>Bulbasaur is a small quadrupedal Pokémon with blue-green skin and darker patches.
     It spends most of its time sleeping in the sun to absorb nutrients through its bulb.</p>
</div>
"""


class FakeTransport:
    """Minimal MediaWiki stand-in that counts real network-equivalent calls."""

    def __init__(self) -> None:
        self.calls = 0

    def get_json(self, params: dict) -> dict:
        self.calls += 1
        if params.get("action") == "query":
            return {
                "query": {
                    "pages": [
                        {
                            "pageid": 1,
                            "ns": 0,
                            "title": "Bulbasaur (Pokémon)",
                            "revisions": [
                                {
                                    "revid": 10,
                                    "timestamp": "2026-01-01T00:00:00Z",
                                    "sha1": "deadbeef",
                                    "size": len(WIKITEXT),
                                    "slots": {"main": {"content": WIKITEXT}},
                                }
                            ],
                        }
                    ]
                }
            }
        if params.get("action") == "parse":
            return {"parse": {"revid": 10, "text": HTML}}
        raise AssertionError(f"unexpected params {params}")


SELECTION = [
    SelectedPage(
        dex=1,
        name="Bulbasaur",
        title="Bulbasaur (Pokémon)",
        url="https://example.invalid/wiki/Bulbasaur_(Pok%C3%A9mon)",
    )
]


def _bytes(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def test_atomic_writes_are_stable(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    rows = [{"b": 2, "a": 1}, {"b": 3, "a": 0}]
    write_jsonl(path, rows)
    first = _bytes(path)
    write_jsonl(path, rows)
    assert _bytes(path) == first
    assert not (tmp_path / "rows.jsonl.tmp").exists()


def test_fetch_and_clean_are_idempotent(cfg, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("POKERAG_ARTIFACTS_ROOT", str(tmp_path))
    transport = FakeTransport()

    records = fetch_wikitext(cfg, transport, SELECTION)
    fetch_html(cfg, transport, records)
    first_calls = transport.calls
    assert first_calls == 2

    index_path = cfg.snapshot_dir / "pages_index.jsonl"
    html_index_path = cfg.snapshot_dir / "html_index.jsonl"
    index_hash = _bytes(index_path)

    records_again = fetch_wikitext(cfg, transport, SELECTION)
    fetch_html(cfg, transport, records_again)
    assert transport.calls == first_calls, "second fetch must not hit the network"
    assert _bytes(index_path) == index_hash
    assert sum(1 for _ in read_jsonl(index_path)) == 1
    assert sum(1 for _ in read_jsonl(html_index_path)) == 1

    cleaned_path = cfg.snapshot_dir / "cleaned.jsonl"
    assert run_clean(cfg) == 1
    cleaned_hash = _bytes(cleaned_path)
    assert run_clean(cfg) == 0, "second clean must not rewrite the document"
    assert _bytes(cleaned_path) == cleaned_hash


def test_freeze_is_byte_identical(cfg, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("POKERAG_ARTIFACTS_ROOT", str(tmp_path))
    transport = FakeTransport()
    records = fetch_wikitext(cfg, transport, SELECTION)
    fetch_html(cfg, transport, records)
    run_clean(cfg)
    run_dedup(cfg)
    assert run_validation(cfg)["passed"] is True

    first = run_freeze(cfg)
    documents_hash = _bytes(cfg.snapshot_dir / "documents.parquet")
    second = run_freeze(cfg)

    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["content_files"] == second["content_files"]
    assert _bytes(cfg.snapshot_dir / "documents.parquet") == documents_hash
    assert second["counts"]["documents"] == 1
