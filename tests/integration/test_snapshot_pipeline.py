"""End-to-end snapshot chain on synthetic data (no network).

Covers the part that is easy to get wrong: dedup -> validation -> freeze, and
the refusal path when a gate fails.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pokerag.core.hashing import sha256_text
from pokerag.core.jsonl import read_json, write_jsonl
from pokerag.ingest.config import CorpusConfig
from pokerag.ingest.dedup import run_dedup
from pokerag.ingest.freeze import SnapshotNotValid, run_freeze
from pokerag.ingest.validate import run_validation

BASE_TEXT = (
    "Bulbasaur is a dual-type Grass and Poison Pokemon introduced in Generation I. "
    "It evolves into Ivysaur starting at level 16. It has a bulb on its back. "
) * 3


def _doc(doc_id: str, pageid: int, text: str, *, stub: bool = False, disambig: bool = False) -> dict:
    start = len(text) // 2
    return {
        "doc_id": doc_id,
        "pageid": pageid,
        "revid": 1000 + pageid,
        "title": f"Doc {pageid}",
        "url": f"https://example.invalid/{pageid}",
        "text": text,
        "content_sha256": sha256_text(text),
        "n_chars": len(text),
        "n_sections": 4,
        "sections": [
            {"heading_path": ["Lead"], "level": 0, "kind": "lead", "char_start": 0, "char_end": start},
            {"heading_path": ["Body"], "level": 2, "kind": "section", "char_start": start + 2, "char_end": len(text)},
        ],
        "infobox": ["Grass | Poison"],
        "is_redirect": False,
        "redirect_target": None,
        "is_disambiguation": disambig,
        "is_stub": stub,
        "dropped": {},
        "license": {
            "name": "CC-BY-NC-SA-2.5",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.5/",
            "allow_retrieval": True,
            "allow_redistribute": False,
            "allow_ft_training": False,
            "attribution": f"Bulbapedia, CC BY-NC-SA 2.5, revision {1000 + pageid}",
        },
    }


def _prepare(cfg: CorpusConfig, tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("POKERAG_ARTIFACTS_ROOT", str(tmp_path))
    snap = cfg.snapshot_dir
    snap.mkdir(parents=True, exist_ok=True)
    docs = [
        _doc("bulba:1", 1, BASE_TEXT),
        _doc("bulba:2", 2, BASE_TEXT + " extra evidence"),
        _doc("bulba:3", 3, "Pikachu is an Electric-type Pokemon. " * 10),
        _doc("bulba:4", 4, "Short stub.", stub=True),
        _doc("bulba:5", 5, "Disambiguation page.", disambig=True),
    ]
    write_jsonl(snap / "cleaned.jsonl", docs)
    write_jsonl(
        snap / "pages_index.jsonl",
        [
            {
                "doc_id": doc["doc_id"],
                "pageid": doc["pageid"],
                "revid": doc["revid"],
                "title": doc["title"],
                "url": doc["url"],
                "rev_timestamp": "2026-11-14T00:00:00Z",
                "api_sha1": "abc",
                "api_size": 100,
            }
            for doc in docs
        ],
    )
    write_jsonl(snap / "fetch_errors.jsonl", [])
    write_jsonl(
        snap / "fetch_log.jsonl",
        [{"status": 200, "gap_seconds": 5.0}, {"status": 200, "gap_seconds": 5.1}],
    )
    return snap


def test_chain_produces_a_frozen_snapshot(cfg: CorpusConfig, tmp_path: Path, monkeypatch):
    snap = _prepare(cfg, tmp_path, monkeypatch)
    run_dedup(cfg)
    validation = run_validation(cfg)
    assert validation["passed"], [g for g in validation["gates"] if not g["passed"]]
    manifest = run_freeze(cfg)

    documents = pd.read_parquet(snap / "documents.parquet")
    register = pd.read_parquet(snap / "register.parquet")
    assert len(documents) == 2, "near-duplicate and stub/disambiguation must be excluded"
    assert len(register) == 5
    assert documents["allow_ft_training"].eq(False).all()
    assert manifest["snapshot_id"] == read_json(snap / "snapshot.json")["snapshot_id"]
    assert (snap / "manifest.json").exists()


def test_freeze_refuses_a_failed_gate(cfg: CorpusConfig, tmp_path: Path, monkeypatch):
    snap = _prepare(cfg, tmp_path, monkeypatch)
    run_dedup(cfg)
    strict = cfg.model_copy(
        update={"validation": cfg.validation.model_copy(update={"min_docs": 999})}
    )
    validation = run_validation(strict)
    assert not validation["passed"]

    with pytest.raises(SnapshotNotValid):
        run_freeze(strict)

    manifest = run_freeze(strict, force=True)
    assert manifest["snapshot_id"]
    assert read_json(snap / "snapshot.json")["forced"] is True
