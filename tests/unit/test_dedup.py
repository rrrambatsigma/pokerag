"""Near-duplicate detection keeps one canonical document and records the rest."""

from __future__ import annotations

from pathlib import Path

from pokerag.core.hashing import sha256_text
from pokerag.core.jsonl import write_jsonl
from pokerag.ingest.dedup import jaccard, run_dedup, shingles

BASE = (
    "bulbasaur is a grass poison type pokemon with a bulb on its back "
    "it evolves into ivysaur at level sixteen and then into venusaur "
) * 3
NEAR = BASE + " a short extra sentence about sunlight"
OTHER = (
    "pikachu is an electric type pokemon that stores electricity in its cheeks "
    "and evolves when exposed to a thunder stone "
) * 3


def _row(doc_id: str, text: str) -> dict:
    return {
        "doc_id": doc_id,
        "text": text,
        "n_chars": len(text),
        "content_sha256": sha256_text(text),
        "is_redirect": False,
        "is_disambiguation": False,
    }


def test_shingles_and_jaccard_are_deterministic():
    left = shingles(BASE, 3)
    right = shingles(BASE, 3)
    assert left == right
    assert jaccard(left, right) == 1.0
    assert jaccard(shingles(BASE, 3), shingles(OTHER, 3)) < 0.2


def test_run_dedup_marks_only_the_copy(cfg, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("POKERAG_ARTIFACTS_ROOT", str(tmp_path))
    snapshot = cfg.snapshot_dir
    snapshot.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        snapshot / "cleaned.jsonl",
        [_row("bulba:1", BASE), _row("bulba:2", NEAR), _row("bulba:3", OTHER)],
    )

    groups = run_dedup(cfg)

    assert groups, "expected at least one duplicate group"
    from pokerag.core.jsonl import read_json

    report = read_json(snapshot / "dedup_report.json")
    duplicates = report["duplicates"]
    assert len(duplicates) == 1
    canonical = groups[0].canonical
    assert canonical == "bulba:2", "the longer document wins"
    assert list(duplicates.values()) == [canonical]
