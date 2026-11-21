"""Freeze stage: turn a validated snapshot into immutable, hashed artefacts.

The snapshot id is derived from the document identities (doc_id, revid, clean
hash) and the config hash, so any later change to the corpus or the cleaning
rules necessarily changes the id.

Idempotency rules:

* rows are sorted by ``doc_id`` before writing, so the same data produces the
  same bytes regardless of discovery order;
* no timestamps or git metadata enter the hashed artefacts; those go to
  ``provenance.json``, which is explicitly outside the content manifest;
* ``config_snapshot.json`` records the exact config used, so the id can be
  recomputed later without trusting the current file on disk.
"""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone

import pandas as pd

from pokerag.core.hashing import sha256_bytes, sha256_json
from pokerag.core.jsonl import read_json, read_jsonl, write_json
from pokerag.core.logging import get_logger
from pokerag.ingest.config import CorpusConfig

logger = get_logger("pokerag.ingest.freeze")

CONTENT_FILES = ("documents.parquet", "register.parquet", "dedup_report.json", "validation.json", "config_snapshot.json")


class SnapshotNotValid(RuntimeError):
    """Raised when freezing a snapshot whose validation gates did not pass."""


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unknown"


def compute_snapshot_id(config_dump: dict, identities: list[tuple[str, int, str]]) -> str:
    """Hash of the config plus every document identity, in a fixed order."""
    return sha256_json({"config": config_dump, "documents": sorted(identities)})


def run_freeze(cfg: CorpusConfig, force: bool = False) -> dict:
    """Write ``documents.parquet``, ``register.parquet`` and the content manifest."""
    snap = cfg.snapshot_dir
    validation = read_json(snap / "validation.json")
    if not validation.get("passed") and not force:
        failed = [gate["gate"] for gate in validation.get("gates", []) if not gate["passed"]]
        raise SnapshotNotValid(
            f"validation belum lulus: {failed}. Perbaiki dulu, atau pakai --force dengan alasan tertulis."
        )

    dedup = read_json(snap / "dedup_report.json")
    duplicates: dict[str, str] = dedup.get("duplicates", {})
    cleaned = {row["doc_id"]: row for row in read_jsonl(snap / "cleaned.jsonl")}
    register_rows = {row["doc_id"]: row for row in read_jsonl(snap / "pages_index.jsonl")}

    final = [
        cleaned[doc_id]
        for doc_id in sorted(cleaned)
        if not cleaned[doc_id]["is_redirect"]
        and not cleaned[doc_id]["is_disambiguation"]
        and not cleaned[doc_id]["is_stub"]
        and cleaned[doc_id]["n_chars"] > 0
        and doc_id not in duplicates
    ]

    documents = pd.DataFrame(
        [
            {
                "doc_id": doc["doc_id"],
                "pageid": doc["pageid"],
                "revid": doc["revid"],
                "title": doc["title"],
                "url": doc["url"],
                "rev_timestamp": register_rows.get(doc["doc_id"], {}).get("rev_timestamp"),
                "api_sha1": register_rows.get(doc["doc_id"], {}).get("api_sha1"),
                "content_sha256": doc["content_sha256"],
                "n_chars": doc["n_chars"],
                "n_sections": doc["n_sections"],
                "license_name": doc["license"]["name"],
                "license_url": doc["license"]["url"],
                "allow_retrieval": doc["license"]["allow_retrieval"],
                "allow_redistribute": doc["license"]["allow_redistribute"],
                "allow_ft_training": doc["license"]["allow_ft_training"],
                "attribution": doc["license"]["attribution"],
                "infobox_json": pd.Series([doc["infobox"]]).to_json(force_ascii=False),
                "sections_json": pd.Series([doc["sections"]]).to_json(force_ascii=False),
                "text": doc["text"],
            }
            for doc in final
        ]
    ).sort_values("doc_id", ignore_index=True)

    register = pd.DataFrame(
        [
            {
                "doc_id": doc["doc_id"],
                "pageid": doc["pageid"],
                "revid": doc["revid"],
                "title": doc["title"],
                "url": doc["url"],
                "rev_timestamp": register_rows.get(doc["doc_id"], {}).get("rev_timestamp"),
                "api_sha1": register_rows.get(doc["doc_id"], {}).get("api_sha1"),
                "api_size": register_rows.get(doc["doc_id"], {}).get("api_size"),
                "content_sha256": doc["content_sha256"],
                "n_chars": doc["n_chars"],
                "n_sections": doc["n_sections"],
                "is_redirect": doc["is_redirect"],
                "is_disambiguation": doc["is_disambiguation"],
                "is_stub": doc["is_stub"],
                "duplicate_of": duplicates.get(doc["doc_id"]),
                "license_name": doc["license"]["name"],
                "allow_ft_training": doc["license"]["allow_ft_training"],
                "attribution": doc["license"]["attribution"],
            }
            for doc in (cleaned[doc_id] for doc_id in sorted(cleaned))
        ]
    )

    documents_path = snap / "documents.parquet"
    register_path = snap / "register.parquet"
    documents.to_parquet(documents_path, index=False)
    register.to_parquet(register_path, index=False)

    config_dump = cfg.model_dump()
    write_json(snap / "config_snapshot.json", config_dump)

    identities = [(doc["doc_id"], int(doc["revid"]), doc["content_sha256"]) for doc in final]
    snapshot_id = compute_snapshot_id(config_dump, identities)

    manifest = {
        "snapshot_id": snapshot_id,
        "snapshot": cfg.name,
        "counts": {
            "documents": len(documents),
            "register_rows": len(register),
            "characters": int(documents["n_chars"].sum()),
        },
        "content_files": {name: sha256_bytes((snap / name).read_bytes()) for name in CONTENT_FILES},
        "config_sha256": sha256_json(config_dump),
        "license": cfg.license.model_dump(),
    }
    write_json(snap / "manifest.json", manifest)

    write_json(
        snap / "snapshot.json",
        {
            "snapshot_id": snapshot_id,
            "name": cfg.name,
            "validation_passed": validation.get("passed", False),
            "forced": bool(force and not validation.get("passed", False)),
            "stats": validation.get("stats", {}),
        },
    )
    write_json(
        snap / "provenance.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
    )
    logger.info("frozen %s documents, snapshot_id=%s", len(documents), snapshot_id[:16])
    return manifest
