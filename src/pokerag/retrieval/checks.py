"""Guards between the index on disk and the snapshot it claims to describe.

Every failure here would otherwise be silent: a stale collection returns
confident neighbours, and a shuffled BM25 ``chunk_ids.json`` returns hits that
point at the wrong text. Both produce plausible-looking output, so they are
checked before the first query rather than trusted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient

from pokerag.core.hashing import sha256_bytes
from pokerag.core.jsonl import read_json
from pokerag.index.build import identity_hash
from pokerag.index.config import IndexConfig
from pokerag.index.qdrant_store import collection_count
from pokerag.retrieval.corpus import chunks_path


def verify_manifest(cfg: IndexConfig, strategy: str) -> dict:
    """Check the manifest exists, matches the embedder and the chunk bytes."""
    path = cfg.manifest_path(strategy)
    if not path.exists():
        raise RuntimeError(
            f"koleksi `{cfg.collection_name(strategy)}` belum dibangun; jalankan "
            f"`pokerag-index build --strategy {strategy}` dulu"
        )
    manifest = read_json(path)

    expected = identity_hash(cfg)
    if manifest.get("embedder_identity_sha256") != expected:
        raise RuntimeError(
            f"koleksi `{manifest.get('collection', strategy)}` dibangun dengan embedder lain "
            f"({manifest.get('embedder', {}).get('model')}, "
            f"{manifest.get('embedder', {}).get('dimension')} dim). "
            "Pakai config yang sama dengan saat build; vektor query tidak boleh beda model."
        )

    chunk_file = chunks_path(cfg, strategy)
    if not chunk_file.exists():
        raise RuntimeError(f"chunk `{strategy}` tidak ada di {chunk_file}")
    digest = sha256_bytes(chunk_file.read_bytes())
    if manifest.get("chunks_sha256") != digest:
        raise RuntimeError(
            f"index `{strategy}` basi: isi {chunk_file.name} sudah berubah sejak build. "
            f"Jalankan `pokerag-index build --strategy {strategy} --force` dulu."
        )
    return manifest


def verify_bm25_alignment(cfg: IndexConfig, strategy: str, order: Sequence[str]) -> None:
    """BM25 maps results by row number, so the id order must be identical."""
    ids_path = Path(cfg.bm25_dir(strategy)) / "chunk_ids.json"
    if not ids_path.exists():
        raise RuntimeError(
            f"BM25 `{strategy}` tidak ada di {ids_path.parent}; jalankan "
            f"`pokerag-index build --strategy {strategy}` dulu"
        )
    stored = json.loads(ids_path.read_text(encoding="utf-8"))
    if len(stored) != len(order):
        raise RuntimeError(
            f"BM25 `{strategy}` berisi {len(stored)} id, parquet berisi {len(order)}. "
            "Urutan baris tidak bisa dipetakan; bangun ulang index."
        )
    mismatch = next(
        (position for position, (left, right) in enumerate(zip(stored, order)) if left != right),
        None,
    )
    if mismatch is not None:
        raise RuntimeError(
            f"urutan chunk id BM25 `{strategy}` berbeda dari parquet pada baris {mismatch} "
            f"({stored[mismatch]} != {order[mismatch]}). BM25 memetakan hasil lewat nomor "
            "baris, jadi bangun ulang index sebelum query."
        )


def verify_points(client: QdrantClient, cfg: IndexConfig, strategy: str, rows: int) -> int:
    """The collection must contain exactly one point per chunk row."""
    count = collection_count(client, cfg.collection_name(strategy))
    if count != rows:
        raise RuntimeError(
            f"koleksi `{cfg.collection_name(strategy)}` berisi {count} titik, parquet berisi "
            f"{rows}. Jalankan `pokerag-index build --strategy {strategy} --force` dulu."
        )
    return count
