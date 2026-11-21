#!/usr/bin/env python3
"""Verify that the local artifacts match the manifests shipped with the corpus.

Checks four things per chunking strategy: the Qdrant collection exists at the
built path, its point count matches the manifest, the BM25 sidecar loads with the
expected document count, and every chunk is still a valid slice of its document.
Nothing is rebuilt and no embedding model is loaded.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import bm25s
import pandas as pd
from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table

from pokerag.core.config import settings

STRATEGIES = ("fixed", "heading", "hierarchical")

warnings.filterwarnings("ignore", message="Local mode is not recommended.*")
CHUNK_COLUMNS = ("chunk_id", "doc_id", "char_start", "char_end", "text")
LICENSE_COLUMNS = ("allow_retrieval", "allow_redistribute", "allow_ft_training")


def load_manifest(indexes: Path, strategy: str) -> dict:
    path = indexes / strategy / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_qdrant(indexes: Path, manifests: dict[str, dict], console: Console) -> bool:
    path = indexes / "qdrant"
    if not path.is_dir():
        console.print(f"[red]missing Qdrant storage: {path}[/red]")
        return False

    console.print(f"qdrant storage: {path}")
    client = QdrantClient(path=str(path))
    table = Table(title="Qdrant collections")
    table.add_column("Strategy")
    table.add_column("Collection")
    table.add_column("Expected", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Status")

    ok = True
    try:
        available = {item.name for item in client.get_collections().collections}
        for strategy in STRATEGIES:
            manifest = manifests[strategy]
            collection = manifest["collection"]
            expected = int(manifest["counts"]["points"])
            if collection not in available:
                table.add_row(strategy, collection, str(expected), "-", "missing")
                ok = False
                continue
            actual = client.count(collection_name=collection, exact=True).count
            status = "ok" if actual == expected else "mismatch"
            ok = ok and actual == expected
            table.add_row(strategy, collection, str(expected), str(actual), status)
    finally:
        client.close()

    console.print(table)
    return ok


def verify_bm25(indexes: Path, manifests: dict[str, dict], console: Console) -> bool:
    table = Table(title="BM25 sidecars")
    table.add_column("Strategy")
    table.add_column("Documents", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Status")

    ok = True
    for strategy in STRATEGIES:
        directory = indexes / "bm25" / strategy
        expected = int(manifests[strategy]["counts"]["chunks"])
        try:
            params = json.loads((directory / "params.index.json").read_text(encoding="utf-8"))
            chunk_ids = json.loads((directory / "chunk_ids.json").read_text(encoding="utf-8"))
            bm25s.BM25.load(str(directory))
            actual = len(chunk_ids)
            docs_match = actual == expected and int(params["num_docs"]) == expected
            status = "ok" if docs_match else "mismatch"
            ok = ok and docs_match
            table.add_row(strategy, str(actual), str(expected), status)
        except Exception as error:  # noqa: BLE001 - report and keep checking
            ok = False
            table.add_row(strategy, "-", str(expected), f"error: {str(error)[:48]}")

    console.print(table)
    return ok


def verify_datasets(
    raw: Path, indexes: Path, manifests: dict[str, dict], console: Console
) -> bool:
    docs_path = raw / "documents.parquet"
    snapshot_path = raw / "manifests" / "snapshot.json"
    if not docs_path.is_file() or not snapshot_path.is_file():
        console.print("[red]missing documents.parquet or snapshot manifest[/red]")
        return False

    docs = pd.read_parquet(docs_path, columns=["doc_id", "text"])
    doc_text = docs.set_index("doc_id")["text"]
    expected_docs = int(json.loads(snapshot_path.read_text(encoding="utf-8"))["stats"]["n_final"])

    table = Table(title="Dataset and span integrity")
    table.add_column("Strategy")
    table.add_column("Rows", justify="right")
    table.add_column("Indexed", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Broken spans", justify="right")
    table.add_column("Status")

    ok = len(docs) == expected_docs
    table.add_row(
        "documents", str(len(docs)), "-", str(expected_docs), "-", "ok" if ok else "mismatch"
    )

    for strategy in STRATEGIES:
        path = raw / "chunks" / f"{strategy}.parquet"
        expected = int(manifests[strategy]["counts"]["chunks"])
        chunk_ids = set(
            json.loads(
                (indexes / "bm25" / strategy / "chunk_ids.json").read_text(encoding="utf-8")
            )
        )
        chunks = pd.read_parquet(path, columns=list(CHUNK_COLUMNS))
        indexed = chunks[chunks["chunk_id"].isin(chunk_ids)]
        broken = 0
        for row in chunks.itertuples(index=False):
            text = doc_text.get(row.doc_id)
            if text is None or text[row.char_start : row.char_end] != row.text:
                broken += 1
        rows_ok = len(indexed) == expected
        spans_ok = broken == 0
        status = "ok" if rows_ok and spans_ok else "mismatch"
        ok = ok and rows_ok and spans_ok
        table.add_row(
            strategy, str(len(chunks)), str(len(indexed)), str(expected), str(broken), status
        )

    console.print(table)
    return ok


def verify_licenses(raw: Path, console: Console) -> bool:
    docs = pd.read_parquet(raw / "documents.parquet", columns=list(LICENSE_COLUMNS))

    table = Table(title="License flags")
    table.add_column("Flag")
    table.add_column("Values")
    for column in LICENSE_COLUMNS:
        counts = docs[column].value_counts(dropna=False).to_dict()
        rendered = ", ".join(f"{key}: {value}" for key, value in counts.items())
        table.add_row(column, rendered)
    console.print(table)

    retrieval_ok = bool(docs["allow_retrieval"].all())
    if not retrieval_ok:
        console.print("[red]some documents are not allowed for retrieval[/red]")
    if not bool(docs["allow_ft_training"].all()):
        console.print(
            "[yellow]allow_ft_training is false for some documents; "
            "B5 adapter training needs a separate licensed dataset[/yellow]"
        )
    return retrieval_ok


def main() -> int:
    console = Console(width=100)
    indexes: Path = settings.indexes_dir
    raw: Path = settings.raw_dir
    if not indexes.is_dir():
        console.print(f"[red]missing index directory: {indexes}[/red]")
        console.print("run: uv run python scripts/fetch_artifacts.py --extract")
        return 1

    manifests = {strategy: load_manifest(indexes, strategy) for strategy in STRATEGIES}
    qdrant_ok = verify_qdrant(indexes, manifests, console)
    bm25_ok = verify_bm25(indexes, manifests, console)
    data_ok = verify_datasets(raw, indexes, manifests, console)
    license_ok = verify_licenses(raw, console)

    console.print(f"snapshot_id: {manifests['fixed'].get('snapshot_id', 'unknown')}")
    console.print(f"index_version (fixed): {manifests['fixed'].get('index_version', 'unknown')}")

    if qdrant_ok and bm25_ok and data_ok and license_ok:
        console.print("[green]M0 index verification passed[/green]")
        return 0
    console.print("[red]M0 index verification failed[/red]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
