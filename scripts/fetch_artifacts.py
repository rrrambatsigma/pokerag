#!/usr/bin/env python3
"""Fetch the frozen PokeRAG artifacts and verify every digest.

Default set: manifests, reports, chunk parquets, documents, and ``indexes.tar.gz``.
Use ``--all`` to mirror the whole dataset (raw HTML and the embedding cache included).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from pokerag.core.config import settings
from pokerag.core.integrity import parse_integrity, sha256_file
from pokerag.ingest.hf import default_progress, download_file, select_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=settings.hf_dataset)
    parser.add_argument("--revision", default=settings.hf_revision)
    parser.add_argument("--dest", type=Path, default=settings.artifacts_dir)
    parser.add_argument("--all", action="store_true", help="download every dataset file")
    parser.add_argument("--extract", action="store_true", help="extract indexes.tar.gz")
    parser.add_argument("--force", action="store_true", help="redownload and re-extract")
    return parser.parse_args()


def extract_indexes(archive: Path, dest: Path, force: bool) -> None:
    if not archive.is_file():
        raise SystemExit(f"missing archive: {archive}")
    target = dest / "indexes"
    marker = target / "qdrant" / "meta.json"
    if marker.is_file() and not force:
        print(f"skip extract, {marker} already exists")
        return
    stale_paths = ("qdrant", "bm25", "fixed", "heading", "hierarchical")
    for name in stale_paths:
        candidate = target / name
        if candidate.exists() and force:
            shutil.rmtree(candidate)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    print(f"extracted {archive.name} -> {target}")


def main() -> int:
    args = parse_args()
    artifacts: Path = args.dest.resolve()
    raw = artifacts / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    repo_commit: str | None = None

    with default_progress() as progress:
        integrity_file = download_file(
            dataset=args.dataset,
            remote_path="INTEGRITY.sha256",
            dest=raw / "INTEGRITY.sha256",
            revision=args.revision,
            force=args.force,
            progress=progress,
        )
        repo_commit = integrity_file.repo_commit

        entries = parse_integrity((raw / "INTEGRITY.sha256").read_text(encoding="utf-8"))
        if not entries:
            raise SystemExit("INTEGRITY.sha256 parsed to zero entries")

        paths = select_paths(entries, include_all=args.all)
        print(f"{len(paths)} of {len(entries)} dataset files selected")

        for path in paths:
            entry = entries[path]
            result = download_file(
                dataset=args.dataset,
                remote_path=path,
                dest=raw / path,
                revision=args.revision,
                expected_sha256=entry.sha256,
                force=args.force,
                progress=progress,
            )
            repo_commit = repo_commit or result.repo_commit
            records.append(
                {
                    "path": path,
                    "sha256": result.sha256,
                    "bytes": result.path.stat().st_size,
                    "skipped": result.skipped,
                }
            )

    manifest = {
        "dataset": args.dataset,
        "revision": args.revision,
        "repo_commit": repo_commit,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "verified": True,
        "files": records,
    }
    (raw / "FETCH.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    downloaded = sum(1 for record in records if not record["skipped"])
    reused = len(records) - downloaded
    print(f"verified {len(records)} files, downloaded {downloaded}, reused {reused}")
    print(f"sha256 of INTEGRITY.sha256: {sha256_file(raw / 'INTEGRITY.sha256')}")

    if args.extract:
        extract_indexes(raw / "indexes.tar.gz", artifacts, args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
