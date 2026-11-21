"""Upload the chunking result to a focused Hugging Face dataset repo.

Only what belongs to the chunking stage is uploaded: the three chunk files, the
two corpus tables needed to resolve spans (documents, register), the manifests,
the audit, and a dataset card. Embeddings and the vector index stay out.

Chunk text is verbatim Bulbapedia text (CC BY-NC-SA 2.5), so the default is a
private repo with attribution and a licence notice, consistent with
``allow_redistribute = false`` in the corpus config.

Usage
-----
    python scripts/upload_chunks_hf.py --dry-run
    python scripts/upload_chunks_hf.py
    python scripts/upload_chunks_hf.py --repo someone/other-name --public
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

ARTIFACTS = Path("artifacts")
SNAPSHOT = ARTIFACTS / "corpus" / "pokerag-corpus-v1"
STAGING = ARTIFACTS / "hf_chunks_staging"
DEFAULT_REPO = "meivawijaya/pokerag-chunks-v1"
STRATEGIES = ("fixed", "heading", "hierarchical")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def link_or_copy(source: Path, target: Path) -> None:
    """Hard-link when possible (same volume), otherwise copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def strategy_rows(manifest: dict, audit: dict) -> list[str]:
    rows = []
    for name in STRATEGIES:
        entry = manifest.get("strategies", {}).get(name)
        detail = audit.get("strategies", {}).get(name)
        if not entry or not detail:
            continue
        boundary = detail["boundary"]
        rows.append(
            f"| `{name}` | {detail['units']:,} | {detail['parents']:,} | "
            f"{detail['chars_p50']} / {detail['chars_p95']} | "
            f"{detail['tokens_p50']} / {detail['tokens_p95']} | "
            f"{entry.get('coverage_ratio', 0):.1%} | {detail['dropped']:,} | "
            f"{boundary['crosses_h2']:,} |"
        )
    return rows


def build_card(manifest: dict, audit: dict, snapshot: dict) -> str:
    stats = snapshot.get("stats", {})
    snapshot_id = snapshot.get("snapshot_id", "")
    return "\n".join(
        [
            "---",
            "license: other",
            "license_name: cc-by-nc-sa-2.5",
            "license_link: https://creativecommons.org/licenses/by-nc-sa/2.5/",
            "language:",
            "  - en",
            "task_categories:",
            "  - text-retrieval",
            "tags:",
            "  - rag",
            "  - pokemon",
            "  - bulbapedia",
            "  - chunking",
            "pretty_name: PokeRAG Chunks v1",
            "---",
            "",
            "# PokeRAG Chunks v1",
            "",
            "Three chunkings of the same 320 Bulbapedia Pokemon articles, built for",
            "ablation A1 (fixed vs heading-aware vs hierarchical chunking). The chunk",
            "sets differ only in their boundaries: same documents, same text, same",
            "character offsets.",
            "",
            "## Files",
            "",
            "| Path | Description |",
            "|---|---|",
            "| `fixed.parquet` | sliding window, 2048 chars / 256 overlap (baseline B2) |",
            "| `heading.parquet` | heading-aware, 1200 chars / 120 overlap, min 240 |",
            "| `hierarchical.parquet` | parent blocks + 1000-char child windows |",
            "| `documents.parquet` | 320 cleaned documents; `text` is what offsets index into |",
            "| `register.parquet` | pageid, revid, revision timestamp, licence, hashes |",
            "| `chunks_manifest.json` | per-strategy counts, config hash, snapshot id |",
            "| `chunks_audit.json` | boundary integrity, section fragmentation, overlap redundancy |",
            "| `chunks_report.md` | human-readable A1 table |",
            "| `INTEGRITY.sha256` | SHA-256 of every file in this repository |",
            "",
            "## Chunking results",
            "",
            "| Strategy | Units | Parents | chars p50 / p95 | tokens p50 / p95 | coverage | dropped (<80 chars) | crosses h2 |",
            "|---|---|---|---|---|---|---|---|",
            *strategy_rows(manifest, audit),
            "",
            "`Units` are the chunks meant for embedding. In `hierarchical`, `parents`",
            "are larger blocks kept for context expansion (role = `parent`) and are",
            "**not** meant to be embedded: filtering `role != 'parent'` gives the",
            "24,444 retrieval units.",
            "",
            "Token counts use the embedding model tokenizer (`BAAI/bge-m3`), not a",
            "chars/4 estimate. The longest chunk is 1,005 tokens, which is why the",
            "index uses a 1,024-token window.",
            "",
            "## Schema",
            "",
            "Every chunk file has the same columns:",
            "",
            "| Column | Meaning |",
            "|---|---|",
            "| `chunk_id` | content-addressed id: doc + strategy + span (+ parent for children) |",
            "| `doc_id` | document key, joins to `documents.parquet` and `register.parquet` |",
            "| `revid` | pinned Bulbapedia revision of that page |",
            "| `strategy` | `fixed`, `heading`, or `hierarchical` |",
            "| `parent_id` | parent chunk id for `hierarchical` children, else null |",
            "| `role` | `unit` (retrieval unit) or `parent` (context expansion only) |",
            "| `heading_path` | JSON list of section headings above the chunk |",
            "| `char_start`, `char_end` | absolute character offsets into `documents.text` |",
            "| `text` | the chunk text itself |",
            "| `n_chars`, `n_tokens` | length in characters and model tokens |",
            "| `order` | reading order within the document |",
            "",
            "Invariant enforced at build time: `chunk.text == documents.text[char_start:char_end]`.",
            "It is what keeps citations and span-based qrels valid across strategies.",
            "",
            "## Source corpus",
            "",
            f"* Snapshot: `pokerag-corpus-v1` — `{snapshot_id}`",
            f"* {stats.get('n_final', 320)} documents, {stats.get('total_chars', 0):,} characters, "
            f"{audit.get('strategies', {}).get('fixed', {}).get('sections', {}).get('sections', 0):,} sections",
            "* Selection: National Pokedex 1-320 (one page per species), pinned to one revision each",
            "",
            "## Licence and attribution",
            "",
            "* Article text is derived from [Bulbapedia](https://bulbapedia.bulbagarden.net) and is",
            "  licensed **CC BY-NC-SA 2.5** (<https://creativecommons.org/licenses/by-nc-sa/2.5/>).",
            "* Attribution: `Bulbapedia, CC BY-NC-SA 2.5, revision <revid>`; the revision per",
            "  document is in `register.parquet`.",
            "* **Non-commercial use only.** Derivatives must remain under CC BY-NC-SA 2.5.",
            "* Images and fair-use media are not included, only article text.",
            "* Not affiliated with or endorsed by Bulbapedia, Nintendo, Game Freak, or The Pokemon Company.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "pokerag-chunk build       # writes chunks/<strategy>.parquet",
            "pokerag-chunk report      # writes the audit table",
            "```",
            "",
            "## Limitations",
            "",
            "* Chunk boundaries are character-based, so a chunk can start mid-sentence; the",
            "  audit reports how often that happens per strategy instead of hiding it.",
            "* `hierarchical` drops 1,296 link stubs shorter than 80 characters, so its text",
            "  coverage is 99.9% rather than 100%.",
            "* The chunk sets contain no question/answer pairs; retrieval scores are measured",
            "  separately with qrels that are not part of this repository.",
            "",
        ]
    )


def build_staging() -> tuple[list[Path], dict, dict, dict]:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    manifest = read_json(SNAPSHOT / "chunks_manifest.json")
    audit = read_json(SNAPSHOT / "chunks_audit.json")
    snapshot = read_json(SNAPSHOT / "snapshot.json")

    files: list[Path] = []

    def add(source: Path, remote: str) -> None:
        if not source.exists():
            print(f"  lewati (tidak ada): {remote}")
            return
        target = STAGING / remote
        link_or_copy(source, target)
        files.append(target)

    for name in STRATEGIES:
        add(SNAPSHOT / "chunks" / f"{name}.parquet", f"{name}.parquet")
    add(SNAPSHOT / "documents.parquet", "documents.parquet")
    add(SNAPSHOT / "register.parquet", "register.parquet")
    add(SNAPSHOT / "chunks_manifest.json", "chunks_manifest.json")
    add(SNAPSHOT / "chunks_report.md", "chunks_report.md")
    add(SNAPSHOT / "chunks_audit.json", "chunks_audit.json")

    card = build_card(manifest, audit, snapshot)
    (STAGING / "README.md").write_text(card, encoding="utf-8")
    files.append(STAGING / "README.md")

    integrity = [
        f"{sha256_file(path)}  {path.relative_to(STAGING).as_posix()}" for path in sorted(files)
    ]
    (STAGING / "INTEGRITY.sha256").write_text("\n".join(integrity) + "\n", encoding="utf-8")
    files.append(STAGING / "INTEGRITY.sha256")
    return files, manifest, audit, snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--public", action="store_true", help="default: private (CC BY-NC-SA)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    who = api.whoami()
    print(f"login sebagai {who.get('name')}")

    print("menyiapkan staging (hard link, tanpa duplikasi disk)...")
    files, manifest, audit, _snapshot = build_staging()
    total = sum(path.stat().st_size for path in files)
    print(f"staging: {len(files)} file, {total / 1024 / 1024:.1f} MB")
    for path in sorted(files):
        print(f"  {path.stat().st_size / 1024 / 1024:7.2f} MB  {path.relative_to(STAGING).as_posix()}")

    if args.dry_run:
        print("dry-run: tidak ada yang diunggah")
        return 0

    private = not args.public
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=private, exist_ok=True)
    print(f"repo siap: https://huggingface.co/datasets/{args.repo} (private={private})")

    api.upload_folder(
        folder_path=str(STAGING),
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=(
            "PokeRAG chunks v1: fixed, heading, hierarchical (+ documents, audit, manifest)"
        ),
        ignore_patterns=["*.tmp", "*.lock", ".gitkeep"],
    )

    remote = sorted(api.list_repo_files(args.repo, repo_type="dataset"))
    print(f"file di remote ({len(remote)}):")
    for name in remote:
        print(f"  {name}")

    stamp = {
        "repo": args.repo,
        "private": private,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": len(files),
        "bytes": total,
        "remote_files": remote,
    }
    (ARTIFACTS / "upload_manifest_chunks.json").write_text(
        json.dumps(stamp, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"selesai: https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
