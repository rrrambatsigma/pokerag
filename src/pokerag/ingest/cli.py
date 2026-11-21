"""Command line entry point for the ingestion stage.

Examples:
    pokerag-ingest select
    pokerag-ingest fetch --limit 5      # smoke test
    pokerag-ingest fetch                # resumable, 0 requests when nothing changed
    pokerag-ingest clean dedup validate freeze
    pokerag-ingest verify               # recompute hashes, prove the snapshot is intact
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from pokerag.core.hashing import sha256_bytes
from pokerag.core.jsonl import read_json, read_jsonl, write_json
from pokerag.core.logging import get_logger
from pokerag.ingest.clean import run_clean
from pokerag.ingest.config import CorpusConfig, load_config
from pokerag.ingest.dedup import run_dedup
from pokerag.ingest.fetch import fetch_html, fetch_wikitext
from pokerag.ingest.freeze import compute_snapshot_id, run_freeze
from pokerag.ingest.models import SelectedPage
from pokerag.ingest.select import build_selection
from pokerag.ingest.transport import MediaWikiTransport
from pokerag.ingest.validate import run_validation

logger = get_logger("pokerag.ingest.cli")

DEFAULT_CONFIG = "configs/corpus.v2.yaml"


def _transport(cfg: CorpusConfig) -> MediaWikiTransport:
    if "POKERAG_CONTACT_EMAIL" in os.environ:
        cfg.fetch.user_agent = cfg.fetch.user_agent.replace("set-from-env", os.environ["POKERAG_CONTACT_EMAIL"])
    return MediaWikiTransport(cfg.fetch, cfg.snapshot_dir / "fetch_log.jsonl")


def _load_selection(cfg: CorpusConfig) -> list[SelectedPage]:
    path = cfg.snapshot_dir / "selection.json"
    if not path.exists():
        raise SystemExit("selection.json belum ada; jalankan `select` dulu")
    return [SelectedPage.model_validate(row) for row in read_json(path)["pages"]]


def cmd_select(cfg: CorpusConfig, limit: int | None = None) -> None:
    transport = _transport(cfg)
    pages = build_selection(cfg, transport)
    write_json(
        cfg.snapshot_dir / "selection.json",
        {
            "snapshot": cfg.name,
            "method": cfg.selection.model_dump(),
            "count": len(pages),
            "pages": [page.model_dump() for page in pages],
        },
    )
    print(f"selected {len(pages)} pages")


def cmd_fetch(cfg: CorpusConfig, limit: int | None = None, resume: bool = True) -> None:
    selection = _load_selection(cfg)
    transport = _transport(cfg)
    records = fetch_wikitext(cfg, transport, selection, limit=limit, resume=resume)
    html_paths = fetch_html(cfg, transport, records, limit=limit, resume=resume)
    print(
        f"fetched wikitext={len(records)} html={len(html_paths)} "
        f"requests={transport.stats.requests} retries={transport.stats.retries} "
        f"bytes={transport.stats.bytes_received}"
    )


def cmd_clean(cfg: CorpusConfig, limit: int | None = None, rebuild: bool = False) -> None:
    written = run_clean(cfg, limit=limit, rebuild=rebuild)
    total = sum(1 for _ in read_jsonl(cfg.snapshot_dir / "cleaned.jsonl"))
    print(f"cleaned {written} documents (total {total})")


def cmd_dedup(cfg: CorpusConfig) -> None:
    groups = run_dedup(cfg)
    print(f"dedup groups={len(groups)}")


def cmd_validate(cfg: CorpusConfig) -> None:
    payload = run_validation(cfg)
    for gate in payload["gates"]:
        mark = "PASS" if gate["passed"] else "FAIL"
        print(f"[{mark}] {gate['gate']}: {gate['observed']} (ambang {gate['threshold']})")
    print(f"validation passed={payload['passed']}")


def cmd_freeze(cfg: CorpusConfig, force: bool = False) -> None:
    manifest = run_freeze(cfg, force=force)
    print(f"snapshot_id={manifest['snapshot_id']}")
    print(f"documents={manifest['counts']['documents']} characters={manifest['counts']['characters']}")


def cmd_verify(cfg: CorpusConfig) -> int:
    """Recompute every content hash and the snapshot id."""
    snap = cfg.snapshot_dir
    manifest = read_json(snap / "manifest.json")
    problems: list[str] = []
    for name, expected in manifest["content_files"].items():
        path = snap / name
        if not path.exists():
            problems.append(f"{name}: hilang")
        elif sha256_bytes(path.read_bytes()) != expected:
            problems.append(f"{name}: hash berubah")
    documents = pd.read_parquet(snap / "documents.parquet")
    identities = [
        (row.doc_id, int(row.revid), row.content_sha256) for row in documents.itertuples()
    ]
    recomputed = compute_snapshot_id(read_json(snap / "config_snapshot.json"), identities)
    if recomputed != manifest["snapshot_id"]:
        problems.append("snapshot_id tidak cocok dengan isi documents.parquet")
    if len(documents) != manifest["counts"]["documents"]:
        problems.append("jumlah dokumen tidak cocok dengan manifest")

    if problems:
        print("VERIFY FAILED")
        for problem in problems:
            print(f" - {problem}")
        return 1
    print(f"VERIFY OK snapshot_id={manifest['snapshot_id']} documents={len(documents)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pokerag-ingest", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path ke YAML konfigurasi korpus")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "fetch", "clean", "dedup", "validate", "freeze", "verify"):
        sp = sub.add_parser(name)
        sp.add_argument("--limit", type=int, default=None, dest="limit", help="batasi jumlah halaman")
        sp.add_argument("--no-resume", action="store_true", dest="no_resume", help="abaikan cache")
        sp.add_argument("--force", action="store_true", dest="force", help="freeze walau gate gagal")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(Path(args.config))
    cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    if args.command == "select":
        cmd_select(cfg, limit=args.limit)
    elif args.command == "fetch":
        cmd_fetch(cfg, limit=args.limit, resume=resume)
    elif args.command == "clean":
        cmd_clean(cfg, limit=args.limit, rebuild=not resume)
    elif args.command == "dedup":
        cmd_dedup(cfg)
    elif args.command == "validate":
        cmd_validate(cfg)
    elif args.command == "freeze":
        cmd_freeze(cfg, force=args.force)
    elif args.command == "verify":
        return cmd_verify(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
