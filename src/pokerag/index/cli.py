"""CLI for the index stage.

    pokerag-index build                          # semua strategi di config
    pokerag-index build --strategy heading       # satu strategi
    pokerag-index build --limit 20               # smoke test
    pokerag-index search "which ability does Charizard have"
    pokerag-index stats
    pokerag-index identity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pokerag.core.jsonl import read_json
from pokerag.index.build import build_all, truncation_summary
from pokerag.index.config import STRATEGIES, IndexConfig, load_index_config
from pokerag.index.embedders import make_embedder
from pokerag.index.identity import build_identity, load_identity, save_identity
from pokerag.retrieval.config import RetrievalConfig
from pokerag.retrieval.pipeline import Retriever

DEFAULT_CONFIG = "configs/index.gpu.v2.yaml"


def _manifest(cfg: IndexConfig, strategy: str) -> dict:
    path = cfg.manifest_path(strategy)
    if not path.exists():
        raise SystemExit(
            f"koleksi `{cfg.collection_name(strategy)}` belum dibangun; jalankan "
            f"`pokerag-index build --strategy {strategy}` dulu"
        )
    return read_json(path)


def cmd_build(args) -> int:
    cfg = load_index_config(Path(args.config))
    if args.allow_truncation:
        cfg = cfg.model_copy(update={"allow_truncation": True})
    strategies = None if args.strategy == "all" else [args.strategy]
    print(
        f"embedder={cfg.embedder.provider}/{cfg.embedder.model} "
        f"device={cfg.embedder.device} dim={cfg.embedder.dimension} "
        f"dtype={cfg.embedder.dtype} max_seq={cfg.embedder.max_seq_length} "
        f"revision={cfg.embedder.revision}"
    )
    result = build_all(cfg, strategies=strategies, limit=args.limit, force=args.force)
    for strategy, manifest in result["manifests"].items():
        facts = result["run"][strategy]
        print(
            f"{strategy:13} collection={manifest['collection']:18} points={manifest['counts']['points']:6d} "
            f"dim={manifest['embedder']['dimension']} embedded_now={facts['embedded_now']} "
            f"build={facts['build_seconds']}s cached={facts['cached']}"
        )
    print(f"database={cfg.qdrant_dir}")
    return 0


def cmd_search(args) -> int:
    """Debug entry point; the real query path lives in ``pokerag.retrieval``."""
    load_index_config(Path(args.config))  # fail early on a malformed config
    depth = max(args.top, 20)
    retrieval = RetrievalConfig(
        index=args.config,
        strategy=args.strategy,
        top_dense=depth,
        top_bm25=depth,
        top_k=args.top,
    )
    with Retriever(retrieval) as retriever:
        hits = retriever.search(args.query, mode=args.mode, top_k=args.top)
        print(
            f"mode={args.mode} koleksi {retriever.collection} · index "
            f"{retriever.index_version[:12]} · {len(hits)} hasil"
        )
        for hit in hits:
            ranks = " ".join(f"{source}#{rank}" for source, rank in sorted(hit.ranks.items()))
            heading = " > ".join(hit.heading_path)[:60]
            print(f"  {hit.rank:2d}. {hit.score:.4f}  {ranks:12s} {hit.doc_id}  {heading}")
    return 0


def cmd_stats(args) -> int:
    cfg = load_index_config(Path(args.config))
    strategies = [name for name in cfg.strategies if cfg.manifest_path(name).exists()]
    if args.strategy:
        strategies = [args.strategy]
    if not strategies:
        raise SystemExit("belum ada index; jalankan `pokerag-index build` dulu")
    payload = {name: _manifest(cfg, name) for name in strategies}
    provenance = read_json(cfg.provenance_path) if cfg.provenance_path.exists() else {}
    print(json.dumps({"manifests": payload, "provenance": provenance}, indent=2, ensure_ascii=False))
    return 0


def cmd_identity(args) -> int:
    cfg = load_index_config(Path(args.config))
    identity = load_identity(cfg)
    if args.refresh or identity is None:
        embedder = make_embedder(cfg.embedder)
        truncation = truncation_summary(cfg)
        identity = build_identity(cfg, embedder.describe(), truncation)
        save_identity(cfg, identity, cfg.models_docs_dir)
        print(f"identitas diperbarui: {cfg.identity_path}")
    print(
        json.dumps(
            {
                "model": identity["model"],
                "digest": identity["digest"],
                "architecture": identity["architecture"],
                "parameters": identity["parameters"],
                "dimension": identity["dimension"],
                "quantization": identity["quantization"],
                "context": identity["context"],
                "license": identity["license"],
                "truncation": identity["truncation"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerag-index", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--strategy", default="all", choices=["all", *STRATEGIES])
    build.add_argument("--limit", type=int, default=None, help="batasi jumlah chunk (smoke test)")
    build.add_argument("--force", action="store_true", help="bangun ulang walau index sudah ada")
    build.add_argument(
        "--allow-truncation",
        action="store_true",
        help="izinkan chunk melewati max_seq_length (dipotong, harus dilaporkan)",
    )

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--strategy", default="heading", choices=list(STRATEGIES))
    search.add_argument("--mode", default="hybrid", choices=["dense", "bm25", "hybrid"])
    search.add_argument("--top", type=int, default=5)

    stats = sub.add_parser("stats")
    stats.add_argument("--strategy", default=None, choices=list(STRATEGIES))

    identity = sub.add_parser("identity")
    identity.add_argument("--refresh", action="store_true", help="ukur ulang dari model yang dimuat")

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        # Chunk text and headings are arbitrary Unicode; a cp1252 console must
        # not turn a successful query into a UnicodeEncodeError.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.command == "build":
        return cmd_build(args)
    if args.command == "search":
        return cmd_search(args)
    if args.command == "stats":
        return cmd_stats(args)
    if args.command == "identity":
        return cmd_identity(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
