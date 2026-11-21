"""CLI for the retrieval stage.

    pokerag-retrieve search "which ability does Charizard have" --mode hybrid
    pokerag-retrieve search "..." --mode bm25 --json
    pokerag-retrieve check                       # manifest, titik, urutan BM25
    pokerag-retrieve eval --qrels data/benchmark/qrels/dev.jsonl --strategy heading
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pokerag.retrieval.config import load_retrieval_config
from pokerag.retrieval.eval import evaluate, load_qrels, write_reports
from pokerag.retrieval.pipeline import Retriever
from pokerag.retrieval.types import MODES

DEFAULT_CONFIG = "configs/retrieval.v2.yaml"


def _utf8_stdout() -> None:
    """Chunk text is arbitrary Unicode; the cp1252 console must not crash on it."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _print_hits(hits, *, as_json: bool, header: str) -> None:
    if as_json:
        print(json.dumps([hit.as_dict() for hit in hits], indent=2, ensure_ascii=False))
        return
    print(header)
    for hit in hits:
        ranks = " ".join(f"{source}#{rank}" for source, rank in sorted(hit.ranks.items()))
        heading = " > ".join(hit.heading_path)[:70]
        print(f"{hit.rank:2d}. [{hit.score:.4f}] {ranks:18s} {hit.doc_id}  {heading}")
        snippet = " ".join(hit.text.split())[:150]
        print(f"    {snippet}")


def cmd_search(args) -> int:
    cfg = load_retrieval_config(Path(args.config))
    strategy = args.strategy or cfg.strategy
    with Retriever(cfg, strategy=strategy) as retriever:
        hits = retriever.search(args.query, mode=args.mode, top_k=args.top)
        header = (
            f"mode={args.mode} strategy={strategy} collection={retriever.collection} "
            f"index={retriever.index_version[:12]} · {len(hits)} hasil"
        )
        _print_hits(hits, as_json=args.json, header=header)
    return 0


def cmd_check(args) -> int:
    cfg = load_retrieval_config(Path(args.config))
    strategy = args.strategy or cfg.strategy
    with Retriever(cfg, strategy=strategy) as retriever:
        facts = retriever.check()
    print(
        f"{facts['collection']} index={facts['index_version'][:12]} "
        f"chunks={facts['chunks']} points={facts['points']} "
        f"bm25={facts['bm25_chunks']} urutan=ok"
    )
    print("manifest, jumlah titik, dan urutan BM25 cocok dengan snapshot")
    return 0


def _list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def cmd_eval(args) -> int:
    cfg = load_retrieval_config(Path(args.config))
    strategy = args.strategy or cfg.strategy
    modes = _list(args.modes) if args.modes else list(cfg.modes)
    ks = [int(value) for value in _list(args.k)] if args.k else list(cfg.ks)
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise SystemExit(f"mode tidak dikenal: {unknown}; pilihan: {list(MODES)}")

    qrels = load_qrels(Path(args.qrels))
    with Retriever(cfg, strategy=strategy) as retriever:
        result = evaluate(retriever, qrels, modes=modes, ks=ks)
        meta = {
            "qrels": str(args.qrels),
            "strategy": strategy,
            "collection": retriever.collection,
            "index_version": retriever.index_version,
            "embedder": f"{cfg.index_config().embedder.model}",
            "rrf_k": cfg.rrf_k,
            "top_dense": cfg.top_dense,
            "top_bm25": cfg.top_bm25,
            "weights": cfg.weights,
            "ks": result["ks"],
        }
        write_reports(
            result,
            meta,
            artifacts=Path(args.output),
            docs=None if args.no_docs else Path(args.docs),
        )

    print(f"qrels={len(qrels)} strategy={strategy} ks={result['ks']}")
    columns = [f"{metric}@{k}" for metric in ("recall", "mrr", "ndcg") for k in result["ks"]]
    print("mode      " + " ".join(f"{column:>10s}" for column in columns))
    for mode, payload in result["modes"].items():
        cells = " ".join(f"{payload['summary'][column]:10.3f}" for column in columns)
        print(f"{mode:9s} {cells}")
    print(f"json: {args.output}")
    if args.docs:
        print(f"tabel: {args.docs}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerag-retrieve", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--mode", default="hybrid", choices=list(MODES))
    search.add_argument("--strategy", default=None)
    search.add_argument("--top", type=int, default=None)
    search.add_argument("--json", action="store_true", help="keluaran JSON lengkap")

    check = sub.add_parser("check")
    check.add_argument("--strategy", default=None)

    evaluate_parser = sub.add_parser("eval")
    evaluate_parser.add_argument("--qrels", default="data/benchmark/qrels/dev.jsonl")
    evaluate_parser.add_argument("--strategy", default=None)
    evaluate_parser.add_argument("--modes", default=None, help="mis. dense,bm25,hybrid")
    evaluate_parser.add_argument("--k", default=None, help="mis. 5,10")
    evaluate_parser.add_argument("--output", default="artifacts/reports/retrieval.json")
    evaluate_parser.add_argument(
        "--docs", default="docs/experiments/B1-retrieval.md", help="tabel markdown yang di-commit"
    )
    evaluate_parser.add_argument(
        "--no-docs", action="store_true", help="jangan tulis tabel markdown (run eksperimen)"
    )

    args = parser.parse_args(argv)
    _utf8_stdout()
    if args.command == "search":
        return cmd_search(args)
    if args.command == "check":
        return cmd_check(args)
    if args.command == "eval":
        return cmd_eval(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
