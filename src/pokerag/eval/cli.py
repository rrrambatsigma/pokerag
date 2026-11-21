"""CLI for the frozen test benchmark.

    pokerag-eval build --export A1=exports/a1.jsonl --export A2=exports/a2.jsonl \
        --queries queries.csv --policy union --out data/benchmark/qrels/test.jsonl

    pokerag-eval score --qrels data/benchmark/qrels/test.jsonl \
        --strategy heading --modes dense,bm25,hybrid --k 5,10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pokerag.core.jsonl import atomic_write_text, write_json
from pokerag.eval.metrics import evaluate_gold, render_markdown
from pokerag.eval.qrels import (
    POLICIES,
    build_gold,
    load_gold,
    load_queries,
    read_export,
    summary,
    write_gold,
)
from pokerag.retrieval.config import load_retrieval_config
from pokerag.retrieval.pipeline import Retriever
from pokerag.retrieval.types import MODES

DEFAULT_CONFIG = "configs/retrieval.v2.yaml"


def _utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _export(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("format export harus NAME=PATH")
    return name.strip(), Path(path.strip())


def cmd_build(args: argparse.Namespace) -> int:
    exports: dict[str, list] = {}
    for name, path in args.export:
        if name in exports:
            raise SystemExit(f"nama export duplikat: {name}")
        exports[name] = read_export(path, name)
        print(f"{name}: {len(exports[name])} baris dari {path}")

    queries = load_queries(Path(args.queries))
    golds = build_gold(
        exports,
        queries,
        policy=args.policy,
        annotator=args.annotator,
    )
    write_gold(Path(args.out), golds)
    counts = summary(golds)
    print(
        f"queries={counts['queries']} answerable={counts['answerable']} "
        f"unanswerable={counts['unanswerable']} skipped={counts['skipped']} "
        f"gold_docs={counts['gold_docs']} gold_spans={counts['gold_spans']}"
    )
    print(f"tulis {args.out}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    cfg = load_retrieval_config(Path(args.config))
    strategy = args.strategy or cfg.strategy
    modes = _list(args.modes) or list(cfg.modes)
    ks = [int(value) for value in _list(args.k)] or list(cfg.ks)
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise SystemExit(f"mode tidak dikenal: {unknown}; pilihan: {list(MODES)}")

    golds = load_gold(Path(args.qrels))
    slices = _list(args.slice) or None
    with Retriever(cfg, strategy=strategy) as retriever:
        result = evaluate_gold(retriever, golds, modes=modes, ks=ks, slices=slices)
        meta = {
            "qrels": str(args.qrels),
            "strategy": strategy,
            "collection": retriever.collection,
            "index_version": retriever.index_version,
            "embedder": f"{cfg.index_config().embedder.model}",
            "policy": args.policy_label,
            "ks": result["ks"],
        }
        write_json(Path(args.output), {"meta": meta, **result})
        if not args.no_docs:
            atomic_write_text(Path(args.docs), render_markdown(result, meta))

    print(
        f"qrels={result['queries']} usable={result['usable']} "
        f"unanswerable={result['unanswerable']} strategy={strategy} ks={result['ks']}"
    )
    columns = [
        f"{metric}@{k}"
        for metric in ("doc_recall", "doc_mrr", "doc_ndcg", "span_hit", "span_mrr")
        for k in result["ks"]
    ]
    print("mode      " + " ".join(f"{column:>13s}" for column in columns))
    for mode, payload in result["modes"].items():
        cells = " ".join(f"{payload['summary'][column]:13.3f}" for column in columns)
        print(f"{mode:9s} {cells}")
    print(f"json: {args.output}")
    if not args.no_docs:
        print(f"tabel: {args.docs}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerag-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="merge annotator exports into gold qrels")
    build.add_argument(
        "--export",
        action="append",
        type=_export,
        required=True,
        metavar="NAME=PATH",
        help="ulangi untuk setiap anotator, mis. A1=exports/a1.jsonl",
    )
    build.add_argument("--queries", required=True, help="query set CSV atau JSON")
    build.add_argument("--policy", choices=list(POLICIES), default="single")
    build.add_argument("--annotator", default=None, help="wajib untuk policy single")
    build.add_argument("--out", default="data/benchmark/qrels/test.jsonl")

    score = sub.add_parser("score", help="run retrieval against gold qrels")
    score.add_argument("--qrels", default="data/benchmark/qrels/test.jsonl")
    score.add_argument("--config", default=DEFAULT_CONFIG)
    score.add_argument("--strategy", default=None)
    score.add_argument("--modes", default=None, help="mis. dense,bm25,hybrid")
    score.add_argument("--k", default=None, help="mis. 5,10")
    score.add_argument("--slice", default=None, help="filter slice, mis. multi_hop")
    score.add_argument("--policy-label", default="", help="label policy untuk laporan")
    score.add_argument("--output", default="artifacts/reports/test-retrieval.json")
    score.add_argument("--docs", default="docs/experiments/test-retrieval.md")
    score.add_argument("--no-docs", action="store_true", help="jangan tulis tabel markdown")

    args = parser.parse_args(argv)
    _utf8_stdout()
    if args.command == "build":
        return cmd_build(args)
    if args.command == "score":
        return cmd_score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
