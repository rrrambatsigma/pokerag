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

import pyarrow.parquet as pq

from pokerag.core.hashing import sha256_file
from pokerag.core.jsonl import atomic_write_text, read_jsonl, write_json
from pokerag.eval.generation import (
    Judgment,
    Prediction,
    PredictionCitation,
    evaluate_generation,
    render_markdown as render_generation_markdown,
)
from pokerag.eval.metrics import evaluate_gold, render_markdown
from pokerag.eval.pool import METHODS, build_pool
from pokerag.eval.pool import write as write_pool
from pokerag.eval.qrels import (
    POLICIES,
    build_gold,
    load_gold,
    load_queries,
    read_export,
    summary,
    write_gold,
)
from pokerag.llm.config import load_ladder
from pokerag.retrieval.config import load_retrieval_config
from pokerag.retrieval.pipeline import Retriever
from pokerag.retrieval.rerank import CrossEncoderReranker
from pokerag.retrieval.types import MODES

DEFAULT_CONFIG = "configs/retrieval.v2.yaml"
DEFAULT_LADDER = "configs/ladder.yaml"
DEFAULT_QUERIES = "../pokerag-annotator/data/queries.csv"
DEFAULT_POOL = "artifacts/pool"


def _utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_predictions(path: Path) -> list[Prediction]:
    """Read model predictions from JSONL (camelCase or snake_case)."""
    predictions: list[Prediction] = []
    for raw in read_jsonl(path):
        query_id = str(raw.get("query_id") or "")
        if not query_id:
            raise ValueError(f"{path}: baris tanpa query_id")
        citations = tuple(
            PredictionCitation(
                chunk_id=str(item.get("chunkId") or item.get("chunk_id") or ""),
                doc_id=str(item.get("docId") or item.get("doc_id") or ""),
                char_start=int(item.get("charStart") or item.get("char_start") or 0),
                char_end=int(item.get("charEnd") or item.get("char_end") or 0),
            )
            for item in raw.get("citations", [])
        )
        predictions.append(
            Prediction(
                query_id=query_id,
                text=str(raw.get("text") or ""),
                abstained=bool(raw.get("abstained", False)),
                citations=citations,
            )
        )
    if not predictions:
        raise ValueError(f"{path}: predictions kosong")
    return predictions


def load_judgments(path: Path) -> list[Judgment]:
    """Read optional correctness labels: {query_id, correct, supported}."""
    judgments: list[Judgment] = []
    for raw in read_jsonl(path):
        query_id = str(raw.get("query_id") or "")
        if not query_id:
            raise ValueError(f"{path}: baris tanpa query_id")
        judgments.append(
            Judgment(
                query_id=query_id,
                correct=_optional_bool(raw.get("correct")),
                supported=_optional_bool(raw.get("supported")),
            )
        )
    return judgments


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


def cmd_generation(args: argparse.Namespace) -> int:
    golds = load_gold(Path(args.qrels))
    predictions = load_predictions(Path(args.predictions))
    judgments = (
        load_judgments(Path(args.judgments)) if args.judgments else []
    )
    result = evaluate_generation(golds, predictions, judgments)

    meta = {
        "qrels": str(args.qrels),
        "predictions": str(args.predictions),
        "judgments": str(args.judgments or ""),
    }
    write_json(Path(args.output), {"meta": meta, **result})
    if not args.no_docs:
        atomic_write_text(Path(args.docs), render_generation_markdown(result, meta))

    print(
        f"queries={result['queries']} predictions={result['predictions']} "
        f"missing={result['missing']} judged={result['judged']}"
    )
    for metric, value in result["metrics"].items():
        rendered = "n/a" if value is None else f"{value:.3f}"
        print(f"  {metric:22s} {rendered}")
    print(f"json: {args.output}")
    if not args.no_docs:
        print(f"tabel: {args.docs}")
    return 0




def cmd_pool(args: argparse.Namespace) -> int:
    cfg = load_retrieval_config(Path(args.config))
    ladder = load_ladder(Path(args.ladder))
    strategy = args.strategy or cfg.strategy
    queries = load_queries(Path(args.queries))
    pairs = [(query_id, meta.text) for query_id, meta in queries.items()]
    if args.limit:
        pairs = pairs[: args.limit]

    documents = pq.read_table(
        cfg.index_config().snapshot_dir / "documents.parquet",
        columns=["doc_id", "revid"],
    ).to_pylist()
    revid_by_doc = {str(row["doc_id"]): int(row["revid"] or 0) for row in documents}

    candidates = args.candidates or (
        ladder.reranker.candidates if ladder.reranker else 20
    )
    with Retriever(cfg, strategy=strategy) as retriever:
        reranker = CrossEncoderReranker(ladder.reranker, device=args.device)
        records = build_pool(
            pairs,
            lambda query, mode, top_k: retriever.search(query, mode=mode, top_k=top_k),
            lambda query, hits, top_k: reranker.rerank(query, hits, top_k),
            depth=args.depth,
            candidates=candidates,
            seed=args.seed,
            revid_by_doc=revid_by_doc,
        )
        payload = write_pool(
            Path(args.out_dir),
            records,
            depth=args.depth,
            seed=args.seed,
            meta={
                "strategy": strategy,
                "collection": retriever.collection,
                "index_version": retriever.index_version,
                "annotation_rule": args.annotation_rule,
                "candidates": candidates,
                "queries": str(args.queries),
                "queries_sha256": sha256_file(Path(args.queries)),
                "config": str(args.config),
                "config_sha256": sha256_file(Path(args.config)),
                "ladder": str(args.ladder),
                "ladder_sha256": sha256_file(Path(args.ladder)),
            },
        )

    print(
        f"queries={payload['n_queries']} passages={payload['n_passages']} "
        f"per_query={payload['passages_per_query']} depth={payload['depth']}"
    )
    print("metode      " + " ".join(f"{method:>10s}" for method in METHODS))
    print(
        "ditemukan   "
        + " ".join(f"{payload['retrieved_by_method'][m]:10d}" for m in METHODS)
    )
    print(
        "eksklusif   "
        + " ".join(f"{payload['exclusive_by_method'][m]:10d}" for m in METHODS)
    )
    print(f"pool: {Path(args.out_dir) / 'pool.jsonl'}")
    print(f"manifest: {Path(args.out_dir) / 'manifest.json'}")
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

    generation = sub.add_parser(
        "generation", help="abstention, citation and judged-answer metrics"
    )
    generation.add_argument("--qrels", default="data/benchmark/qrels/test.jsonl")
    generation.add_argument("--predictions", required=True, help="JSONL prediksi model")
    generation.add_argument("--judgments", default=None, help="JSONL {query_id, correct, supported}")
    generation.add_argument("--output", default="artifacts/reports/generation.json")
    generation.add_argument("--docs", default="docs/experiments/generation.md")
    generation.add_argument("--no-docs", action="store_true")

    pool = sub.add_parser("pool", help="bangun pool anotasi 4 metode A2")
    pool.add_argument("--queries", default=DEFAULT_QUERIES)
    pool.add_argument("--config", default=DEFAULT_CONFIG)
    pool.add_argument("--ladder", default=DEFAULT_LADDER)
    pool.add_argument("--strategy", default=None)
    pool.add_argument("--depth", type=int, default=10, help="top-k per metode")
    pool.add_argument("--candidates", type=int, default=None, help="kandidat hybrid untuk rerank")
    pool.add_argument("--seed", type=int, default=7)
    pool.add_argument("--out-dir", default=DEFAULT_POOL)
    pool.add_argument("--device", default="cuda")
    pool.add_argument("--annotation-rule", default="v3")
    pool.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    _utf8_stdout()
    if args.command == "pool":
        return cmd_pool(args)
    if args.command == "build":
        return cmd_build(args)
    if args.command == "score":
        return cmd_score(args)
    if args.command == "generation":
        return cmd_generation(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
