"""Run a ladder variant: ``pokerag-generate --variant B2 --queries queries.csv``."""

from __future__ import annotations

import argparse
import sys

from pokerag.llm.runner import run_variant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerag-generate", description=__doc__)
    parser.add_argument("--ladder", default="configs/ladder.yaml")
    parser.add_argument("--variant", required=True, help="B0 sampai B6")
    parser.add_argument("--queries", required=True, help="CSV atau JSON query set")
    parser.add_argument("--run-index", type=int, default=0, help="indeks seed (0..2)")
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah query")
    parser.add_argument("--device", default=None, help="cuda atau cpu")
    parser.add_argument(
        "--mode",
        choices=("off", "dense", "bm25", "hybrid"),
        default=None,
        help="override mode retrieval, dipakai untuk A2",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override reranker, dipakai untuk A2",
    )
    parser.add_argument("--out-root", default="artifacts/runs")
    parser.add_argument("--retrieval-config", default="configs/retrieval.v2.yaml")
    args = parser.parse_args(argv)

    run_dir = run_variant(
        ladder_path=args.ladder,
        variant_name=args.variant,
        queries_path=args.queries,
        run_index=args.run_index,
        limit=args.limit,
        device=args.device,
        mode=args.mode,
        rerank=args.rerank,
        out_root=args.out_root,
        retrieval_config=args.retrieval_config,
    )
    print(f"run: {run_dir}")
    print(f"predictions: {run_dir / 'predictions.jsonl'}")
    print(f"manifest: {run_dir / 'manifest.json'}")
    print(f"requests: {run_dir / 'requests.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
