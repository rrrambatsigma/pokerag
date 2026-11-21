"""Command line entry points for PokeRAG."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from pokerag.stages.retrieve import MODES, STRATEGIES, RetrievalConfig, Retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pokerag", description="PokeRAG tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    retrieve = subparsers.add_parser("retrieve", help="retrieve chunks for a query")
    retrieve.add_argument("query", help="natural language question")
    retrieve.add_argument("--strategy", choices=STRATEGIES, default="fixed")
    retrieve.add_argument("--mode", choices=MODES, default="hybrid")
    retrieve.add_argument("-k", "--top-k", type=int, default=10)
    retrieve.add_argument("--candidates", type=int, default=50)
    retrieve.add_argument("--rrf-k", type=int, default=60)
    retrieve.add_argument("--device", default=None, help="cpu, cuda, mps")
    retrieve.add_argument("--json", action="store_true", help="emit JSON")

    return parser


def snippet(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def run_retrieve(args: argparse.Namespace, console: Console) -> int:
    config = RetrievalConfig(
        strategy=args.strategy,
        mode=args.mode,
        top_k=args.top_k,
        candidates=args.candidates,
        rrf_k=args.rrf_k,
    )

    try:
        retriever = Retriever.build(config, device=args.device)
    except (FileNotFoundError, RuntimeError) as error:
        console.print(f"[red]{error}[/red]")
        return 1

    try:
        results = retriever.search(args.query)
        titles = {item.chunk.doc_id: retriever.title(item.chunk.doc_id) for item in results}
    finally:
        retriever.close()

    if not results:
        console.print("[yellow]tidak ada hasil[/yellow]")
        return 1

    if args.json:
        payload = [
            {
                "rank": rank,
                "chunk_id": item.chunk.chunk_id,
                "doc_id": item.chunk.doc_id,
                "title": titles[item.chunk.doc_id],
                "heading_path": list(item.chunk.heading_path),
                "char_start": item.chunk.char_start,
                "char_end": item.chunk.char_end,
                "score": item.score,
                "ranks": item.ranks,
                "text": item.chunk.text,
            }
            for rank, item in enumerate(results, start=1)
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    table = Table(title=f"{args.mode} / {args.strategy} / k={args.top_k}")
    table.add_column("#", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Dokumen")
    table.add_column("Heading")
    table.add_column("Cuplikan")
    for rank, item in enumerate(results, start=1):
        table.add_row(
            str(rank),
            f"{item.score:.4f}",
            titles[item.chunk.doc_id],
            item.chunk.label,
            snippet(item.chunk.text),
        )
    console.print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if args.command == "retrieve":
        return run_retrieve(args, console)

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
