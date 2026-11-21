"""CLI for the chunking stage.

    pokerag-chunk build                      # all three strategies
    pokerag-chunk build --strategy heading   # one strategy
    pokerag-chunk report                     # audit A1 + docs/experiments/A1-chunking.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pokerag.chunking.audit import write_report
from pokerag.chunking.build import build
from pokerag.chunking.config import load_chunking_config

DEFAULT_CONFIG = "configs/chunking.v2.yaml"


def cmd_build(cfg) -> int:
    manifest = build(cfg)
    for name, entry in manifest["strategies"].items():
        print(
            f"{name:13} chunks={entry['n_chunks']:6d} p95={entry['p95_chars']:5d} "
            f"over_budget={entry['chunks_over_budget']:4d} coverage={entry['coverage_ratio']:.1%}"
        )
    return 0


def cmd_report(cfg, docs_dir: Path | None) -> int:
    audit = write_report(cfg, docs_dir)
    print(f"snapshot {audit['snapshot']} · tokenizer {audit['tokenizer']} · budget {audit['budget_tokens']} token")
    for name, stats in audit["strategies"].items():
        boundary = stats["boundary"]
        sections = stats["sections"]
        overlap = stats["overlap"]
        print(
            f"{name:13} units={stats['units']:6d} token p50/p95={stats['tokens_p50']:4d}/{stats['tokens_p95']:4d} "
            f"mulai_bersih={boundary['clean_start']:.0%} tengah_kata={boundary['mid_word_start']:.0%} "
            f"potong_h2={boundary['crosses_h2']} section_utuh={sections['intact']:.0%} "
            f"redundansi={overlap['redundancy']:.1%} dibuang={stats['dropped']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerag-chunk", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument(
        "--strategy", default="all", choices=["all", "fixed", "heading", "hierarchical"]
    )
    report_parser = sub.add_parser("report")
    report_parser.add_argument(
        "--docs-dir", default="docs/experiments", help="tempat salinan laporan yang di-commit"
    )
    args = parser.parse_args(argv)

    cfg = load_chunking_config(Path(args.config))
    if args.command == "build":
        return cmd_build(cfg)
    if args.command == "report":
        return cmd_report(cfg, Path(args.docs_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
