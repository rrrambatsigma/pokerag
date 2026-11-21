"""Red-team fixture tools: ``pokerag-redteam check`` and ``run``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pokerag.core.jsonl import atomic_write_text, write_json
from pokerag.redteam.runner import (
    evaluate,
    load_fixtures,
    load_responses,
    render_markdown,
    validate_suite,
)

DEFAULT_FIXTURES = "data/redteam/fixtures/suite.jsonl"


def _utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _response_set(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("format respons harus NAME=PATH")
    return name.strip(), Path(path.strip())


def cmd_check(args: argparse.Namespace) -> int:
    fixtures = load_fixtures(Path(args.fixtures))
    problems = validate_suite(fixtures)
    print(f"fixtures={len(fixtures)}")
    for category in sorted({fixture.category for fixture in fixtures}):
        count = sum(1 for fixture in fixtures if fixture.category == category)
        print(f"  {category:12s} {count}")
    if problems:
        print("masalah:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("suite valid")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    fixtures = load_fixtures(Path(args.fixtures))
    problems = validate_suite(fixtures)
    if problems:
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    responses = {
        name: load_responses(path) for name, path in args.responses
    }
    result = evaluate(fixtures, responses)
    meta = {
        "fixtures": str(args.fixtures),
        "responses": {name: str(path) for name, path in args.responses},
    }
    write_json(Path(args.output), {"meta": meta, **result})
    if not args.no_docs:
        atomic_write_text(Path(args.docs), render_markdown(result, meta))

    labels = list(result["labels"])
    print(f"fixtures={result['fixtures']} labels={','.join(labels)}")
    categories = sorted(
        {
            category
            for payload in result["labels"].values()
            for category in payload["by_category"]
        }
    )
    print("category    " + " ".join(f"{label:>12s}" for label in labels))
    for category in categories:
        cells = []
        for label in labels:
            block = result["labels"][label]["by_category"].get(category)
            cells.append(f"{block['passed']}/{block['total']}" if block else "n/a")
        print(f"{category:12s} " + " ".join(f"{cell:>12s}" for cell in cells))
    print("json: " + args.output)
    if not args.no_docs:
        print("tabel: " + args.docs)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerag-redteam", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="validate the fixture suite")
    check.add_argument("--fixtures", default=DEFAULT_FIXTURES)

    run = sub.add_parser("run", help="score recorded responses")
    run.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    run.add_argument(
        "--responses",
        action="append",
        type=_response_set,
        required=True,
        metavar="NAME=PATH",
        help="ulangi untuk baseline dan mitigation",
    )
    run.add_argument("--output", default="artifacts/reports/redteam.json")
    run.add_argument("--docs", default="docs/experiments/redteam.md")
    run.add_argument("--no-docs", action="store_true")

    args = parser.parse_args(argv)
    _utf8_stdout()
    if args.command == "check":
        return cmd_check(args)
    if args.command == "run":
        return cmd_run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
