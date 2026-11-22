"""CLI for the B5 supervised dataset.

    pokerag-sft build --corpus artifacts/corpus/pokerag-corpus-v2
    pokerag-sft check --items data/sft/items.jsonl

``build`` extracts facts from the frozen snapshot, writes ``items.jsonl`` and
the chat-format ``train.jsonl``/``val.jsonl``, then runs the leak gates before
reporting success. ``check`` re-runs the gates against an existing dataset and
verifies every item against the snapshot it came from.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow.parquet as pq

from pokerag.core.hashing import sha256_file, sha256_json, sha256_text
from pokerag.core.jsonl import read_jsonl, write_json, write_jsonl
from pokerag.llm.prompts import PromptSet, load_prompt_set, render_system, render_user
from pokerag.retrieval.types import Hit
from pokerag.sft import facts as F
from pokerag.sft.dataset import (
    Example,
    Item,
    build_examples,
    build_items,
    counts,
    split_items,
)
from pokerag.sft.leak import (
    MAX_COMMON_CHARS,
    NGRAM,
    Overlap,
    check_text,
    entity_collisions,
    normalise,
    question_collisions,
)

DEFAULT_CORPUS = "artifacts/corpus/pokerag-corpus-v2"
DEFAULT_QUERIES = "../pokerag-annotator/data/queries.csv"
DEFAULT_DEV_QRELS = "data/benchmark/qrels/dev.jsonl"
DEFAULT_OUT = "data/sft"
DEFAULT_EXCLUDED = "artifacts/sft/excluded_entities.txt"
DEFAULT_SYSTEM = "configs/prompts/grounded_system.txt"
DEFAULT_USER = "configs/prompts/grounded_user.txt"

QUOTA = {F.TYPE: 40, F.CATCH_RATE: 25, F.EGG_GROUP: 20, F.EVOLUTION: 15}


# --------------------------------------------------------------------------- io


def _heading(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return [str(part) for part in parsed]
    return [str(part) for part in value]  # type: ignore[union-attr]


def load_corpus(root: Path) -> tuple[list[dict], dict[str, list[tuple[list[str], str]]]]:
    documents = pq.read_table(root / "documents.parquet").to_pylist()
    chunks: dict[str, list[tuple[list[str], str]]] = defaultdict(list)
    for row in pq.read_table(root / "chunks" / "heading.parquet").to_pylist():
        chunks[row["doc_id"]].append((_heading(row["heading_path"]), row["text"] or ""))
    return documents, chunks


def load_queries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_dev(path: Path) -> list[dict]:
    return list(read_jsonl(path)) if path.exists() else []


def exclusions(queries: Sequence[dict], dev: Sequence[dict], names: Iterable[str]) -> dict:
    """Entities and documents that training must not touch."""
    entities: set[str] = set()
    docs: set[str] = set()
    for row in queries:
        for name in (row.get("entity") or "").split(";"):
            if name.strip():
                entities.add(name.strip())
        if row.get("expected_doc"):
            docs.add(row["expected_doc"].strip())
    dev_queries: list[str] = []
    for row in dev:
        if row.get("doc_id"):
            docs.add(row["doc_id"])
        if row.get("query"):
            dev_queries.append(row["query"])
    known = sorted(names)
    for query in dev_queries:
        folded = normalise(query)
        for name in known:
            if re.search(rf"\b{re.escape(normalise(name))}\b", folded):
                entities.add(name)
    return {"entities": entities, "docs": docs, "dev_queries": dev_queries}


def collect_facts(
    documents: Sequence[dict],
    chunks: dict[str, list[tuple[list[str], str]]],
    banned_docs: set[str],
    banned_entities: set[str],
) -> dict[str, dict[str, F.Fact]]:
    banned = {normalise(name) for name in banned_entities}
    by_doc: dict[str, dict[str, F.Fact]] = {}
    for document in documents:
        doc_id = document["doc_id"]
        if doc_id in banned_docs:
            continue
        entity = F.entity_from_title(document["title"])
        if normalise(entity) in banned:
            continue
        found = F.extract(entity, doc_id, int(document.get("revid") or 0), chunks.get(doc_id, []))
        if found:
            by_doc[doc_id] = {fact.kind: fact for fact in found}
    return by_doc


# ------------------------------------------------------------------- messages


def to_messages(prompt_set: PromptSet, example: Example) -> list[dict]:
    hits = [
        Hit(
            chunk_id=f"sft-{index}",
            score=0.0,
            rank=index,
            source="sft",
            doc_id=doc_id,
            text=text,
            heading_path=[heading] if heading else [],
            char_start=0,
            char_end=len(text),
            url="",
        )
        for index, (doc_id, heading, text) in enumerate(example.evidence, start=1)
    ]
    user = render_user(prompt_set, example.question, hits)
    target = json.dumps(example.target(), ensure_ascii=False, sort_keys=True)
    return [
        {"role": "system", "content": render_system(prompt_set)},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]


def example_rows(prompt_set: PromptSet, examples: Sequence[Example]) -> list[dict]:
    return [
        {
            "item_id": example.item_id,
            "variant": example.variant,
            "messages": to_messages(prompt_set, example),
        }
        for example in examples
    ]


# ----------------------------------------------------------------------- gates


def run_gates(
    items: Sequence[Item],
    chunks: dict[str, list[tuple[list[str], str]]],
    banned_entities: set[str],
    test_questions: Sequence[str],
) -> dict:
    collisions = entity_collisions(items, banned_entities)
    questions = question_collisions(items, test_questions)
    worst = Overlap((), 0, None)
    per_item: dict[str, dict] = {}
    for item in items:
        hits: list[str] = []
        longest = 0
        where: str | None = None
        for text in (item.question, item.evidence, item.answer):
            probe = check_text(text, chunks.get(item.doc_id, []))
            hits.extend(probe.ngram_hits)
            if probe.longest > longest:
                longest, where = probe.longest, probe.where
        per_item[item.item_id] = Overlap(tuple(hits), longest, where).as_dict()
        if hits or longest > worst.longest:
            worst = Overlap(tuple(hits), longest, where)
    return {
        "entity_collisions": collisions,
        "question_collisions": questions,
        "max_common_chars": worst.longest,
        "max_common_where": worst.where,
        "ngram": NGRAM,
        "max_common_limit": MAX_COMMON_CHARS,
        "copied_items": sorted(
            item_id for item_id, row in per_item.items() if row["ngram_hits"] or row["longest"] >= MAX_COMMON_CHARS
        ),
        "per_item": per_item,
    }


def gates_ok(report: dict) -> bool:
    return (
        not report["entity_collisions"]
        and not report["question_collisions"]
        and not report["copied_items"]
    )


# ---------------------------------------------------------------------- build


def cmd_build(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    documents, chunks = load_corpus(corpus)
    queries = load_queries(Path(args.queries))
    dev = load_dev(Path(args.dev_qrels))
    names = [F.entity_from_title(document["title"]) for document in documents]
    banned = exclusions(queries, dev, names)

    facts_by_doc = collect_facts(documents, chunks, banned["docs"], banned["entities"])
    items = build_items(facts_by_doc, QUOTA, seed=args.seed)
    if len(items) < sum(QUOTA.values()):
        print(
            f"warning: only {len(items)} items available, quota asks {sum(QUOTA.values())}",
            file=sys.stderr,
        )

    prompt_set = load_prompt_set(Path(args.system), Path(args.user))
    examples = build_examples(items, seed=args.seed)
    train_items, val_items = split_items(items)
    train_ids = {item.item_id for item in train_items}
    train = [example for example in examples if example.item_id in train_ids]
    val = [example for example in examples if example.item_id not in train_ids]

    gates = run_gates(items, chunks, banned["entities"], [row["text"] for row in queries if row.get("text")])

    out = Path(args.out)
    write_jsonl(out / "items.jsonl", [item.as_dict() for item in items])
    write_jsonl(out / "train.jsonl", example_rows(prompt_set, train))
    write_jsonl(out / "val.jsonl", example_rows(prompt_set, val))

    excluded_path = Path(args.excluded)
    excluded_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_path.write_text("\n".join(sorted(banned["entities"])) + "\n", encoding="utf-8")

    manifest = {
        "format": "sft-v1",
        "seed": args.seed,
        "quota": QUOTA,
        "n_items": len(items),
        "n_examples": {"train": len(train), "val": len(val)},
        "counts_by_kind": counts(items),
        "counts_by_variant": {
            name: sum(1 for example in examples if example.variant == name) for name in ("clean", "hard", "abstain")
        },
        "prompt_version": prompt_set.version,
        "prompts": {"system": args.system, "user": args.user},
        "corpus": {
            "root": str(corpus),
            "documents_sha256": sha256_file(corpus / "documents.parquet"),
            "chunks_sha256": sha256_file(corpus / "chunks" / "heading.parquet"),
        },
        "benchmark": {
            "queries": str(args.queries),
            "queries_sha256": sha256_file(Path(args.queries)) if Path(args.queries).exists() else None,
            "n_test_entities": len(banned["entities"]),
            "n_test_docs": len(banned["docs"]),
        },
        "exclusions_sha256": sha256_text("\n".join(sorted(banned["entities"]))),
        "items_sha256": sha256_json([item.as_dict() for item in items]),
        "gates": {
            "ok": gates_ok(gates),
            "entity_collisions": len(gates["entity_collisions"]),
            "question_collisions": len(gates["question_collisions"]),
            "copied_items": len(gates["copied_items"]),
            "max_common_chars": gates["max_common_chars"],
            "ngram": gates["ngram"],
            "max_common_limit": gates["max_common_limit"],
        },
    }
    write_json(out / "manifest.json", manifest)

    print(
        json.dumps(
            {
                "items": manifest["n_items"],
                "kinds": manifest["counts_by_kind"],
                "examples": manifest["n_examples"],
                "gates": manifest["gates"],
            },
            indent=2,
        )
    )
    if not gates_ok(gates):
        print("gate failure: dataset must not be trained on", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------- check


def cmd_check(args: argparse.Namespace) -> int:
    items_path = Path(args.items)
    rows = list(read_jsonl(items_path))
    if not rows:
        print(f"no items in {items_path}", file=sys.stderr)
        return 1
    items = [
        Item(
            item_id=row["item_id"],
            entity=row["entity"],
            kind=row["kind"],
            question=row["question"],
            evidence=row["evidence"],
            answer=row["answer"],
            doc_id=row["provenance"]["doc_id"],
            revid=row["provenance"]["revid"],
            locator=row["provenance"]["locator"],
            values=tuple(row["provenance"]["values"]),
        )
        for row in rows
    ]

    documents, chunks = load_corpus(Path(args.corpus))
    queries = load_queries(Path(args.queries))
    dev = load_dev(Path(args.dev_qrels))
    names = [F.entity_from_title(document["title"]) for document in documents]
    banned = exclusions(queries, dev, names)

    mismatched: list[str] = []
    missing: list[str] = []
    for item in items:
        doc_chunks = chunks.get(item.doc_id)
        if doc_chunks is None:
            missing.append(item.item_id)
            continue
        fresh = F.verify(
            F.Fact(item.kind, item.entity, item.doc_id, item.revid, item.locator, item.values),
            doc_chunks,
        )
        if fresh is None:
            mismatched.append(item.item_id)

    report = run_gates(items, chunks, banned["entities"], [row["text"] for row in queries if row.get("text")])
    summary = {
        "items": len(items),
        "missing_docs": missing,
        "mismatched_facts": mismatched,
        "gates": {
            "ok": gates_ok(report),
            "entity_collisions": report["entity_collisions"],
            "question_collisions": report["question_collisions"],
            "copied_items": report["copied_items"],
            "max_common_chars": report["max_common_chars"],
            "max_common_where": report["max_common_where"],
        },
        "worst_items": sorted(
            report["per_item"].items(), key=lambda pair: pair[1]["longest"], reverse=True
        )[:5],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    ok = gates_ok(report) and not missing and not mismatched
    return 0 if ok else 1


# ------------------------------------------------------------------------ cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerag-sft", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="extract facts and write the dataset")
    build.add_argument("--corpus", default=DEFAULT_CORPUS)
    build.add_argument("--queries", default=DEFAULT_QUERIES)
    build.add_argument("--dev-qrels", default=DEFAULT_DEV_QRELS)
    build.add_argument("--out", default=DEFAULT_OUT)
    build.add_argument("--excluded", default=DEFAULT_EXCLUDED)
    build.add_argument("--system", default=DEFAULT_SYSTEM)
    build.add_argument("--user", default=DEFAULT_USER)
    build.add_argument("--seed", type=int, default=7)

    check = sub.add_parser("check", help="re-run provenance and leak gates")
    check.add_argument("--items", default=f"{DEFAULT_OUT}/items.jsonl")
    check.add_argument("--corpus", default=DEFAULT_CORPUS)
    check.add_argument("--queries", default=DEFAULT_QUERIES)
    check.add_argument("--dev-qrels", default=DEFAULT_DEV_QRELS)

    args = parser.parse_args(argv)
    if args.command == "build":
        return cmd_build(args)
    if args.command == "check":
        return cmd_check(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
