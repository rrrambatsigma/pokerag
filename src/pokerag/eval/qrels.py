"""Build gold qrels from blind annotator exports.

The annotation tool stores one row per quoted span. A multi-hop query therefore
produces several rows, and an unanswerable query produces one row with no
document. This module groups those rows into one gold record per query, so the
evaluator can score multi-document evidence and keep the unanswerable slice.

Three policies are available because there is no adjudication yet:

* ``single`` uses one annotator, the honest default while people disagree.
* ``union`` keeps every span any annotator marked.
* ``intersection`` keeps only spans every annotator marked, which is the
  conservative lower bound. A query whose intersection is empty is reported as
  skipped, not silently turned into an unanswerable one.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from pokerag.core.jsonl import read_jsonl, write_jsonl

POLICIES = ("single", "union", "intersection")


@dataclass(frozen=True)
class QueryMeta:
    query_id: str
    text: str
    slice: str = ""


@dataclass(frozen=True)
class ExportRow:
    query_id: str
    annotator: str
    doc_id: str | None
    start: int | None
    end: int | None
    answerable: bool
    quote: str = ""
    slice: str = ""


@dataclass(frozen=True)
class GoldSpan:
    start: int
    end: int
    quote: str = ""


@dataclass(frozen=True)
class GoldDoc:
    doc_id: str
    spans: tuple[GoldSpan, ...]


@dataclass(frozen=True)
class GoldQuery:
    query_id: str
    query: str
    slice: str
    answerable: bool
    gold: tuple[GoldDoc, ...]
    annotators: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "slice": self.slice,
            "answerable": self.answerable,
            "annotators": list(self.annotators),
            "gold": [
                {
                    "doc_id": doc.doc_id,
                    "spans": [[span.start, span.end] for span in doc.spans],
                    "quotes": [span.quote for span in doc.spans],
                }
                for doc in self.gold
            ],
        }


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _rows_from_path(path: Path) -> Iterable[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing export: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    yield from read_jsonl(path)


def read_export(path: Path, annotator: str | None = None) -> list[ExportRow]:
    """Read one annotator export (JSONL or CSV) into typed rows."""
    rows: list[ExportRow] = []
    for raw in _rows_from_path(path):
        status = str(raw.get("status") or "done").strip().lower()
        if status and status != "done":
            continue
        query_id = str(raw.get("query_id") or "").strip()
        if not query_id:
            continue
        name = str(raw.get("annotator") or annotator or "").strip()
        if not name:
            raise ValueError(f"{path}: baris tanpa annotator dan --annotator tidak diisi")

        doc_id = raw.get("doc_id")
        doc_id = str(doc_id).strip() if doc_id not in (None, "") else None
        start = raw.get("char_start")
        end = raw.get("char_end")
        if doc_id is not None:
            if start in (None, "") or end in (None, ""):
                raise ValueError(f"{path}: {query_id} punya doc_id tanpa offset")
            start = int(start)
            end = int(end)
            if end <= start:
                raise ValueError(f"{path}: {query_id} punya span tidak valid {start}-{end}")
        else:
            start = end = None

        relevance = int(raw.get("relevance") or (1 if doc_id else 0))
        if "answerable" in raw:
            answerable = _truthy(raw["answerable"])
        else:
            answerable = relevance == 1 and doc_id is not None
        if not answerable:
            doc_id, start, end = None, None, None

        rows.append(
            ExportRow(
                query_id=query_id,
                annotator=name,
                doc_id=doc_id,
                start=start,
                end=end,
                answerable=answerable and doc_id is not None,
                quote=str(raw.get("quote") or ""),
                slice=str(raw.get("slice") or ""),
            )
        )
    return rows


def load_queries(path: Path) -> dict[str, QueryMeta]:
    """Read query text and slice from the annotator query set (CSV or JSON)."""
    if not path.is_file():
        raise FileNotFoundError(f"missing query set: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".jsonl":
        raw_rows = list(read_jsonl(path))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "queries" in payload:
            payload = payload["queries"]
        raw_rows = list(payload)

    queries: dict[str, QueryMeta] = {}
    for raw in raw_rows:
        query_id = str(raw.get("query_id") or raw.get("id") or "").strip()
        text = str(raw.get("text") or "").strip()
        if not query_id or not text:
            raise ValueError(f"{path}: baris tanpa query_id atau text")
        if query_id in queries:
            raise ValueError(f"{path}: query_id duplikat {query_id}")
        queries[query_id] = QueryMeta(
            query_id=query_id,
            text=text,
            slice=str(raw.get("slice") or ""),
        )
    if not queries:
        raise ValueError(f"{path}: query set kosong")
    return queries


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def intersect_groups(groups: Sequence[Sequence[tuple[int, int]]]) -> list[tuple[int, int]]:
    """Ranges covered by at least one interval from every group."""
    if not groups:
        return []
    current = merge_intervals(groups[0])
    for group in groups[1:]:
        candidates: list[tuple[int, int]] = []
        for left in current:
            for right in merge_intervals(group):
                start = max(left[0], right[0])
                end = min(left[1], right[1])
                if start < end:
                    candidates.append((start, end))
        current = merge_intervals(candidates)
        if not current:
            return []
    return current


def _query_slice(per_annotator: dict[str, list[ExportRow]], names: Sequence[str]) -> str:
    for name in names:
        for row in per_annotator[name]:
            if row.slice:
                return row.slice
    return ""


def _spans_for(rows: Sequence[ExportRow]) -> list[GoldSpan]:
    seen: set[tuple[int, int]] = set()
    spans: list[GoldSpan] = []
    for row in rows:
        if row.doc_id is None or row.start is None or row.end is None:
            continue
        key = (row.start, row.end)
        if key in seen:
            continue
        seen.add(key)
        spans.append(GoldSpan(row.start, row.end, row.quote))
    return sorted(spans, key=lambda span: (span.start, span.end))


def _docs_by_annotator(rows: Sequence[ExportRow]) -> dict[str, dict[str, list[ExportRow]]]:
    grouped: dict[str, dict[str, list[ExportRow]]] = {}
    for row in rows:
        if row.doc_id is None:
            continue
        grouped.setdefault(row.annotator, {}).setdefault(row.doc_id, []).append(row)
    return grouped


def build_gold(
    exports: dict[str, Sequence[ExportRow]],
    queries: dict[str, QueryMeta],
    *,
    policy: str = "single",
    annotator: str | None = None,
) -> list[GoldQuery]:
    """Merge annotator rows into one gold record per query."""
    if policy not in POLICIES:
        raise ValueError(f"policy tidak dikenal: {policy}")
    if not exports:
        raise ValueError("tidak ada export")

    names = sorted(exports)
    if policy == "single":
        chosen = annotator or (names[0] if len(names) == 1 else None)
        if chosen is None or chosen not in exports:
            raise ValueError(f"policy single butuh --annotator di antara {names}")
        names = [chosen]

    by_query: dict[str, list[ExportRow]] = {}
    for name in names:
        for row in exports[name]:
            by_query.setdefault(row.query_id, []).append(row)

    golds: list[GoldQuery] = []
    for query_id in sorted(by_query):
        if query_id not in queries:
            raise ValueError(f"query {query_id} tidak ada di query set")
        meta = queries[query_id]
        rows = by_query[query_id]
        present = {row.annotator for row in rows}

        if policy == "intersection" and present != set(names):
            continue

        per_annotator = {
            name: [row for row in rows if row.annotator == name] for name in names
        }
        flags = {
            name: any(row.answerable for row in per_annotator[name]) for name in names
        }

        if policy == "single":
            answerable = flags[names[0]]
        elif policy == "union":
            answerable = any(flags.values())
        else:
            answerable = all(flags.values())

        if not answerable:
            golds.append(
                GoldQuery(
                    query_id=query_id,
                    query=meta.text,
                    slice=meta.slice or _query_slice(per_annotator, names),
                    answerable=False,
                    gold=(),
                    annotators=tuple(names),
                )
            )
            continue

        docs_by_annotator = _docs_by_annotator(
            [row for name in names for row in per_annotator[name]]
        )
        if policy == "single":
            doc_ids = sorted(docs_by_annotator.get(names[0], {}))
        elif policy == "union":
            doc_ids = sorted({doc for docs in docs_by_annotator.values() for doc in docs})
        else:
            doc_ids = sorted(
                set.intersection(*(set(docs_by_annotator.get(name, {})) for name in names))
                if all(docs_by_annotator.get(name) for name in names)
                else set()
            )

        gold_docs: list[GoldDoc] = []
        for doc_id in doc_ids:
            if policy == "single":
                spans = _spans_for(docs_by_annotator[names[0]][doc_id])
            elif policy == "union":
                spans = _spans_for(
                    [row for name in names for row in docs_by_annotator.get(name, {}).get(doc_id, [])]
                )
            else:
                intervals = intersect_groups(
                    [
                        [(row.start, row.end) for row in docs_by_annotator[name][doc_id]]
                        for name in names
                    ]
                )
                spans = [GoldSpan(start, end) for start, end in intervals]
            if spans:
                gold_docs.append(GoldDoc(doc_id=doc_id, spans=tuple(spans)))

        golds.append(
            GoldQuery(
                query_id=query_id,
                query=meta.text,
                slice=meta.slice or _query_slice(per_annotator, names),
                answerable=answerable,
                gold=tuple(gold_docs),
                annotators=tuple(names),
            )
        )

    return golds


def write_gold(path: Path, golds: Sequence[GoldQuery]) -> int:
    return write_jsonl(path, [gold.to_dict() for gold in golds])


def load_gold(path: Path) -> list[GoldQuery]:
    golds: list[GoldQuery] = []
    for raw in read_jsonl(path):
        golds.append(
            GoldQuery(
                query_id=str(raw["query_id"]),
                query=str(raw["query"]),
                slice=str(raw.get("slice") or ""),
                answerable=bool(raw.get("answerable")),
                annotators=tuple(raw.get("annotators") or ()),
                gold=tuple(
                    GoldDoc(
                        doc_id=str(doc["doc_id"]),
                        spans=tuple(
                            GoldSpan(int(start), int(end))
                            for start, end in doc.get("spans", [])
                        ),
                    )
                    for doc in raw.get("gold", [])
                ),
            )
        )
    return golds


def summary(golds: Sequence[GoldQuery]) -> dict[str, int]:
    answerable = sum(1 for gold in golds if gold.answerable and gold.gold)
    unanswerable = sum(1 for gold in golds if not gold.answerable)
    skipped = len(golds) - answerable - unanswerable
    docs = sum(len(gold.gold) for gold in golds)
    spans = sum(len(doc.spans) for gold in golds for doc in gold.gold)
    return {
        "queries": len(golds),
        "answerable": answerable,
        "unanswerable": unanswerable,
        "skipped": skipped,
        "gold_docs": docs,
        "gold_spans": spans,
    }
