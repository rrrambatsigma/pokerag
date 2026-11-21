"""Retrieval metrics for multi-document, span-anchored gold qrels.

The dev baseline judges relevance per document and checks the cited string.
The test benchmark instead carries exact character spans, so a chunk counts as
evidence only when it overlaps a gold span. Multi-hop queries contribute more
than one gold document, which is why recall is a fraction over all gold docs
rather than a single hit-or-miss.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Iterable

from pokerag.eval.qrels import GoldQuery
from pokerag.retrieval.types import Hit

METRICS = ("doc_recall", "doc_mrr", "doc_ndcg", "span_hit", "span_mrr")


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def score_query(hits: Sequence[Hit], gold: GoldQuery, ks: Iterable[int]) -> dict[str, float]:
    """Score one ranked list against one gold record at every cut-off."""
    gold_docs = {doc.doc_id: doc for doc in gold.gold}
    if not gold_docs:
        raise ValueError(f"{gold.query_id}: gold kosong")

    ordered = sorted(hits, key=lambda hit: hit.rank)
    doc_first: dict[str, int] = {}
    span_first: dict[str, int] = {}
    first_span_rank: int | None = None

    for hit in ordered:
        doc = gold_docs.get(hit.doc_id)
        if doc is None:
            continue
        doc_first.setdefault(hit.doc_id, hit.rank)
        if any(
            overlaps(hit.char_start, hit.char_end, span.start, span.end)
            for span in doc.spans
        ):
            span_first.setdefault(hit.doc_id, hit.rank)
            if first_span_rank is None:
                first_span_rank = hit.rank

    n_gold = len(gold_docs)
    scores: dict[str, float] = {}
    for k in sorted({int(value) for value in ks}):
        found = {doc_id for doc_id, rank in doc_first.items() if rank <= k}
        doc_rank = min(doc_first.values(), default=None)

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for doc_id, rank in doc_first.items()
            if rank <= k
        )
        ideal = sum(
            1.0 / math.log2(index + 1) for index in range(1, min(n_gold, k) + 1)
        )

        scores[f"doc_recall@{k}"] = len(found) / n_gold
        scores[f"doc_mrr@{k}"] = (
            1.0 / doc_rank if doc_rank is not None and doc_rank <= k else 0.0
        )
        scores[f"doc_ndcg@{k}"] = dcg / ideal if ideal else 0.0
        scores[f"span_hit@{k}"] = (
            sum(1 for rank in span_first.values() if rank <= k) / n_gold
        )
        scores[f"span_mrr@{k}"] = (
            1.0 / first_span_rank
            if first_span_rank is not None and first_span_rank <= k
            else 0.0
        )
    return scores


def _mean(rows: Sequence[dict], keys: Sequence[str]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in keys}
    return {
        key: sum(row["scores"][key] for row in rows) / len(rows) for key in keys
    }


def evaluate_gold(
    retriever,
    golds: Sequence[GoldQuery],
    *,
    modes: Sequence[str],
    ks: Sequence[int],
    slices: Sequence[str] | None = None,
) -> dict:
    """Run answerable queries in every mode and aggregate overall and per slice.

    Unanswerable queries carry no retrieval gold, so they are counted and left
    out of the means. Generation metrics for abstention need the generator and
    are reported separately later.
    """
    cuts = sorted({int(k) for k in ks})
    if not cuts or cuts[0] < 1:
        raise ValueError("ks harus berisi angka >= 1")
    depth = cuts[-1]

    allowed = set(slices) if slices else None
    usable = [
        gold
        for gold in golds
        if gold.answerable and gold.gold and (allowed is None or gold.slice in allowed)
    ]
    if not usable:
        raise ValueError("tidak ada query answerable untuk dievaluasi")

    keys = [f"{metric}@{k}" for metric in METRICS for k in cuts]
    result: dict = {
        "queries": len(golds),
        "usable": len(usable),
        "answerable": sum(1 for gold in golds if gold.answerable),
        "unanswerable": sum(1 for gold in golds if not gold.answerable),
        "skipped": sum(1 for gold in golds if gold.answerable and not gold.gold),
        "ks": cuts,
        "slices": sorted({gold.slice for gold in usable}),
        "modes": {},
    }

    for mode in modes:
        rows: list[dict] = []
        for gold in usable:
            hits = retriever.search(gold.query, mode=mode, top_k=depth)
            rows.append(
                {
                    "query_id": gold.query_id,
                    "slice": gold.slice,
                    "scores": score_query(hits, gold, cuts),
                    "top_doc_ids": [hit.doc_id for hit in hits],
                    "top_chunk_ids": [hit.chunk_id for hit in hits],
                }
            )

        by_slice: dict[str, dict] = {}
        for slice_name in sorted({row["slice"] for row in rows}):
            subset = [row for row in rows if row["slice"] == slice_name]
            by_slice[slice_name] = {
                "queries": len(subset),
                "summary": _mean(subset, keys),
            }

        result["modes"][mode] = {
            "summary": _mean(rows, keys),
            "by_slice": by_slice,
            "per_query": rows,
        }
    return result


def _metric_columns(ks: Sequence[int]) -> list[str]:
    return [f"{metric}@{k}" for metric in METRICS for k in ks]


def render_markdown(result: dict, meta: dict | None = None) -> str:
    meta = meta or {}
    ks = result.get("ks", [])
    columns = _metric_columns(ks)
    lines = [
        "# Test retrieval",
        "",
        f"- queries: {result.get('queries', 0)}",
        f"- answerable: {result.get('answerable', 0)}",
        f"- unanswerable: {result.get('unanswerable', 0)}",
        f"- qrels: `{meta.get('qrels', '')}`",
        f"- strategy: `{meta.get('strategy', '')}`",
        f"- index_version: `{str(meta.get('index_version', ''))[:12]}`",
        f"- policy: `{meta.get('policy', '')}`",
        "",
        "Gold dinilai pada level span. `span_hit` menuntut chunk beririsan",
        "dengan rentang karakter gold, bukan sekadar dokumen yang sama.",
        "",
        "| Mode | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for mode, payload in result.get("modes", {}).items():
        summary = payload["summary"]
        lines.append(
            f"| `{mode}` | " + " | ".join(f"{summary[column]:.3f}" for column in columns) + " |"
        )

    if ks:
        last = ks[-1]
        lines.extend(
            [
                "",
                f"## Per slice (doc_recall@{last} / span_hit@{last})",
                "",
                "| Slice | Mode | doc_recall | span_hit | Queries |",
                "|---|---|---|---|---|",
            ]
        )
        for slice_name in result.get("slices", []):
            for mode, payload in result.get("modes", {}).items():
                block = payload["by_slice"].get(slice_name)
                if not block:
                    continue
                summary = block["summary"]
                lines.append(
                    f"| `{slice_name}` | `{mode}` | "
                    f"{summary[f'doc_recall@{last}']:.3f} | "
                    f"{summary[f'span_hit@{last}']:.3f} | {block['queries']} |"
                )
    lines.append("")
    return "\n".join(lines)
