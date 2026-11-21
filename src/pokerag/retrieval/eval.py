"""Retrieval evaluation: qrels in, recall/MRR/nDCG out.

Relevance is judged at the **document** level on purpose. Chunk boundaries
differ between strategies (A1), so a chunk-level judgement would make fixed,
heading and hierarchical incomparable. The optional ``evidence`` string adds a
second, stricter check: the chunk must also contain the cited passage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from pokerag.core.jsonl import atomic_write_text, read_jsonl, write_json
from pokerag.retrieval.types import Hit

METRICS = ("recall", "mrr", "ndcg", "span_hit", "span_mrr")


@dataclass(frozen=True)
class Qrel:
    query_id: str
    query: str
    doc_id: str
    evidence: str = ""


def load_qrels(path: Path) -> list[Qrel]:
    """Read one JSON object per line; order is preserved for reproducible runs."""
    qrels: list[Qrel] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        try:
            qrel = Qrel(
                query_id=str(row["query_id"]),
                query=str(row["query"]),
                doc_id=str(row["doc_id"]),
                evidence=str(row.get("evidence") or ""),
            )
        except KeyError as exc:
            raise ValueError(f"qrel di {path} kehilangan kolom {exc}") from exc
        if not qrel.query.strip():
            raise ValueError(f"query kosong di qrel {qrel.query_id}")
        if qrel.query_id in seen:
            raise ValueError(f"query_id duplikat: {qrel.query_id}")
        seen.add(qrel.query_id)
        qrels.append(qrel)
    if not qrels:
        raise ValueError(f"qrels kosong: {path}")
    return qrels


def _normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def score_hits(hits: Sequence[Hit], qrel: Qrel, ks: Iterable[int]) -> dict[str, float]:
    """Score one ranked list against one qrel at every cut-off ``k``."""
    evidence = _normalise(qrel.evidence)
    doc_rank: int | None = None
    span_rank: int | None = None
    for hit in hits:
        if hit.doc_id != qrel.doc_id:
            continue
        if doc_rank is None:
            doc_rank = hit.rank
        if evidence and evidence in _normalise(hit.text):
            span_rank = hit.rank
            break
        if not evidence and doc_rank is not None:
            span_rank = doc_rank
            break

    scores: dict[str, float] = {}
    for k in ks:
        in_doc = doc_rank is not None and doc_rank <= k
        in_span = span_rank is not None and span_rank <= k
        scores[f"recall@{k}"] = 1.0 if in_doc else 0.0
        scores[f"mrr@{k}"] = (1.0 / doc_rank) if in_doc else 0.0
        scores[f"ndcg@{k}"] = (1.0 / math.log2(doc_rank + 1)) if in_doc else 0.0
        scores[f"span_hit@{k}"] = 1.0 if in_span else 0.0
        scores[f"span_mrr@{k}"] = (1.0 / span_rank) if in_span else 0.0
    return scores


def evaluate(
    retriever,
    qrels: Sequence[Qrel],
    modes: Sequence[str],
    ks: Sequence[int],
    top_k: int | None = None,
) -> dict:
    """Run every query in every mode and aggregate the metrics.

    ``retriever`` only needs a ``search(query, mode=..., top_k=...)`` method,
    which keeps this testable with a stub.
    """
    cuts = sorted({int(k) for k in ks})
    if not cuts or cuts[0] < 1:
        raise ValueError("ks harus berisi angka >= 1")
    depth = max(cuts) if top_k is None else max(top_k, max(cuts))
    results: dict[str, dict] = {}
    for mode in modes:
        per_query: list[dict] = []
        for qrel in qrels:
            hits = retriever.search(qrel.query, mode=mode, top_k=depth)
            scores = score_hits(hits, qrel, cuts)
            per_query.append(
                {
                    "query_id": qrel.query_id,
                    "query": qrel.query,
                    "doc_id": qrel.doc_id,
                    "evidence": qrel.evidence,
                    "scores": scores,
                    "top_doc_ids": [hit.doc_id for hit in hits],
                    "top_chunk_ids": [hit.chunk_id for hit in hits],
                }
            )
        summary = {
            key: sum(row["scores"][key] for row in per_query) / len(per_query)
            for key in per_query[0]["scores"]
        }
        results[mode] = {"summary": summary, "per_query": per_query}
    return {"queries": len(qrels), "ks": cuts, "modes": dict(results)}


def _metric_columns(ks: Sequence[int]) -> list[str]:
    return [f"{metric}@{k}" for metric in METRICS for k in ks]


def render_markdown(result: dict, meta: dict | None = None) -> str:
    """A committed report: one row per mode, no per-query noise."""
    meta = meta or {}
    ks = result.get("ks", [])
    columns = _metric_columns(ks)
    lines = [
        "# B1 — Retrieval baseline",
        "",
        f"- queries: {result.get('queries', 0)}",
        f"- qrels: `{meta.get('qrels', '')}`",
        f"- strategy: `{meta.get('strategy', '')}`",
        f"- index_version: `{str(meta.get('index_version', ''))[:12]}`",
        f"- embedder: `{meta.get('embedder', '')}`",
        "",
        "Relevance dinilai di level dokumen; `span_*` menuntut chunk juga memuat",
        "kutipan `evidence`. Metrik dihitung pada cut-off yang sama untuk semua mode.",
        "",
        "| Mode | " + " | ".join(columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for mode, payload in result.get("modes", {}).items():
        summary = payload["summary"]
        cells = [f"{summary[column]:.3f}" for column in columns]
        lines.append(f"| `{mode}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_reports(result: dict, meta: dict | None = None, *, artifacts: Path, docs: Path | None) -> None:
    """Write the machine-readable JSON and, optionally, the committed table."""
    payload = {"meta": meta or {}, **result}
    write_json(artifacts, payload)
    if docs is not None:
        atomic_write_text(docs, render_markdown(result, meta))
