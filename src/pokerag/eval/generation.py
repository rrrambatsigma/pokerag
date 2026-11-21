"""Generation metrics: abstention, citations, and judged correctness.

Two families are kept apart on purpose:

* automatic, computed from the prediction and the gold spans: abstention on the
  right slice, citation precision and recall;
* judged, computed only when a human or verifier supplied ``correct`` and
  ``supported``. A missing judgment yields ``None`` and is excluded from the
  mean, never counted as zero. Otherwise a missing judgment would look like a
  failure and quietly flatter the systems that were actually judged.

``unsupported_correct`` is the parametric-leakage signal: the answer is factually
right but no cited evidence supports it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pokerag.eval.metrics import overlaps
from pokerag.eval.qrels import GoldQuery

METRICS = (
    "abstain_correct",
    "false_confident",
    "false_refusal",
    "citation_precision",
    "citation_recall",
    "supported_correct",
    "unsupported_correct",
    "parametric_leak",
)


@dataclass(frozen=True)
class PredictionCitation:
    chunk_id: str
    doc_id: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Prediction:
    query_id: str
    text: str
    abstained: bool
    citations: tuple[PredictionCitation, ...] = ()


@dataclass(frozen=True)
class Judgment:
    query_id: str
    correct: bool | None = None
    supported: bool | None = None


def _citation_matches(gold: GoldQuery, citation: PredictionCitation) -> bool:
    for doc in gold.gold:
        if doc.doc_id != citation.doc_id:
            continue
        return any(
            overlaps(citation.char_start, citation.char_end, span.start, span.end)
            for span in doc.spans
        )
    return False


def score_generation(
    gold: GoldQuery,
    prediction: Prediction,
    judgment: Judgment | None = None,
) -> dict[str, float | None]:
    """Score one prediction. ``None`` means the metric does not apply here."""
    if not gold.answerable:
        answered = not prediction.abstained
        scores: dict[str, float | None] = {
            "abstain_correct": 0.0 if answered else 1.0,
            "false_confident": 1.0 if answered else 0.0,
            "false_refusal": None,
            "citation_precision": None,
            "citation_recall": None,
            "supported_correct": None,
            "unsupported_correct": None,
            "parametric_leak": None,
        }
        if judgment is not None and judgment.correct is not None:
            scores["parametric_leak"] = (
                1.0 if answered and judgment.correct else 0.0
            )
        return scores

    scores = {
        "abstain_correct": None,
        "false_confident": None,
        "false_refusal": 1.0 if prediction.abstained else 0.0,
        "citation_precision": 0.0,
        "citation_recall": 0.0,
        "supported_correct": 0.0,
        "unsupported_correct": 0.0,
        "parametric_leak": None,
    }
    if prediction.abstained:
        return scores

    citations = prediction.citations
    matched = [citation for citation in citations if _citation_matches(gold, citation)]
    scores["citation_precision"] = len(matched) / len(citations) if citations else 0.0

    covered = {citation.doc_id for citation in matched}
    gold_docs = {doc.doc_id for doc in gold.gold}
    scores["citation_recall"] = len(covered & gold_docs) / len(gold_docs) if gold_docs else 0.0

    if judgment is not None and judgment.correct is not None and judgment.supported is not None:
        scores["supported_correct"] = (
            1.0 if judgment.correct and judgment.supported else 0.0
        )
        scores["unsupported_correct"] = (
            1.0 if judgment.correct and not judgment.supported else 0.0
        )
    else:
        scores["supported_correct"] = None
        scores["unsupported_correct"] = None
    return scores


def _mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _summary(rows: Sequence[Mapping]) -> dict[str, float | None]:
    return {metric: _mean([row["scores"][metric] for row in rows]) for metric in METRICS}


def evaluate_generation(
    golds: Sequence[GoldQuery],
    predictions: Sequence[Prediction],
    judgments: Sequence[Judgment] | None = None,
) -> dict:
    """Aggregate generation metrics overall, per slice, and per note."""
    by_query = {prediction.query_id: prediction for prediction in predictions}
    judged = {judgment.query_id: judgment for judgment in judgments or ()}

    rows: list[dict] = []
    missing = 0
    for gold in golds:
        prediction = by_query.get(gold.query_id)
        if prediction is None:
            missing += 1
            continue
        rows.append(
            {
                "query_id": gold.query_id,
                "slice": gold.slice,
                "note": gold.note,
                "answerable": gold.answerable,
                "scores": score_generation(gold, prediction, judged.get(gold.query_id)),
            }
        )

    by_slice: dict[str, dict] = {}
    for slice_name in sorted({row["slice"] for row in rows}):
        subset = [row for row in rows if row["slice"] == slice_name]
        by_slice[slice_name] = {
            "queries": len(subset),
            "metrics": _summary(subset),
        }

    by_note: dict[str, dict] = {}
    for note in sorted({row["note"] for row in rows if row["note"]}):
        subset = [row for row in rows if row["note"] == note]
        by_note[note] = {"queries": len(subset), "metrics": _summary(subset)}

    return {
        "queries": len(golds),
        "predictions": len(predictions),
        "missing": missing,
        "judged": sum(1 for gold in golds if gold.query_id in judged),
        "answerable": sum(1 for gold in golds if gold.answerable),
        "unanswerable": sum(1 for gold in golds if not gold.answerable),
        "metrics": _summary(rows),
        "by_slice": by_slice,
        "by_note": by_note,
        "per_query": [
            {
                "query_id": row["query_id"],
                "slice": row["slice"],
                "note": row["note"],
                "scores": row["scores"],
            }
            for row in rows
        ],
    }


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(result: dict, meta: dict | None = None) -> str:
    meta = meta or {}
    lines = [
        "# Generation evaluation",
        "",
        f"- queries: {result.get('queries', 0)}",
        f"- predictions: {result.get('predictions', 0)}",
        f"- missing predictions: {result.get('missing', 0)}",
        f"- judged: {result.get('judged', 0)}",
        f"- qrels: `{meta.get('qrels', '')}`",
        f"- predictions file: `{meta.get('predictions', '')}`",
        f"- judgments file: `{meta.get('judgments', '')}`",
        "",
        "Metrik `supported_correct` dan `unsupported_correct` hanya dihitung bila",
        "ada judgment. Nilai `n/a` berarti tidak ada data, bukan nol.",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for metric in METRICS:
        lines.append(f"| `{metric}` | {_cell(result['metrics'][metric])} |")

    lines.extend(
        [
            "",
            "## Per slice",
            "",
            "| Slice | Queries | abstain_correct | false_confident | false_refusal | citation_precision | citation_recall |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for slice_name, block in result.get("by_slice", {}).items():
        metrics = block["metrics"]
        lines.append(
            f"| `{slice_name}` | {block['queries']} | "
            f"{_cell(metrics['abstain_correct'])} | {_cell(metrics['false_confident'])} | "
            f"{_cell(metrics['false_refusal'])} | {_cell(metrics['citation_precision'])} | "
            f"{_cell(metrics['citation_recall'])} |"
        )

    if result.get("by_note"):
        lines.extend(
            [
                "",
                "## Per note",
                "",
                "| Note | Queries | abstain_correct | false_confident | parametric_leak |",
                "|---|---|---|---|---|",
            ]
        )
        for note, block in result["by_note"].items():
            metrics = block["metrics"]
            lines.append(
                f"| {note} | {block['queries']} | "
                f"{_cell(metrics['abstain_correct'])} | {_cell(metrics['false_confident'])} | "
                f"{_cell(metrics['parametric_leak'])} |"
            )
    lines.append("")
    return "\n".join(lines)
