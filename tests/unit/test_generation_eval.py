"""Generation metrics: abstention, citation precision/recall and judged answers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokerag.eval.cli import load_judgments, load_predictions
from pokerag.eval.generation import (
    Judgment,
    Prediction,
    PredictionCitation,
    evaluate_generation,
    render_markdown,
    score_generation,
)
from pokerag.eval.qrels import GoldDoc, GoldQuery, GoldSpan


def gold(
    query_id: str = "q1",
    *,
    answerable: bool = True,
    note: str = "",
    slice_name: str = "single_hop",
    docs: list[tuple[str, list[tuple[int, int]]]] | None = None,
) -> GoldQuery:
    docs = docs or [("d1", [(100, 150)])]
    return GoldQuery(
        query_id=query_id,
        query="question",
        slice=slice_name,
        answerable=answerable,
        gold=tuple(
            GoldDoc(doc_id=doc_id, spans=tuple(GoldSpan(s, e) for s, e in spans))
            for doc_id, spans in docs
        ),
        annotators=("A1",),
        note=note,
    )


def prediction(
    query_id: str = "q1",
    *,
    abstained: bool = False,
    citations: list[tuple[str, int, int]] | None = None,
) -> Prediction:
    citations = citations if citations is not None else [("d1", 100, 150)]
    return Prediction(
        query_id=query_id,
        text="Cloyster is Water/Ice.",
        abstained=abstained,
        citations=tuple(
            PredictionCitation(f"c{index}", doc_id, start, end)
            for index, (doc_id, start, end) in enumerate(citations)
        ),
    )


def test_supported_correct_requires_correct_and_supported() -> None:
    scores = score_generation(
        gold(), prediction(), Judgment("q1", correct=True, supported=True)
    )

    assert scores["supported_correct"] == 1.0
    assert scores["unsupported_correct"] == 0.0
    assert scores["citation_precision"] == 1.0
    assert scores["citation_recall"] == 1.0
    assert scores["false_refusal"] == 0.0


def test_correct_but_unsupported_is_flagged() -> None:
    scores = score_generation(
        gold(), prediction(), Judgment("q1", correct=True, supported=False)
    )

    assert scores["supported_correct"] == 0.0
    assert scores["unsupported_correct"] == 1.0


def test_false_refusal_on_answerable_query() -> None:
    scores = score_generation(
        gold(), prediction(abstained=True), Judgment("q1", correct=False, supported=False)
    )

    assert scores["false_refusal"] == 1.0
    assert scores["citation_precision"] == 0.0
    assert scores["supported_correct"] == 0.0


def test_unanswerable_abstention() -> None:
    scores = score_generation(
        gold(answerable=False, note="type (outside corpus)"),
        prediction(abstained=True, citations=[]),
    )

    assert scores["abstain_correct"] == 1.0
    assert scores["false_confident"] == 0.0
    assert scores["parametric_leak"] is None


def test_unanswerable_answered_correctly_is_parametric_leak() -> None:
    scores = score_generation(
        gold(answerable=False),
        prediction(citations=[]),
        Judgment("q1", correct=True, supported=False),
    )

    assert scores["false_confident"] == 1.0
    assert scores["abstain_correct"] == 0.0
    assert scores["parametric_leak"] == 1.0


def test_multi_document_citation_recall_and_precision() -> None:
    query = gold(docs=[("d1", [(100, 150)]), ("d2", [(200, 250)])], slice_name="multi_hop")
    pred = prediction(citations=[("d1", 120, 130), ("d2", 200, 260), ("d3", 0, 10)])

    scores = score_generation(query, pred, Judgment("q1", correct=True, supported=True))

    assert scores["citation_recall"] == 1.0
    assert scores["citation_precision"] == pytest.approx(2 / 3)


def test_missing_judgment_is_none_and_ignored_in_the_mean() -> None:
    answerable = gold("q1")
    unjudged = gold("q2")

    result = evaluate_generation(
        [answerable, unjudged],
        [prediction("q1"), prediction("q2")],
        judgments=[Judgment("q1", correct=True, supported=True)],
    )

    assert result["judged"] == 1
    assert result["metrics"]["supported_correct"] == 1.0
    assert result["metrics"]["unsupported_correct"] == 0.0


def test_missing_predictions_are_counted() -> None:
    result = evaluate_generation([gold("q1"), gold("q2")], [prediction("q1")])

    assert result["missing"] == 1
    assert result["per_query"][0]["query_id"] == "q1"


def test_metrics_group_by_slice_and_note() -> None:
    golds = [
        gold("q1", answerable=False, note="type (outside corpus)", slice_name="unanswerable"),
        gold("q2", answerable=False, note="catch rate (outside corpus)", slice_name="unanswerable"),
        gold("q3", note="catch rate", slice_name="single_hop"),
    ]
    predictions = [
        prediction("q1", abstained=True, citations=[]),
        prediction("q2", citations=[]),
        prediction("q3"),
    ]
    result = evaluate_generation(golds, predictions)

    assert result["by_slice"]["unanswerable"]["queries"] == 2
    assert result["by_slice"]["unanswerable"]["metrics"]["abstain_correct"] == 0.5
    assert set(result["by_note"]) == {
        "type (outside corpus)",
        "catch rate (outside corpus)",
        "catch rate",
    }


def test_render_markdown_marks_missing_judgment_as_na() -> None:
    result = evaluate_generation([gold()], [prediction()])
    markdown = render_markdown(result, {"qrels": "x", "predictions": "y"})

    assert "n/a" in markdown
    assert "Generation evaluation" in markdown


def test_load_predictions_accepts_camel_case(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "text": "answer",
                "abstained": False,
                "citations": [
                    {"chunkId": "c1", "docId": "d1", "charStart": 10, "charEnd": 20}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    predictions = load_predictions(path)

    assert predictions[0].citations[0].chunk_id == "c1"
    assert predictions[0].citations[0].char_start == 10


def test_load_judgments_reads_optional_labels(tmp_path: Path) -> None:
    path = tmp_path / "judgments.jsonl"
    path.write_text(
        '{"query_id": "q1", "correct": true, "supported": false}\n'
        '{"query_id": "q2", "correct": null}\n',
        encoding="utf-8",
    )

    judgments = load_judgments(path)

    assert judgments[0].correct is True and judgments[0].supported is False
    assert judgments[1].correct is None
