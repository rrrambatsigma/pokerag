"""Qrels adapter and multi-document metrics, with no model and no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokerag.core.jsonl import write_jsonl
from pokerag.eval.metrics import score_query, evaluate_gold, render_markdown
from pokerag.eval.qrels import (
    ExportRow,
    GoldDoc,
    GoldQuery,
    GoldSpan,
    QueryMeta,
    build_gold,
    intersect_groups,
    load_gold,
    load_queries,
    merge_intervals,
    read_export,
    summary,
    write_gold,
)
from pokerag.retrieval.types import Hit


def export_rows() -> dict[str, list[ExportRow]]:
    return {
        "A1": [
            ExportRow("q1", "A1", "d1", 100, 150, True, "alpha", "multi_hop"),
            ExportRow("q1", "A1", "d2", 200, 250, True, "beta", "multi_hop"),
            ExportRow("q2", "A1", None, None, None, False, "", "unanswerable"),
            ExportRow("q3", "A1", "d1", 10, 20, True, "gamma", "single_hop"),
        ],
        "A2": [
            ExportRow("q1", "A2", "d1", 120, 160, True, "delta", "multi_hop"),
            ExportRow("q2", "A2", None, None, None, False, "", "unanswerable"),
            ExportRow("q3", "A2", "d1", 15, 25, True, "epsilon", "single_hop"),
        ],
    }


def queries() -> dict[str, QueryMeta]:
    return {
        "q1": QueryMeta("q1", "two docs", "multi_hop"),
        "q2": QueryMeta("q2", "no evidence", "unanswerable"),
        "q3": QueryMeta("q3", "one doc", "single_hop"),
    }


def test_read_export_jsonl_and_csv(tmp_path: Path) -> None:
    rows = [
        {
            "query_id": "q1",
            "doc_id": "d1",
            "char_start": 10,
            "char_end": 20,
            "relevance": 1,
            "answerable": True,
            "status": "done",
            "quote": "x",
            "slice": "single_hop",
            "annotator": "A1",
        },
        {
            "query_id": "q2",
            "doc_id": None,
            "relevance": 0,
            "answerable": False,
            "status": "done",
            "slice": "unanswerable",
            "annotator": "A1",
        },
    ]
    jsonl = tmp_path / "a1.jsonl"
    write_jsonl(jsonl, rows)

    parsed = read_export(jsonl)
    assert [row.query_id for row in parsed] == ["q1", "q2"]
    assert parsed[0].start == 10 and parsed[0].end == 20
    assert parsed[0].answerable is True
    assert parsed[1].answerable is False and parsed[1].doc_id is None

    csv_path = tmp_path / "a1.csv"
    csv_path.write_text(
        "query_id,doc_id,char_start,char_end,relevance,answerable,status,quote,slice,annotator\n"
        "q1,d1,10,20,1,true,done,x,single_hop,A1\n",
        encoding="utf-8",
    )
    from_csv = read_export(csv_path)
    assert len(from_csv) == 1
    assert from_csv[0].end == 20 and from_csv[0].annotator == "A1"


def test_load_queries_csv_and_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "queries.csv"
    csv_path.write_text(
        "query_id,text,slice\nq1,First?,single_hop\n", encoding="utf-8"
    )
    assert load_queries(csv_path)["q1"].text == "First?"

    json_path = tmp_path / "queries.json"
    json_path.write_text(
        json.dumps([{"id": "q1", "text": "First?", "slice": "single_hop"}]),
        encoding="utf-8",
    )
    assert load_queries(json_path)["q1"].slice == "single_hop"


def test_single_policy_uses_one_annotator() -> None:
    golds = build_gold(export_rows(), queries(), policy="single", annotator="A1")
    by_id = {gold.query_id: gold for gold in golds}

    assert [doc.doc_id for doc in by_id["q1"].gold] == ["d1", "d2"]
    assert by_id["q2"].answerable is False
    assert by_id["q1"].annotators == ("A1",)


def test_union_policy_keeps_every_document() -> None:
    golds = build_gold(export_rows(), queries(), policy="union")
    q1 = next(gold for gold in golds if gold.query_id == "q1")

    assert [doc.doc_id for doc in q1.gold] == ["d1", "d2"]
    d1 = next(doc for doc in q1.gold if doc.doc_id == "d1")
    assert [(span.start, span.end) for span in d1.spans] == [(100, 150), (120, 160)]


def test_intersection_keeps_overlap_only() -> None:
    golds = build_gold(export_rows(), queries(), policy="intersection")
    q1 = next(gold for gold in golds if gold.query_id == "q1")

    assert [doc.doc_id for doc in q1.gold] == ["d1"]
    assert [(span.start, span.end) for span in q1.gold[0].spans] == [(120, 150)]


def test_interval_helpers() -> None:
    assert merge_intervals([(10, 20), (15, 30), (40, 50)]) == [(10, 30), (40, 50)]
    assert intersect_groups([[(0, 10)], [(5, 15)]]) == [(5, 10)]
    assert intersect_groups([[(0, 5)], [(6, 10)]]) == []


def test_summary_counts() -> None:
    golds = build_gold(export_rows(), queries(), policy="union")
    counts = summary(golds)
    assert counts["queries"] == 3
    assert counts["answerable"] == 2
    assert counts["unanswerable"] == 1
    assert counts["skipped"] == 0


def make_gold(qid: str, text: str, slice_name: str, docs: list[tuple[str, list[tuple[int, int]]]]) -> GoldQuery:
    return GoldQuery(
        query_id=qid,
        query=text,
        slice=slice_name,
        answerable=True,
        gold=tuple(
            GoldDoc(doc_id=doc_id, spans=tuple(GoldSpan(s, e) for s, e in spans))
            for doc_id, spans in docs
        ),
        annotators=("A1",),
    )


def hit(rank: int, doc_id: str, start: int, end: int) -> Hit:
    return Hit(
        chunk_id=f"{doc_id}-{rank}",
        score=1.0,
        rank=rank,
        source="stub",
        doc_id=doc_id,
        text="",
        heading_path=[],
        char_start=start,
        char_end=end,
        url="",
    )


def test_multi_document_recall_and_span_overlap() -> None:
    gold = make_gold("q1", "two docs", "multi_hop", [("d1", [(100, 150)]), ("d2", [(200, 250)])])
    hits = [hit(1, "d1", 0, 100), hit(2, "d2", 200, 260), hit(3, "d1", 120, 130)]

    scores = score_query(hits, gold, [1, 2, 3])

    assert scores["doc_recall@1"] == pytest.approx(0.5)
    assert scores["doc_recall@2"] == pytest.approx(1.0)
    assert scores["span_hit@1"] == 0.0
    assert scores["span_hit@2"] == pytest.approx(0.5)
    assert scores["span_hit@3"] == pytest.approx(1.0)
    assert scores["span_mrr@2"] == pytest.approx(0.5)


def test_unanswerable_is_excluded_from_means() -> None:
    answerable = make_gold("q1", "one", "single_hop", [("d1", [(0, 10)])])
    unanswerable = GoldQuery("q2", "none", "unanswerable", False, (), ("A1",))

    class Stub:
        def search(self, query: str, mode: str = "hybrid", top_k: int = 5):
            return [hit(1, "d1", 0, 10)]

    result = evaluate_gold(Stub(), [answerable, unanswerable], modes=["dense"], ks=[1, 5])

    assert result["queries"] == 2
    assert result["usable"] == 1
    assert result["unanswerable"] == 1
    assert result["modes"]["dense"]["summary"]["doc_recall@5"] == 1.0
    assert len(result["modes"]["dense"]["per_query"]) == 1


def test_slice_aggregation_and_markdown() -> None:
    multi = make_gold("q1", "multi", "multi_hop", [("d1", [(0, 10)]), ("d2", [(0, 10)])])
    single = make_gold("q2", "single", "single_hop", [("d3", [(0, 10)])])

    class Stub:
        def search(self, query: str, mode: str = "hybrid", top_k: int = 5):
            if query == "multi":
                return [hit(1, "d1", 0, 10)]
            return [hit(1, "d3", 0, 10)]

    result = evaluate_gold(Stub(), [multi, single], modes=["dense"], ks=[5])
    assert set(result["modes"]["dense"]["by_slice"]) == {"multi_hop", "single_hop"}
    assert result["modes"]["dense"]["by_slice"]["multi_hop"]["summary"]["doc_recall@5"] == 0.5

    markdown = render_markdown(result, {"qrels": "x", "strategy": "fixed"})
    assert "multi_hop" in markdown and "doc_recall@5" in markdown


def test_write_and_load_gold_roundtrip(tmp_path: Path) -> None:
    golds = build_gold(export_rows(), queries(), policy="union")
    path = tmp_path / "gold.jsonl"
    write_gold(path, golds)

    loaded = load_gold(path)
    assert [gold.query_id for gold in loaded] == [gold.query_id for gold in golds]
    assert loaded[0].gold[0].spans[0].start == golds[0].gold[0].spans[0].start
