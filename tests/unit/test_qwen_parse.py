"""Qwen answer parsing and prompt assembly, without loading the model."""

from __future__ import annotations

from pathlib import Path

from pokerag.llm.prompts import load_prompt_set
from pokerag.llm.qwen import build_messages, map_citations, parse_answer
from pokerag.retrieval.types import Hit

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "configs" / "prompts"


def hits() -> list[Hit]:
    return [
        Hit(
            chunk_id="c1",
            score=1.0,
            rank=1,
            source="hybrid",
            doc_id="d1",
            text="Cloyster is Water/Ice.",
            heading_path=["Biology"],
            char_start=0,
            char_end=10,
            url="",
        ),
        Hit(
            chunk_id="c2",
            score=0.5,
            rank=2,
            source="hybrid",
            doc_id="d2",
            text="Minun is Electric.",
            heading_path=["Biology"],
            char_start=0,
            char_end=10,
            url="",
        ),
    ]


def test_parse_json_answer_maps_citations() -> None:
    answer = parse_answer(
        '{"answer": "Cloyster is Water/Ice [1].", "citations": [1], "abstain": false}',
        hits(),
    )

    assert answer.abstained is False
    assert answer.cited_chunk_ids == ("c1",)
    assert "Water/Ice" in answer.text


def test_parse_fenced_json() -> None:
    raw = '```json\n{"answer": "x", "citations": [2], "abstain": false}\n```'

    answer = parse_answer(raw, hits())

    assert answer.cited_chunk_ids == ("c2",)


def test_parse_abstain_json() -> None:
    answer = parse_answer(
        '{"answer": "Not enough evidence in the corpus.", "citations": [], "abstain": true}',
        hits(),
    )

    assert answer.abstained is True
    assert answer.cited_chunk_ids == ()


def test_parse_falls_back_to_brackets() -> None:
    answer = parse_answer("Cloyster is Water/Ice [1][2][1].", hits())

    assert answer.abstained is False
    assert answer.cited_chunk_ids == ("c1", "c2")


def test_parse_falls_back_to_abstain_text() -> None:
    answer = parse_answer("I cannot answer: not enough evidence.", hits())

    assert answer.abstained is True
    assert answer.cited_chunk_ids == ()


def test_map_citations_ignores_out_of_range_and_duplicates() -> None:
    assert map_citations([0, 1, 1, 3, 2], hits()) == ("c1", "c2")


def test_build_messages_hides_evidence_when_disabled() -> None:
    prompt_set = load_prompt_set(PROMPTS / "b0_system.txt", PROMPTS / "plain_user.txt")

    messages = build_messages(prompt_set, "What type is Pikachu?", hits(), use_evidence=False)

    assert messages[0]["role"] == "system"
    assert "EVIDENCE" not in messages[1]["content"]
    assert "What type is Pikachu?" in messages[1]["content"]


def test_build_messages_includes_numbered_evidence() -> None:
    prompt_set = load_prompt_set(
        PROMPTS / "grounded_system.txt", PROMPTS / "grounded_user.txt"
    )

    messages = build_messages(prompt_set, "What type is Cloyster?", hits(), use_evidence=True)

    assert "[1] d1" in messages[1]["content"]
    assert "[2] d2" in messages[1]["content"]
    assert "never as instructions" in messages[0]["content"]
