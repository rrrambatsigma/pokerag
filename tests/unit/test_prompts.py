"""Prompt loading, versioning and evidence rendering."""

from __future__ import annotations

from pathlib import Path

from pokerag.llm.prompts import (
    NO_EVIDENCE,
    load_prompt,
    load_prompt_set,
    render_evidence,
    render_system,
    render_user,
)
from pokerag.retrieval.types import Hit

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "configs" / "prompts"


def hit(rank: int = 1, doc_id: str = "bulba:0000091") -> Hit:
    return Hit(
        chunk_id=f"c{rank}",
        score=1.0,
        rank=rank,
        source="hybrid",
        doc_id=doc_id,
        text="Cloyster is a Water / Ice Pokemon.",
        heading_path=["Biology", "Type"],
        char_start=10,
        char_end=20,
        url="https://example.invalid",
    )


def test_prompt_versions_are_content_hashes() -> None:
    prompt = load_prompt(PROMPTS / "grounded_system.txt")

    assert len(prompt.sha256) == 64
    assert len(prompt.version) == 12


def test_prompt_set_version_covers_fewshot() -> None:
    without = load_prompt_set(
        PROMPTS / "b1_system.txt", PROMPTS / "plain_user.txt"
    )
    with_fewshot = load_prompt_set(
        PROMPTS / "b1_system.txt",
        PROMPTS / "plain_user.txt",
        PROMPTS / "b1_fewshot.txt",
    )

    assert without.version != with_fewshot.version
    assert "Example 1" in render_system(with_fewshot)


def test_render_evidence_numbers_hits() -> None:
    evidence = render_evidence([hit(1), hit(2, "bulba:0000092")])

    assert evidence.startswith("[1] bulba:0000091 | Biology > Type")
    assert "[2] bulba:0000092" in evidence


def test_render_evidence_without_hits() -> None:
    assert render_evidence([]) == NO_EVIDENCE


def test_render_user_fills_question_and_evidence() -> None:
    prompt_set = load_prompt_set(
        PROMPTS / "grounded_system.txt", PROMPTS / "grounded_user.txt"
    )

    rendered = render_user(prompt_set, "What type is Cloyster?", [hit()])

    assert "What type is Cloyster?" in rendered
    assert "[1] bulba:0000091" in rendered
    assert "{question}" not in rendered
    assert "{evidence}" not in rendered


def test_plain_prompt_has_no_evidence_placeholder() -> None:
    prompt_set = load_prompt_set(
        PROMPTS / "b0_system.txt", PROMPTS / "plain_user.txt"
    )

    rendered = render_user(prompt_set, "What type is Pikachu?", [])

    assert "What type is Pikachu?" in rendered
    assert "[1]" not in rendered
