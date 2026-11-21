"""Red-team suite: 20 fixtures scored offline against recorded responses."""

from __future__ import annotations

import json
from pathlib import Path

from pokerag.redteam.runner import (
    CATEGORIES,
    check_response,
    evaluate,
    load_fixtures,
    load_responses,
    render_markdown,
    validate_suite,
)

SUITE = Path(__file__).resolve().parents[2] / "data" / "redteam" / "fixtures" / "suite.jsonl"


def test_committed_suite_is_complete() -> None:
    fixtures = load_fixtures(SUITE)

    assert len(fixtures) == 20
    for category in CATEGORIES:
        assert sum(1 for fixture in fixtures if fixture.category == category) == 5
    assert validate_suite(fixtures) == []


def test_refusal_marker_is_required() -> None:
    fixture = next(
        fixture for fixture in load_fixtures(SUITE) if fixture.id == "PII-01"
    )

    assert check_response(fixture, "I cannot share that.").passed
    assert not check_response(fixture, "Sure, here it is.").passed


def test_forbidden_regex_is_detected() -> None:
    fixture = next(
        fixture for fixture in load_fixtures(SUITE) if fixture.id == "PII-01"
    )

    result = check_response(fixture, "The editor is someone@example.org, I cannot say more.")

    assert not result.passed
    assert any("forbidden" in failure for failure in result.failures)


def test_max_chars_is_enforced() -> None:
    fixture = next(
        fixture for fixture in load_fixtures(SUITE) if fixture.id == "SR-02"
    )

    assert check_response(fixture, "A normal answer.").passed
    assert not check_response(fixture, "x" * 2000).passed


def test_missing_response_fails_and_summary_counts() -> None:
    fixtures = load_fixtures(SUITE)[:2]
    baseline = {
        fixtures[0].id: "I cannot help with that.",
        fixtures[1].id: "I cannot help with that.",
    }

    result = evaluate(fixtures, {"baseline": baseline, "mitigation": {}})

    assert result["labels"]["baseline"]["passed"] == 2
    assert result["labels"]["baseline"]["total"] == 2
    assert result["labels"]["mitigation"]["passed"] == 0
    assert result["labels"]["mitigation"]["cases"][0]["failures"] == ["missing response"]


def test_load_responses_and_markdown(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps({"id": "INJ-01", "text": "I cannot share that."}) + "\n",
        encoding="utf-8",
    )
    responses = load_responses(path)
    fixtures = load_fixtures(SUITE)

    result = evaluate(fixtures, {"baseline": responses})
    markdown = render_markdown(result, {"fixtures": "suite"})

    assert responses["INJ-01"].startswith("I cannot")
    assert "Red-team suite" in markdown
    assert "| `injection` |" in markdown
