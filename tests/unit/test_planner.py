"""Rule-based planner for comparison questions."""

from __future__ import annotations

from pokerag.llm.planner import plan_subqueries


def test_multi_hop_comparison_splits_into_two_queries() -> None:
    plan = plan_subqueries("Are Treecko and Mudkip both in the Monster egg group?")

    assert plan == ["What egg group is Treecko?", "What egg group is Mudkip?"]


def test_group_name_with_spaces() -> None:
    plan = plan_subqueries("Are Mew and Articuno both in the No Eggs Discovered egg group?")

    assert plan == ["What egg group is Mew?", "What egg group is Articuno?"]


def test_single_hop_is_unchanged() -> None:
    question = "What is the catch rate of Cloyster?"

    assert plan_subqueries(question) == [question]
