"""Write training sentences from extracted facts, in our own words.

Nothing here reads corpus text: the input is a :class:`pokerag.sft.facts.Fact`
(value only) and the output is a short question, one evidence sentence, and the
target answer. Wording is deliberately plain so it does not track Bulbapedia's
phrasing; :mod:`pokerag.sft.leak` checks that it does not.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokerag.sft.facts import CATCH_RATE, EGG_GROUP, EVOLUTION, TYPE, Fact

QUESTIONS: dict[str, tuple[str, ...]] = {
    TYPE: ("What type is {entity}?", "Which types does {entity} have?"),
    CATCH_RATE: (
        "What is the catch rate of {entity}?",
        "What catch rate does {entity} have?",
    ),
    EGG_GROUP: (
        "Which egg group is {entity} in?",
        "What egg group does {entity} belong to?",
    ),
    EVOLUTION: ("What does {entity} evolve into?", "Into which Pokémon does {entity} evolve?"),
}


@dataclass(frozen=True)
class Rendered:
    question: str
    evidence: str
    answer: str


def _type_parts(fact: Fact) -> tuple[str, ...]:
    return fact.values


def _article(word: str) -> str:
    return "an" if word[:1].upper() in "AEIOU" else "a"


def render_evidence(fact: Fact) -> str:
    entity = fact.entity
    if fact.kind == TYPE:
        parts = _type_parts(fact)
        if len(parts) == 1:
            return f"{entity} is {_article(parts[0])} {parts[0]}-type Pokémon."
        return f"{entity} has two types, {parts[0]} and {parts[1]}."
    if fact.kind == CATCH_RATE:
        value, percent = fact.values
        return f"{entity} has a catch rate of {value}, which equals {percent}%."
    if fact.kind == EGG_GROUP:
        groups = fact.values
        if len(groups) == 1:
            return f"{entity} belongs to the {groups[0]} egg group."
        return f"{entity} belongs to the {groups[0]} and {groups[1]} egg groups."
    if fact.kind == EVOLUTION:
        target, level = fact.values[0], fact.values[1]
        return f"{entity} evolves into {target} at level {level}."
    raise ValueError(f"unknown fact kind: {fact.kind}")


def render_answer(fact: Fact) -> str:
    entity = fact.entity
    if fact.kind == TYPE:
        parts = _type_parts(fact)
        if len(parts) == 1:
            return f"{entity} is {_article(parts[0])} {parts[0]}-type Pokémon"
        return f"{entity} has two types, {parts[0]} and {parts[1]}"
    if fact.kind == CATCH_RATE:
        value, percent = fact.values
        return f"{entity} has a catch rate of {value} ({percent}%)"
    if fact.kind == EGG_GROUP:
        groups = fact.values
        if len(groups) == 1:
            return f"{entity} belongs to the {groups[0]} egg group"
        return f"{entity} belongs to the {groups[0]} and {groups[1]} egg groups"
    if fact.kind == EVOLUTION:
        target, level = fact.values[0], fact.values[1]
        sentence = f"{entity} evolves into {target} at level {level}"
        if len(fact.values) >= 4:
            sentence += f", and that form evolves into {fact.values[2]} at level {fact.values[3]}"
        return sentence
    raise ValueError(f"unknown fact kind: {fact.kind}")


def render(fact: Fact, index: int = 0, answer: str | None = None) -> Rendered:
    """Render one item. ``answer`` overrides the derived answer (used for holdout)."""
    templates = QUESTIONS[fact.kind]
    question = templates[index % len(templates)].format(entity=fact.entity)
    return Rendered(
        question=question,
        evidence=render_evidence(fact),
        answer=answer if answer is not None else render_answer(fact),
    )
