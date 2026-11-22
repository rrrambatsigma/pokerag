"""Fact extraction for B5 supervised data.

The corpus is CC BY-NC-SA 2.5 with ``allow_ft_training = false``, so training
data may not contain corpus sentences. Facts are not protected expression: what
this module pulls out is a value (type list, catch rate, egg group list,
evolution level) and :mod:`pokerag.sft.render` writes the training sentence from
that value in our own words.

Extraction is deterministic. :func:`verify` re-extracts from the source chunks
and compares, so an item that no longer matches its snapshot fails loudly
instead of silently drifting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

TYPE = "type"
CATCH_RATE = "catch_rate"
EGG_GROUP = "egg_group"
EVOLUTION = "evolution"

KINDS: tuple[str, ...] = (TYPE, CATCH_RATE, EGG_GROUP, EVOLUTION)

#: Lead sentence, e.g. "is a dual-type Grass / Poison Pokémon" (no "-type" suffix).
DUAL_TYPE_RE = re.compile(r"is a dual-type ([A-Z][a-z]+) / ([A-Z][a-z]+) Pok")
#: Lead sentence, e.g. "is a Fire-type Pokémon".
SINGLE_TYPE_RE = re.compile(r"is an? ([A-Z][a-z]+)-type Pok")
#: Infobox cell, e.g. "Catch rate 45 (11.9%)".
CATCH_RE = re.compile(r"Catch rate (\d+) \((\d+(?:\.\d+)?)%\)")
#: Infobox cell, e.g. "Egg Groups Monster and Grass | Hatch time ...".
EGG_RE = re.compile(r"Egg Groups? ([A-Za-z0-9 ]+?) \|")
#: Lead sentence, e.g. "evolves into Ivysaur starting at level 16".
EVO_RE = re.compile(r"evolves into ([A-Z][A-Za-z0-9♀♂.'’\- ]*?) starting at level (\d+)")

Chunk = tuple[Sequence[str], str]


@dataclass(frozen=True)
class Fact:
    """A single value extracted from one document, with its provenance."""

    kind: str
    entity: str
    doc_id: str
    revid: int
    locator: str
    values: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.doc_id}:{self.kind}"

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "entity": self.entity,
            "doc_id": self.doc_id,
            "revid": self.revid,
            "locator": self.locator,
            "values": list(self.values),
        }


def entity_from_title(title: str) -> str:
    """``"Bulbasaur (Pokémon)"`` -> ``"Bulbasaur"``."""
    return title.replace(" (Pokémon)", "").strip()


def _infobox(chunks: Iterable[Chunk]) -> str:
    return "\n".join(text for path, text in chunks if list(path) == ["Infobox"])


def _lead(chunks: Iterable[Chunk]) -> str:
    return "\n".join(text for path, text in chunks if not list(path))


def extract(entity: str, doc_id: str, revid: int, chunks: Sequence[Chunk]) -> list[Fact]:
    """All facts this document supports, in a stable order."""
    lead = _lead(chunks)
    box = _infobox(chunks)
    facts: list[Fact] = []

    match = DUAL_TYPE_RE.search(lead)
    if match:
        facts.append(Fact(TYPE, entity, doc_id, revid, "Lead", (match.group(1), match.group(2))))
    else:
        match = SINGLE_TYPE_RE.search(lead)
        if match:
            facts.append(Fact(TYPE, entity, doc_id, revid, "Lead", (match.group(1),)))

    match = CATCH_RE.search(box)
    if match:
        facts.append(
            Fact(CATCH_RATE, entity, doc_id, revid, "Infobox", (match.group(1), match.group(2)))
        )

    match = EGG_RE.search(box)
    if match:
        groups = tuple(part.strip() for part in match.group(1).split(" and ") if part.strip())
        if groups:
            facts.append(Fact(EGG_GROUP, entity, doc_id, revid, "Infobox", groups))

    stages = EVO_RE.findall(lead)
    if stages:
        values: list[str] = []
        for target, level in stages:
            values.extend([target.strip(), level])
        facts.append(Fact(EVOLUTION, entity, doc_id, revid, "Lead", tuple(values)))

    return facts


def verify(fact: Fact, chunks: Sequence[Chunk]) -> Fact | None:
    """Freshly extracted copy when it still agrees with ``fact``, else ``None``."""
    for candidate in extract(fact.entity, fact.doc_id, fact.revid, chunks):
        if candidate.kind == fact.kind and candidate.values == fact.values:
            return candidate
    return None


def find(facts: Iterable[Fact], kind: str) -> Fact | None:
    for fact in facts:
        if fact.kind == kind:
            return fact
    return None
