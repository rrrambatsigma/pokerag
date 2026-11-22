"""Build the B5 supervised dataset from extracted facts.

Layout of one item (``items.jsonl``): a question, one evidence sentence written
by us, the target answer, and provenance (``doc_id`` + ``revid`` + locator) so a
reviewer can confirm the value against the frozen snapshot.

Each item becomes three training examples:

* ``clean``    evidence that contains the answer plus unrelated distractors,
* ``hard``     evidence that contains the answer plus same-kind near misses,
* ``abstain``  only same-kind near misses, so the target is the abstain JSON.

The split is by item, never by example, so a held-out entity never leaks into
training through another variant.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from pokerag.sft.facts import Fact
from pokerag.sft.render import render

VARIANTS: tuple[str, ...] = ("clean", "hard", "abstain")

ABSTAIN_ANSWER = "Not enough evidence in the corpus."


@dataclass(frozen=True)
class Item:
    item_id: str
    entity: str
    kind: str
    question: str
    evidence: str
    answer: str
    doc_id: str
    revid: int
    locator: str
    values: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "entity": self.entity,
            "kind": self.kind,
            "question": self.question,
            "evidence": self.evidence,
            "answer": self.answer,
            "provenance": {
                "doc_id": self.doc_id,
                "revid": self.revid,
                "locator": self.locator,
                "values": list(self.values),
            },
        }


@dataclass(frozen=True)
class Example:
    item_id: str
    variant: str
    question: str
    evidence: tuple[tuple[str, str, str], ...]
    answer: str
    citations: tuple[int, ...]
    abstain: bool

    def target(self) -> dict:
        return {"answer": self.answer, "citations": list(self.citations), "abstain": self.abstain}

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "variant": self.variant,
            "question": self.question,
            "evidence": [
                {"doc_id": doc_id, "heading": heading, "text": text}
                for doc_id, heading, text in self.evidence
            ],
            "target": self.target(),
        }


def _rng(*parts: object) -> random.Random:
    """Deterministic RNG from stable text, independent of PYTHONHASHSEED."""
    seed = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return random.Random(int(seed[:16], 16))


def build_items(
    facts_by_doc: Mapping[str, Mapping[str, Fact]],
    quota: Mapping[str, int],
    seed: int = 7,
    prefix: str = "sft",
) -> list[Item]:
    """Pick at most one fact per document, balanced by kind rather than by name."""
    order = sorted(facts_by_doc)
    _rng(seed, "order").shuffle(order)

    remaining = {kind: int(count) for kind, count in quota.items()}
    items: list[Item] = []
    for doc_id in order:
        if not any(remaining.values()):
            break
        available = [kind for kind in facts_by_doc[doc_id] if remaining.get(kind, 0) > 0]
        if not available:
            continue
        kind = max(available, key=lambda name: (remaining[name], -list(remaining).index(name)))
        fact = facts_by_doc[doc_id][kind]
        rendered = render(fact, index=len(items))
        items.append(
            Item(
                item_id=f"{prefix}{len(items) + 1:04d}",
                entity=fact.entity,
                kind=fact.kind,
                question=rendered.question,
                evidence=rendered.evidence,
                answer=rendered.answer,
                doc_id=fact.doc_id,
                revid=fact.revid,
                locator=fact.locator,
                values=fact.values,
            )
        )
        remaining[kind] -= 1
    return items


def split_items(items: Sequence[Item], val_every: int = 7) -> tuple[list[Item], list[Item]]:
    train = [item for index, item in enumerate(items) if (index + 1) % val_every != 0]
    val = [item for index, item in enumerate(items) if (index + 1) % val_every == 0]
    return train, val


def _evidence(doc_id: str, item: Item) -> tuple[str, str, str]:
    return (doc_id, item.locator, item.evidence)


def build_examples(items: Sequence[Item], seed: int = 7) -> list[Example]:
    pool: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        pool[item.kind].append(item)
    all_items = list(items)

    examples: list[Example] = []
    for item in all_items:
        rng = _rng(seed, item.item_id)
        same = [other for other in pool[item.kind] if other.item_id != item.item_id]
        other = [candidate for candidate in all_items if candidate.kind != item.kind]
        if len(same) < 2 or len(other) < 1:
            continue
        same_pick = rng.sample(same, 2)
        other_pick = rng.sample(other, min(2, len(other)))

        plans: dict[str, list[Item]] = {
            "clean": [item, *other_pick[:2]],
            "hard": [item, *same_pick],
            "abstain": [*same_pick, other_pick[0]],
        }
        for variant, members in plans.items():
            rng.shuffle(members)
            if variant == "abstain":
                examples.append(
                    Example(
                        item_id=item.item_id,
                        variant=variant,
                        question=item.question,
                        evidence=tuple(_evidence(member.doc_id, member) for member in members),
                        answer=ABSTAIN_ANSWER,
                        citations=(),
                        abstain=True,
                    )
                )
                continue
            position = next(
                index for index, member in enumerate(members) if member.item_id == item.item_id
            )
            examples.append(
                Example(
                    item_id=item.item_id,
                    variant=variant,
                    question=item.question,
                    evidence=tuple(_evidence(member.doc_id, member) for member in members),
                    answer=f"{item.answer} [{position + 1}]",
                    citations=(position + 1,),
                    abstain=False,
                )
            )
    return examples


def counts(items: Iterable[Item]) -> dict[str, int]:
    tally: dict[str, int] = defaultdict(int)
    for item in items:
        tally[item.kind] += 1
    return dict(sorted(tally.items()))
