"""Leak gates for the B5 dataset.

Three things must hold before an adapter is trained:

1. no training entity is used by the test or dev split,
2. no training question is the test question verbatim,
3. no training string copies the corpus, checked against the provenance
   document with an 8-word shingle and a longest-common-substring limit.

The third gate compares against the provenance document only. That is where a
copy would come from, and cross-document boilerplate is not evidence of
copying. Thresholds live here so a report can quote one number.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NGRAM = 8
MAX_COMMON_CHARS = 50


def normalise(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(WORD.findall(folded))


def word_ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = normalise(text).split()
    if len(words) < n:
        return set()
    return {tuple(words[index : index + n]) for index in range(len(words) - n + 1)}


def longest_common(a: str, b: str) -> int:
    """Length in characters of the longest shared substring."""
    if not a or not b:
        return 0
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    return matcher.find_longest_match(0, len(a), 0, len(b)).size


@dataclass(frozen=True)
class Overlap:
    ngram_hits: tuple[str, ...]
    longest: int
    where: str | None

    @property
    def copied(self) -> bool:
        return bool(self.ngram_hits) or self.longest >= MAX_COMMON_CHARS

    def as_dict(self) -> dict:
        return {"ngram_hits": list(self.ngram_hits), "longest": self.longest, "where": self.where}


def check_text(text: str, chunks: Sequence[tuple[str, str]]) -> Overlap:
    """Compare one training string with the chunks of its provenance document."""
    shingles = word_ngrams(text)
    worst = Overlap((), 0, None)
    for heading, chunk_text in chunks:
        if shingles:
            shared = shingles & word_ngrams(chunk_text)
            if shared:
                label = " > ".join(heading) if heading else "(lead)"
                return Overlap(tuple(" ".join(item) for item in sorted(shared))[:3], NGRAM, label)
        size = longest_common(text, chunk_text)
        if size > worst.longest:
            worst = Overlap((), size, heading or "(lead)")
    return worst


def entity_collisions(items: Iterable[object], forbidden: Iterable[str]) -> list[tuple[str, str]]:
    banned = {normalise(name) for name in forbidden}
    collisions: list[tuple[str, str]] = []
    for item in items:
        entity = getattr(item, "entity", None)
        if entity is not None and normalise(entity) in banned:
            collisions.append((getattr(item, "item_id", "?"), entity))
    return collisions


def question_collisions(items: Iterable[object], questions: Iterable[str]) -> list[tuple[str, str]]:
    known = {normalise(question): question for question in questions}
    collisions: list[tuple[str, str]] = []
    for item in items:
        question = getattr(item, "question", None)
        if question is None:
            continue
        if normalise(question) in known:
            collisions.append((getattr(item, "item_id", "?"), question))
    return collisions
