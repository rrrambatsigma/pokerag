"""Deduplication: exact hashes plus MinHash near-duplicate detection.

Pokémon articles are highly templated, so near-duplicates are common (for
example a species page and its redirect target, or a page duplicated after a
move). Duplicates are not deleted: the report records which document is
canonical and which ones point to it, so the decision stays auditable.
"""

from __future__ import annotations

import re
from collections import defaultdict

from datasketch import MinHash, MinHashLSH

from pokerag.core.jsonl import read_jsonl, write_json
from pokerag.core.logging import get_logger
from pokerag.ingest.config import CorpusConfig
from pokerag.ingest.models import DuplicateGroup

logger = get_logger("pokerag.ingest.dedup")

WORD_RE = re.compile(r"\w+", re.UNICODE)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def shingles(text: str, size: int) -> set[str]:
    """Return the set of word n-grams of ``text``."""
    words = WORD_RE.findall(text.lower())
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    """Exact Jaccard similarity of two shingle sets."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / len(left | right)


def run_dedup(cfg: CorpusConfig) -> list[DuplicateGroup]:
    """Group documents that are identical or near-identical."""
    snap = cfg.snapshot_dir
    docs = [row for row in read_jsonl(snap / "cleaned.jsonl") if row["n_chars"] > 0]
    candidates = [
        row
        for row in docs
        if not row["is_redirect"] and not row["is_disambiguation"]
    ]
    candidates.sort(key=lambda row: row["doc_id"])

    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in candidates:
        by_hash[row["content_sha256"]].append(row["doc_id"])

    union = _UnionFind()
    exact_groups: list[DuplicateGroup] = []
    for digest, members in by_hash.items():
        if len(members) > 1:
            for member in members[1:]:
                union.union(members[0], member)
            exact_groups.append(
                DuplicateGroup(reason="exact", canonical=members[0], members=sorted(members), max_similarity=1.0)
            )

    # Near-duplicate candidates via LSH, then verified with exact Jaccard so the
    # reported similarity is not an estimate. LSH runs at a slightly looser
    # threshold: it only proposes candidates, the exact comparison decides.
    candidate_threshold = max(0.5, cfg.dedup.threshold - 0.15)
    lsh = MinHashLSH(threshold=candidate_threshold, num_perm=cfg.dedup.num_perm)
    minhashes: dict[str, MinHash] = {}
    shingle_sets: dict[str, set[str]] = {}
    for row in candidates:
        if any(row["doc_id"] in group.members for group in exact_groups):
            continue
        values = shingles(row["text"], cfg.dedup.shingle_size)
        if not values:
            continue
        signature = MinHash(num_perm=cfg.dedup.num_perm)
        for value in values:
            signature.update(value.encode("utf-8"))
        minhashes[row["doc_id"]] = signature
        shingle_sets[row["doc_id"]] = values
        lsh.insert(row["doc_id"], signature)

    near_groups: list[DuplicateGroup] = []
    seen: set[str] = set()
    for doc_id in sorted(minhashes):
        if doc_id in seen:
            continue
        signature = minhashes[doc_id]
        neighbours = sorted(set(lsh.query(signature)) - {doc_id})
        members = [doc_id]
        best = 0.0
        for neighbour in neighbours:
            score = jaccard(shingle_sets[doc_id], shingle_sets[neighbour])
            if score >= cfg.dedup.threshold:
                members.append(neighbour)
                best = max(best, score)
        if len(members) > 1:
            members = sorted(set(members))
            seen.update(members)
            near_groups.append(
                DuplicateGroup(reason="near", canonical=members[0], members=members, max_similarity=round(best, 4))
            )

    groups = sorted(exact_groups + near_groups, key=lambda group: (group.members[0], group.reason))
    sizes = {row["doc_id"]: row["n_chars"] for row in candidates}
    duplicates: dict[str, str] = {}
    for group in groups:
        canonical = max(group.members, key=lambda doc_id: (sizes.get(doc_id, 0), doc_id))
        group.canonical = canonical
        group.members = sorted(group.members)
        for member in group.members:
            if member != canonical:
                duplicates[member] = canonical

    write_json(
        snap / "dedup_report.json",
        {
            "config": cfg.dedup.model_dump(),
            "n_candidates": len(candidates),
            "n_groups": len(groups),
            "n_duplicates": len(duplicates),
            "groups": [group.model_dump() for group in groups],
            "duplicates": duplicates,
        },
    )
    logger.info("dedup: %s groups, %s documents marked as duplicates", len(groups), len(duplicates))
    return groups
