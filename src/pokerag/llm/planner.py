"""Rule-based query planner for comparison questions.

The benchmark's multi-hop slice asks "Are X and Y both in the G egg group?".
Answering it requires evidence for X and for Y separately, so the planner turns
one question into two retrieval sub-queries. It is deliberately deterministic:
a planner call before retrieval would force the generator to load before the
embedder, which the 6 GB memory budget does not allow.
"""

from __future__ import annotations

import re

COMPARISON = re.compile(r"^Are (.+?) and (.+?) both in the (.+?) egg group\?$", re.IGNORECASE)


def plan_subqueries(question: str) -> list[str]:
    match = COMPARISON.match(question.strip())
    if not match:
        return [question]
    left, right, _group = (part.strip() for part in match.groups())
    return [f"What egg group is {left}?", f"What egg group is {right}?"]
