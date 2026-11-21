"""Prompt loading, versioning and rendering.

Evidence is numbered so the model can cite ``[n]`` and the parser can map the
number back to a chunk id. The prompt treats evidence as data, not instructions;
that policy is also what the injection fixtures test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pokerag.core.hashing import sha256_text
from pokerag.retrieval.types import Hit

NO_EVIDENCE = "No evidence retrieved."


@dataclass(frozen=True)
class Prompt:
    path: Path
    text: str
    sha256: str

    @property
    def version(self) -> str:
        return self.sha256[:12]


@dataclass(frozen=True)
class PromptSet:
    system: Prompt
    user: Prompt
    fewshot: Prompt | None = None

    @property
    def version(self) -> str:
        parts = [self.system.text, self.user.text, self.fewshot.text if self.fewshot else ""]
        return sha256_text("\n---\n".join(parts))[:12]


def load_prompt(path: str | Path) -> Prompt:
    prompt_path = Path(path)
    text = prompt_path.read_text(encoding="utf-8")
    return Prompt(path=prompt_path, text=text, sha256=sha256_text(text))


def load_prompt_set(
    system_path: str | Path,
    user_path: str | Path,
    fewshot_path: str | Path | None = None,
) -> PromptSet:
    return PromptSet(
        system=load_prompt(system_path),
        user=load_prompt(user_path),
        fewshot=load_prompt(fewshot_path) if fewshot_path else None,
    )


def render_system(prompt_set: PromptSet) -> str:
    if prompt_set.fewshot is None:
        return prompt_set.system.text
    return prompt_set.system.text.rstrip() + "\n\n" + prompt_set.fewshot.text.strip()


def render_evidence(hits: Sequence[Hit]) -> str:
    if not hits:
        return NO_EVIDENCE
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        heading = " > ".join(hit.heading_path) if hit.heading_path else "(no heading)"
        blocks.append(f"[{index}] {hit.doc_id} | {heading}\n{hit.text.strip()}")
    return "\n\n".join(blocks)


def render_user(prompt_set: PromptSet, question: str, hits: Sequence[Hit]) -> str:
    return (
        prompt_set.user.text.replace("{evidence}", render_evidence(hits))
        .replace("{question}", question.strip())
    )
