"""Claim verification and bounded repair for B4.

The verifier asks the same model a different question: is the answer supported
by the evidence? When it says no, the generator gets one chance to re-read the
evidence with the verifier's reason in view. If it still fails, the system
abstains instead of shipping an unsupported answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pokerag.llm.base import Answer, Completion
from pokerag.llm.prompts import render_evidence
from pokerag.llm.qwen import QwenGenerator, _extract_payload, parse_answer
from pokerag.retrieval.types import Hit

SUPPORTED_FIELD = re.compile(r'"supported"\s*:\s*(true|false)', re.IGNORECASE)


@dataclass(frozen=True)
class Verdict:
    supported: bool
    reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Verifier(Protocol):
    def verify(
        self, question: str, answer: Answer, evidence: Sequence[Hit]
    ) -> Verdict: ...


def parse_verdict(text: str) -> Verdict:
    payload = _extract_payload(text)
    if payload is not None and "supported" in payload:
        supported = payload.get("supported")
        if isinstance(supported, str):
            supported = supported.strip().lower() == "true"
        return Verdict(bool(supported), str(payload.get("reason") or "").strip())

    match = SUPPORTED_FIELD.search(text)
    if match:
        return Verdict(match.group(1).lower() == "true", text.strip()[:200])

    lowered = text.lower()
    if "not supported" in lowered or "unsupported" in lowered:
        return Verdict(False, text.strip()[:200])
    if "supported" in lowered:
        return Verdict(True, text.strip()[:200])
    return Verdict(False, "unparseable verifier output")


class QwenVerifier:
    def __init__(
        self,
        model: QwenGenerator,
        system_prompt: str,
        user_template: str,
    ) -> None:
        self._model = model
        self._system = system_prompt
        self._user = user_template

    def verify(
        self, question: str, answer: Answer, evidence: Sequence[Hit]
    ) -> Verdict:
        user = (
            self._user.replace("{evidence}", render_evidence(evidence))
            .replace("{question}", question)
            .replace("{answer}", answer.text)
        )
        completion = self._model.complete(
            [
                {"role": "system", "content": self._system},
                {"role": "user", "content": user},
            ],
            max_new_tokens=256,
        )
        verdict = parse_verdict(completion.text)
        return Verdict(
            supported=verdict.supported,
            reason=verdict.reason,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
        )


class VerifyingGenerator:
    name = "qwen-b4"

    def __init__(
        self,
        base: QwenGenerator,
        verifier: Verifier,
        *,
        system_prompt: str,
        repair_template: str,
        max_repairs: int = 1,
    ) -> None:
        self._base = base
        self._verifier = verifier
        self._system = system_prompt
        self._repair = repair_template
        self._max_repairs = max_repairs

    def verify(self, question: str, answer: Answer, evidence: Sequence[Hit]) -> Verdict:
        return self._verifier.verify(question, answer, evidence)

    def generate(self, question: str, evidence: Sequence[Hit]) -> Answer:
        answer = self._base.generate(question, evidence)
        prompt_tokens = answer.prompt_tokens or 0
        completion_tokens = answer.completion_tokens or 0

        verdict = self._verifier.verify(question, answer, evidence)
        prompt_tokens += verdict.prompt_tokens or 0
        completion_tokens += verdict.completion_tokens or 0

        repairs = 0
        while not verdict.supported and repairs < self._max_repairs:
            user = (
                self._repair.replace("{evidence}", render_evidence(evidence))
                .replace("{question}", question)
                .replace("{answer}", answer.text)
                .replace("{reason}", verdict.reason)
            )
            completion: Completion = self._base.complete(
                [
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": user},
                ]
            )
            prompt_tokens += completion.prompt_tokens or 0
            completion_tokens += completion.completion_tokens or 0
            answer = parse_answer(completion.text, evidence)
            verdict = self._verifier.verify(question, answer, evidence)
            prompt_tokens += verdict.prompt_tokens or 0
            completion_tokens += verdict.completion_tokens or 0
            repairs += 1

        if not verdict.supported:
            return Answer(
                text="Not enough evidence in the corpus.",
                cited_chunk_ids=(),
                abstained=True,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                raw=answer.raw,
                verified=False,
                repairs=repairs,
            )
        return Answer(
            text=answer.text,
            cited_chunk_ids=answer.cited_chunk_ids,
            abstained=answer.abstained,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=answer.raw,
            verified=True,
            repairs=repairs,
        )
