"""Qwen2.5-Instruct backend.

The model is pinned by revision and loaded in fp16. Generation is deterministic
for a given seed and temperature; the runner records both. The parser accepts the
JSON contract from the prompt, and falls back to bracket citations when the model
ignores the schema, which small models sometimes do.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from pokerag.llm.base import Answer
from pokerag.llm.config import ModelRef, SamplingRef
from pokerag.llm.prompts import PromptSet, render_system, render_user
from pokerag.retrieval.types import Hit

FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
BRACKET = re.compile(r"\[(\d+)\]")
NOT_ENOUGH = "not enough evidence"

# Small models sometimes forget to escape quotes inside the answer string, which
# makes json.loads fail. Pull the fields out with tolerant patterns instead of
# throwing the whole answer away.
ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"(.*?)"\s*,?\s*"citations"', re.DOTALL)
CITATIONS_FIELD = re.compile(r'"citations"\s*:\s*\[(.*?)\]', re.DOTALL)
ABSTAIN_FIELD = re.compile(r'"abstain"\s*:\s*(true|false)', re.IGNORECASE)


def _extract_payload(text: str) -> dict | None:
    fenced = FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_fields(text: str) -> tuple[str, list[int], bool] | None:
    match = ANSWER_FIELD.search(text)
    if not match:
        return None
    answer = match.group(1).strip()
    citations = CITATIONS_FIELD.search(text)
    numbers = [int(value) for value in re.findall(r"\d+", citations.group(1))] if citations else []
    abstain = ABSTAIN_FIELD.search(text)
    abstained = bool(abstain and abstain.group(1).lower() == "true")
    return answer, numbers, abstained


def _numbers(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    numbers: list[int] = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0:
            numbers.append(number)
    return numbers


def map_citations(numbers: Sequence[int], hits: Sequence[Hit]) -> tuple[str, ...]:
    chunk_ids: list[str] = []
    for number in numbers:
        if 1 <= number <= len(hits):
            chunk_id = hits[number - 1].chunk_id
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)
    return tuple(chunk_ids)


def parse_answer(text: str, hits: Sequence[Hit]) -> Answer:
    cleaned = text.strip()
    payload = _extract_payload(cleaned)

    if payload is not None:
        answer = str(payload.get("answer") or "").strip()
        abstained = bool(payload.get("abstain", False))
        numbers = _numbers(payload.get("citations"))
        if abstained:
            return Answer(
                text=answer or "Not enough evidence in the corpus.",
                cited_chunk_ids=(),
                abstained=True,
                raw=text,
            )
        return Answer(
            text=answer or cleaned,
            cited_chunk_ids=map_citations(numbers, hits),
            abstained=False,
            raw=text,
        )

    fields = _extract_fields(cleaned)
    if fields is not None:
        answer, numbers, abstained = fields
        if not numbers:
            numbers = [int(match) for match in BRACKET.findall(answer)]
        abstained = abstained or (NOT_ENOUGH in answer.lower() and not numbers)
        return Answer(
            text=answer,
            cited_chunk_ids=() if abstained else map_citations(numbers, hits),
            abstained=abstained,
            raw=text,
        )

    numbers = [int(match) for match in BRACKET.findall(cleaned)]
    abstained = NOT_ENOUGH in cleaned.lower() and not numbers
    return Answer(
        text=cleaned,
        cited_chunk_ids=() if abstained else map_citations(numbers, hits),
        abstained=abstained,
        raw=text,
    )


def build_messages(
    prompt_set: PromptSet,
    question: str,
    hits: Sequence[Hit],
    *,
    use_evidence: bool,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": render_system(prompt_set)},
        {
            "role": "user",
            "content": render_user(prompt_set, question, hits if use_evidence else []),
        },
    ]


class QwenGenerator:
    name = "qwen"

    def __init__(
        self,
        model: ModelRef,
        sampling: SamplingRef,
        prompt_set: PromptSet,
        *,
        seed: int | None = None,
        use_evidence: bool = True,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - depends on local install
            raise RuntimeError("generator needs the gpu extra: uv sync --extra gpu") from error

        target = device or model.device
        if target.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA tidak tersedia; jalankan dengan --device cpu untuk uji kecil")

        self._torch = torch
        self._sampling = sampling
        self._prompt_set = prompt_set
        self._seed = seed if seed is not None else sampling.seeds[0]
        self._use_evidence = use_evidence
        self._target = target

        dtype = getattr(torch, model.dtype) if model.dtype else None
        self._tokenizer = AutoTokenizer.from_pretrained(
            model.model_id, revision=model.revision
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model.model_id, revision=model.revision, dtype=dtype
        )
        self._model.to(target)
        self._model.eval()

    def generate(self, question: str, evidence: Sequence[Hit]) -> Answer:
        messages = build_messages(
            self._prompt_set, question, evidence, use_evidence=self._use_evidence
        )
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._target)

        do_sample = self._sampling.temperature > 0
        self._torch.manual_seed(self._seed)
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._sampling.max_new_tokens,
                do_sample=do_sample,
                temperature=self._sampling.temperature if do_sample else None,
                top_p=self._sampling.top_p if do_sample else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion_tokens = int(output.shape[1]) - prompt_tokens
        raw = self._tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)
        answer = parse_answer(raw, evidence if self._use_evidence else [])
        return Answer(
            text=answer.text,
            cited_chunk_ids=answer.cited_chunk_ids,
            abstained=answer.abstained,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=raw,
        )
