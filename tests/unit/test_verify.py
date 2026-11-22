"""Verifier parsing and bounded repair, with fake model/verifier."""

from __future__ import annotations

from pokerag.llm.base import Answer, Completion
from pokerag.llm.verify import Verdict, VerifyingGenerator, parse_verdict


def test_parse_verdict_json() -> None:
    verdict = parse_verdict('{"supported": false, "reason": "Slowbro is Water 1, so the answer No is wrong."}')

    assert verdict.supported is False
    assert "Water 1" in verdict.reason


def test_parse_verdict_field_fallback() -> None:
    verdict = parse_verdict('The result is {"supported": true, "reason": "both match"}')

    assert verdict.supported is True


def test_parse_verdict_unparseable_is_conservative() -> None:
    assert parse_verdict("looks fine to me").supported is False


class FakeBase:
    system_text = "system"

    def __init__(self, repair_text: str = '{"answer": "Yes", "citations": [1], "abstain": false}') -> None:
        self.repair_text = repair_text
        self.generate_calls = 0
        self.complete_calls = 0

    def generate(self, question, evidence):
        self.generate_calls += 1
        return Answer(
            text="No",
            cited_chunk_ids=("c1",),
            abstained=False,
            prompt_tokens=10,
            completion_tokens=5,
        )

    def complete(self, messages, max_new_tokens=None):
        self.complete_calls += 1
        return Completion(text=self.repair_text, prompt_tokens=30, completion_tokens=15)


class FakeVerifier:
    def __init__(self, verdicts: list[Verdict]) -> None:
        self.verdicts = iter(verdicts)
        self.calls = 0

    def verify(self, question, answer, evidence):
        self.calls += 1
        verdict = next(self.verdicts)
        return Verdict(
            supported=verdict.supported,
            reason=verdict.reason,
            prompt_tokens=20,
            completion_tokens=10,
        )


def make_generator(verifier: FakeVerifier, base: FakeBase | None = None) -> VerifyingGenerator:
    return VerifyingGenerator(
        base or FakeBase(),
        verifier,
        system_prompt="system",
        repair_template="{evidence}|{question}|{answer}|{reason}",
        max_repairs=1,
    )


def test_supported_answer_is_returned_without_repair() -> None:
    verifier = FakeVerifier([Verdict(True, "ok")])
    generator = make_generator(verifier)

    answer = generator.generate("q", [])

    assert answer.text == "No"
    assert verifier.calls == 1


def test_unsupported_answer_is_repaired_once() -> None:
    verifier = FakeVerifier(
        [Verdict(False, "Slowbro is Water 1"), Verdict(True, "now correct")]
    )
    base = FakeBase()
    generator = make_generator(verifier, base)

    answer = generator.generate("q", [])

    assert answer.text == "Yes"
    assert base.generate_calls == 1
    assert base.complete_calls == 1
    assert verifier.calls == 2
    assert answer.prompt_tokens == 80
    assert answer.completion_tokens == 40
    assert answer.repairs == 1
    assert answer.verified is True


def test_still_unsupported_abstains() -> None:
    verifier = FakeVerifier(
        [Verdict(False, "wrong"), Verdict(False, "still wrong")]
    )
    generator = make_generator(verifier)

    answer = generator.generate("q", [])

    assert answer.abstained is True
    assert answer.text == "Not enough evidence in the corpus."
    assert answer.cited_chunk_ids == ()
    assert verifier.calls == 2
    assert answer.repairs == 1
    assert answer.verified is False
