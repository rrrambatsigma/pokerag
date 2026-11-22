"""QLoRA helper functions that do not need a GPU."""

from __future__ import annotations

import pytest

from pokerag.sft.train import encode_example, mask_labels


class FakeTokenizer:
    """Length-only tokenizer: ids are 1..n, so prefixes stay prefixes."""

    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(1, len(text) + 1))}


MESSAGES = [
    {"role": "system", "content": "rules"},
    {"role": "user", "content": "EVIDENCE ... QUESTION"},
    {"role": "assistant", "content": '{"answer": "x"}'},
]


def test_mask_labels_masks_the_prompt_only():
    labels = mask_labels([10, 11, 12], [10, 11, 12, 13, 14])
    assert labels == [-100, -100, -100, 13, 14]


def test_mask_labels_rejects_a_mismatched_prefix():
    with pytest.raises(ValueError):
        mask_labels([10, 11], [9, 10, 11, 12])


def test_encode_example_masks_system_and_user():
    ids, labels = encode_example(FakeTokenizer(), MESSAGES, max_seq=1024)
    assert len(ids) == len(labels)
    assert labels[0] == -100
    assert labels[-1] != -100
    assert sum(label != -100 for label in labels) > 0


def test_encode_example_truncates_but_keeps_the_completion():
    ids, labels = encode_example(FakeTokenizer(), MESSAGES, max_seq=50)
    assert len(ids) == 50
    assert len(labels) == 50
    assert any(label != -100 for label in labels)


def test_encode_example_fails_when_truncation_eats_the_prompt():
    with pytest.raises(ValueError):
        encode_example(FakeTokenizer(), MESSAGES, max_seq=20)
