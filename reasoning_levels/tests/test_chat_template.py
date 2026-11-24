import pytest

from reasoning_levels.chat_template import (
    EFFORT_TOKENS,
    add_effort_tokens,
    format_sft_text,
)


class FakeTokenizer:
    eos_token = "<|im_end|>"

    def __init__(self):
        self.added_tokens = []

    def add_special_tokens(self, mapping):
        self.added_tokens.extend(mapping["additional_special_tokens"])
        return len(mapping["additional_special_tokens"])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is True
        return (
            "<|im_start|>user\n"
            f"{messages[0]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )


def test_add_effort_tokens_registers_the_three_control_tokens():
    tokenizer = FakeTokenizer()

    added = add_effort_tokens(tokenizer)

    assert added == 3
    assert tokenizer.added_tokens == list(EFFORT_TOKENS.values())


def test_format_sft_text_places_effort_token_before_thinking():
    tokenizer = FakeTokenizer()
    row = {
        "problem": "What is 2+2?",
        "effort": "high",
        "thinking": "Add two and two.",
        "response": r"\boxed{4}",
    }

    rendered = format_sft_text(row, tokenizer)

    assert "<|im_start|>assistant\n<|reasoning_effort_high|><think>\n" in rendered
    assert "Add two and two.\n</think>\n\n\\boxed{4}<|im_end|>" in rendered


def test_format_sft_text_rejects_unknown_effort():
    with pytest.raises(ValueError, match="unknown effort"):
        format_sft_text(
            {"problem": "x", "effort": "x", "thinking": "t", "response": "r"},
            FakeTokenizer(),
        )
