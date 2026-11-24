from reasoning_levels.chat_template import THINKING_CLOSE
from reasoning_levels.evaluate_sft import (
    DEFAULT_DO_SAMPLE,
    DEFAULT_MAX_NEW_TOKENS,
    aggregate_metrics,
    format_inference_prompt,
    parse_generated_tokens,
)


class FakeTokenizer:
    eos_token_id = 999

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is True
        return "<|im_start|>assistant\n<think>\n"

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        assert text == THINKING_CLOSE
        return {"input_ids": [98, 97]}

    def decode(self, token_ids, skip_special_tokens=False):
        pieces = {
            1: "r",
            2: "e",
            3: "s",
            4: "t",
            5: "l",
            6: "t",
            98: "\n",
            97: "</think>\n\n",
        }
        return "".join(pieces[token_id] for token_id in token_ids)


def test_inference_prompt_inserts_effort_token_before_thinking():
    prompt = format_inference_prompt("What is 2+2?", "high", FakeTokenizer())

    assert prompt == "<|im_start|>assistant\n<|reasoning_effort_high|><think>\n"


def test_parse_generated_tokens_counts_reasoning_tokens_at_boundary():
    parsed = parse_generated_tokens([1, 2, 98, 97, 3, 4], FakeTokenizer())

    assert parsed["thinking"] == "re"
    assert parsed["response"] == "st"
    assert parsed["thinking_tokens"] == 2
    assert parsed["total_tokens"] == 6
    assert parsed["boundary_found"] is True


def test_aggregate_metrics_reports_effort_medians_and_ordering():
    rows = [
        {"effort": "low", "thinking_tokens": 10, "total_tokens": 15, "correct": True},
        {"effort": "medium", "thinking_tokens": 30, "total_tokens": 35, "correct": True},
        {"effort": "high", "thinking_tokens": 50, "total_tokens": 55, "correct": False},
    ]

    metrics = aggregate_metrics(rows)

    assert metrics["low"]["accuracy"] == 1.0
    assert metrics["high"]["accuracy"] == 0.0
    assert metrics["median_thinking_tokens"] == {"low": 10, "medium": 30, "high": 50}
    assert metrics["thinking_tokens_strictly_increasing"] is True


def test_evaluation_uses_one_greedy_completion_ceiling():
    assert DEFAULT_DO_SAMPLE is False
    assert DEFAULT_MAX_NEW_TOKENS == 4096
