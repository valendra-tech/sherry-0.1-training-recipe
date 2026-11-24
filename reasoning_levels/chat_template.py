from __future__ import annotations


EFFORT_TOKENS = {
    "low": "<|reasoning_effort_low|>",
    "medium": "<|reasoning_effort_medium|>",
    "high": "<|reasoning_effort_high|>",
}

THINKING_OPEN = "<think>\n"
THINKING_CLOSE = "\n</think>\n\n"


def add_effort_tokens(tokenizer) -> int:
    return tokenizer.add_special_tokens({
        "additional_special_tokens": list(EFFORT_TOKENS.values()),
    })


def format_sft_text(row: dict, tokenizer) -> str:
    """Serialize a verified row using the real Qwen3.6 text-only template."""
    effort = row.get("effort")
    if effort not in EFFORT_TOKENS:
        raise ValueError(f"unknown effort: {effort}")

    thinking = row.get("thinking")
    response = row.get("response")
    if not isinstance(thinking, str) or not thinking.strip():
        raise ValueError("row must contain non-empty thinking text")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("row must contain non-empty response text")

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["problem"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not isinstance(prompt, str) or not prompt.endswith(THINKING_OPEN):
        raise ValueError(
            "Qwen3.6 chat template did not end with the expected thinking opener"
        )
    if not getattr(tokenizer, "eos_token", None):
        raise ValueError("tokenizer must define eos_token")

    assistant_prefix = prompt[:-len(THINKING_OPEN)]
    return (
        assistant_prefix
        + EFFORT_TOKENS[effort]
        + THINKING_OPEN
        + thinking.strip()
        + THINKING_CLOSE
        + response.strip()
        + tokenizer.eos_token
    )
