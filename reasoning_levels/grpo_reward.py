from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

from .boxed import _numeric_value, answers_match, extract_answer
from .chat_template import THINKING_CLOSE


GLOBAL_GRPO_MAX_NEW_TOKENS = 65536
EFFORT_REWARD_TARGETS = {"low": 1024, "medium": 4096, "high": 32768}
CORRECTNESS_REWARD = 2.0
PARTIAL_CORRECTNESS_REWARD = 0.05
STRUCTURE_REWARD = 0.20
REWARD_WEIGHTS = {
    "correctness": CORRECTNESS_REWARD,
    "partial_correctness": PARTIAL_CORRECTNESS_REWARD,
    "structure": STRUCTURE_REWARD,
}


def _as_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(item) for item in value]


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(
            item.get("content", "")
            for item in completion
            if isinstance(item, dict) and isinstance(item.get("content", ""), str)
        )
    return str(completion)


_CUED_NUMERIC_CANDIDATE = re.compile(
    r"(?:\b(?:answer|result|conclusion|therefore|thus|hence)\b|"
    r"\b(?:maximum|minimum)\b)"
    r"[^.!?\n]{0,120}?"
    r"(?P<number>\\frac\{[^{}\n]+\}\{[^{}\n]+\}|"
    r"[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))"
    r"(?![A-Za-z0-9_.])",
    re.IGNORECASE,
)


def _partial_candidate(text: str) -> str | None:
    explicit = extract_answer(text)
    if explicit is not None:
        return explicit
    matches = list(_CUED_NUMERIC_CANDIDATE.finditer(text))
    return matches[-1].group("number") if matches else None


def _tokenize(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    value = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(token_id) for token_id in value]


def _find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(values):
        return None
    for start in range(len(values) - len(pattern) + 1):
        if values[start:start + len(pattern)] == pattern:
            return start
    return None


def _parse_generated_tokens(generated_ids: Iterable[int], tokenizer) -> dict[str, Any]:
    ids = [int(token_id) for token_id in generated_ids]
    close_ids = _tokenize(tokenizer, THINKING_CLOSE)
    boundary = _find_subsequence(ids, close_ids)
    if boundary is not None:
        thinking_ids = ids[:boundary]
        response_ids = ids[boundary + len(close_ids):]
        return {
            "thinking": tokenizer.decode(thinking_ids, skip_special_tokens=False).strip(),
            "response": tokenizer.decode(response_ids, skip_special_tokens=False).strip(),
            "thinking_tokens": len(thinking_ids),
            "total_tokens": len(ids),
            "boundary_found": True,
        }

    text = tokenizer.decode(ids, skip_special_tokens=False)
    if THINKING_CLOSE in text:
        thinking, response = text.split(THINKING_CLOSE, 1)
        thinking = thinking.strip()
        response = response.strip()
        thinking_tokens = len(_tokenize(tokenizer, thinking)) if thinking else 0
    else:
        thinking = text.strip()
        response = ""
        thinking_tokens = len(ids)
    return {
        "thinking": thinking,
        "response": response,
        "thinking_tokens": thinking_tokens,
        "total_tokens": len(ids),
        "boundary_found": False,
    }


def _eos_ids(tokenizer) -> set[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(item) for item in eos_token_id}
    return set()


def _length_reward(
    effort: str,
    thinking_tokens: int,
    boundary_found: bool,
    hit_cap_without_eos: bool,
) -> float:
    if effort == "high" or not boundary_found or thinking_tokens <= 0 or hit_cap_without_eos:
        return 0.0
    if effort == "low" and thinking_tokens <= EFFORT_REWARD_TARGETS[effort]:
        return STRUCTURE_REWARD
    if thinking_tokens <= 1024:
        return STRUCTURE_REWARD * thinking_tokens / 1024.0
    if thinking_tokens <= EFFORT_REWARD_TARGETS[effort]:
        return STRUCTURE_REWARD
    # Grade over-target spans so the structure bonus cannot hide length drift.
    overrun = thinking_tokens - EFFORT_REWARD_TARGETS[effort]
    return STRUCTURE_REWARD * (1.0 - 2.0 * overrun / thinking_tokens)


def _partial_correctness_reward(
    predicted: str | None, ground_truth: str | None
) -> float:
    if predicted is None or ground_truth is None or answers_match(predicted, ground_truth):
        return 0.0

    predicted_value = _numeric_value(predicted)
    ground_truth_value = _numeric_value(ground_truth)
    if predicted_value is None or ground_truth_value is None:
        return 0.0
    if not math.isfinite(predicted_value) or not math.isfinite(ground_truth_value):
        return 0.0

    relative_error = abs(predicted_value - ground_truth_value) / (
        abs(ground_truth_value) + 1.0
    )
    return PARTIAL_CORRECTNESS_REWARD / (1.0 + relative_error)


def score_completion(
    completion: Any,
    completion_ids: Sequence[int] | None,
    ground_truth: str,
    effort: str,
    tokenizer,
    max_new_tokens: int = GLOBAL_GRPO_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    if effort not in EFFORT_REWARD_TARGETS:
        raise ValueError(f"unknown effort: {effort}")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    text = _completion_text(completion)
    ids = _as_ids(completion_ids) if completion_ids is not None else None
    if ids is None:
        ids = _as_ids(tokenizer(text, add_special_tokens=False)["input_ids"])

    parsed = _parse_generated_tokens(ids, tokenizer)
    predicted = extract_answer(parsed["response"])
    correct = answers_match(predicted, ground_truth)
    structurally_valid = bool(parsed["boundary_found"] and predicted is not None)
    hit_cap_without_eos = len(ids) >= max_new_tokens and (
        not ids or ids[-1] not in _eos_ids(tokenizer)
    )
    correctness_reward = CORRECTNESS_REWARD if correct else 0.0
    partial_candidate = predicted or _partial_candidate(parsed["thinking"])
    partial_correctness_reward = _partial_correctness_reward(
        partial_candidate, ground_truth
    )
    structure_reward = STRUCTURE_REWARD if structurally_valid else -STRUCTURE_REWARD
    length_reward = _length_reward(
        effort,
        parsed["thinking_tokens"],
        parsed["boundary_found"],
        hit_cap_without_eos,
    )
    return {
        "reward": float(
            correctness_reward
            + partial_correctness_reward
            + structure_reward
            + length_reward
        ),
        "correct": correct,
        "predicted_answer": predicted,
        "boundary_found": parsed["boundary_found"],
        "thinking_tokens": parsed["thinking_tokens"],
        "total_tokens": parsed["total_tokens"],
        "correctness_reward": correctness_reward,
        "partial_correctness_reward": partial_correctness_reward,
        "structure_reward": structure_reward,
        "length_reward": length_reward,
        "hit_cap_without_eos": hit_cap_without_eos,
    }


def make_reasoning_effort_reward(
    tokenizer, max_new_tokens: int = GLOBAL_GRPO_MAX_NEW_TOKENS
):
    def reasoning_effort_reward(
        prompts: Iterable[str],
        completions: Iterable[Any],
        completion_ids: Iterable[Sequence[int]] | None = None,
        ground_truth: Iterable[str] = (),
        effort: Iterable[str] = (),
        **kwargs,
    ) -> list[float]:
        del prompts, kwargs
        completions = list(completions)
        truths = list(ground_truth)
        efforts = list(effort)
        ids = list(completion_ids) if completion_ids is not None else [None] * len(completions)
        if len(truths) != len(completions) or len(efforts) != len(completions):
            raise ValueError("reward metadata must have one ground truth and effort per completion")
        if len(ids) != len(completions):
            raise ValueError("completion_ids must have one entry per completion")
        return [
            float(
                score_completion(
                    completion,
                    row_ids,
                    truth,
                    level,
                    tokenizer,
                    max_new_tokens=max_new_tokens,
                )["reward"]
            )
            for completion, row_ids, truth, level in zip(completions, ids, truths, efforts)
        ]

    reasoning_effort_reward.__name__ = "reasoning_effort_reward"
    return reasoning_effort_reward
