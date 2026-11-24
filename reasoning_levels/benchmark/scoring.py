"""Scoring helpers for the public benchmark battery.

``last_boxed_only_string`` and ``remove_boxed`` are adapted from
lm-evaluation-harness ``lm_eval/tasks/minerva_math/utils.py`` (MIT license).

Thinking markers come from ``reasoning_levels.chat_template`` so the benchmark
splits generations with the exact tokens the Qwen3.6 template was trained with.
"""

from __future__ import annotations

import re
import threading

from reasoning_levels.chat_template import THINKING_CLOSE, THINKING_OPEN

THINK_OPEN = THINKING_OPEN.strip()
THINK_CLOSE = THINKING_CLOSE.strip()
_BOXED_LETTER = re.compile(r"\\boxed\s*\{\s*([A-Da-d])\s*\}")
_PAREN_LETTER = re.compile(r"\(([A-Da-d])\)")
_MATH_CANDIDATE = re.compile(
    r"^(?:\\[a-zA-Z]+|[0-9+\-\*/(),.:;\\{}\[\]\s^_=|!<>\u00b7\u03c0\u221a])+$"
)
_MATH_CANDIDATE_MAX = 80
_MATH_VERIFY_LOCK = threading.Lock()


def split_thinking(generation: str) -> tuple[str, str]:
    """Return ``(reasoning, final_content)`` from a raw generation.

    When the close marker is absent the whole text is reasoning and the final
    content is empty; callers treat an empty content as a missing answer.
    """
    text = generation or ""
    if THINK_CLOSE in text:
        reasoning, content = text.rsplit(THINK_CLOSE, 1)
    else:
        reasoning, content = text, ""
    reasoning = reasoning.removeprefix(THINK_OPEN).strip()
    return reasoning, content.strip()


def last_boxed_only_string(text: str) -> str | None:
    """Return the last ``\\boxed{...}`` (or ``\\fbox{...}``) span, if any."""
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Return the payload of a ``\\boxed{...}`` string."""
    if "\\boxed " in s:
        left = "\\boxed "
        if not s.startswith(left):
            raise ValueError(f"boxed content malformed: {s!r}")
        return s[len(left):]
    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        raise ValueError(f"boxed content malformed: {s!r}")
    return s[len(left):-1]


def _is_math_candidate(text: str) -> bool:
    if not text or len(text) > _MATH_CANDIDATE_MAX:
        return False
    return bool(_MATH_CANDIDATE.fullmatch(text))


def extract_math_answer(content: str) -> str | None:
    """Extract the final answer from the model's non-thinking content.

    Preference order: the last ``\\boxed{}`` payload, a short numeric-looking
    dollar segment, then the last number in the text. The last-number fallback
    exists because unboxed prose answers are common; it is a heuristic and is
    applied identically to every condition.
    """
    boxed = last_boxed_only_string(content)
    if boxed is not None:
        try:
            return remove_boxed(boxed).strip()
        except ValueError:
            pass
    indices = [position for position, char in enumerate(content) if char == "$"]
    if len(indices) >= 2:
        candidate = content[indices[0] + 1 : indices[-1]].strip()
        if _is_math_candidate(candidate):
            return candidate
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", content)
    if numbers:
        return numbers[-1]
    return None


def gpqa_letter(text: str) -> str | None:
    """Extract an (A)-(D) letter from boxed spans or a final parenthesized choice."""
    boxed = _BOXED_LETTER.findall(text)
    if boxed:
        return boxed[-1].upper()
    parenthesized = _PAREN_LETTER.findall(text)
    if parenthesized:
        return parenthesized[-1].upper()
    return None


def math_verify_match(gold_text: str | None, candidate_text: str) -> bool | None:
    """Symbolic verification through ``math_verify``.

    Returns ``None`` when the package is unavailable so callers can record the
    metric as missing instead of failing. ``math_verify`` parsing is not
    thread-safe, so calls are serialized with a module-level lock.
    """
    if not gold_text or not candidate_text:
        return None
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    with _MATH_VERIFY_LOCK:
        try:
            return bool(verify(gold=parse(gold_text), target=parse(candidate_text)))
        except Exception:
            return False


def _numeric_equivalence(predicted: str | None, expected: str | None) -> bool:
    from reasoning_levels.boxed import answers_match

    return answers_match(predicted, expected)


def score_math(row: dict, content: str) -> dict:
    """Score an AIME 2025 or MATH-500 sample from its final content."""
    expected = row.get("answer")
    predicted = extract_math_answer(content)
    gold_text = row.get("solution") or row.get("answer")
    return {
        "answer_extracted": predicted,
        "expected": expected,
        "exact_match": _numeric_equivalence(predicted, expected),
        "math_verify": math_verify_match(gold_text, content) if gold_text else None,
    }


def score_gpqa(row: dict, content: str) -> dict:
    """Score a GPQA-Diamond sample from its final content."""
    expected = row.get("answer")
    predicted = gpqa_letter(content)
    return {
        "answer_extracted": predicted,
        "expected": expected,
        "exact_match": bool(expected) and predicted == expected,
        "math_verify": None,
    }
