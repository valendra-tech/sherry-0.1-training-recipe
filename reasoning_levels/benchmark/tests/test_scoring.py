from reasoning_levels.benchmark.scoring import (
    THINK_CLOSE,
    THINK_OPEN,
    extract_math_answer,
    gpqa_letter,
    last_boxed_only_string,
    remove_boxed,
    split_thinking,
)
from reasoning_levels.chat_template import (
    THINKING_CLOSE as CANONICAL_CLOSE,
    THINKING_OPEN as CANONICAL_OPEN,
)


def test_think_markers_match_the_canonical_chat_template():
    assert THINK_OPEN == CANONICAL_OPEN.strip()
    assert THINK_CLOSE == CANONICAL_CLOSE.strip()
    assert THINK_OPEN != THINK_CLOSE


def test_split_thinking_with_close_marker():
    generation = f"{THINK_OPEN}Let me reason.{THINK_CLOSE}\n\nThe answer is 70."
    reasoning, content = split_thinking(generation)
    assert reasoning == "Let me reason."
    assert content == "The answer is 70."


def test_split_thinking_without_close_marker():
    generation = "reasoning out loud but truncated"
    reasoning, content = split_thinking(generation)
    assert reasoning == "reasoning out loud but truncated"
    assert content == ""


def test_split_thinking_uses_last_marker():
    generation = f"a {THINK_CLOSE} b {THINK_CLOSE} c"
    reasoning, content = split_thinking(generation)
    assert reasoning == f"a {THINK_CLOSE} b"
    assert content == "c"


def test_last_boxed_only_string_handles_nested_braces():
    text = "so \\boxed{\\frac{1}{2}} is the value"
    assert last_boxed_only_string(text) == "\\boxed{\\frac{1}{2}}"


def test_extract_math_answer_prefers_boxed():
    text = "The answer is \\boxed{70}."
    assert extract_math_answer(text) == "70"


def test_extract_math_answer_falls_back_to_dollar_segment():
    text = "Therefore the result is $42$ units."
    assert extract_math_answer(text) == "42"


def test_extract_math_answer_returns_none_without_candidate():
    assert extract_math_answer("No final answer here.") is None


def test_extract_math_answer_prefers_short_dollar_segment_over_last_number():
    text = "The final value is $\\frac{1}{2}$ after simplifying 4 terms."
    assert extract_math_answer(text) == "\\frac{1}{2}"


def test_extract_math_answer_rejects_prose_dollar_segment():
    text = "So $C$, $V$, and $S$ satisfy the constraints. The answer is 16."
    assert extract_math_answer(text) == "16"


def test_extract_math_answer_falls_back_to_last_number_in_prose():
    text = (
        "We simplify and get $9 - \\frac{56}{b + 7}$. "
        "The sum of all such integer bases is 70."
    )
    assert extract_math_answer(text) == "70"


def test_extract_math_answer_prose_without_dollars_uses_last_number():
    text = "The total number of ordered pairs is 117, so the answer is 117."
    assert extract_math_answer(text) == "117"


def test_remove_boxed_rejects_malformed():
    import pytest

    with pytest.raises(ValueError):
        remove_boxed("not boxed")


def test_gpqa_letter_from_boxed():
    assert gpqa_letter("The answer is \\boxed{C}.") == "C"


def test_gpqa_letter_from_parenthesized_fallback():
    assert gpqa_letter("The final answer is (D).") == "D"


def test_gpqa_letter_none_when_absent():
    assert gpqa_letter("I am not sure.") is None


from reasoning_levels.benchmark.scoring import score_gpqa, score_math


def test_score_math_aime_exact_match():
    row = {"answer": "70", "solution": None}
    result = score_math(row, "The answer is \\boxed{70}.")
    assert result["answer_extracted"] == "70"
    assert result["expected"] == "70"
    assert result["exact_match"] is True


def test_score_math_aime_wrong_answer():
    row = {"answer": "70", "solution": None}
    result = score_math(row, "The answer is \\boxed{71}.")
    assert result["exact_match"] is False


def test_score_math_uses_numeric_equivalence():
    row = {"answer": "\\frac{1}{2}", "solution": None}
    result = score_math(row, "So the value is \\boxed{0.5}.")
    assert result["exact_match"] is True


def test_score_math_missing_answer():
    row = {"answer": "70", "solution": None}
    result = score_math(row, "I could not solve it.")
    assert result["answer_extracted"] is None
    assert result["exact_match"] is False


def test_score_math_verify_with_math_verify_package():
    import pytest

    pytest.importorskip("math_verify")
    row = {"answer": "42", "solution": "$x = 42$"}
    result = score_math(row, "The answer is \\boxed{42}.")
    assert result["math_verify"] is True


def test_score_math_verify_uses_answer_when_solution_missing(monkeypatch):
    from reasoning_levels.benchmark import scoring

    captured = {}

    def fake_match(gold, candidate):
        captured["gold"] = gold
        captured["candidate"] = candidate
        return True

    monkeypatch.setattr(scoring, "math_verify_match", fake_match)
    row = {"answer": "70", "solution": None}
    result = scoring.score_math(row, "The answer is 70.")
    assert result["math_verify"] is True
    assert captured["gold"] == "70"


def test_score_gpqa_letter_match():
    row = {"answer": "D", "solution": "\\boxed{D}"}
    result = score_gpqa(row, "The final answer is \\boxed{D}.")
    assert result["answer_extracted"] == "D"
    assert result["exact_match"] is True


def test_score_gpqa_wrong_letter():
    row = {"answer": "D", "solution": "\\boxed{D}"}
    result = score_gpqa(row, "The final answer is (B).")
    assert result["exact_match"] is False


def test_score_gpqa_missing_expected_answer():
    row = {"answer": None, "solution": ""}
    result = score_gpqa(row, "The final answer is (A).")
    assert result["expected"] is None
    assert result["exact_match"] is False


def test_math_verify_match_serializes_concurrent_calls(monkeypatch):
    import sys
    import threading
    import time
    import types
    from concurrent.futures import ThreadPoolExecutor

    from reasoning_levels.benchmark import scoring

    fake = types.ModuleType("math_verify")
    state = {"active": 0, "interleaved": False, "lock": threading.Lock()}

    def parse(text):
        return text

    def verify(gold, target):
        with state["lock"]:
            state["active"] += 1
            if state["active"] > 1:
                state["interleaved"] = True
        time.sleep(0.002)
        with state["lock"]:
            state["active"] -= 1
        return True

    fake.parse = parse
    fake.verify = verify
    monkeypatch.setitem(sys.modules, "math_verify", fake)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(lambda index: scoring.math_verify_match("g", f"c{index}"), range(32))
        )

    assert all(result is True for result in results)
    assert state["interleaved"] is False
