from reasoning_levels.sft_data import effort_token, split_by_problem, validate_rows


def rows_for(problem, effort="low", correct=True):
    return {
        "problem": problem,
        "source": "synthetic_math",
        "effort": effort,
        "thinking": "short reasoning",
        "thinking_tokens": 12,
        "response": r"\boxed{42}",
        "predicted_answer": "42",
        "ground_truth": "42",
        "correct": correct,
        "finish_reason": "stop",
    }


def test_effort_token_is_deterministic():
    assert effort_token("low") == "<|reasoning_effort_low|>"
    assert effort_token("medium") == "<|reasoning_effort_medium|>"
    assert effort_token("high") == "<|reasoning_effort_high|>"


def test_validate_rows_requires_three_correct_efforts_per_problem():
    rows = [rows_for("p", effort) for effort in ["low", "medium", "high"]]
    assert validate_rows(rows) == rows
    assert validate_rows(rows[:-1]) == []


def test_validate_rows_rejects_invalid_rows_and_duplicate_efforts():
    invalid_rows = [
        rows_for("p", "low"),
        rows_for("p", "medium"),
        rows_for("p", "high", correct=False),
    ]
    assert validate_rows(invalid_rows) == []

    duplicate_rows = [
        rows_for("p", effort)
        for effort in ["low", "medium", "high", "high"]
    ]
    assert validate_rows(duplicate_rows) == []


def test_split_by_problem_never_leaks_a_problem_between_splits():
    rows = [
        *(rows_for("p1", effort) for effort in ["low", "medium", "high"]),
        *(rows_for("p2", effort) for effort in ["low", "medium", "high"]),
    ]
    train, evaluation = split_by_problem(rows, eval_fraction=0.5, seed=42)
    assert {row["problem"] for row in train}.isdisjoint({row["problem"] for row in evaluation})
    assert len(train) + len(evaluation) == 6
