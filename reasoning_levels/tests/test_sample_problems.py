from reasoning_levels.sample_problems import (
    has_boxed_answer,
    is_numeric_answer,
    is_proof_problem,
    sample_problems,
)


def test_has_boxed_answer():
    assert has_boxed_answer("The answer is \\boxed{42}.") is True
    assert has_boxed_answer("No answer here.") is False


def test_sample_problems_returns_list():
    """Integration test: loads a small slice of NuminaMath-CoT and samples."""
    problems = sample_problems(n=20, seed=42, balance_sources=True)
    assert len(problems) == 20
    ex = problems[0]
    assert "problem" in ex and "ground_truth" in ex and "source" in ex
    assert ex["ground_truth"] is not None


def test_sample_problems_balanced_sources():
    problems = sample_problems(n=100, seed=42, balance_sources=True)
    sources = [p["source"] for p in problems]
    assert len(set(sources)) >= 3


def test_sample_problems_can_limit_reference_solution_length():
    problems = sample_problems(
        n=20,
        seed=42,
        balance_sources=True,
        max_solution_chars=1200,
    )
    assert len(problems) == 20
    assert all(len(problem["solution"]) <= 1200 for problem in problems)


def test_problem_filters_reject_proofs_and_unsupported_values():
    assert is_proof_problem("Prove that the triangle is equilateral.") is True
    assert is_proof_problem("What is the probability of 7 successes?") is False
    assert is_numeric_answer(r"\frac{9}{400}") is True
    assert is_numeric_answer(r"\frac{1}{2}(\sqrt{5}+1)") is False
    assert is_numeric_answer("(-1,1)") is False


def test_sample_problems_excludes_normalized_problem(monkeypatch):
    rows = [
        {"problem": "excluded   problem", "solution": r"\boxed{1}", "source": "synthetic_math"},
        {"problem": "kept problem", "solution": r"\boxed{2}", "source": "synthetic_math"},
    ]
    monkeypatch.setattr(
        "reasoning_levels.sample_problems.load_dataset",
        lambda *args, **kwargs: rows,
    )

    problems = sample_problems(
        n=1,
        seed=42,
        balance_sources=False,
        sources=["synthetic_math"],
        exclude_problems=["  excluded problem  "],
    )

    assert [problem["problem"] for problem in problems] == ["kept problem"]
