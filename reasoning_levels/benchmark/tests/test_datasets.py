from reasoning_levels.benchmark.datasets import (
    TASKS,
    load_task,
    normalize_aime25,
    normalize_gpqa,
    normalize_math500,
    normalize_rows,
)


def test_task_specs_are_pinned():
    for task, spec in TASKS.items():
        if spec["path"].startswith("local/"):
            assert len(spec["revision"]) == 64, task
            assert spec["split"] == "local", task
            continue
        assert len(spec["revision"]) == 40, task
        assert spec["split"] == "test", task


def test_normalize_aime25():
    row = normalize_aime25(3, {"problem": "  What is 1+1?  ", "answer": 2, "id": 7})
    assert row == {
        "problem_id": "aime25-003",
        "problem": "What is 1+1?",
        "solution": None,
        "answer": "2",
    }


def test_normalize_math500_derives_expected_from_solution():
    solution = "We compute it. The final answer is \\boxed{\\frac{3}{4}}."
    row = normalize_math500(11, {"problem": "Compute.", "solution": solution})
    assert row["problem_id"] == "math500-0011"
    assert row["answer"] == "\\frac{3}{4}"


def test_normalize_math500_without_boxed_solution():
    row = normalize_math500(0, {"problem": "Compute.", "solution": "No boxed here."})
    assert row["answer"] is None


def test_normalize_gpqa():
    row = normalize_gpqa(2, {"problem": "Q?", "solution": "The answer is \\boxed{C}."})
    assert row == {
        "problem_id": "gpqa-002",
        "problem": "Q?",
        "solution": "The answer is \\boxed{C}.",
        "answer": "C",
    }


def test_normalize_rows_skips_malformed_rows():
    def normalizer(index, row):
        if "problem" not in row:
            raise KeyError("problem")
        return {"problem_id": str(index), "problem": row["problem"]}

    normalized, skipped = normalize_rows(normalizer, [{"problem": "a"}, {}, {"problem": "b"}])
    assert [row["problem"] for row in normalized] == ["a", "b"]
    assert skipped == 1


def test_load_task_unknown_name_raises():
    import pytest

    with pytest.raises(ValueError):
        load_task("nope")
