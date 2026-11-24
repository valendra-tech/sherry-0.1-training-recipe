import pytest

from reasoning_levels.benchmark.prompts import STOPS, doc_to_text


def test_aime25_prompt_matches_lm_eval_format():
    row = {"problem": "  Find the sum of all integer bases.  "}
    assert doc_to_text("aime25", row) == (
        "Question: Find the sum of all integer bases.\nAnswer:"
    )


def test_math500_prompt_matches_minerva_format():
    row = {"problem": "  Compute 1 + 1.  "}
    assert doc_to_text("math500", row) == "Problem:\nCompute 1 + 1.\n\nSolution:"


def test_gpqa_prompt_is_verbatim_dataset_text():
    row = {"problem": "Q?\n\n(A) 1\n(B) 2\n\nWrite \\boxed{A}."}
    assert doc_to_text("gpqa_diamond", row) == row["problem"]


def test_stop_sequences_match_lm_eval():
    assert STOPS["aime25"] == ["Question:", "</s>", "<|im_end|>", "<|eot_id|>"]
    assert STOPS["math500"] == ["Problem:", "</s>", "<|im_end|>"]
    assert STOPS["gpqa_diamond"] == ["Question:", "</s>", "<|im_end|>"]


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        doc_to_text("unknown", {"problem": "x"})
