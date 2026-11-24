from reasoning_levels.grpo_reward import (
    EFFORT_REWARD_TARGETS,
    GLOBAL_GRPO_MAX_NEW_TOKENS,
    make_reasoning_effort_reward,
    score_completion,
)


class Tokenizer:
    eos_token_id = 999

    def __call__(self, text, add_special_tokens=False):
        if text == "\n</think>\n\n":
            return {"input_ids": [98, 97]}
        return {"input_ids": []}

    def decode(self, token_ids, skip_special_tokens=False):
        pieces = {
            1: "r",
            2: "e",
            5: "\\boxed{42}",
            6: "\\boxed{41}",
            7: "\\boxed{0}",
            8: "therefore the maximum possible answer is 41",
            9: "therefore the answer is 42",
            10: "2019 43 2025",
            98: "\n",
            97: "</think>\n\n",
        }
        return "".join(pieces[token_id] for token_id in token_ids)


def completion_ids(thinking=(1, 2), response=(5,)):
    return [*thinking, 98, 97, *response]


def test_reward_targets_are_fixed_by_effort():
    assert EFFORT_REWARD_TARGETS == {"low": 1024, "medium": 4096, "high": 32768}


def test_effort_target_is_not_treated_as_a_physical_generation_cap():
    result = score_completion(
        completion="r" * 1024,
        completion_ids=[1] * 1024,
        ground_truth="42",
        effort="low",
        tokenizer=Tokenizer(),
    )

    assert GLOBAL_GRPO_MAX_NEW_TOKENS == 65536
    assert result["hit_cap_without_eos"] is False


def test_correct_completion_has_dominant_correctness_reward():
    result = score_completion(
        completion="re\n</think>\n\\boxed{42}",
        completion_ids=completion_ids(),
        ground_truth="42",
        effort="medium",
        tokenizer=Tokenizer(),
    )

    assert result["correct"] is True
    assert result["boundary_found"] is True
    assert result["thinking_tokens"] == 2
    assert result["correctness_reward"] == 2.0
    assert result["reward"] > 2.0


def test_wrong_answer_does_not_receive_correctness_reward():
    result = score_completion(
        completion="re\n</think>\n\\boxed{41}",
        completion_ids=completion_ids(response=(6,)),
        ground_truth="42",
        effort="high",
        tokenizer=Tokenizer(),
    )

    assert result["correct"] is False
    assert result["correctness_reward"] == 0.0


def test_wrong_numeric_answer_gets_bounded_proximity_reward():
    near = score_completion(
        completion="re\n</think>\n\\boxed{41}",
        completion_ids=completion_ids(response=(6,)),
        ground_truth="42",
        effort="high",
        tokenizer=Tokenizer(),
    )
    far = score_completion(
        completion="re\n</think>\n\\boxed{0}",
        completion_ids=completion_ids(response=(7,)),
        ground_truth="42",
        effort="high",
        tokenizer=Tokenizer(),
    )

    assert near["correct"] is False
    assert 0.0 < near["partial_correctness_reward"] <= 0.05
    assert near["partial_correctness_reward"] > far["partial_correctness_reward"]
    assert near["correctness_reward"] == 0.0


def test_truncated_numeric_attempt_gets_only_partial_proximity_reward():
    result = score_completion(
        completion="reasoning \\boxed{41}",
        completion_ids=[1, 2, 6],
        ground_truth="42",
        effort="low",
        tokenizer=Tokenizer(),
    )

    assert result["boundary_found"] is False
    assert result["correct"] is False
    assert 0.0 < result["partial_correctness_reward"] <= 0.05
    assert result["correctness_reward"] == 0.0


def test_truncated_cued_numeric_attempt_gets_bounded_partial_reward():
    result = score_completion(
        completion="therefore the maximum possible answer is 41",
        completion_ids=[8],
        ground_truth="42",
        effort="low",
        tokenizer=Tokenizer(),
    )

    assert result["boundary_found"] is False
    assert result["correct"] is False
    assert 0.0 < result["partial_correctness_reward"] <= 0.05
    assert result["correctness_reward"] == 0.0


def test_truncated_cued_exact_number_does_not_receive_exact_credit():
    result = score_completion(
        completion="therefore the answer is 42",
        completion_ids=[9],
        ground_truth="42",
        effort="low",
        tokenizer=Tokenizer(),
    )

    assert result["boundary_found"] is False
    assert result["correct"] is False
    assert result["correctness_reward"] == 0.0
    assert result["partial_correctness_reward"] == 0.0
    assert result["structure_reward"] == -0.20


def test_truncated_bare_numbers_do_not_receive_partial_reward():
    result = score_completion(
        completion="2019 43 2025",
        completion_ids=[10],
        ground_truth="42",
        effort="low",
        tokenizer=Tokenizer(),
    )

    assert result["boundary_found"] is False
    assert result["partial_correctness_reward"] == 0.0
    assert result["reward"] == -0.20


def test_high_has_no_length_penalty_when_correct_and_long():
    result = score_completion(
        completion="re\n</think>\n\\boxed{42}",
        completion_ids=completion_ids(),
        ground_truth="42",
        effort="high",
        tokenizer=Tokenizer(),
    )

    assert result["length_reward"] == 0.0
    assert result["correct"] is True


def test_medium_short_completion_gets_partial_length_reward():
    result = score_completion(
        completion="re\n</think>\n\\boxed{42}",
        completion_ids=completion_ids(),
        ground_truth="42",
        effort="medium",
        tokenizer=Tokenizer(),
    )

    assert 0.0 < result["length_reward"] < 0.20


def test_over_target_length_reward_is_bounded_and_graduated():
    near = score_completion(
        completion="long reasoning",
        completion_ids=[1] * 2000 + [98, 97, 5],
        ground_truth="42",
        effort="low",
        tokenizer=Tokenizer(),
    )
    far = score_completion(
        completion="longer reasoning",
        completion_ids=[1] * 4000 + [98, 97, 5],
        ground_truth="42",
        effort="low",
        tokenizer=Tokenizer(),
    )

    assert -0.20 < near["length_reward"] < 0.20
    assert far["length_reward"] < near["length_reward"]
    assert near["reward"] > far["reward"]


def test_trl_reward_factory_passes_one_scalar_per_completion():
    reward = make_reasoning_effort_reward(Tokenizer())
    values = reward(
        prompts=["p1", "p2"],
        completions=["re\n</think>\n\\boxed{42}", "re\n</think>\n\\boxed{41}"],
        completion_ids=[completion_ids(), completion_ids(response=(6,))],
        ground_truth=["42", "42"],
        effort=["low", "low"],
    )

    assert len(values) == 2
    assert all(isinstance(value, float) for value in values)
    assert values[0] > values[1]
