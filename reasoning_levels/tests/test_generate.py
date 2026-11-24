from unittest.mock import patch, MagicMock
from reasoning_levels.generate_dataset import (
    CANDIDATE_SYSTEM_PROMPT,
    CANDIDATE_GENERATION_EFFORT,
    EFFORT_ENABLE_THINKING,
    EFFORT_LEVELS,
    EFFORT_MAX_TOKENS,
    build_messages,
    generate_trace,
    parse_thinking_response,
    problem_batch_size,
    select_effort_traces,
    completed_problem_keys,
)
from reasoning_levels.config import APIConfig, DatasetConfig


def test_build_messages_low():
    msgs = build_messages("What is 2+2?", "low")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "reasoning effort is set to low" in msgs[0]["content"].lower()
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "What is 2+2?"


def test_build_messages_high():
    msgs = build_messages("What is 2+2?", "high")
    assert "reasoning effort is set to xhigh" in msgs[0]["content"].lower()


def test_generate_trace_returns_thinking_and_response():
    """Mock the OpenAI client and verify the trace structure."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "The answer is \\boxed{4}."
    mock_response.choices[0].message.reasoning = "2+2=4, quick."
    mock_client.chat.completions.create.return_value = mock_response

    result = generate_trace(
        client=mock_client,
        model="Qwen3.6-35B-A3B",
        problem="What is 2+2?",
        effort="low",
        max_tokens=512,
    )
    assert "thinking" in result
    assert "response" in result
    assert result["thinking"] == "2+2=4, quick."
    assert result["response"] == "The answer is \\boxed{4}."


def test_effort_uses_thinking_mode_not_different_token_budgets():
    """The token ceiling must not be the mechanism that defines effort."""
    assert EFFORT_ENABLE_THINKING == {
        "low": True,
        "medium": True,
        "high": True,
    }
    assert len(set(EFFORT_MAX_TOKENS.values())) == 1
    assert next(iter(EFFORT_MAX_TOKENS.values())) >= 32768


def test_generate_trace_reports_finish_reason():
    """A length stop must be visible so truncated traces can be diagnosed."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].finish_reason = "length"
    mock_response.choices[0].message.content = ""
    mock_response.choices[0].message.reasoning = "unfinished reasoning"
    mock_client.chat.completions.create.return_value = mock_response

    result = generate_trace(
        client=mock_client,
        model="Qwen3.6-35B-A3B",
        problem="What is 2+2?",
        effort="medium",
        max_tokens=8192,
    )

    assert result["finish_reason"] == "length"


def test_select_effort_traces_uses_observed_thinking_length():
    """Assign labels from actual lengths instead of trusting prompt labels."""
    candidates = [
        {"thinking": "short", "thinking_tokens": 10, "response": "\\boxed{42}", "correct": True, "finish_reason": "stop"},
        {"thinking": "medium", "thinking_tokens": 30, "response": "\\boxed{42}", "correct": True, "finish_reason": "stop"},
        {"thinking": "long", "thinking_tokens": 50, "response": "\\boxed{42}", "correct": True, "finish_reason": "stop"},
    ]

    selected = select_effort_traces(candidates)

    assert [trace["effort"] for trace in selected] == ["low", "medium", "high"]
    assert [trace["thinking_tokens"] for trace in selected] == [10, 30, 50]


def test_select_effort_traces_discards_incomplete_candidates():
    candidates = [
        {"thinking": "short", "thinking_tokens": 10, "response": "\\boxed{42}", "correct": True, "finish_reason": "stop"},
        {"thinking": "cut off", "thinking_tokens": 20, "response": "", "correct": True, "finish_reason": "length"},
        {"thinking": "wrong", "thinking_tokens": 30, "response": "\\boxed{0}", "correct": False, "finish_reason": "stop"},
        {"thinking": "long", "thinking_tokens": 40, "response": "\\boxed{42}", "correct": True, "finish_reason": "stop"},
    ]

    assert select_effort_traces(candidates) == []


def test_dataset_config_has_candidate_pool_and_bounded_parallelism():
    config = DatasetConfig()
    assert config.candidates_per_problem >= 3
    assert config.max_workers >= 1
    assert config.max_solution_chars == 2000


def test_candidate_pool_uses_one_base_prompt():
    assert CANDIDATE_GENERATION_EFFORT == "medium"
    assert CANDIDATE_SYSTEM_PROMPT == "Put your final answer in \\boxed{}."


def test_build_messages_accepts_candidate_prompt_override():
    msgs = build_messages("What is 2+2?", "medium", system_prompt=CANDIDATE_SYSTEM_PROMPT)
    assert msgs[0]["content"] == CANDIDATE_SYSTEM_PROMPT


def test_api_config_can_override_completion_ceiling(monkeypatch):
    monkeypatch.setenv("QWEN_MAX_COMPLETION_TOKENS", "8192")
    config = APIConfig.from_env()
    assert set(config.max_tokens.values()) == {8192}


def test_problem_batch_size_keeps_candidate_requests_bounded():
    assert problem_batch_size(max_workers=32, candidates_per_problem=3) == 10
    assert problem_batch_size(max_workers=8, candidates_per_problem=3) == 2


def test_completed_problem_keys_requires_all_effort_labels():
    results = [
        {"problem": "complete", "effort": "low"},
        {"problem": "complete", "effort": "medium"},
        {"problem": "complete", "effort": "high"},
        {"problem": "partial", "effort": "low"},
        {"problem": "partial", "effort": "medium"},
    ]

    assert completed_problem_keys(results) == {"complete"}


def test_generate_trace_handles_no_reasoning_content():
    """If the API doesn't separate thinking, fall back to parsing from content."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = " Müd\n2+2=4\n\n\n\nThe answer is \\boxed{4}."
    mock_response.choices[0].message.reasoning_content = None
    mock_response.choices[0].message.reasoning = None
    mock_client.chat.completions.create.return_value = mock_response

    result = generate_trace(
        client=mock_client,
        model="Qwen3.6-35B-A3B",
        problem="What is 2+2?",
        effort="low",
        max_tokens=512,
    )
    assert result["thinking"] is not None
    assert "2+2=4" in result["thinking"]
    assert "\\boxed{4}" in result["response"]


def test_parse_thinking_response_separated():
    """When reasoning_content is provided, use it directly."""
    msg = MagicMock()
    msg.reasoning = "My reasoning."
    msg.reasoning_content = None
    msg.content = "The answer is \\boxed{42}."
    thinking, response = parse_thinking_response(msg)
    assert thinking == "My reasoning."
    assert response == "The answer is \\boxed{42}."


def test_parse_thinking_response_reasoning_field():
    """vLLM Qwen3.6 format: reasoning field has thinking, content has response."""
    msg = MagicMock()
    msg.reasoning = "Step 1: multiply. Step 2: add."
    msg.reasoning_content = None
    msg.content = "The answer is \\boxed{391}."
    thinking, response = parse_thinking_response(msg)
    assert thinking == "Step 1: multiply. Step 2: add."
    assert response == "The answer is \\boxed{391}."


def test_parse_thinking_response_inline():
    """When reasoning_content is None, parse from content with Mud format."""
    msg = MagicMock()
    msg.reasoning = None
    msg.reasoning_content = None
    msg.content = " Müd\nStep 1: add.\nStep 2: result.\n\n\n\n\\boxed{7}"
    thinking, response = parse_thinking_response(msg)
    assert "Step 1" in thinking
    assert "\\boxed{7}" in response
