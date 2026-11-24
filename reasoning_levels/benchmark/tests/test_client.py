from reasoning_levels.benchmark.client import completion_to_telemetry, estimate_tokens
from reasoning_levels.benchmark.scoring import THINK_CLOSE, THINK_OPEN


def test_telemetry_with_inline_thinking():
    data = {
        "choices": [
            {
                "message": {"content": f"{THINK_OPEN}reason{THINK_CLOSE}\n\nfinal"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }
    telemetry = completion_to_telemetry(data, tokenizer=None)
    assert telemetry["reasoning"] == "reason"
    assert telemetry["content"] == "final"
    assert telemetry["prompt_tokens"] == 11
    assert telemetry["completion_tokens"] == 7
    assert telemetry["finish_reason"] == "stop"
    assert telemetry["reasoning_tokens_estimate"] is None


def test_telemetry_with_reasoning_content_field():
    data = {
        "choices": [
            {
                "message": {"content": "final", "reasoning_content": "reason"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 9},
    }
    telemetry = completion_to_telemetry(data, tokenizer=None)
    assert telemetry["reasoning"] == "reason"
    assert telemetry["content"] == "final"
    assert telemetry["finish_reason"] == "length"


def test_telemetry_with_truncated_thinking():
    data = {
        "choices": [
            {
                "message": {"content": f"{THINK_OPEN}reasoning cut off mid-thought"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 50},
    }
    telemetry = completion_to_telemetry(data, tokenizer=None)
    assert telemetry["reasoning"] == "reasoning cut off mid-thought"
    assert telemetry["content"] == ""
    assert telemetry["finish_reason"] == "length"


def test_telemetry_without_usage():
    data = {
        "choices": [{"message": {"content": "final"}, "finish_reason": "stop"}],
    }
    telemetry = completion_to_telemetry(data, tokenizer=None)
    assert telemetry["prompt_tokens"] is None
    assert telemetry["completion_tokens"] is None


def test_telemetry_without_any_marker_is_all_reasoning():
    data = {
        "choices": [
            {"message": {"content": "no markers here"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
    }
    telemetry = completion_to_telemetry(data, tokenizer=None)
    assert telemetry["reasoning"] == "no markers here"
    assert telemetry["content"] == ""


def test_estimate_tokens_with_fake_tokenizer():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    assert estimate_tokens(FakeTokenizer(), "a b c") == 3
    assert estimate_tokens(None, "a b c") is None
    assert estimate_tokens(FakeTokenizer(), "") is None

from reasoning_levels.benchmark import client as client_module
from reasoning_levels.benchmark.client import build_payload, generate_one


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class RecordingClient:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = 0
        self.payloads = []

    def post(self, path, json):
        self.calls += 1
        self.payloads.append(json)
        if self.calls <= self.failures:
            raise RuntimeError("transient")
        return FakeResponse(
            {
                "choices": [
                    {"message": {"content": "final"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }
        )


def test_build_payload_adds_effort_marker_only_for_explicit_efforts():
    payload = build_payload(
        model="m",
        prompt="p",
        stops=["s"],
        effort="high",
        max_tokens=32,
        temperature=0.0,
        seed=0,
    )
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert payload["max_tokens"] == 32
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0


def test_build_payload_omits_marker_for_none_and_auto():
    for effort in ("none", "auto"):
        payload = build_payload(
            model="m",
            prompt="p",
            stops=["s"],
            effort=effort,
            max_tokens=32,
            temperature=0.0,
            seed=0,
        )
        assert "chat_template_kwargs" not in payload


def test_generate_one_returns_data_and_latency():
    fake = RecordingClient()
    data, latency_ms = generate_one(
        fake,
        model="m",
        prompt="p",
        stops=["s"],
        effort="none",
        max_tokens=32,
        temperature=0.0,
        seed=0,
    )
    assert data["choices"][0]["finish_reason"] == "stop"
    assert isinstance(latency_ms, int)
    assert fake.calls == 1


def test_generate_one_retries_and_succeeds(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)
    fake = RecordingClient(failures=2)
    data, _ = generate_one(
        fake,
        model="m",
        prompt="p",
        stops=["s"],
        effort="none",
        max_tokens=32,
        temperature=0.0,
        seed=0,
    )
    assert fake.calls == 3
    assert data["usage"]["completion_tokens"] == 2


def test_generate_one_raises_after_retries_exhausted(monkeypatch):
    import pytest

    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)
    fake = RecordingClient(failures=99)
    with pytest.raises(RuntimeError):
        generate_one(
            fake,
            model="m",
            prompt="p",
            stops=["s"],
            effort="none",
            max_tokens=32,
            temperature=0.0,
            seed=0,
            retries=2,
        )
