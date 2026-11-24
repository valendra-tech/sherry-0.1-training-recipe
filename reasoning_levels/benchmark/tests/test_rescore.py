import json
import sys
import types

from reasoning_levels.benchmark import rescore as module
from reasoning_levels.benchmark.scoring import THINK_CLOSE, THINK_OPEN, split_thinking


def _write(path, record):
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_rescore_moves_unclosed_thinking_out_of_content(tmp_path, monkeypatch):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *args, **kwargs: [
        {"problem": "Q?", "solution": "The answer is \\boxed{70}."}
    ]
    monkeypatch.setitem(sys.modules, "datasets", fake)

    raw = tmp_path / "sherry-high__math500.jsonl"
    _write(
        raw,
        {
            "condition": "sherry-high",
            "task": "math500",
            "model": "m",
            "effort": "high",
            "problem_id": "math500-0000",
            "dataset_revision": "rev",
            "prompt_tokens": 3,
            "completion_tokens": 50,
            "reasoning_tokens_estimate": None,
            "finish_reason": "length",
            "latency_ms": 1,
            "generation_text": "thinking without a close marker",
            "reasoning_text": "",
            "answer_extracted": "70",
            "expected": "70",
            "exact_match": True,
            "math_verify": None,
            "correct": True,
        },
    )

    total, changed = module.rescore_task(raw, "math500")

    assert (total, changed) == (1, 1)
    stored = json.loads(raw.read_text(encoding="utf-8").splitlines()[0])
    assert stored["generation_text"] == ""
    assert stored["reasoning_text"] == "thinking without a close marker"
    assert stored["correct"] is False
    assert stored["exact_match"] is False


def test_rescore_preserves_closed_thinking_and_answer(tmp_path, monkeypatch):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *args, **kwargs: [
        {"problem": "Q?", "solution": "The answer is \\boxed{70}."}
    ]
    monkeypatch.setitem(sys.modules, "datasets", fake)

    generation = f"{THINK_OPEN}reasoning{THINK_CLOSE}\n\nThe answer is \\boxed{{70}}."
    reasoning, content = split_thinking(generation)
    raw = tmp_path / "sherry-high__math500.jsonl"
    _write(
        raw,
        {
            "condition": "sherry-high",
            "task": "math500",
            "model": "m",
            "effort": "high",
            "problem_id": "math500-0000",
            "dataset_revision": "rev",
            "prompt_tokens": 3,
            "completion_tokens": 50,
            "reasoning_tokens_estimate": None,
            "finish_reason": "stop",
            "latency_ms": 1,
            "generation_text": content,
            "reasoning_text": reasoning,
            "answer_extracted": "70",
            "expected": "70",
            "exact_match": True,
            "math_verify": None,
            "correct": True,
        },
    )

    total, changed = module.rescore_task(raw, "math500")

    assert (total, changed) == (1, 0)
    stored = json.loads(raw.read_text(encoding="utf-8").splitlines()[0])
    assert stored["correct"] is True
    assert stored["generation_text"] == "The answer is \\boxed{70}."
