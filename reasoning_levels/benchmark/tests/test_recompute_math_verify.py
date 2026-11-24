import json
import sys
import types

from reasoning_levels.benchmark import recompute_math_verify as module


def test_recompute_task_updates_stored_metric(tmp_path, monkeypatch):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *args, **kwargs: [
        {"problem": "Q?", "solution": "The answer is \\boxed{7}."}
    ]
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(module, "math_verify_match", lambda gold, candidate: True)

    raw = tmp_path / "sherry-high__math500.jsonl"
    raw.write_text(
        json.dumps(
            {
                "problem_id": "math500-0000",
                "math_verify": False,
                "generation_text": "answer",
                "correct": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    total, changed = module.recompute_task(raw, "math500")

    assert (total, changed) == (1, 1)
    stored = json.loads(raw.read_text(encoding="utf-8").splitlines()[0])
    assert stored["math_verify"] is True
    assert stored["correct"] is True


def test_recompute_task_rejects_unknown_task(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        module.recompute_task(tmp_path / "raw.jsonl", "nope")
