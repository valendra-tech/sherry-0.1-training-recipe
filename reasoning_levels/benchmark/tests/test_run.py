import json
from pathlib import Path

from reasoning_levels.benchmark.run import (
    build_parser,
    done_problem_ids,
    raw_path,
    record_for,
    score_for_task,
)


def test_parser_defaults():
    args = build_parser().parse_args(
        ["--model", "m", "--condition", "sherry-high", "--output-dir", "/tmp/out"]
    )
    assert args.base_url == "http://127.0.0.1:8000/v1"
    assert args.effort == "none"
    assert args.tasks == "aime25,math500,gpqa_diamond"
    assert args.concurrency == 16
    assert args.max_tokens == 32768
    assert args.temperature == 0.0
    assert args.seed == 0
    assert args.limit == 0
    assert args.tokenizer == ""


def test_done_problem_ids_reads_existing_jsonl(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps({"problem_id": "a"}) + "\n" + json.dumps({"problem_id": "b"}) + "\n",
        encoding="utf-8",
    )
    assert done_problem_ids(path) == {"a", "b"}


def test_done_problem_ids_missing_file(tmp_path):
    assert done_problem_ids(tmp_path / "nope.jsonl") == set()


def test_raw_path_naming(tmp_path):
    assert raw_path(Path("/out"), "sherry-high", "aime25") == Path(
        "/out/raw/sherry-high__aime25.jsonl"
    )


def test_score_for_task_dispatch():
    result = score_for_task("aime25", {"answer": "70", "solution": None}, "\\boxed{70}")
    assert result["exact_match"] is True


def test_record_for_schema():
    row = {"problem_id": "aime25-000", "problem": "p", "solution": None, "answer": "70"}
    telemetry = {
        "reasoning": "thinking",
        "content": "The answer is \\boxed{70}.",
        "prompt_tokens": 20,
        "completion_tokens": 40,
        "finish_reason": "stop",
        "reasoning_tokens_estimate": 10,
    }
    record = record_for(
        task="aime25",
        row=row,
        dataset_revision="rev",
        condition="sherry-high",
        model="m",
        effort="high",
        telemetry=telemetry,
        latency_ms=1234,
    )
    assert record["condition"] == "sherry-high"
    assert record["task"] == "aime25"
    assert record["problem_id"] == "aime25-000"
    assert record["correct"] is True
    assert record["exact_match"] is True
    assert record["completion_tokens"] == 40
    assert len(record["generation_sha256"]) == 64
    assert record["reasoning_text"] == "thinking"
    assert record["generation_text"].endswith("\\boxed{70}.")


def test_done_problem_ids_ignores_partial_trailing_line(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps({"problem_id": "a"}) + "\n" + '{"problem_id": "b"',
        encoding="utf-8",
    )
    assert done_problem_ids(path) == {"a"}


def test_done_problem_ids_ignores_lines_without_problem_id(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps({"problem_id": "a"}) + "\n" + json.dumps({"other": 1}) + "\n",
        encoding="utf-8",
    )
    assert done_problem_ids(path) == {"a"}


from reasoning_levels.benchmark import run as run_module
from reasoning_levels.benchmark.run import run_task
from reasoning_levels.benchmark.scoring import THINK_CLOSE, THINK_OPEN


def _fake_generation(**kwargs):
    return (
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            f"{THINK_OPEN}reason{THINK_CLOSE}\n\n"
                            "The answer is \\boxed{70}."
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        },
        100,
    )


def test_run_task_writes_records_and_skips_cached(tmp_path, monkeypatch):
    calls = []

    def counting_generate(*args, **kwargs):
        calls.append(kwargs["prompt"])
        return _fake_generation(**kwargs)

    monkeypatch.setattr(run_module, "generate_one", counting_generate)
    rows = [
        {"problem_id": "aime25-000", "problem": "p1", "solution": None, "answer": "70"},
        {"problem_id": "aime25-001", "problem": "p2", "solution": None, "answer": "70"},
    ]
    written = run_task(
        client=object(),
        model="m",
        condition="sherry-high",
        effort="high",
        task="aime25",
        rows=rows,
        dataset_revision="rev",
        output_dir=tmp_path,
        concurrency=2,
        max_tokens=32,
        temperature=0.0,
        seed=0,
        tokenizer=None,
    )
    assert written == 2
    assert len(calls) == 2

    written_again = run_task(
        client=object(),
        model="m",
        condition="sherry-high",
        effort="high",
        task="aime25",
        rows=rows,
        dataset_revision="rev",
        output_dir=tmp_path,
        concurrency=2,
        max_tokens=32,
        temperature=0.0,
        seed=0,
        tokenizer=None,
    )
    assert written_again == 0
    assert len(calls) == 2

    path = raw_path(tmp_path, "sherry-high", "aime25")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["problem_id"] for record in records] == ["aime25-000", "aime25-001"]
    assert all(record["correct"] for record in records)
