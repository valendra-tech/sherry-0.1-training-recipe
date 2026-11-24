import json

from reasoning_levels.benchmark.report import (
    BASE_REFERENCE,
    load_records,
    paired_delta_ci,
    percentile,
    summarize,
    wilson_interval,
)


def _record(condition, task, problem_id, correct, tokens=100, finish="stop"):
    return {
        "condition": condition,
        "task": task,
        "problem_id": problem_id,
        "correct": correct,
        "exact_match": correct,
        "math_verify": correct,
        "completion_tokens": tokens,
        "reasoning_tokens_estimate": tokens // 2,
        "finish_reason": finish,
    }


def test_load_records_skips_blank_lines(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps(_record("sherry-high", "aime25", "a", True)) + "\n\n",
        encoding="utf-8",
    )
    assert len(load_records(tmp_path)) == 1


def test_percentile_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.5) == 30.0
    assert percentile(values, 0.9) == 40.0
    assert percentile([], 0.5) is None


def test_wilson_interval_all_correct_is_bounded():
    low, high = wilson_interval(30, 30)
    assert low < 1.0
    assert high >= 0.999
    assert low > 0.85


def test_wilson_interval_zero_correct():
    low, high = wilson_interval(0, 30)
    assert low == 0.0
    assert 0.0 < high < 0.2


def test_wilson_interval_empty_sample():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_summarize_groups_by_task_and_condition():
    records = [
        _record("sherry-high", "aime25", "b", False, tokens=200),
        _record("sherry-high", "aime25", "a", True, tokens=200),
    ]
    summary = summarize(records)
    stats = summary["aime25"]["sherry-high"]
    assert stats["n"] == 2
    assert stats["correct"] == 1
    assert stats["accuracy"] == 0.5
    assert stats["mean_completion_tokens"] == 200.0
    assert stats["truncation_share"] == 0.0
    assert stats["accuracy_ci_low"] < 0.5 < stats["accuracy_ci_high"]


def test_summarize_counts_truncation():
    records = [
        _record("sherry-high", "aime25", "a", True, finish="length"),
        _record("sherry-high", "aime25", "b", True, finish="stop"),
    ]
    summary = summarize(records)
    assert summary["aime25"]["sherry-high"]["truncation_share"] == 0.5


def test_base_reference_values_are_pinned_to_the_model_card():
    gpqa = BASE_REFERENCE["gpqa_diamond"]
    assert gpqa["value"] == 86.0
    assert gpqa["comparable"] is True
    aime = BASE_REFERENCE["aime25"]
    assert aime["value"] is None
    assert aime["comparable"] is False
    assert "AIME 26" in aime["note"]


from reasoning_levels.benchmark.report import build_markdown, build_report


def test_build_markdown_contains_tables_cis_reference_and_caveats():
    records = [
        _record("sherry-high", "aime25", "a", True),
        _record("sherry-high", "aime25", "b", False),
        _record("sherry-high", "math500", "m", True),
    ]
    summary = summarize(records)
    markdown = build_markdown(summary, records)
    assert "# Sherry 35B-A3B 0.1 GRPO Preview at Maximum Reasoning - Benchmark Report" in markdown
    assert "## AIME 2025" in markdown
    assert "## MATH-500" in markdown
    assert "sherry-high" in markdown
    assert "0.500" in markdown
    assert "## Base Reference (Published Model Card, Not Re-Run)" in markdown
    assert "86.0" in markdown
    assert "AIME 26" in markdown
    assert "not re-run" in markdown.lower()
    assert "## Caveats" in markdown


def test_paired_delta_ci_zero_when_identical():
    flags = {f"p{index}": True for index in range(20)}
    assert paired_delta_ci(flags, flags, iterations=500, seed=7) == (0.0, 0.0)


def test_paired_delta_ci_requires_shared_problems():
    import pytest

    with pytest.raises(ValueError):
        paired_delta_ci({"a": True}, {"b": True}, iterations=10, seed=7)


def test_build_markdown_includes_paired_delta_when_base_present():
    records = [
        _record("base", "aime25", "a", True),
        _record("base", "aime25", "b", False),
        _record("sherry-high", "aime25", "a", True),
        _record("sherry-high", "aime25", "b", True),
    ]
    summary = summarize(records)
    markdown = build_markdown(summary, records)
    assert "vs Base" in markdown
    assert "Paired delta vs base on 2 problems" in markdown
    assert "+0.500" in markdown


def test_build_markdown_shows_math_verify_count_and_delta():
    records = [
        _record("base", "aime25", "a", True),
        _record("base", "aime25", "b", False),
        _record("sherry-high", "aime25", "a", True),
        _record("sherry-high", "aime25", "b", True),
    ]
    summary = summarize(records)
    markdown = build_markdown(summary, records)
    assert "1.000 (n=2)" in markdown
    assert "Paired delta vs base on 2 problems (math_verify)" in markdown


def test_build_markdown_skips_math_verify_delta_without_values():
    records = [
        _record("base", "aime25", "a", True),
        _record("sherry-high", "aime25", "a", True),
    ]
    for record in records:
        record["math_verify"] = None
    summary = summarize(records)
    markdown = build_markdown(summary, records)
    assert "(math_verify)" not in markdown


def test_build_report_writes_files(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sherry-high__aime25.jsonl").write_text(
        json.dumps(_record("sherry-high", "aime25", "a", True)) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "report"
    result = build_report(raw_dir, output_dir)
    assert result["records"] == 1
    assert result["base_reference"]["gpqa_diamond"]["value"] == 86.0
    assert result["base_model_re_run"] is False
    assert (output_dir / "report.json").exists()
    assert (output_dir / "REPORT.md").exists()
    assert "AIME 2025" in (output_dir / "REPORT.md").read_text(encoding="utf-8")


def test_build_report_marks_base_model_re_run(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "base__aime25.jsonl").write_text(
        json.dumps(_record("base", "aime25", "a", True)) + "\n",
        encoding="utf-8",
    )
    result = build_report(raw_dir, tmp_path / "report")
    assert result["base_model_re_run"] is True
