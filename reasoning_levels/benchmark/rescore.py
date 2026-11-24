"""Rescore stored raw records after a scoring-policy fix.

Reapplies the thinking split and the task scoring to existing JSONL telemetry,
so historical runs can be corrected without re-running the model. Files are
rewritten atomically and every record's scoring fields are recomputed. When a
tokenizer id is provided, reasoning-token estimates are recomputed too.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from reasoning_levels.benchmark.datasets import TASKS, load_task
from reasoning_levels.benchmark.run import record_for
from reasoning_levels.benchmark.scoring import split_thinking


def rescore_task(
    raw_path: Path,
    task: str,
    tokenizer: Any = None,
) -> tuple[int, int]:
    """Rewrite one raw JSONL file with recomputed split and scoring fields."""
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    spec = TASKS[task]
    rows = load_task(task)
    by_id = {row["problem_id"]: row for row in rows}

    raw_path = Path(raw_path)
    records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    updated: list[dict] = []
    changed = 0
    for record in records:
        row = by_id.get(record["problem_id"])
        if row is None:
            updated.append(record)
            continue
        reasoning, content = split_thinking(record.get("generation_text") or "")
        if record.get("reasoning_text"):
            reasoning = record["reasoning_text"]
            content = record.get("generation_text") or ""
        telemetry = {
            "reasoning": reasoning,
            "content": content,
            "prompt_tokens": record.get("prompt_tokens"),
            "completion_tokens": record.get("completion_tokens"),
            "finish_reason": record.get("finish_reason"),
            "reasoning_tokens_estimate": tokenizer_estimate(tokenizer, reasoning),
        }
        rescored = record_for(
            task=task,
            row=row,
            dataset_revision=record.get("dataset_revision") or spec["revision"],
            condition=record["condition"],
            model=record["model"],
            effort=record["effort"],
            telemetry=telemetry,
            latency_ms=record.get("latency_ms") or 0,
        )
        if any(
            rescored[key] != record.get(key)
            for key in ("correct", "exact_match", "answer_extracted")
        ):
            changed += 1
        updated.append(rescored)

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=raw_path.parent
    ) as handle:
        for record in updated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_name = handle.name
    os.replace(temporary_name, raw_path)
    return len(updated), changed


def tokenizer_estimate(tokenizer: Any, text: str) -> int | None:
    from reasoning_levels.benchmark.client import estimate_tokens

    return estimate_tokens(tokenizer, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--tasks", default="aime25,math500,gpqa_diamond")
    parser.add_argument("--tokenizer", default="")
    args = parser.parse_args(argv)
    tokenizer = None
    if args.tokenizer:
        from reasoning_levels.benchmark.run import load_tokenizer

        tokenizer = load_tokenizer(args.tokenizer)
    raw_dir = Path(args.raw_dir)
    for task in [name.strip() for name in args.tasks.split(",") if name.strip()]:
        for path in sorted(raw_dir.glob(f"*__{task}.jsonl")):
            total, changed = rescore_task(path, task, tokenizer=tokenizer)
            print(f"{path.name}: {total} records, {changed} changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
