"""Recompute the math_verify metric for stored math-task records.

``math_verify`` is not thread-safe, so historical runs that evaluated it from
a thread pool stored ``False`` for every concurrent sample. This utility
recomputes the metric serially from the stored generations and rewrites the
JSONL file atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from reasoning_levels.benchmark.datasets import NORMALIZERS, TASKS, normalize_rows
from reasoning_levels.benchmark.scoring import math_verify_match

DEFAULT_TASKS = "math500"


def recompute_task(raw_path: Path, task: str) -> tuple[int, int]:
    """Rewrite one raw JSONL file with a serial math_verify recomputation."""
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    spec = TASKS[task]
    from datasets import load_dataset

    dataset = load_dataset(spec["path"], split=spec["split"], revision=spec["revision"])
    rows, _ = normalize_rows(NORMALIZERS[task], list(dataset))
    by_id = {row["problem_id"]: row for row in rows}

    raw_path = Path(raw_path)
    records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed = 0
    updated: list[dict] = []
    for record in records:
        row = by_id.get(record["problem_id"])
        if row is not None and row.get("solution"):
            value = math_verify_match(
                row["solution"], record.get("generation_text") or ""
            )
            if value != record.get("math_verify"):
                changed += 1
            record = dict(record)
            record["math_verify"] = value
        updated.append(record)

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=raw_path.parent
    ) as handle:
        for record in updated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_name = handle.name
    os.replace(temporary_name, raw_path)
    return len(updated), changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    args = parser.parse_args(argv)
    raw_dir = Path(args.raw_dir)
    for task in [name.strip() for name in args.tasks.split(",") if name.strip()]:
        for path in sorted(raw_dir.glob(f"*__{task}.jsonl")):
            total, changed = recompute_task(path, task)
            print(f"{path.name}: {total} records, {changed} updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
