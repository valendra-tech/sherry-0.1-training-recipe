"""Resumable public-benchmark runner.

Every condition and task writes one JSONL file under ``<output-dir>/raw``.
Completed ``problem_id`` values are skipped on re-runs, so a long evaluation
can be resumed after an SSH disconnect without losing completed samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reasoning_levels.benchmark import code_suite
from reasoning_levels.benchmark.client import (
    completion_to_telemetry,
    generate_one,
    make_client,
)
from reasoning_levels.benchmark.datasets import TASKS, load_task
from reasoning_levels.benchmark.prompts import STOPS, doc_to_text
from reasoning_levels.benchmark.scoring import score_gpqa, score_math

TASK_NAMES = ("aime25", "math500", "gpqa_diamond")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--condition", required=True, help="record label")
    parser.add_argument(
        "--effort",
        default="none",
        choices=("none", "auto", "low", "medium", "high"),
    )
    parser.add_argument("--tasks", default=",".join(TASK_NAMES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument(
        "--tokenizer",
        default="",
        help="optional HF tokenizer id used to estimate reasoning tokens",
    )
    return parser


def raw_path(output_dir: Path, condition: str, task: str) -> Path:
    return output_dir / "raw" / f"{condition}__{task}.jsonl"


def done_problem_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            problem_id = record.get("problem_id")
            if problem_id is not None:
                done.add(problem_id)
    return done


def score_for_task(task: str, row: dict, content: str) -> dict:
    if task == "code_basic":
        return code_suite.grade(row, content)
    if task == "gpqa_diamond":
        return score_gpqa(row, content)
    return score_math(row, content)


def record_for(
    *,
    task: str,
    row: dict,
    dataset_revision: str,
    condition: str,
    model: str,
    effort: str,
    telemetry: dict,
    latency_ms: int,
) -> dict:
    scoring = score_for_task(task, row, telemetry["content"])
    digest = hashlib.sha256(
        (telemetry["reasoning"] + "\n" + telemetry["content"]).encode("utf-8")
    ).hexdigest()
    return {
        "condition": condition,
        "task": task,
        "model": model,
        "effort": effort,
        "problem_id": row["problem_id"],
        "dataset_revision": dataset_revision,
        "prompt_tokens": telemetry["prompt_tokens"],
        "completion_tokens": telemetry["completion_tokens"],
        "reasoning_tokens_estimate": telemetry["reasoning_tokens_estimate"],
        "finish_reason": telemetry["finish_reason"],
        "latency_ms": latency_ms,
        "generation_sha256": digest,
        "generation_text": telemetry["content"],
        "reasoning_text": telemetry["reasoning"],
        "answer_extracted": scoring["answer_extracted"],
        "expected": scoring["expected"],
        "exact_match": scoring["exact_match"],
        "math_verify": scoring["math_verify"],
        "correct": scoring["exact_match"],
    }


def load_tokenizer(model_id: str) -> Any:
    if not model_id:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        print(f"warning: tokenizer {model_id} unavailable ({exc}); estimates disabled")
        return None


def run_task(
    *,
    client: Any,
    model: str,
    condition: str,
    effort: str,
    task: str,
    rows: list[dict],
    dataset_revision: str,
    output_dir: Path,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    seed: int,
    tokenizer: Any,
) -> int:
    """Run one task, appending new records and skipping cached problems."""
    output_dir = Path(output_dir)
    path = raw_path(output_dir, condition, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = done_problem_ids(path)
    pending = [row for row in rows if row["problem_id"] not in done]
    if not pending:
        print(f"{condition}/{task}: all {len(rows)} problems cached")
        return 0

    stops = STOPS[task]

    def generate(row: dict) -> dict:
        prompt = doc_to_text(task, row)
        data, latency_ms = generate_one(
            client,
            model=model,
            prompt=prompt,
            stops=stops,
            effort=effort,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        telemetry = completion_to_telemetry(data, tokenizer=tokenizer)
        return record_for(
            task=task,
            row=row,
            dataset_revision=dataset_revision,
            condition=condition,
            model=model,
            effort=effort,
            telemetry=telemetry,
            latency_ms=latency_ms,
        )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    written = 0
    with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        futures = {executor.submit(generate, row): row for row in pending}
        for future in as_completed(futures):
            record = future.result()
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            if written % 25 == 0 or written == len(pending):
                print(f"{condition}/{task}: {written}/{len(pending)} new records")
    return written


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    tokenizer = load_tokenizer(args.tokenizer)
    client = make_client(args.base_url)
    tasks = [name.strip() for name in args.tasks.split(",") if name.strip()]
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"unknown task: {task}")
        rows = load_task(task, limit=args.limit)
        written = run_task(
            client=client,
            model=args.model,
            condition=args.condition,
            effort=args.effort,
            task=task,
            rows=rows,
            dataset_revision=TASKS[task]["revision"],
            output_dir=output_dir,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            tokenizer=tokenizer,
        )
        print(f"{task}: {written} new records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
