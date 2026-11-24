"""Pinned dataset specs and pure row normalizers.

Revisions are pinned so a re-run years later loads the same bytes. The
``datasets`` import is lazy so unit tests never touch the network.
"""

from __future__ import annotations

from typing import Callable

from reasoning_levels.benchmark import code_suite
from reasoning_levels.benchmark.scoring import (
    gpqa_letter,
    last_boxed_only_string,
    remove_boxed,
)

TASKS: dict[str, dict[str, str]] = {
    "aime25": {
        "path": "math-ai/aime25",
        "revision": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
        "split": "test",
    },
    "math500": {
        "path": "HuggingFaceH4/MATH-500",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "split": "test",
    },
    "gpqa_diamond": {
        "path": "hendrydong/gpqa_diamond_mc",
        "revision": "284143babc24a94fbac45d143333b2307e64ff80",
        "split": "test",
    },
    "code_basic": {
        "path": "local/code-basic-suite",
        "revision": code_suite.suite_revision(),
        "split": "local",
    },
}


def normalize_aime25(index: int, row: dict) -> dict:
    return {
        "problem_id": f"aime25-{index:03d}",
        "problem": row["problem"].strip(),
        "solution": None,
        "answer": str(row["answer"]).strip(),
    }


def normalize_math500(index: int, row: dict) -> dict:
    solution = row["solution"]
    boxed = last_boxed_only_string(solution)
    answer = None
    if boxed is not None:
        try:
            answer = remove_boxed(boxed).strip() or None
        except ValueError:
            answer = None
    return {
        "problem_id": f"math500-{index:04d}",
        "problem": row["problem"].strip(),
        "solution": solution,
        "answer": answer,
    }


def normalize_gpqa(index: int, row: dict) -> dict:
    return {
        "problem_id": f"gpqa-{index:03d}",
        "problem": row["problem"].strip(),
        "solution": row["solution"],
        "answer": gpqa_letter(row["solution"]),
    }


NORMALIZERS: dict[str, Callable[[int, dict], dict]] = {
    "aime25": normalize_aime25,
    "math500": normalize_math500,
    "gpqa_diamond": normalize_gpqa,
}


def normalize_rows(
    normalizer: Callable[[int, dict], dict], rows: list[dict]
) -> tuple[list[dict], int]:
    """Normalize rows, skipping malformed ones and counting the skips."""
    normalized: list[dict] = []
    skipped = 0
    for index, row in enumerate(rows):
        try:
            normalized.append(normalizer(index, row))
        except Exception:
            skipped += 1
    return normalized, skipped


def load_task(task: str, limit: int = 0) -> list[dict]:
    """Load and normalize a pinned task from the Hugging Face Hub."""
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    if task == "code_basic":
        return code_suite.load_tasks(limit)
    spec = TASKS[task]
    from datasets import load_dataset

    dataset = load_dataset(spec["path"], split=spec["split"], revision=spec["revision"])
    normalized, skipped = normalize_rows(NORMALIZERS[task], list(dataset))
    if skipped:
        print(f"warning: {task}: skipped {skipped} malformed rows")
    if limit:
        normalized = normalized[:limit]
    return normalized
