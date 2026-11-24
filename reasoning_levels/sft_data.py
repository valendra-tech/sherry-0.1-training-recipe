from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


DATASET_ID = "valendra/sherry-reasoning-effort-0.1"
EFFORTS = ["low", "medium", "high"]


def effort_token(effort: str) -> str:
    if effort not in EFFORTS:
        raise ValueError(f"unknown effort: {effort}")
    return f"<|reasoning_effort_{effort}|>"


def validate_rows(rows: list[dict]) -> list[dict]:
    """Keep only complete, correct one-row-per-effort problem triplets."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("correct") is not True or row.get("finish_reason") != "stop":
            continue
        if row.get("effort") not in EFFORTS or not row.get("thinking"):
            continue
        problem = row.get("problem")
        if problem:
            grouped[problem].append(row)

    valid = []
    for problem_rows in grouped.values():
        by_effort = {row["effort"]: row for row in problem_rows}
        if len(problem_rows) == len(EFFORTS) and set(by_effort) == set(EFFORTS):
            valid.extend(by_effort[effort] for effort in EFFORTS)
    return valid


def split_by_problem(
    rows: list[dict], eval_fraction: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split complete examples while keeping every problem in one partition."""
    if not 0 <= eval_fraction <= 1:
        raise ValueError("eval_fraction must be between 0 and 1")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["problem"]].append(row)

    problems = list(grouped)
    random.Random(seed).shuffle(problems)
    n_eval = min(len(problems), max(1, round(len(problems) * eval_fraction))) if problems else 0
    eval_problems = set(problems[:n_eval])
    evaluation = [row for row in rows if row["problem"] in eval_problems]
    train = [row for row in rows if row["problem"] not in eval_problems]
    return train, evaluation


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_hf_jsonl(dataset_id: str = DATASET_ID) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=dataset_id,
        filename="reasoning_effort_math.jsonl",
        repo_type="dataset",
    )
    return load_jsonl(path)
