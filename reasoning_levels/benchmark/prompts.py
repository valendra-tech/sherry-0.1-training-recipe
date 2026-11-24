"""Benchmark prompt rendering.

The formats follow lm-evaluation-harness task definitions so results stay
comparable with public numbers:

- ``aime25``: ``lm_eval/tasks/aime/aime25.yaml``
- ``math500``: the ``minerva_math`` ``doc_to_text``
- ``gpqa_diamond``: the ``hendrydong/gpqa_diamond_mc`` mirror text, which
  already embeds the question, the (A)-(D) choices, and the instruction to
  answer with ``\\boxed{<letter>}``
"""

from __future__ import annotations

STOPS: dict[str, list[str]] = {
    "aime25": ["Question:", "</s>", "<|im_end|>", "<|eot_id|>"],
    "math500": ["Problem:", "</s>", "<|im_end|>"],
    "gpqa_diamond": ["Question:", "</s>", "<|im_end|>"],
    "code_basic": ["<|im_end|>"],
}


def doc_to_text(task: str, row: dict) -> str:
    """Render the task prompt for one dataset row."""
    if task == "code_basic":
        return row["prompt"]
    problem = row["problem"].strip()
    if task == "aime25":
        return f"Question: {problem}\nAnswer:"
    if task == "math500":
        return f"Problem:\n{problem}\n\nSolution:"
    if task == "gpqa_diamond":
        return problem
    raise ValueError(f"unknown task: {task}")
