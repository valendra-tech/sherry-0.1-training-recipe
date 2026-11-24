from __future__ import annotations
from collections.abc import Collection
import random
import re
from datasets import load_dataset
from .boxed import extract_boxed


def has_boxed_answer(solution: str) -> bool:
    """Check if a solution contains a \\boxed{} answer."""
    return extract_boxed(solution) is not None


def is_proof_problem(problem: str) -> bool:
    """Return whether a prompt asks for a proof rather than a scalar answer."""
    return bool(re.search(
        r"\b(?:prove|proof|demonstrate|establish)\b|\bshow\s+that\b",
        problem.lower(),
    ))


def is_numeric_answer(answer: str) -> bool:
    """Check if an answer is a scalar format supported by ``answers_match``."""
    a = answer.strip().replace(",", "")
    a = a.replace(r"\dfrac", r"\frac")
    a = a.replace(r"\left", "").replace(r"\right", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", a):
        return True
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?\s*/\s*[+-]?\d+(?:\.\d+)?", a):
        return True
    if re.fullmatch(r"\\frac\{[+-]?\d+\}\{[+-]?\d+\}", a):
        return True
    if re.fullmatch(
        r"\(?[+-]?\d+(?:\.\d+)?\s*/\s*[+-]?\d+(?:\.\d+)?\)?\s*\^\s*[+-]?\d+",
        a,
    ):
        return True
    return False


def normalize_problem(problem: str) -> str:
    """Normalize whitespace for exact-problem leakage checks."""
    return re.sub(r"\s+", " ", problem).strip()


def sample_problems(n: int = 1000, seed: int = 42,
                    balance_sources: bool = True,
                    sources: list = None,
                    max_solution_chars: int | None = 2000,
                    exclude_problems: Collection[str] | None = None) -> list[dict]:
    """Sample n problems from NuminaMath-CoT, filtered to those with numeric \\boxed{} answers.
    Excludes proof/demonstration problems (which can't be verified by exact-match).
    If balance_sources, samples equally across available sources.
    Returns list of {"problem": str, "ground_truth": str, "source": str, "solution": str}."""
    if sources is None:
        sources = ["olympiads", "synthetic_math", "cn_k12", "orca_math"]

    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    rng = random.Random(seed)
    excluded = {
        normalize_problem(problem)
        for problem in (exclude_problems or ())
    }

    by_source: dict[str, list[dict]] = {s: [] for s in sources}
    for ex in ds:
        src = ex.get("source", "")
        if src not in by_source:
            continue
        if normalize_problem(ex["problem"]) in excluded:
            continue
        if is_proof_problem(ex["problem"]):
            continue
        gt = extract_boxed(ex["solution"])
        if gt is None:
            continue
        if not is_numeric_answer(gt):
            continue
        if max_solution_chars is not None and len(ex["solution"]) > max_solution_chars:
            continue
        by_source[src].append({
            "problem": ex["problem"],
            "ground_truth": gt,
            "source": src,
            "solution": ex["solution"],
        })

    if balance_sources:
        per_source = n // len(sources)
        remainder = n % len(sources)
        result = []
        for i, src in enumerate(sources):
            count = per_source + (1 if i < remainder else 0)
            available = by_source.get(src, [])
            count = min(count, len(available))
            result.extend(rng.sample(available, count))
        while len(result) < n:
            largest = max(by_source.values(), key=len)
            if not largest:
                break
            pick = rng.choice(largest)
            if pick not in result:
                result.append(pick)
        return result[:n]
    else:
        all_valid = [p for ps in by_source.values() for p in ps]
        return rng.sample(all_valid, min(n, len(all_valid)))
