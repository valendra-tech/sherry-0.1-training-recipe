from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any, Callable, Iterable

from .boxed import answers_match, extract_answer
from .chat_template import EFFORT_TOKENS, THINKING_CLOSE, THINKING_OPEN
from .sample_problems import normalize_problem, sample_problems
from .sft_data import load_jsonl, write_jsonl
from .sft_train import DEFAULT_MAX_LENGTH, MODEL_NAME


EFFORT_LEVELS = ["low", "medium", "high"]
DEFAULT_PROBLEM_COUNT = 100
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_DO_SAMPLE = False


def _sequence(value: Any) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return list(value)


def _tokenize(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        value = encoded["input_ids"]
    else:
        value = encoded.input_ids
    return [int(token_id) for token_id in _sequence(value)]


def _apply_chat_template(tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": problem}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": True},
        )


def format_inference_prompt(problem: str, effort: str, tokenizer) -> str:
    """Serialize the user prompt and insert one trained effort token."""
    if effort not in EFFORT_TOKENS:
        raise ValueError(f"unknown effort: {effort}")

    prompt = _apply_chat_template(tokenizer, problem)
    if not isinstance(prompt, str) or not prompt.endswith(THINKING_OPEN):
        raise ValueError(
            "Qwen3.6 chat template did not end with the expected thinking opener"
        )
    return (
        prompt[:-len(THINKING_OPEN)]
        + EFFORT_TOKENS[effort]
        + THINKING_OPEN
    )


def effort_token_ids(tokenizer) -> list[int]:
    ids = [tokenizer.convert_tokens_to_ids(token) for token in EFFORT_TOKENS.values()]
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if any(token_id is None or token_id == unknown_id for token_id in ids):
        raise ValueError(f"effort tokens did not receive valid ids: {ids}")
    return [int(token_id) for token_id in ids]


def _find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(values):
        return None
    for start in range(len(values) - len(pattern) + 1):
        if values[start:start + len(pattern)] == pattern:
            return start
    return None


def parse_generated_tokens(generated_ids: Iterable[int], tokenizer) -> dict[str, Any]:
    """Split generated IDs at the thinking boundary and count exact token spans."""
    ids = [int(token_id) for token_id in generated_ids]
    close_ids = _tokenize(tokenizer, THINKING_CLOSE)
    boundary = _find_subsequence(ids, close_ids)

    if boundary is not None:
        thinking_ids = ids[:boundary]
        response_ids = ids[boundary + len(close_ids):]
        return {
            "thinking": tokenizer.decode(thinking_ids, skip_special_tokens=False).strip(),
            "response": tokenizer.decode(response_ids, skip_special_tokens=False).strip(),
            "thinking_tokens": len(thinking_ids),
            "total_tokens": len(ids),
            "boundary_found": True,
        }

    text = tokenizer.decode(ids, skip_special_tokens=False)
    if THINKING_CLOSE in text:
        thinking, response = text.split(THINKING_CLOSE, 1)
        thinking = thinking.strip()
        response = response.strip()
        thinking_tokens = len(_tokenize(tokenizer, thinking)) if thinking else 0
    else:
        thinking = text.strip()
        response = ""
        thinking_tokens = len(ids)
    return {
        "thinking": thinking,
        "response": response,
        "thinking_tokens": thinking_tokens,
        "total_tokens": len(ids),
        "boundary_found": False,
    }


def completion_finish_reason(
    generated_ids: Iterable[int], tokenizer, max_new_tokens: int
) -> str:
    ids = [int(token_id) for token_id in generated_ids]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_ids = {int(token_id) for token_id in eos_token_id}
    else:
        eos_ids = set()
    if ids and ids[-1] in eos_ids:
        return "stop"
    if len(ids) >= max_new_tokens:
        return "length"
    return "stop"


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def generate_local_trace(
    model,
    processor,
    tokenizer,
    problem: str,
    effort: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, Any]:
    """Generate one deterministic local trace for a problem and effort."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt = format_inference_prompt(problem, effort, tokenizer)
    encoded = processor(
        text=prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    input_length = int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else len(_sequence(input_ids))
    device = _model_device(model)
    if device is not None:
        encoded = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": DEFAULT_DO_SAMPLE,
        "use_cache": True,
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    started = time.perf_counter()
    import torch

    with torch.inference_mode():
        generated = model.generate(**encoded, **generation_kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000

    sequences = getattr(generated, "sequences", generated)
    sequence = _sequence(sequences[0])
    generated_ids = [int(token_id) for token_id in sequence[input_length:]]
    parsed = parse_generated_tokens(generated_ids, tokenizer)
    parsed.update(
        {
            "finish_reason": completion_finish_reason(
                generated_ids, tokenizer, max_new_tokens
            ),
            "prompt_tokens": input_length,
            "latency_ms": elapsed_ms,
        }
    )
    return parsed


def load_sft_adapter(
    adapter_path: str | Path,
    max_length: int = DEFAULT_MAX_LENGTH,
    model_name: str | None = None,
):
    """Load the trained adapter on the same Unsloth base used for SFT."""
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor
    from unsloth import FastModel

    adapter_path = Path(adapter_path)
    processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    effort_token_ids(tokenizer)

    if model_name is None:
        peft_config = PeftConfig.from_pretrained(adapter_path)
        model_name = peft_config.base_model_name_or_path or MODEL_NAME

    model, _ = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None:
        raise RuntimeError("SFT base model must expose input embeddings")
    if input_embeddings.num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model, processor, tokenizer


def load_excluded_problems(paths: Iterable[str | Path]) -> set[str]:
    excluded = set()
    for path in paths:
        for row in load_jsonl(path):
            problem = row.get("problem")
            if isinstance(problem, str) and problem:
                excluded.add(normalize_problem(problem))
    return excluded


def sample_unseen_problems(
    n: int = DEFAULT_PROBLEM_COUNT,
    seed: int = DEFAULT_SEED,
    excluded_paths: Iterable[str | Path] = (),
) -> list[dict]:
    excluded = load_excluded_problems(excluded_paths)
    problems = sample_problems(
        n=n,
        seed=seed,
        balance_sources=True,
        max_solution_chars=2000,
        exclude_problems=excluded,
    )
    if len(problems) != n:
        raise RuntimeError(
            f"could only sample {len(problems)} unseen problems, expected {n}"
        )
    return problems


def _numeric_values(rows: list[dict], field: str) -> list[int | float]:
    return [
        row[field]
        for row in rows
        if isinstance(row.get(field), (int, float))
    ]


def aggregate_metrics(rows: list[dict]) -> dict[str, Any]:
    """Aggregate accuracy, token counts, finish reasons, and median ordering."""
    metrics: dict[str, Any] = {}
    medians: dict[str, int | float | None] = {}
    for effort in EFFORT_LEVELS:
        effort_rows = [row for row in rows if row.get("effort") == effort]
        correct_values = [
            row["correct"]
            for row in effort_rows
            if isinstance(row.get("correct"), bool)
        ]
        thinking_values = _numeric_values(effort_rows, "thinking_tokens")
        total_values = _numeric_values(effort_rows, "total_tokens")
        effort_metrics = {
            "count": len(effort_rows),
            "completed_count": len(correct_values),
            "accuracy": (
                sum(correct_values) / len(correct_values)
                if correct_values
                else None
            ),
            "median_thinking_tokens": median(thinking_values) if thinking_values else None,
            "mean_thinking_tokens": fmean(thinking_values) if thinking_values else None,
            "median_total_tokens": median(total_values) if total_values else None,
            "mean_total_tokens": fmean(total_values) if total_values else None,
            "finish_reasons": dict(Counter(
                row.get("finish_reason") for row in effort_rows
                if row.get("finish_reason") is not None
            )),
        }
        metrics[effort] = effort_metrics
        medians[effort] = effort_metrics["median_thinking_tokens"]

    ordered = [medians[effort] for effort in EFFORT_LEVELS]
    metrics["median_thinking_tokens"] = medians
    metrics["thinking_tokens_strictly_increasing"] = (
        all(value is not None for value in ordered)
        and ordered[0] < ordered[1] < ordered[2]
    )
    return metrics


def load_result_rows(path: str | Path) -> list[dict]:
    output_path = Path(path)
    if not output_path.exists():
        return []
    rows = []
    with output_path.open(encoding="utf-8") as file:
        for line in file:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _append_result(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_problems(
    model,
    processor,
    tokenizer,
    problems: list[dict],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_length: int = DEFAULT_MAX_LENGTH,
    existing_rows: list[dict] | None = None,
    raw_output_path: str | Path | None = None,
    generator: Callable[..., dict[str, Any]] = generate_local_trace,
) -> list[dict]:
    """Generate all efforts, persisting each row so interrupted runs can resume."""
    results = list(existing_rows or [])
    completed = {
        (row.get("problem_index"), row.get("effort"))
        for row in results
    }
    raw_path = Path(raw_output_path) if raw_output_path else None

    for problem_index, problem in enumerate(problems):
        for effort in EFFORT_LEVELS:
            key = (problem_index, effort)
            if key in completed:
                continue
            started = time.perf_counter()
            trace = generator(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                problem=problem["problem"],
                effort=effort,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
            )
            predicted = extract_answer(trace["response"])
            row = {
                "problem_index": problem_index,
                "problem": problem["problem"],
                "source": problem.get("source"),
                "effort": effort,
                "thinking": trace["thinking"],
                "thinking_tokens": trace["thinking_tokens"],
                "response": trace["response"],
                "total_tokens": trace["total_tokens"],
                "predicted_answer": predicted,
                "ground_truth": problem["ground_truth"],
                "correct": answers_match(predicted, problem["ground_truth"]),
                "finish_reason": trace["finish_reason"],
                "boundary_found": trace["boundary_found"],
                "prompt_tokens": trace.get("prompt_tokens"),
                "latency_ms": trace.get("latency_ms"),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "max_new_tokens": max_new_tokens,
            }
            results.append(row)
            completed.add(key)
            if raw_path:
                _append_result(raw_path, row)
            print(
                f"{problem_index + 1}/{len(problems)} {effort}: "
                f"{row['thinking_tokens']} thinking tokens, correct={row['correct']}"
            )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Qwen3.6 SFT adapter")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--train", default="data/sft/train.jsonl")
    parser.add_argument("--sft-eval", default="data/sft/eval.jsonl")
    parser.add_argument("--problems", default=None)
    parser.add_argument("--problems-output", default="data/sft/unseen-100.jsonl")
    parser.add_argument("--output", default="data/sft/evaluation.json")
    parser.add_argument("--max-eval-rows", type=int, default=DEFAULT_PROBLEM_COUNT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_eval_rows <= 0:
        raise ValueError("--max-eval-rows must be positive")

    if args.problems:
        problems = load_jsonl(args.problems)[:args.max_eval_rows]
    else:
        problems = sample_unseen_problems(
            n=args.max_eval_rows,
            seed=args.seed,
            excluded_paths=[args.train, args.sft_eval],
        )
        write_jsonl(problems, args.problems_output)
    if len(problems) != args.max_eval_rows:
        raise ValueError(
            f"expected {args.max_eval_rows} evaluation problems, got {len(problems)}"
        )

    model, processor, tokenizer = load_sft_adapter(
        args.adapter,
        max_length=args.max_length,
    )
    output_path = Path(args.output)
    raw_output_path = output_path.with_suffix(".jsonl")
    existing_rows = load_result_rows(raw_output_path)
    rows = evaluate_problems(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        problems=problems,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        existing_rows=existing_rows,
        raw_output_path=raw_output_path,
    )
    summary = {
        "adapter": args.adapter,
        "model": MODEL_NAME,
        "problem_count": len(problems),
        "efforts": EFFORT_LEVELS,
        "do_sample": DEFAULT_DO_SAMPLE,
        "max_new_tokens": args.max_new_tokens,
        "max_length": args.max_length,
        "seed": args.seed,
        "metrics": aggregate_metrics(rows),
        "raw_output": str(raw_output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
