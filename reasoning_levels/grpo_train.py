from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .chat_template import EFFORT_TOKENS, THINKING_OPEN
from .grpo_reward import (
    EFFORT_REWARD_TARGETS,
    GLOBAL_GRPO_MAX_NEW_TOKENS,
    REWARD_WEIGHTS,
    make_reasoning_effort_reward,
)
from .sft_data import write_jsonl
from .sft_train import MODEL_NAME


EFFORT_LEVELS = ["low", "medium", "high"]
DEFAULT_GRPO_STEPS = 200
DEFAULT_GRPO_PROBLEMS = math.ceil(DEFAULT_GRPO_STEPS / len(EFFORT_LEVELS))
GRPO_OUTPUT = "checkpoints/qwen36-reasoning-effort-grpo-0.1"
SFT_OUTPUT = "checkpoints/qwen36-reasoning-effort-sft-0.2"


def validate_output_paths(adapter_path: str | Path, output_path: str | Path) -> None:
    if Path(adapter_path).resolve() == Path(output_path).resolve():
        raise ValueError("GRPO output must not equal the SFT adapter path")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen3.6 reasoning-effort GRPO adapter")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--train", default="data/sft/train.jsonl")
    parser.add_argument("--sft-eval", default="data/sft/eval.jsonl")
    parser.add_argument("--problems-output", default="data/sft/grpo-problems.jsonl")
    parser.add_argument("--problems-manifest", default="data/sft/grpo-problems-manifest.json")
    parser.add_argument("--output", default=GRPO_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_GRPO_STEPS)
    parser.add_argument("--max-new-tokens", type=int, default=GLOBAL_GRPO_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _normalize_trl_optional_flags():
    # Transformers 5.x returns (available, version) for some optional-package
    # checks while TRL 0.24 expects a boolean, making false flags truthy.
    try:
        import trl.import_utils as import_utils
    except ModuleNotFoundError:
        import_utils = None
    if import_utils is not None:
        for name, value in vars(import_utils).items():
            if name.endswith("_available") and isinstance(value, tuple):
                setattr(import_utils, name, value[0])



def _load_grpo_config():
    _normalize_trl_optional_flags()
    from trl import GRPOConfig

    return GRPOConfig


def _load_grpo_trainer():
    _normalize_trl_optional_flags()
    from trl import GRPOTrainer

    return GRPOTrainer


def build_grpo_config(
    output_dir: str | Path,
    max_steps: int = DEFAULT_GRPO_STEPS,
    max_completion_length: int = GLOBAL_GRPO_MAX_NEW_TOKENS,
):
    GRPOConfig = _load_grpo_config()

    return GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        max_steps=max_steps,
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=max_completion_length,
        generation_batch_size=4,
        num_iterations=1,
        temperature=0.7,
        top_p=0.95,
        beta=0.0,
        loss_type="dr_grpo",
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=4,
        shuffle_dataset=False,
        remove_unused_columns=False,
        use_vllm=False,
        report_to=["tensorboard"],
        seed=42,
        data_seed=42,
    )


def load_trainable_sft_adapter(
    adapter_path: str | Path,
    max_length: int = GLOBAL_GRPO_MAX_NEW_TOKENS,
):
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoProcessor
    from unsloth import FastModel

    adapter_path = Path(adapter_path)
    processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
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
        raise RuntimeError("GRPO base model must expose input embeddings")
    if input_embeddings.num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model.config.use_cache = False
    return model, processor, tokenizer


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


def build_grpo_prompt(problem: str, effort: str, tokenizer) -> str:
    if effort not in EFFORT_TOKENS:
        raise ValueError(f"unknown effort: {effort}")
    prompt = _apply_chat_template(tokenizer, problem)
    if not isinstance(prompt, str) or not prompt.endswith(THINKING_OPEN):
        raise ValueError("chat template did not end with the thinking opener")
    return prompt[:-len(THINKING_OPEN)] + EFFORT_TOKENS[effort] + THINKING_OPEN


def effort_from_prompt(prompt: str) -> str:
    matches = [effort for effort, token in EFFORT_TOKENS.items() if token in prompt]
    if len(matches) != 1:
        raise ValueError(f"prompt must contain exactly one effort token: {prompt!r}")
    return matches[0]


def build_grpo_rows(problems: list[dict], tokenizer, max_groups: int) -> list[dict]:
    if max_groups <= 0:
        raise ValueError("max_groups must be positive")
    rows = []
    for problem_index, problem in enumerate(problems):
        for effort in EFFORT_LEVELS:
            rows.append(
                {
                    "prompt": build_grpo_prompt(problem["problem"], effort, tokenizer),
                    "problem": problem["problem"],
                    "problem_index": problem_index,
                    "effort": effort,
                    "ground_truth": problem["ground_truth"],
                    "source": problem.get("source"),
                }
            )
            if len(rows) == max_groups:
                return rows
    return rows


def sample_grpo_problems(
    train_path: str | Path,
    eval_path: str | Path,
    n_problems: int = DEFAULT_GRPO_PROBLEMS,
    seed: int = 42,
) -> list[dict]:
    from .evaluate_sft import sample_unseen_problems

    return sample_unseen_problems(
        n=n_problems,
        seed=seed,
        excluded_paths=[train_path, eval_path],
    )


def write_problem_manifest(
    problems: list[dict],
    path: str | Path,
    seed: int,
    excluded_paths: list[str | Path],
) -> None:
    manifest = {
        "seed": seed,
        "problem_count": len(problems),
        "excluded_paths": [str(item) for item in excluded_paths],
        "problems": [
            {
                "index": index,
                "source": problem.get("source"),
                "sha256": hashlib.sha256(
                    " ".join(problem["problem"].split()).encode("utf-8")
                ).hexdigest(),
            }
            for index, problem in enumerate(problems)
        ],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_run_metadata(
    output_dir: str | Path,
    args: argparse.Namespace,
    rows: list[dict],
    metrics: dict,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "adapter": str(args.adapter),
        "model": MODEL_NAME,
        "output": str(output_path),
        "problem_rows": len(rows),
        "effort_counts": {
            effort: sum(row.get("effort") == effort for row in rows)
            for effort in EFFORT_LEVELS
        },
        "global_max_new_tokens": args.max_new_tokens,
        "effort_reward_targets": EFFORT_REWARD_TARGETS,
        "reward_weights": REWARD_WEIGHTS,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "train_result": metrics,
    }
    (output_path / "grpo_run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_output_paths(args.adapter, args.output)
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    max_steps = 1 if args.dry_run else args.max_steps
    max_tokens = 128 if args.dry_run else args.max_new_tokens
    model, processor, tokenizer = load_trainable_sft_adapter(
        args.adapter,
        max_length=GLOBAL_GRPO_MAX_NEW_TOKENS,
    )
    problems = sample_grpo_problems(
        args.train,
        args.sft_eval,
        n_problems=math.ceil(max_steps / len(EFFORT_LEVELS)),
        seed=args.seed,
    )
    rows = build_grpo_rows(problems, tokenizer, max_groups=max_steps)
    if len(rows) != max_steps:
        raise RuntimeError(f"expected {max_steps} GRPO rows, got {len(rows)}")
    write_jsonl(rows, args.problems_output)
    write_problem_manifest(
        problems,
        args.problems_manifest,
        args.seed,
        [args.train, args.sft_eval],
    )

    from datasets import Dataset

    reward = make_reasoning_effort_reward(tokenizer, max_new_tokens=max_tokens)
    trainer_class = _load_grpo_trainer()
    trainer = trainer_class(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward,
        train_dataset=Dataset.from_list(rows),
        args=build_grpo_config(
            args.output,
            max_steps=max_steps,
            max_completion_length=max_tokens,
        ),
    )
    result = trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    write_run_metadata(args.output, args, rows, result.metrics)


if __name__ == "__main__":
    main()
