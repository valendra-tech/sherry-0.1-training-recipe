from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from .chat_template import EFFORT_TOKENS, THINKING_CLOSE, THINKING_OPEN, add_effort_tokens, format_sft_text
from .sft_data import load_jsonl, validate_rows
from .sft_environment import TARGET_MODULES, validate_target_modules


MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_MAX_LENGTH = 4096
DEFAULT_MAX_STEPS = 500
DEFAULT_OUTPUT = "checkpoints/qwen36-reasoning-effort-sft-0.1"
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "shared_expert.gate_proj",
    "shared_expert.up_proj",
    "shared_expert.down_proj",
]


def build_unsloth_lora_kwargs(
    token_ids: list[int],
    input_embedding_name: str,
    output_embedding_name: str,
    max_length: int,
) -> dict[str, Any]:
    """Build the Unsloth LoRA arguments, including selective token rows."""
    if not input_embedding_name or not output_embedding_name:
        raise ValueError("embedding module names must be non-empty")

    token_ids = list(token_ids)
    return {
        "r": 32,
        "lora_alpha": 64,
        # PEFT's ParamWrapper for trainable token rows requires zero dropout.
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": list(LORA_TARGET_MODULES),
        "trainable_token_indices": {
            input_embedding_name: token_ids,
            output_embedding_name: token_ids,
        },
        "use_gradient_checkpointing": "unsloth",
        "random_state": 42,
        "max_seq_length": max_length,
    }


def build_training_arguments(output_dir: str | Path, max_steps: int | None = None):
    from transformers import TrainingArguments

    if max_steps is None:
        max_steps = int(os.environ.get("SFT_MAX_STEPS", str(DEFAULT_MAX_STEPS)))
    save_steps = max(1, min(100, max_steps)) if max_steps > 0 else 100
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        max_steps=max_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        logging_strategy="steps",
        logging_steps=1 if max_steps == 1 else 10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        eval_strategy="no",
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=42,
        data_seed=42,
    )


def _sequence(value: Any) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return list(value)


def _encode_ids(processor, text: str) -> list[int]:
    encoded = processor(
        text=text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    return [int(item) for item in _sequence(encoded["input_ids"])]


def _find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(values):
        return None
    for start in range(len(values) - len(pattern) + 1):
        if values[start:start + len(pattern)] == pattern:
            return start
    return None


def _answer_preserving_indices(
    processor,
    text: str,
    full_ids: list[int],
    max_length: int,
) -> list[int]:
    """Keep the answer suffix while trimming only the thinking token span."""
    prefix_text, thinking_marker, remainder = text.partition(THINKING_OPEN)
    if not thinking_marker:
        raise ValueError("formatted SFT text is missing the thinking opener")
    _, close_marker, _ = remainder.partition(THINKING_CLOSE)
    if not close_marker:
        raise ValueError("formatted SFT text is missing the thinking closer")

    prefix_ids = _encode_ids(processor, prefix_text + THINKING_OPEN)
    close_ids = _encode_ids(processor, THINKING_CLOSE)
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("SFT prefix tokenization changed across text boundaries")
    close_start = _find_subsequence(full_ids, close_ids)
    if close_start is None or close_start < len(prefix_ids):
        raise ValueError("formatted SFT text has no usable thinking boundary")

    suffix_length = len(full_ids) - close_start
    available_thinking = max_length - len(prefix_ids) - suffix_length
    if available_thinking < 0:
        raise ValueError("SFT answer suffix is longer than max_length")

    thinking_end = len(prefix_ids) + available_thinking
    return list(range(thinking_end)) + list(range(close_start, len(full_ids)))


class TokenizedSFTDataset:
    def __init__(self, rows: list[dict], processor, max_length: int):
        self.examples: list[dict[str, list[int]]] = []
        self.truncated_count = 0
        self.thinking_truncated_count = 0

        tokenizer = getattr(processor, "tokenizer", processor)
        for row in rows:
            text = format_sft_text(row, tokenizer)
            untruncated = processor(
                text=text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )
            encoded = processor(
                text=text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            input_ids = _sequence(encoded["input_ids"])
            original_ids = _sequence(untruncated["input_ids"])
            if len(original_ids) > len(input_ids):
                self.truncated_count += 1
                keep_indices = _answer_preserving_indices(
                    processor,
                    text,
                    original_ids,
                    max_length,
                )
                source = untruncated
                input_ids = [original_ids[index] for index in keep_indices]
                self.thinking_truncated_count += 1
            else:
                keep_indices = None
                source = encoded

            example: dict[str, list[int]] = {}
            for key, value in source.items():
                sequence = _sequence(value)
                if sequence and all(isinstance(item, int) for item in sequence):
                    if keep_indices is not None:
                        if len(sequence) != len(original_ids):
                            continue
                        sequence = [sequence[index] for index in keep_indices]
                    example[key] = sequence
            example["labels"] = list(input_ids)
            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class SFTDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            max_length = math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of

        keys = set().union(*(feature.keys() for feature in features))
        batch = {}
        for key in keys:
            if key == "labels":
                pad_value = -100
            elif key == "input_ids":
                pad_value = self.pad_token_id
            else:
                pad_value = 0
            values = []
            for feature in features:
                value = list(feature.get(key, []))
                values.append(value + [pad_value] * (max_length - len(value)))
            batch[key] = torch.tensor(values, dtype=torch.long)
        return batch


def _token_ids(tokenizer) -> list[int]:
    ids = [tokenizer.convert_tokens_to_ids(token) for token in EFFORT_TOKENS.values()]
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if any(token_id is None or token_id == unknown_id for token_id in ids):
        raise ValueError(f"effort tokens did not receive valid ids: {ids}")
    return ids


def _unique_module_name(model, target_module, description: str) -> str:
    matches = [name for name, module in model.named_modules() if module is target_module]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {description} module, found: {matches}")
    return matches[0]


def validate_trainable_lora_scope(model) -> None:
    routed_experts = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".experts." in name
    ]
    if routed_experts:
        raise ValueError(
            "routed MoE expert adapters are outside the intended LoRA scope: "
            f"{routed_experts[:3]}"
        )


def load_model_and_processor(
    model_name: str = MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
):
    import torch
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    tokenizer = getattr(processor, "tokenizer", processor)
    added = add_effort_tokens(tokenizer)
    if added != len(EFFORT_TOKENS):
        raise ValueError(f"expected to add {len(EFFORT_TOKENS)} effort tokens, added {added}")
    token_ids = _token_ids(tokenizer)

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError("Qwen3.6 model must expose input and output embeddings")
    if input_embeddings.num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        if input_embeddings is None or output_embeddings is None:
            raise RuntimeError("Qwen3.6 model must expose input and output embeddings")
    model.config.use_cache = False

    module_names = {name for name, _ in model.named_modules()}
    availability = validate_target_modules(module_names, LORA_TARGET_MODULES)
    if not all(availability.values()):
        missing = [name for name, found in availability.items() if not found]
        raise ValueError(f"base model is missing LoRA targets: {missing}")

    input_embedding_name = _unique_module_name(model, input_embeddings, "input embedding")
    output_embedding_name = _unique_module_name(model, output_embeddings, "output embedding")
    model = FastModel.get_peft_model(
        model,
        **build_unsloth_lora_kwargs(
            token_ids,
            input_embedding_name=input_embedding_name,
            output_embedding_name=output_embedding_name,
            max_length=max_length,
        ),
    )
    validate_trainable_lora_scope(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, processor, tokenizer, token_ids


def build_trainer(model, processor, train_dataset, output_dir: str | Path, max_steps: int | None = None,
                  eval_dataset=None):
    from transformers import Trainer

    tokenizer = getattr(processor, "tokenizer", processor)
    return Trainer(
        model=model,
        args=build_training_arguments(output_dir, max_steps=max_steps),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTDataCollator(tokenizer),
        processing_class=processor,
    )


def _write_run_metadata(output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft_run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    probe = Path("data/sft/template_probe.json")
    if probe.exists():
        shutil.copyfile(probe, output_dir / "template_probe.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Qwen3.6 reasoning-effort SFT adapter")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--train", default="data/sft/train.jsonl")
    parser.add_argument("--eval", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    max_length = args.max_length or int(os.environ.get("SFT_MAX_LENGTH", str(DEFAULT_MAX_LENGTH)))
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = int(os.environ.get("SFT_MAX_STEPS", str(DEFAULT_MAX_STEPS)))

    all_train_rows = load_jsonl(args.train)
    valid_train_rows = validate_rows(all_train_rows)
    if len(valid_train_rows) != len(all_train_rows):
        raise ValueError("train JSONL contains rows that fail the verified SFT validation")
    train_rows = all_train_rows[:args.max_train_rows] if args.max_train_rows else all_train_rows

    eval_rows = None
    if args.eval:
        all_eval_rows = load_jsonl(args.eval)
        valid_eval_rows = validate_rows(all_eval_rows)
        if len(valid_eval_rows) != len(all_eval_rows):
            raise ValueError("eval JSONL contains rows that fail the verified SFT validation")
        eval_rows = all_eval_rows[:args.max_eval_rows] if args.max_eval_rows else all_eval_rows

    model, processor, tokenizer, token_ids = load_model_and_processor(args.model, max_length=max_length)
    train_dataset = TokenizedSFTDataset(train_rows, processor, max_length)
    eval_dataset = TokenizedSFTDataset(eval_rows, processor, max_length) if eval_rows else None
    trainer = build_trainer(
        model,
        processor,
        train_dataset,
        args.output,
        max_steps=max_steps,
        eval_dataset=eval_dataset,
    )
    result = trainer.train()
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    _write_run_metadata(
        Path(args.output),
        {
            "model": args.model,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows) if eval_rows else 0,
            "max_length": max_length,
            "truncated_train_examples": train_dataset.truncated_count,
            "thinking_truncated_train_examples": train_dataset.thinking_truncated_count,
            "truncated_eval_examples": eval_dataset.truncated_count if eval_dataset else 0,
            "thinking_truncated_eval_examples": (
                eval_dataset.thinking_truncated_count if eval_dataset else 0
            ),
            "effort_token_ids": token_ids,
            "effort_tokens": list(EFFORT_TOKENS.values()),
            "thinking_open": THINKING_OPEN,
            "thinking_close": THINKING_CLOSE,
            "target_modules": LORA_TARGET_MODULES,
            "train_result": result.metrics,
            "training_args": trainer.args.to_dict(),
        },
    )
    print(json.dumps(result.metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
