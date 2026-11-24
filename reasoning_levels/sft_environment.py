from __future__ import annotations

import json
from importlib.util import find_spec


DEFAULT_MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def model_auto_class_name() -> str:
    return "AutoModelForImageTextToText"


def effort_token_names() -> list[str]:
    return [
        "<|reasoning_effort_low|>",
        "<|reasoning_effort_medium|>",
        "<|reasoning_effort_high|>",
    ]


def validate_target_modules(module_names: set[str], targets: list[str]) -> dict[str, bool]:
    return {
        target: any(name == target or name.endswith(f".{target}") for name in module_names)
        for target in targets
    }


def inspect_runtime() -> dict:
    import torch
    import transformers

    cuda_available = bool(torch.cuda.is_available())
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else None,
        "bf16_supported": bool(cuda_available and torch.cuda.is_bf16_supported()),
        "transformers": transformers.__version__,
        "optional": {
            name: find_spec(name) is not None
            for name in ["bitsandbytes", "peft", "trl", "unsloth"]
        },
    }


def _template_samples(tokenizer) -> dict[str, str]:
    messages = [{"role": "user", "content": "What is 2+2?"}]
    samples = {}
    for thinking in [True, False]:
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": thinking},
            )
        samples[f"enable_thinking={thinking}"] = rendered
    return samples


def _load_auto_model_class():
    import transformers

    for name in [
        model_auto_class_name(),
        "AutoModelForMultimodalLM",
        "AutoModelForCausalLM",
    ]:
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            return name, model_class
    raise RuntimeError("No compatible Transformers auto model class is available")


def inspect_model(model_name: str = DEFAULT_MODEL_NAME) -> dict:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    class_name, model_class = _load_auto_model_class()
    model = model_class.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    names = {name for name, _ in model.named_modules()}
    return {
        "model_class": type(model).__name__,
        "model_config": type(model.config).__name__,
        "auto_model_class": class_name,
        "processor_class": type(processor).__name__,
        "tokenizer_size": len(tokenizer),
        "chat_template": tokenizer.chat_template,
        "template_samples": _template_samples(tokenizer),
        "target_modules": validate_target_modules(names, TARGET_MODULES),
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
    }


def main() -> None:
    report = {"runtime": inspect_runtime()}
    report["model"] = inspect_model()
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
