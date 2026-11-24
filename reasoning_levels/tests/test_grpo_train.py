import json
import sys
import types

import pytest

from reasoning_levels.grpo_train import (
    EFFORT_LEVELS,
    GLOBAL_GRPO_MAX_NEW_TOKENS,
    GRPO_OUTPUT,
    SFT_OUTPUT,
    build_grpo_prompt,
    build_grpo_rows,
    build_grpo_config,
    effort_from_prompt,
    load_trainable_sft_adapter,
    main,
    parse_args,
    validate_output_paths,
    write_run_metadata,
)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is True
        return "<|im_start|>assistant\n<think>\n"


def test_prompt_contains_selected_effort_token_before_thinking():
    prompt = build_grpo_prompt("What is 2+2?", "high", Tokenizer())
    assert prompt.endswith("<|reasoning_effort_high|><think>\n")
    assert effort_from_prompt(prompt) == "high"


def test_build_rows_omits_reference_solution_and_cycles_levels():
    problems = [
        {
            "problem": f"p{i}",
            "ground_truth": str(i),
            "source": "synthetic_math",
            "solution": "secret",
        }
        for i in range(2)
    ]
    rows = build_grpo_rows(problems, Tokenizer(), max_groups=6)

    assert [row["effort"] for row in rows] == EFFORT_LEVELS * 2
    assert all("solution" not in row for row in rows)
    assert all("ground_truth" in row for row in rows)
    assert all("secret" not in row["prompt"] for row in rows)


def test_build_rows_truncates_to_requested_group_count():
    problems = [
        {"problem": f"p{i}", "ground_truth": str(i), "source": "synthetic_math"}
        for i in range(3)
    ]
    rows = build_grpo_rows(problems, Tokenizer(), max_groups=5)
    assert len(rows) == 5
    assert [row["effort"] for row in rows] == ["low", "medium", "high", "low", "medium"]


def test_global_generation_budget_and_effort_schedule():
    assert GLOBAL_GRPO_MAX_NEW_TOKENS == 65536
    assert EFFORT_LEVELS == ["low", "medium", "high"]


def test_run_metadata_records_reward_weights(tmp_path):
    args = types.SimpleNamespace(
        adapter="sft", max_new_tokens=GLOBAL_GRPO_MAX_NEW_TOKENS, seed=42, dry_run=False
    )

    write_run_metadata(
        tmp_path,
        args,
        [{"effort": "low"}],
        {},
    )

    metadata = json.loads((tmp_path / "grpo_run.json").read_text())
    assert metadata["reward_weights"] == {
        "correctness": 2.0,
        "partial_correctness": 0.05,
        "structure": 0.20,
    }
    assert metadata["global_max_new_tokens"] == GLOBAL_GRPO_MAX_NEW_TOKENS
    assert metadata["effort_reward_targets"] == {
        "low": 1024,
        "medium": 4096,
        "high": 32768,
    }


def test_grpo_config_matches_approved_training_budget(monkeypatch):
    class FakeGRPOConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(GRPOConfig=FakeGRPOConfig))
    config = build_grpo_config("/tmp/grpo", max_steps=200)
    assert config.num_generations == 4
    assert config.max_completion_length == GLOBAL_GRPO_MAX_NEW_TOKENS
    assert config.generation_batch_size == 4
    assert config.max_steps == 200
    assert config.learning_rate == 5e-6
    assert config.temperature == 0.7
    assert config.top_p == 0.95
    assert config.beta == 0.0
    assert config.loss_type == "dr_grpo"
    assert config.shuffle_dataset is False
    assert config.use_vllm is False
    assert not hasattr(config, "steps_per_generation")


def test_sft_and_grpo_output_directories_are_distinct():
    assert SFT_OUTPUT != GRPO_OUTPUT
    assert SFT_OUTPUT.endswith("sft-0.2")
    assert GRPO_OUTPUT.endswith("grpo-0.1")


def test_loader_uses_the_sft_base_and_trainable_adapter(monkeypatch, tmp_path):
    class Tokenizer:
        def __len__(self):
            return 103

    class Processor:
        tokenizer = Tokenizer()

    class Embeddings:
        def __init__(self, num_embeddings):
            self.num_embeddings = num_embeddings

    class Model:
        def __init__(self):
            self.input_embeddings = Embeddings(100)
            self.config = types.SimpleNamespace(use_cache=True)

        def get_input_embeddings(self):
            return self.input_embeddings

        def resize_token_embeddings(self, size):
            self.input_embeddings = Embeddings(size)

    model = Model()
    processor = Processor()
    calls = {}

    class PeftConfig:
        base_model_name_or_path = "Qwen/Qwen3.6-35B-A3B"

        @classmethod
        def from_pretrained(cls, path):
            calls["peft_config_path"] = path
            return cls()

    class PeftModel:
        @classmethod
        def from_pretrained(cls, base, path, is_trainable=False):
            calls["adapter"] = (base, path, is_trainable)
            return base

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, path, trust_remote_code=False):
            calls["processor"] = (path, trust_remote_code)
            return processor

    class FastModel:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["model"] = kwargs
            return model, None

    fake_bfloat16 = object()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16=fake_bfloat16))
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(PeftConfig=PeftConfig, PeftModel=PeftModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoProcessor=AutoProcessor),
    )
    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastModel=FastModel))

    adapter = tmp_path / "sft"
    loaded_model, loaded_processor, tokenizer = load_trainable_sft_adapter(adapter)

    assert loaded_model is model
    assert loaded_processor is processor
    assert tokenizer is processor.tokenizer
    assert model.config.use_cache is False
    assert calls["model"] == {
        "model_name": "Qwen/Qwen3.6-35B-A3B",
        "max_seq_length": GLOBAL_GRPO_MAX_NEW_TOKENS,
        "dtype": fake_bfloat16,
        "load_in_4bit": False,
        "load_in_16bit": True,
        "full_finetuning": False,
    }
    assert calls["adapter"] == (model, adapter, True)


def test_validate_output_paths_rejects_overwriting_sft(tmp_path):
    path = tmp_path / "adapter"
    with pytest.raises(ValueError, match="must not equal"):
        validate_output_paths(path, path)


def test_parse_args_uses_approved_production_defaults():
    args = parse_args(["--adapter", "checkpoints/sft"])
    assert args.max_steps == 200
    assert args.max_new_tokens == GLOBAL_GRPO_MAX_NEW_TOKENS
    assert args.seed == 42
    assert args.output.endswith("grpo-0.1")
    assert args.dry_run is False


def test_grpo_trainer_uses_tokenizer_for_text_only_processing(monkeypatch, tmp_path):
    class Tokenizer:
        pad_token_id = 248044

    class Processor:
        tokenizer = Tokenizer()

        def save_pretrained(self, output):
            pass

    processor = Processor()
    trainer_kwargs = {}

    class Dataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class Trainer:
        def __init__(self, **kwargs):
            trainer_kwargs.update(kwargs)

        def train(self):
            return types.SimpleNamespace(metrics={})

        def save_model(self, output):
            pass

    datasets = types.ModuleType("datasets")
    datasets.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.load_trainable_sft_adapter",
        lambda adapter, max_length: (object(), processor, processor.tokenizer),
    )
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.sample_grpo_problems",
        lambda *args, **kwargs: [{"problem": "p", "ground_truth": "1"}],
    )
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.build_grpo_rows",
        lambda *args, **kwargs: [{"prompt": "p", "problem": "p", "effort": "low"}],
    )
    monkeypatch.setattr(
        "reasoning_levels.grpo_train._load_grpo_trainer",
        lambda: Trainer,
    )
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.build_grpo_config",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.make_reasoning_effort_reward",
        lambda tokenizer, **kwargs: object(),
    )
    monkeypatch.setattr("reasoning_levels.grpo_train.write_jsonl", lambda *args: None)
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.write_problem_manifest", lambda *args: None
    )
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.write_run_metadata", lambda *args: None
    )

    main(
        [
            "--adapter",
            str(tmp_path / "sft"),
            "--output",
            str(tmp_path / "grpo"),
            "--dry-run",
        ]
    )

    assert trainer_kwargs["processing_class"] is processor.tokenizer


def test_full_grpo_uses_one_global_trainer(monkeypatch, tmp_path):
    class Processor:
        tokenizer = types.SimpleNamespace()

        def save_pretrained(self, output):
            pass

    processor = Processor()
    trainer_loads = []

    class Dataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class Trainer:
        def __init__(self, **kwargs):
            pass

        def train(self):
            return types.SimpleNamespace(metrics={})

        def save_model(self, output):
            pass

    datasets = types.ModuleType("datasets")
    datasets.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.load_trainable_sft_adapter",
        lambda adapter, max_length: (object(), processor, processor.tokenizer),
    )
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.sample_grpo_problems",
        lambda *args, **kwargs: [{"problem": "p", "ground_truth": "1"}],
    )
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.build_grpo_rows",
        lambda *args, **kwargs: [
            {"prompt": "p", "problem": "p", "effort": "low"}
        ]
        * kwargs["max_groups"],
    )
    monkeypatch.setattr(
        "reasoning_levels.grpo_train._load_grpo_trainer",
        lambda: trainer_loads.append(True) or Trainer,
    )
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.build_grpo_config",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.make_reasoning_effort_reward",
        lambda tokenizer, **kwargs: object(),
    )
    monkeypatch.setattr("reasoning_levels.grpo_train.write_jsonl", lambda *args: None)
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.write_problem_manifest", lambda *args: None
    )
    monkeypatch.setattr(
        "reasoning_levels.grpo_train.write_run_metadata", lambda *args: None
    )

    main(
        [
            "--adapter",
            str(tmp_path / "sft"),
            "--output",
            str(tmp_path / "grpo"),
            "--max-steps",
            "3",
        ]
    )

    assert trainer_loads == [True]
