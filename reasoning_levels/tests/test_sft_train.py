import sys
import types

import pytest

from reasoning_levels.chat_template import THINKING_CLOSE, THINKING_OPEN
from reasoning_levels.sft_train import (
    MODEL_NAME,
    SFTDataCollator,
    TokenizedSFTDataset,
    build_unsloth_lora_kwargs,
    build_training_arguments,
    load_model_and_processor,
    validate_trainable_lora_scope,
)


def test_trainer_uses_the_original_bf16_qwen_checkpoint():
    assert MODEL_NAME == "Qwen/Qwen3.6-35B-A3B"


def test_unsloth_lora_kwargs_target_modules_and_token_rows():
    token_ids = [248077, 248078, 248079]
    kwargs = build_unsloth_lora_kwargs(
        token_ids,
        input_embedding_name="model.embed_tokens",
        output_embedding_name="lm_head",
        max_length=2048,
    )

    assert kwargs["r"] == 32
    assert kwargs["lora_alpha"] == 64
    assert kwargs["lora_dropout"] == 0.0
    assert kwargs["bias"] == "none"
    assert kwargs["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "shared_expert.gate_proj",
        "shared_expert.up_proj",
        "shared_expert.down_proj",
    ]
    assert kwargs["trainable_token_indices"] == {
        "model.embed_tokens": token_ids,
        "lm_head": token_ids,
    }
    assert kwargs["use_gradient_checkpointing"] == "unsloth"
    assert kwargs["max_seq_length"] == 2048


def test_training_arguments_use_single_bf16_batch_and_checkpointing(tmp_path):
    pytest.importorskip("transformers")

    arguments = build_training_arguments(tmp_path)

    assert arguments.per_device_train_batch_size == 1
    assert arguments.gradient_accumulation_steps == 16
    assert arguments.bf16 is True
    assert arguments.gradient_checkpointing is True
    assert arguments.learning_rate == 1e-4
    assert arguments.max_steps == 500


def test_collator_pads_model_inputs_and_labels_consistently():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    batch = SFTDataCollator(Tokenizer(), pad_to_multiple_of=4)([
        {
            "input_ids": [4, 5, 6],
            "attention_mask": [1, 1, 1],
            "mm_token_type_ids": [0, 0, 0],
            "labels": [4, 5, 6],
        },
        {
            "input_ids": [7, 8],
            "attention_mask": [1, 1],
            "mm_token_type_ids": [0, 0],
            "labels": [7, 8],
        },
    ])

    assert batch["input_ids"].tolist() == [[4, 5, 6, 0], [7, 8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]
    assert batch["mm_token_type_ids"].tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert batch["labels"].tolist() == [[4, 5, 6, -100], [7, 8, -100, -100]]
    assert batch["input_ids"].dtype == torch.long


def test_tokenized_dataset_keeps_multimodal_metadata_without_truncation():
    class Tokenizer:
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            return "<prompt><think>\n"

    class Processor:
        tokenizer = Tokenizer()

        def __call__(self, text, max_length=None, truncation=False, **kwargs):
            ids = list(range(6))
            if truncation:
                ids = ids[:max_length]
            return {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "mm_token_type_ids": [0] * len(ids),
            }

    dataset = TokenizedSFTDataset(
        [{"problem": "p", "effort": "low", "thinking": "reason", "response": "answer"}],
        Processor(),
        max_length=8,
    )

    assert len(dataset) == 1
    assert dataset.truncated_count == 0
    assert dataset[0]["input_ids"] == [0, 1, 2, 3, 4, 5]
    assert dataset[0]["labels"] == [0, 1, 2, 3, 4, 5]
    assert dataset[0]["mm_token_type_ids"] == [0, 0, 0, 0, 0, 0]


def test_tokenized_dataset_preserves_answer_when_thinking_is_truncated():
    class Tokenizer:
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            return "<prompt>" + THINKING_OPEN

    class Processor:
        tokenizer = Tokenizer()

        def __call__(self, text, max_length=None, truncation=False, **kwargs):
            if text == THINKING_CLOSE:
                ids = [90, 91]
            elif text.endswith(THINKING_OPEN):
                ids = [10, 11]
            else:
                ids = [10, 11, 20, 21, 22, 23, 90, 91, 30, 31, 32]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            return {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "mm_token_type_ids": [0] * len(ids),
            }

    dataset = TokenizedSFTDataset(
        [{"problem": "p", "effort": "low", "thinking": "long", "response": "answer"}],
        Processor(),
        max_length=8,
    )

    assert dataset.truncated_count == 1
    assert dataset[0]["input_ids"] == [10, 11, 20, 90, 91, 30, 31, 32]
    assert dataset[0]["labels"] == dataset[0]["input_ids"]
    assert dataset[0]["mm_token_type_ids"] == [0] * 8


def test_unsloth_lora_kwargs_reject_missing_embedding_names():
    with pytest.raises(ValueError, match="embedding module names"):
        build_unsloth_lora_kwargs(
            [248077, 248078, 248079],
            input_embedding_name="",
            output_embedding_name="lm_head",
            max_length=2048,
        )


def test_trainable_scope_rejects_routed_moe_experts():
    class Parameter:
        requires_grad = True

    class Model:
        def named_parameters(self):
            return [
                (
                    "model.layers.0.mlp.experts.gate_up_proj.lora_A.weight",
                    Parameter(),
                )
            ]

    with pytest.raises(ValueError, match="routed MoE expert"):
        validate_trainable_lora_scope(Model())


def test_loader_uses_unsloth_bf16_and_selective_embedding_rows(monkeypatch):
    import torch

    class Tokenizer:
        eos_token = "<eos>"
        unk_token_id = 0

        def __init__(self):
            self.tokens = []

        def __len__(self):
            return 103

        def add_special_tokens(self, mapping):
            self.tokens.extend(mapping["additional_special_tokens"])
            return len(mapping["additional_special_tokens"])

        def convert_tokens_to_ids(self, token):
            return 100 + self.tokens.index(token)

        def save_pretrained(self, output_dir):
            pass

    class Embeddings:
        def __init__(self, num_embeddings):
            self.num_embeddings = num_embeddings

    class Processor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def save_pretrained(self, output_dir):
            pass

    class Model:
        def __init__(self):
            self.input_embeddings = Embeddings(100)
            self.output_embeddings = Embeddings(100)
            self.config = types.SimpleNamespace()

        def get_input_embeddings(self):
            return self.input_embeddings

        def get_output_embeddings(self):
            return self.output_embeddings

        def resize_token_embeddings(self, size):
            self.input_embeddings = Embeddings(size)
            self.output_embeddings = Embeddings(size)

        def named_modules(self):
            return [
                ("model.embed_tokens", self.input_embeddings),
                ("lm_head", self.output_embeddings),
                ("model.layers.0.self_attn.q_proj", object()),
                ("model.layers.0.self_attn.k_proj", object()),
                ("model.layers.0.self_attn.v_proj", object()),
                ("model.layers.0.self_attn.o_proj", object()),
                ("model.layers.0.mlp.gate_proj", object()),
                ("model.layers.0.mlp.up_proj", object()),
                ("model.layers.0.mlp.down_proj", object()),
                ("model.layers.0.mlp.shared_expert.gate_proj", object()),
                ("model.layers.0.mlp.shared_expert.up_proj", object()),
                ("model.layers.0.mlp.shared_expert.down_proj", object()),
            ]

        def named_parameters(self):
            return []

    class FastModel:
        model = Model()
        tokenizer = Tokenizer()
        processor = Processor(tokenizer)
        load_kwargs = None
        lora_kwargs = None

        @classmethod
        def from_pretrained(cls, **kwargs):
            cls.load_kwargs = kwargs
            return cls.model, cls.processor

        @classmethod
        def get_peft_model(cls, model, **kwargs):
            cls.lora_kwargs = kwargs
            return model

    unsloth = types.ModuleType("unsloth")
    unsloth.FastModel = FastModel
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)

    model, processor, tokenizer, token_ids = load_model_and_processor(max_length=2048)

    assert model is FastModel.model
    assert processor is FastModel.processor
    assert tokenizer is FastModel.tokenizer
    assert token_ids == [100, 101, 102]
    assert FastModel.load_kwargs == {
        "model_name": MODEL_NAME,
        "max_seq_length": 2048,
        "dtype": torch.bfloat16,
        "load_in_4bit": False,
        "load_in_16bit": True,
        "full_finetuning": False,
    }
    assert FastModel.lora_kwargs["trainable_token_indices"] == {
        "model.embed_tokens": token_ids,
        "lm_head": token_ids,
    }
