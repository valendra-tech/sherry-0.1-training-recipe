from reasoning_levels.sft_environment import (
    DEFAULT_MODEL_NAME,
    effort_token_names,
    model_auto_class_name,
    validate_target_modules,
)


def test_effort_token_names_are_stable():
    assert effort_token_names() == [
        "<|reasoning_effort_low|>",
        "<|reasoning_effort_medium|>",
        "<|reasoning_effort_high|>",
    ]


def test_validate_target_modules_reports_missing_modules():
    modules = {
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.gate_proj",
    }
    result = validate_target_modules(modules, ["q_proj", "k_proj", "v_proj"])
    assert result == {"q_proj": True, "k_proj": False, "v_proj": False}


def test_validate_target_modules_matches_scoped_suffixes():
    modules = {
        "model.layers.0.mlp.shared_expert.gate_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
    }

    result = validate_target_modules(
        modules,
        ["shared_expert.gate_proj", "shared_expert.up_proj"],
    )

    assert result == {
        "shared_expert.gate_proj": True,
        "shared_expert.up_proj": False,
    }


def test_model_loader_targets_the_multimodal_qwen_class():
    assert model_auto_class_name() == "AutoModelForImageTextToText"


def test_model_probe_uses_the_original_bf16_checkpoint():
    assert DEFAULT_MODEL_NAME == "Qwen/Qwen3.6-35B-A3B"
