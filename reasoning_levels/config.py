import os
from dataclasses import dataclass, field


@dataclass
class APIConfig:
    base_url: str = ""
    api_key: str = ""
    model_name: str = "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast"
    max_tokens: dict = field(default_factory=lambda: {
        "low": 32768,
        "medium": 32768,
        "high": 32768,
    })

    @classmethod
    def from_env(cls):
        max_completion_tokens = int(os.environ.get("QWEN_MAX_COMPLETION_TOKENS", "32768"))
        return cls(
            base_url=os.environ.get("QWEN_API_BASE", ""),
            api_key=os.environ.get("QWEN_API_KEY", ""),
            max_tokens={level: max_completion_tokens for level in ["low", "medium", "high"]},
        )


@dataclass
class DatasetConfig:
    source: str = "AI-MO/NuminaMath-CoT"
    n_problems: int = 1000
    candidates_per_problem: int = 3
    max_workers: int = 32
    max_solution_chars: int | None = 2000
    output_path: str = "data/reasoning_effort_math.jsonl"
    hf_repo: str = "valendra-tech/qwen36-reasoning-effort-math"
    sources_balance: list = field(default_factory=lambda: [
        "olympiads", "synthetic_math", "cn_k12", "orca_math",
    ])


@dataclass
class EffortConfig:
    levels: list = field(default_factory=lambda: ["low", "medium", "high"])
    system_prompts: dict = field(default_factory=lambda: {
        "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration. Put your final answer in \\boxed{}.",
        "medium": "Solve the problem step by step. Put your final answer in \\boxed{}.",
        "high": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer. Put your final answer in \\boxed{}.",
    })


@dataclass
class ExperimentConfig:
    api: APIConfig = field(default_factory=APIConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    effort: EffortConfig = field(default_factory=EffortConfig)

    @classmethod
    def from_env(cls):
        dataset = DatasetConfig(
            candidates_per_problem=int(os.environ.get("QWEN_CANDIDATES_PER_PROBLEM", "3")),
            max_workers=int(os.environ.get("QWEN_MAX_WORKERS", "32")),
            max_solution_chars=int(os.environ.get("QWEN_MAX_SOLUTION_CHARS", "2000")),
        )
        return cls(api=APIConfig.from_env(), dataset=dataset)
