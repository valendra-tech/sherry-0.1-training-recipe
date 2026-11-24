from __future__ import annotations
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from openai import OpenAI
from .config import ExperimentConfig
from .boxed import extract_answer, answers_match
from .sample_problems import sample_problems

EFFORT_LEVELS = ["low", "medium", "high"]
CANDIDATE_GENERATION_EFFORT = "medium"
CANDIDATE_SYSTEM_PROMPT = "Put your final answer in \\boxed{}."

# System prompts matching Qwen3.8's chat template reasoning_effort instructions.
# low: brief thinking, high: thorough thinking (mapped from xhigh).
# All include the \boxed{} instruction since we need verifiable answers.
EFFORT_SYSTEM_PROMPTS = {
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration. Put your final answer in \\boxed{}.",
    "medium": "Solve the problem step by step. Put your final answer in \\boxed{}.",
    "high": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer. Put your final answer in \\boxed{}.",
}

EFFORT_ENABLE_THINKING = {
    "low": True,
    "medium": True,
    "high": True,
}

# This is only a completion ceiling. Effort is controlled by the prompt and
# enable_thinking, not by giving each level a different token budget.
MAX_COMPLETION_TOKENS = 32768
EFFORT_MAX_TOKENS = {level: MAX_COMPLETION_TOKENS for level in EFFORT_LEVELS}


def build_messages(problem: str, effort: str, system_prompt: str | None = None) -> list[dict]:
    """Build the API messages for a given problem and effort level."""
    if system_prompt is None:
        system_prompt = EFFORT_SYSTEM_PROMPTS[effort]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]


def parse_thinking_response(message) -> tuple[str | None, str]:
    """Extract thinking and response from an API message.
    Handles three formats:
    1. 'reasoning' field (vLLM with reasoning parser, Qwen3.6 NVFP4)
    2. 'reasoning_content' field (some OpenAI-compatible APIs)
    3. Inline Mud format in content
    """
    thinking = None
    # Try 'reasoning' field first (vLLM Qwen3.6 format)
    if hasattr(message, "reasoning") and message.reasoning:
        thinking = message.reasoning
        response = message.content or ""
        return thinking, response
    # Try 'reasoning_content' field (some APIs)
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        thinking = message.reasoning_content
        response = message.content or ""
        return thinking, response
    # Parse inline: Mud\n[thinking]\n\n\n\n[response]
    content = message.content or ""
    if " Müd" in content:
        parts = content.split(" Müd", 1)
        after_think = parts[1] if len(parts) > 1 else ""
        if "\n\n\n\n" in after_think:
            thinking, response = after_think.split("\n\n\n\n", 1)
            thinking = thinking.lstrip("\n").strip()
            response = response.strip()
        else:
            thinking = after_think.strip()
            response = ""
    else:
        response = content
    return thinking, response


def generate_trace(client, model: str, problem: str, effort: str,
                   max_tokens: int = None, system_prompt: str | None = None) -> dict:
    """Generate a single trace via the API. Returns {"thinking": str, "response": str}."""
    if max_tokens is None:
        max_tokens = EFFORT_MAX_TOKENS[effort]

    messages = build_messages(problem, effort, system_prompt=system_prompt)
    enable_thinking = EFFORT_ENABLE_THINKING[effort]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    choice = response.choices[0]
    thinking, resp = parse_thinking_response(choice.message)
    finish_reason = getattr(choice, "finish_reason", None)
    if not isinstance(finish_reason, str):
        finish_reason = None
    usage = getattr(response, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    thinking_tokens = getattr(completion_details, "reasoning_tokens", None)
    if not isinstance(thinking_tokens, int):
        thinking_tokens = len((thinking or "").split())
    return {
        "thinking": thinking,
        "response": resp,
        "finish_reason": finish_reason,
        "thinking_tokens": thinking_tokens,
    }


def _thinking_length(trace: dict) -> int:
    """Return an API token count, with a text-length fallback for other APIs."""
    token_count = trace.get("thinking_tokens")
    if isinstance(token_count, int) and token_count >= 0:
        return token_count
    return len((trace.get("thinking") or "").split())


def select_effort_traces(candidates: list[dict]) -> list[dict]:
    """Select verified traces and label them by observed thinking length.

    The shortest correct candidate becomes low, the middle candidate becomes
    medium, and the longest candidate becomes high. Traces stopped by the
    completion limit are never eligible.
    """
    valid = [
        candidate
        for candidate in candidates
        if candidate.get("correct") is True
        and candidate.get("finish_reason") != "length"
        and candidate.get("response")
    ]
    if len(valid) < len(EFFORT_LEVELS):
        return []

    ordered = sorted(valid, key=_thinking_length)
    middle_index = (len(ordered) - 1) // 2
    selected_indexes = [0, middle_index, len(ordered) - 1]
    selected = []
    for effort, index in zip(EFFORT_LEVELS, selected_indexes):
        trace = dict(ordered[index])
        trace["effort"] = effort
        selected.append(trace)
    return selected


def problem_batch_size(max_workers: int, candidates_per_problem: int) -> int:
    """Choose a problem batch that keeps all candidate requests within the pool."""
    requests_per_problem = candidates_per_problem
    return max(1, max_workers // max(requests_per_problem, 1))


def completed_problem_keys(results: list[dict]) -> set[str]:
    """Return problems that already have all three selected effort labels."""
    efforts_by_problem: dict[str, set[str]] = {}
    for result in results:
        problem = result.get("problem")
        effort = result.get("effort")
        if problem and effort:
            efforts_by_problem.setdefault(problem, set()).add(effort)
    required = set(EFFORT_LEVELS)
    return {
        problem
        for problem, efforts in efforts_by_problem.items()
        if required.issubset(efforts)
    }


def load_existing_results(output_path: str) -> list[dict]:
    """Load completed JSONL rows so an interrupted run can resume safely."""
    if not os.path.exists(output_path):
        return []
    results = []
    with open(output_path) as file:
        for line in file:
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(result, dict):
                results.append(result)
    return results


def generate_dataset(config: ExperimentConfig, output_path: str = None):
    """Full pipeline: sample problems, generate traces at 3 levels, verify, save."""
    if output_path is None:
        output_path = config.dataset.output_path

    api_cfg = config.api
    if not api_cfg.base_url or not api_cfg.api_key:
        raise ValueError("QWEN_API_BASE and QWEN_API_KEY must be set (env vars or .env file)")

    client = OpenAI(
        base_url=api_cfg.base_url,
        api_key=api_cfg.api_key,
        http_client=httpx.Client(verify=False),
    )

    print(f"Sampling {config.dataset.n_problems} problems from NuminaMath-CoT...")
    problems = sample_problems(
        n=config.dataset.n_problems,
        balance_sources=True,
        sources=config.dataset.sources_balance,
        max_solution_chars=config.dataset.max_solution_chars,
    )
    print(f"Sampled {len(problems)} problems.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results = load_existing_results(output_path)
    existing_completed = completed_problem_keys(results)
    if existing_completed:
        print(f"Resuming: {len(existing_completed)} problems already complete.")

    stats = {
        "candidate_total": 0,
        "candidate_correct": 0,
        "total": len(results),
        "correct": sum(int(result.get("correct") is True) for result in results),
        "by_level": {l: {"total": 0, "correct": 0} for l in EFFORT_LEVELS},
    }
    for result in results:
        effort = result.get("effort")
        if effort in stats["by_level"]:
            stats["by_level"][effort]["total"] += 1
            stats["by_level"][effort]["correct"] += int(result.get("correct") is True)

    pending = [
        (i, problem)
        for i, problem in enumerate(problems)
        if problem["problem"] not in existing_completed
    ]
    batch_size = problem_batch_size(
        config.dataset.max_workers,
        config.dataset.candidates_per_problem,
    )
    print(f"Generating {len(pending)} problems in batches of {batch_size}...")

    with ThreadPoolExecutor(max_workers=config.dataset.max_workers) as executor:
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start:batch_start + batch_size]
            candidates_by_problem = {problem_index: [] for problem_index, _ in batch}
            requests = {
                executor.submit(
                    generate_trace,
                    client=client,
                    model=api_cfg.model_name,
                    problem=problem["problem"],
                    effort=CANDIDATE_GENERATION_EFFORT,
                    max_tokens=api_cfg.max_tokens.get(CANDIDATE_GENERATION_EFFORT, MAX_COMPLETION_TOKENS),
                    system_prompt=CANDIDATE_SYSTEM_PROMPT,
                ): (problem_index, problem, CANDIDATE_GENERATION_EFFORT, candidate_index)
                for problem_index, problem in batch
                for candidate_index in range(config.dataset.candidates_per_problem)
            }

            for future in as_completed(requests):
                problem_index, problem, requested_effort, candidate_index = requests[future]
                try:
                    trace = future.result()
                except Exception as e:
                    print(f"  [ERROR] problem {problem_index} effort={requested_effort} candidate={candidate_index}: {e}")
                    continue

                if trace["finish_reason"] == "length":
                    print(f"  [TRUNCATED] problem {problem_index} effort={requested_effort} candidate={candidate_index}")

                predicted = extract_answer(trace["response"])
                correct = answers_match(predicted, problem["ground_truth"])
                stats["candidate_total"] += 1
                if correct:
                    stats["candidate_correct"] += 1

                candidates_by_problem[problem_index].append({
                    "problem": problem["problem"],
                    "source": problem["source"],
                    "requested_effort": requested_effort,
                    "thinking": trace["thinking"],
                    "thinking_tokens": trace["thinking_tokens"],
                    "response": trace["response"],
                    "predicted_answer": predicted,
                    "ground_truth": problem["ground_truth"],
                    "correct": correct,
                    "finish_reason": trace["finish_reason"],
                })

            for problem_index, problem in batch:
                selected = select_effort_traces(candidates_by_problem[problem_index])
                if not selected:
                    print(f"  [SKIP] problem {problem_index}: fewer than 3 verified candidates")
                    continue

                for entry in selected:
                    results.append(entry)
                    stats["total"] += 1
                    stats["correct"] += int(entry["correct"])
                    stats["by_level"][entry["effort"]]["total"] += 1
                    stats["by_level"][entry["effort"]]["correct"] += int(entry["correct"])

                processed = len(existing_completed) + batch_start + batch.index((problem_index, problem)) + 1
                if processed % 10 == 0 or processed == len(problems):
                    print(f"  Progress: {processed}/{len(problems)} problems, "
                          f"{len(results)} selected traces from {stats['candidate_total']} candidates")

                with open(output_path, "w") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")

    print(f"\n=== Generation complete ===")
    print(f"Candidate traces: {stats['candidate_total']}")
    print(f"Candidate correct: {stats['candidate_correct']} "
          f"({100*stats['candidate_correct']/max(stats['candidate_total'],1):.1f}%)")
    print(f"Selected traces: {stats['total']}")
    for level in EFFORT_LEVELS:
        s = stats["by_level"][level]
        print(f"  {level}: {s['correct']}/{s['total']} "
              f"({100*s['correct']/max(s['total'],1):.1f}%)")

    return results, stats


if __name__ == "__main__":
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    config = ExperimentConfig.from_env()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else config.dataset.n_problems
    config.dataset.n_problems = n
    output = sys.argv[2] if len(sys.argv) > 2 else config.dataset.output_path
    generate_dataset(config, output_path=output)
