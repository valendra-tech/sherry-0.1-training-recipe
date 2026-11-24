"""Thin httpx client for the vLLM OpenAI-compatible chat completions API.

The harness deliberately avoids enabling a server-side reasoning parser: the
model returns thinking inline, and ``completion_to_telemetry`` splits reasoning
from final content deterministically.
"""

from __future__ import annotations

import time
from typing import Any

from reasoning_levels.benchmark.scoring import THINK_OPEN, split_thinking

EXPLICIT_EFFORTS = ("low", "medium", "high")


def make_client(base_url: str, timeout: float = 3600.0) -> Any:
    """Create an httpx client bound to the served model's OpenAI API root."""
    import httpx

    return httpx.Client(base_url=base_url, timeout=timeout)


def build_payload(
    *,
    model: str,
    prompt: str,
    stops: list[str],
    effort: str,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> dict:
    """Build the chat completions payload shared by every condition."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "seed": seed,
        "stop": stops,
    }
    if effort in EXPLICIT_EFFORTS:
        payload["chat_template_kwargs"] = {"reasoning_effort": effort}
    return payload


def generate_one(
    client: Any,
    *,
    model: str,
    prompt: str,
    stops: list[str],
    effort: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    retries: int = 3,
) -> tuple[dict, int]:
    """Send one chat completion request with exponential backoff.

    Returns ``(response_json, latency_ms)`` and raises ``RuntimeError`` after
    the retry budget is exhausted.
    """
    payload = build_payload(
        model=model,
        prompt=prompt,
        stops=stops,
        effort=effort,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        started = time.perf_counter()
        try:
            response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            return response.json(), int((time.perf_counter() - started) * 1000)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"generation failed after {retries} attempts: {last_error}")


def estimate_tokens(tokenizer: Any, text: str) -> int | None:
    """Approximate the token count of a text span, or ``None`` without a tokenizer."""
    if tokenizer is None or not text:
        return None
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


def completion_to_telemetry(data: dict, tokenizer: Any = None) -> dict:
    """Extract reasoning, final content, usage, and finish reason from a response."""
    choice = data["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if not reasoning:
        reasoning, content = split_thinking(content)
    else:
        reasoning = reasoning.removeprefix(THINK_OPEN).strip()
        content = content.strip()
    usage = data.get("usage") or {}
    return {
        "reasoning": reasoning,
        "content": content,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "reasoning_tokens_estimate": estimate_tokens(tokenizer, reasoning),
    }
