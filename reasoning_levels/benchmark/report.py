"""Aggregate raw benchmark records into JSON and Markdown reports.

Standard library only: the report is regenerated locally from the JSONL
telemetry pulled from the remote host.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TASK_LABELS = {
    "aime25": "AIME 2025",
    "math500": "MATH-500",
    "gpqa_diamond": "GPQA-Diamond",
    "code_basic": "Code (basic suite)",
}

# Published Qwen/Qwen3.6-35B-A3B model-card values. They are an external
# reference only: the published protocol (81,920-token budget for hard math,
# temperature 1.0, top_p 0.95) differs from this harness and the base model is
# not re-run here.
BASE_REFERENCE = {
    "aime25": {
        "value": None,
        "comparable": False,
        "note": "The model card publishes AIME 26 (92.7), not AIME 2025; different year set.",
    },
    "math500": {
        "value": None,
        "comparable": False,
        "note": "No MATH-500 value published in the model card.",
    },
    "gpqa_diamond": {
        "value": 86.0,
        "comparable": True,
        "note": "Model-card GPQA (86.0); published sampling protocol differs from this greedy harness.",
    },
    "code_basic": {
        "value": None,
        "comparable": False,
        "note": "No published value; this 40-task suite is authored in-repo.",
    },
}


def load_records(raw_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(Path(raw_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
    return records


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile with half-up rounding between ranks."""
    if not values:
        return None
    ordered = sorted(values)
    index = int(q * (len(ordered) - 1) + 0.5)
    return ordered[index]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 1.0
    phat = successes / total
    denominator = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def paired_delta_ci(
    base_flags: dict[str, bool],
    candidate_flags: dict[str, bool],
    iterations: int = 10000,
    seed: int = 1234,
) -> tuple[float, float]:
    """Bootstrap a 95% CI for the paired accuracy delta over shared problems."""
    import random

    shared = sorted(set(base_flags) & set(candidate_flags))
    if len(shared) != len(base_flags) or len(shared) != len(candidate_flags):
        raise ValueError("conditions do not share the same problems")
    if not shared:
        raise ValueError("no shared problems")
    base_values = [1.0 if base_flags[key] else 0.0 for key in shared]
    candidate_values = [1.0 if candidate_flags[key] else 0.0 for key in shared]
    rng = random.Random(seed)
    count = len(shared)
    deltas: list[float] = []
    for _ in range(iterations):
        draws = [rng.randrange(count) for _ in range(count)]
        base_mean = sum(base_values[index] for index in draws) / count
        candidate_mean = sum(candidate_values[index] for index in draws) / count
        deltas.append(candidate_mean - base_mean)
    deltas.sort()
    return deltas[int(0.025 * iterations)], deltas[int(0.975 * iterations) - 1]


def summarize(records: list[dict]) -> dict:
    """Group records by task and condition and compute accuracy and token stats."""
    grouped: dict[str, dict[str, list[dict]]] = {}
    for record in records:
        grouped.setdefault(record["task"], {}).setdefault(record["condition"], []).append(
            record
        )

    summary: dict[str, dict[str, dict]] = {}
    for task, conditions in grouped.items():
        summary[task] = {}
        for condition, items in conditions.items():
            completion_tokens = [
                float(item["completion_tokens"])
                for item in items
                if item.get("completion_tokens") is not None
            ]
            reasoning_tokens = [
                float(item["reasoning_tokens_estimate"])
                for item in items
                if item.get("reasoning_tokens_estimate") is not None
            ]
            verify_values = [
                item["math_verify"]
                for item in items
                if item.get("math_verify") is not None
            ]
            correct = sum(1 for item in items if item.get("correct"))
            ci_low, ci_high = wilson_interval(correct, len(items))
            summary[task][condition] = {
                "n": len(items),
                "correct": correct,
                "accuracy": correct / len(items),
                "accuracy_ci_low": ci_low,
                "accuracy_ci_high": ci_high,
                "math_verify_accuracy": (
                    _mean([1.0 if value else 0.0 for value in verify_values])
                ),
                "math_verify_n": len(verify_values),
                "mean_completion_tokens": _mean(completion_tokens),
                "median_completion_tokens": percentile(completion_tokens, 0.5),
                "p90_completion_tokens": percentile(completion_tokens, 0.9),
                "mean_reasoning_tokens": _mean(reasoning_tokens),
                "truncation_share": (
                    sum(1 for item in items if item.get("finish_reason") == "length")
                    / len(items)
                ),
            }
    return summary


CAVEATS = """\
- Preview release: the Sherry model card states independent quality evaluation
  was incomplete.
- Training data derives from NuminaMath, so AIME 2025 and MATH-500 may be
  contaminated; treat the measured scores as an upper bound.
- GPQA-Diamond comes from the public mirror `hendrydong/gpqa_diamond_mc`
  (198 rows) because the official `Idavidrein/gpqa` repository is gated.
- Greedy single-sample decoding with one shared token budget; small tasks have
  wide Wilson confidence intervals.
__BASE__
- AIME 2025 has no published base value; the model card reports AIME 26.
- `reasoning_tokens_estimate` comes from re-tokenizing the reasoning text and
  is an approximation, not the server's exact reasoning token count.
- `exact_match` uses a heuristic extractor for unboxed answers and can
  under-count prose answers; `math_verify` is the more robust math metric.
"""

BASE_REFERENCE_CAVEAT = """\
- The base model was not re-run. The reference table uses published model-card
  values whose sampling protocol differs (temperature 1.0, top_p 0.95, and an
  81,920-token budget for hard math), so it is context, not a controlled delta."""

BASE_RE_RUN_CAVEAT = """\
- The base model was re-run in this harness under the same protocol, so the
  paired deltas above are controlled comparisons. The published model-card
  table is context only, because its sampling protocol differs (temperature
  1.0, top_p 0.95, and an 81,920-token budget for hard math)."""


def caveats_text(has_base: bool) -> str:
    base_line = BASE_RE_RUN_CAVEAT if has_base else BASE_REFERENCE_CAVEAT
    return CAVEATS.replace("__BASE__", base_line)


def _format(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _math_verify_cell(stats: dict) -> str:
    if stats["math_verify_accuracy"] is None:
        return "n/a"
    return f"{_format(stats['math_verify_accuracy'])} (n={stats['math_verify_n']})"


def build_markdown(summary: dict, records: list[dict]) -> str:
    """Render the Markdown report with absolute scores and the base reference."""
    has_base = any(record.get("condition") == "base" for record in records)
    by_task_condition: dict[str, dict[str, dict[str, bool]]] = {}
    by_task_condition_verify: dict[str, dict[str, dict[str, bool]]] = {}
    for record in records:
        by_task_condition.setdefault(record["task"], {}).setdefault(
            record["condition"], {}
        )[record["problem_id"]] = bool(record.get("correct"))
        if record.get("math_verify") is not None:
            by_task_condition_verify.setdefault(record["task"], {}).setdefault(
                record["condition"], {}
            )[record["problem_id"]] = bool(record["math_verify"])
    if has_base:
        title = "# Sherry 35B-A3B 0.1 GRPO Preview vs Base - Public Benchmark Report"
        intro = (
            "All numbers in this section are measured by this harness; the base "
            "model was re-run in this harness under the same protocol."
        )
    else:
        title = "# Sherry 35B-A3B 0.1 GRPO Preview at Maximum Reasoning - Benchmark Report"
        intro = (
            "All numbers in this section are measured by this harness; the base "
            "model was not re-run."
        )
    lines: list[str] = [
        title,
        "",
        intro,
        "",
    ]
    for task in sorted(summary, key=lambda name: list(TASK_LABELS).index(name)):
        label = TASK_LABELS.get(task, task)
        lines.append(f"## {label}")
        lines.append("")
        lines.append(
            "| condition | n | accuracy | 95% CI | math_verify | mean tokens "
            "| median tokens | p90 tokens | mean reasoning | truncated |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for condition in sorted(summary[task]):
            stats = summary[task][condition]
            lines.append(
                f"| {condition} | {stats['n']} | {_format(stats['accuracy'])} "
                f"| {_format(stats['accuracy_ci_low'])} to {_format(stats['accuracy_ci_high'])} "
                f"| {_math_verify_cell(stats)} "
                f"| {_format(stats['mean_completion_tokens'], 1)} "
                f"| {_format(stats['median_completion_tokens'], 1)} "
                f"| {_format(stats['p90_completion_tokens'], 1)} "
                f"| {_format(stats['mean_reasoning_tokens'], 1)} "
                f"| {_format(stats['truncation_share'])} |"
            )
        lines.append("")
        if has_base and task in by_task_condition:
            conditions = by_task_condition[task]
            base_flags = conditions.get("base")
            if base_flags:
                for condition in sorted(conditions):
                    if condition == "base":
                        continue
                    candidate_flags = conditions[condition]
                    try:
                        low, high = paired_delta_ci(base_flags, candidate_flags)
                    except ValueError:
                        lines.append(
                            f"- Paired delta vs base for {condition}: not computable."
                        )
                        continue
                    delta = (
                        sum(candidate_flags.values()) / len(candidate_flags)
                        - sum(base_flags.values()) / len(base_flags)
                    )
                    lines.append(
                        f"- Paired delta vs base on {len(base_flags)} problems: "
                        f"{delta:+.3f} (95% CI {low:+.3f} to {high:+.3f})."
                    )
        if has_base and task in by_task_condition_verify:
            base_verify = by_task_condition_verify[task].get("base", {})
            if base_verify:
                for condition in sorted(by_task_condition_verify[task]):
                    if condition == "base":
                        continue
                    candidate_verify = by_task_condition_verify[task][condition]
                    if not candidate_verify:
                        continue
                    shared = sorted(set(base_verify) & set(candidate_verify))
                    base_shared = {key: base_verify[key] for key in shared}
                    candidate_shared = {key: candidate_verify[key] for key in shared}
                    try:
                        low, high = paired_delta_ci(base_shared, candidate_shared)
                    except ValueError:
                        continue
                    delta = (
                        sum(candidate_shared.values()) / len(candidate_shared)
                        - sum(base_shared.values()) / len(base_shared)
                    )
                    lines.append(
                        f"- Paired delta vs base on {len(shared)} problems "
                        f"(math_verify): {delta:+.3f} "
                        f"(95% CI {low:+.3f} to {high:+.3f})."
                    )
        lines.append("")
        reference = BASE_REFERENCE.get(task)
        if reference:
            if reference["value"] is None:
                lines.append(
                    f"- Published base reference: not available. {reference['note']}"
                )
            else:
                lines.append(
                    f"- Published base reference: {reference['value']:.1f}. "
                    f"{'Comparable benchmark, different protocol.' if reference['comparable'] else 'Not comparable.'} "
                    f"{reference['note']}"
                )
        lines.append("")
    if has_base:
        lines.append("## Base Reference (Published Model Card)")
        lines.append("")
        lines.append(
            "The base model was also re-run in this harness under the same "
            "protocol; the published numbers below are context only."
        )
    else:
        lines.append("## Base Reference (Published Model Card, Not Re-Run)")
    lines.append("")
    lines.append(
        "| task | published base value | comparable | note |"
    )
    lines.append("|---|---|---|---|")
    for task in sorted(BASE_REFERENCE, key=lambda name: list(TASK_LABELS).index(name)):
        reference = BASE_REFERENCE[task]
        value = "n/a" if reference["value"] is None else f"{reference['value']:.1f}"
        comparable = "yes" if reference["comparable"] else "no"
        lines.append(
            f"| {TASK_LABELS.get(task, task)} | {value} | {comparable} "
            f"| {reference['note']} |"
        )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(caveats_text(has_base))
    return "\n".join(lines) + "\n"


def build_report(raw_dir: Path, output_dir: Path) -> dict:
    """Write ``report.json`` and ``REPORT.md`` from raw JSONL telemetry."""
    records = load_records(raw_dir)
    summary = summarize(records)
    revisions = sorted({record.get("dataset_revision") for record in records})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": len(records),
        "dataset_revisions": revisions,
        "base_reference": BASE_REFERENCE,
        "base_model_re_run": any(
            record.get("condition") == "base" for record in records
        ),
        "summary": summary,
    }
    (output_dir / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        build_markdown(summary, records), encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    payload = build_report(Path(args.raw_dir), Path(args.output_dir))
    print(f"wrote report for {payload['records']} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
