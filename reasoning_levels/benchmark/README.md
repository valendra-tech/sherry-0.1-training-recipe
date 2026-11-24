# Sherry 0.1 Benchmark

This is the evaluation harness used to compare the
[merged Sherry 0.1 Preview](https://huggingface.co/valendra/sherry-35b-a3b-0.1-preview)
with [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). It uses
pinned benchmark revisions and an OpenAI-compatible vLLM server. The results
from September 18, 2026 are archived as
[Markdown](reports/2026-09-18-sherry-high-vs-base.md) and
[JSON](reports/2026-09-18-sherry-high-vs-base.json). Raw model responses are
not distributed.

| Task | Source | Problems | Completion budget |
| --- | --- | ---: | ---: |
| `math500` | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 500 | 32,768 |
| `gpqa_diamond` | [GPQA-Diamond public mirror](https://huggingface.co/datasets/hendrydong/gpqa_diamond_mc) | 198 | 32,768 |
| `aime25` | [AIME 2025](https://huggingface.co/datasets/math-ai/aime25) | 30 | 32,768 |
| `code_basic` | [In-repo Python suite](code_tasks/basic_python.json) | 40 | 16,384 |

[`datasets.py`](datasets.py) pins the dataset revisions. The prompts and
scoring are in [`prompts.py`](prompts.py) and [`scoring.py`](scoring.py).
These are single-sample, greedy runs (`temperature=0`, `top_p=1`, seed `0`).
The base condition omits Sherry's `reasoning_effort` setting.

## 1. Install the client and run the tests

From the repository root, after installing
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked --extra benchmark
uv run --locked --offline --extra benchmark python -m pytest -q \
  reasoning_levels/benchmark/tests
```

This extra installs the benchmark client, symbolic math verifier, and
tokenizer dependencies. It does not need CUDA or PyTorch. Actual model
generation requires a separate serving environment and suitable GPU memory.

## 2. Serve Sherry and verify the connection

The published comparison used vLLM 0.29.0 and BF16 weights. Follow the
[vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
in a separate environment. Serve one model at a time. Do not enable a
server-side reasoning parser when matching the historical protocol; the
client splits inline reasoning from the final answer.

```bash
vllm serve valendra/sherry-35b-a3b-0.1-preview \
  --served-model-name sherry --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 40960 --max-num-seqs 16 --seed 0
```

The context must accommodate both the prompt and the completion budget; the
short context used for the inference example in the root README is not
enough for the full benchmark. From another terminal, check readiness and
run a small smoke test into a separate directory:

```bash
curl --fail http://127.0.0.1:8000/v1/models

uv run --locked --extra benchmark python -m reasoning_levels.benchmark.run \
  --base-url http://127.0.0.1:8000/v1 --model sherry \
  --condition sherry-high --effort high --tasks math500 --limit 5 \
  --concurrency 1 --max-tokens 4096 --output-dir runs/benchmark-smoke
```

## 3. Run the math and science tasks

With Sherry still serving:

```bash
uv run --locked --extra benchmark python -m reasoning_levels.benchmark.run \
  --base-url http://127.0.0.1:8000/v1 --model sherry \
  --condition sherry-high --effort high --tasks aime25,math500,gpqa_diamond \
  --concurrency 16 --max-tokens 32768 --temperature 0 --seed 0 \
  --tokenizer valendra/sherry-35b-a3b-0.1-preview --output-dir runs/benchmark
```

Stop your Sherry server, then start the base model with matching settings:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --served-model-name base --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 40960 --max-num-seqs 16 --seed 0
```

When the base server is ready:

```bash
uv run --locked --extra benchmark python -m reasoning_levels.benchmark.run \
  --base-url http://127.0.0.1:8000/v1 --model base \
  --condition base --effort none --tasks aime25,math500,gpqa_diamond \
  --concurrency 16 --max-tokens 32768 --temperature 0 --seed 0 \
  --tokenizer Qwen/Qwen3.6-35B-A3B --output-dir runs/benchmark
```

The runner writes `runs/benchmark/raw/<condition>__<task>.jsonl`. Repeating
an identical command resumes from completed problem IDs. Use a fresh output
directory if model weights, prompts, precision, or generation settings
change. The optional tokenizer estimates reasoning length separately from
the completion token count returned by the server.

## 4. Optionally run the Python suite

**The `code_basic` task executes model-generated Python.** The grader's
subprocess, resource limits, and timeout are not a security boundary.
Only run this task in a disposable, isolated environment with no secrets,
host mounts, or external network access. The default tasks do not execute
generated code.

In that environment, while serving the matching model on loopback, repeat
each condition's command above with `--tasks code_basic --max-tokens 16384`
instead of the math and science task list and budget. Use `sherry-high` for
Sherry and `base` for Qwen so the grader can pair the same 40 problems. Keep
their records with the other task results inside your evaluation environment.

## 5. Recompute Math-Verify and build the report

Before reporting the MATH-500 Math-Verify score, rescore it serially: the
original concurrent generator uses worker threads, where Math-Verify's
timeout handling can mark valid answers incorrect. The recompute command
updates the raw MATH-500 files in place; run it on copies if you need to
retain the initial records.

```bash
uv run --locked --extra benchmark python -m reasoning_levels.benchmark.recompute_math_verify \
  --raw-dir runs/benchmark/raw --tasks math500

uv run --locked --extra benchmark python -m reasoning_levels.benchmark.report \
  --raw-dir runs/benchmark/raw --output-dir runs/benchmark/report
```

The report contains `REPORT.md` and `report.json` with scores, sample
counts, token statistics, Wilson confidence intervals, and paired deltas
when both conditions are present. Do not interpret AIME Math-Verify from a
threaded run; the historical AIME comparison remains preliminary. MATH-500
and AIME may overlap with the training source, and GPQA uses a public mirror
rather than the gated original dataset. See the
[model card](https://huggingface.co/valendra/sherry-35b-a3b-0.1-preview#evaluation)
for additional caveats.
