# Sherry 0.1 Training Recipe

Sherry 0.1 is an experimental fine-tune of Qwen3.6-35B-A3B. It lets callers
request `low`, `medium`, or `high` reasoning effort from the same model. The
levels bias how the model reasons; they are not hard token limits or guarantees
of correctness.

This repository contains the data preparation, supervised fine-tuning (SFT),
adapter evaluation, and Group Relative Policy Optimization (GRPO) scripts,
plus the original harness for comparing the published Sherry model with its
base model. The training traces and weights are hosted on Hugging Face, not
in Git.

| Resource | What it contains |
| --- | --- |
| [Sherry 0.1 collection](https://huggingface.co/collections/valendra/sherry-01) | Both published model artifacts |
| [SFT Preview](https://huggingface.co/valendra/sherry-35b-a3b-0.1-sft-preview) | The intermediate PEFT LoRA adapter; requires the base model |
| [Merged GRPO Preview](https://huggingface.co/valendra/sherry-35b-a3b-0.1-preview) | The standalone BF16 checkpoint with a `reasoning_effort`-aware chat template |
| [Sherry reasoning-effort dataset](https://huggingface.co/datasets/valendra/sherry-reasoning-effort-0.1-dataset) | The 2,061 verified traces used for SFT |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | The base model for both training stages |
| [Training write-up](https://qdiv0.com/en/blog/sherry-reasoning-effort) | Training decisions, benchmark results, and evaluation caveats |

**Reproducibility scope:** You can run the training workflow below from the
published traces. The current GRPO script loads the base weights in BF16,
whereas the merged Preview's model card describes a QLoRA run. This checkout
does not contain the weight-merging script, so the training recipe is not an
exact rebuild of the published checkpoint. The
[benchmark harness](reasoning_levels/benchmark/README.md) evaluates the
published weights directly.

## 1. Set up the environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from a shell:

```bash
git clone https://github.com/valendra-tech/sherry-0.1-training-recipe.git
cd sherry-0.1-training-recipe
uv python install 3.11
uv sync --locked
```

The default environment installs the dataset tools and test runner without
PyTorch. Python 3.11 is the default; the project also allows Python 3.12.
[`pyproject.toml`](pyproject.toml) declares the dependencies and
[`uv.lock`](uv.lock) pins the resolved versions. Run the commands in this guide
from the repository root. `uv run` uses the project's virtual environment;
you do not need to activate it.

**For SFT, adapter evaluation, and GRPO:** use x86-64 Linux with a BF16-capable
NVIDIA GPU and a driver compatible with CUDA 12.8. The 35B base weights alone
take roughly 70 GB in BF16, before activations, optimizer state, or generated
sequences. GRPO's four completions per prompt can require substantially more
memory. Allow disk space for the base weights, Hugging Face cache, and saved
adapters. The `train` extra is only installed on x86-64 Linux.

```bash
uv sync --locked --extra train
uv run --locked --extra train python - <<'PY'
import torch

if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("Training requires a BF16-capable NVIDIA GPU.")
print(torch.cuda.get_device_name(0), torch.version.cuda)
PY
```

The training extra includes PyTorch, Transformers, PEFT, TRL, Unsloth, and
bitsandbytes. See the [Unsloth installation guide](https://unsloth.ai/docs/get-started/install)
for platform requirements. The optional
[`sft_environment`](reasoning_levels/sft_environment.py) probe checks model
loading and the chat template; unlike the CUDA check above, it loads the
full base model.

## 2. Obtain the reasoning traces

### Recommended: download the published dataset

This is the shortest path to training and does not require a generation API
or Hugging Face credentials:

```bash
uv run --locked hf download \
  --repo-type dataset \
  valendra/sherry-reasoning-effort-0.1-dataset \
  reasoning_effort_math.jsonl \
  --local-dir data/published
```

The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
writes the file to `data/published/reasoning_effort_math.jsonl`.

### Optional: generate a new dataset

[`generate_dataset.py`](reasoning_levels/generate_dataset.py) samples numeric
problems from [NuminaMath-CoT](https://huggingface.co/datasets/project-numina/NuminaMath-CoT),
filtering out proofs and unsupported answer formats. It asks the
[Qwen3.6-35B-A3B NVFP4 teacher](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast)
for multiple traces per problem with the same prompt and thinking enabled.
It checks each final answer against the reference, drops truncated and
incorrect completions, and ranks the remaining traces by observed thinking
length. The shortest, middle, and longest selected traces become `low`,
`medium`, and `high`. Problems without three usable traces are skipped.

Run the teacher behind a trusted, local OpenAI-compatible endpoint that
accepts the model ID used in [`config.py`](reasoning_levels/config.py) and
returns the thinking trace and final answer. The teacher's
[NVFP4 serving guide](https://unsloth.ai/docs/models/qwen3.6#nvfp4) describes
the hardware and server setup. Bind the server to `127.0.0.1` and use the
`/v1` base URL. The generator currently disables TLS certificate verification;
use loopback here instead of sending credentials to a remote endpoint.

```bash
export QWEN_API_BASE="http://127.0.0.1:8000/v1"
export QWEN_API_KEY="local"  # Only for a local server that does not check the key.
export QWEN_MAX_WORKERS=4
export QWEN_CANDIDATES_PER_PROBLEM=3

uv run --locked python -m reasoning_levels.generate_dataset \
  20 data/generated/reasoning_effort_math.jsonl
```

The first argument is the number of problems to sample, not the number of
saved traces. Start small: every problem makes multiple generation requests
and the default completion ceiling is 32,768 tokens. The script resumes from
the specified file and skips problems with a complete triplet. A regenerated
dataset is not expected to match the published rows.

## 3. Validate and split the traces

When generating your own traces, pass `data/generated/reasoning_effort_math.jsonl`
in the command below. The [split helper](reasoning_levels/sft_data.py) retains
only complete triplets of correct, normally terminated traces and keeps all
three effort rows for a problem in the same partition.

```bash
uv run --locked python - data/published/reasoning_effort_math.jsonl <<'PY'
import sys

from reasoning_levels.sft_data import (
    load_jsonl,
    split_by_problem,
    validate_rows,
    write_jsonl,
)

rows = validate_rows(load_jsonl(sys.argv[1]))
train, evaluation = split_by_problem(rows, eval_fraction=0.10, seed=42)
if not train or not evaluation:
    raise SystemExit("Need complete verified triplets in both splits.")

write_jsonl(train, "data/sft/train.jsonl")
write_jsonl(evaluation, "data/sft/eval.jsonl")
print(f"{len(rows)} valid rows: {len(train)} train, {len(evaluation)} evaluation")
PY
```

The published file yields 1,854 training rows and 207 evaluation rows. Keep
both JSONL splits: evaluation and GRPO use them to exclude SFT problems.

## 4. Train the SFT adapter

[`sft_train.py`](reasoning_levels/sft_train.py) teaches the model to respond
to a learned effort token placed immediately before `<think>`. It trains
LoRA weights on attention and shared-expert projections and updates the three
effort token rows without fine-tuning the routed experts. The default
configuration uses LoRA rank 32, alpha 64, BF16, and 8-bit AdamW.

```bash
uv run --locked --extra train python -m reasoning_levels.sft_train \
  --model Qwen/Qwen3.6-35B-A3B \
  --train data/sft/train.jsonl \
  --eval data/sft/eval.jsonl \
  --output checkpoints/sft \
  --max-length 4096 \
  --max-steps 500
```

The output directory contains the adapter, processor and tokenizer files,
and `sft_run.json`, which records training settings, effort token IDs, and
truncation counts. It is not a standalone model. Pass the whole directory to
the next steps so that the processor and learned tokens stay with the adapter.

## 5. Evaluate the SFT adapter

[`evaluate_sft.py`](reasoning_levels/evaluate_sft.py) samples fresh
NuminaMath problems, excluding both SFT splits. It generates one greedy
completion per problem and effort level, then reports numeric-answer accuracy
and reasoning length. This is a diagnostic for the SFT adapter, not the
benchmark suite reported for the merged model.

```bash
uv run --locked --extra train python -m reasoning_levels.evaluate_sft \
  --adapter checkpoints/sft \
  --train data/sft/train.jsonl \
  --sft-eval data/sft/eval.jsonl \
  --problems-output data/sft/unseen-problems.jsonl \
  --output data/sft/evaluation.json \
  --max-eval-rows 20 \
  --max-new-tokens 4096
```

The command evaluates all three effort levels on 20 problems. Inspect
`data/sft/evaluation.json` for the summary and `data/sft/evaluation.jsonl`
for the individual generations. The evaluator resumes from the raw JSONL;
use a fresh `--output` path when changing the adapter, seed, or generation
settings. The default evaluation size is 100 problems.

## 6. Continue training with GRPO

[`grpo_train.py`](reasoning_levels/grpo_train.py) reloads the SFT adapter as
trainable, samples new numeric problems outside both SFT splits, and tries
four completions per prompt and effort. The
[reward](reasoning_levels/grpo_reward.py) gives the strongest signal to
correct answers. It adds smaller signals for numeric near misses, a closed
thinking block with a parseable answer, and effort-appropriate length.

| Effort | Learned control | Length shaping in this recipe |
| --- | --- | --- |
| `low` | `<\|reasoning_effort_low\|>` | Favors up to 1,024 thinking tokens |
| `medium` | `<\|reasoning_effort_medium\|>` | Favors roughly 1,024–4,096 thinking tokens |
| `high` | `<\|reasoning_effort_high\|>` | No length bonus or penalty |

Start with a one-step check on the same GPU host. Despite its name,
`--dry-run` loads the model, trains for one step with a 128-token completion
ceiling, and writes an adapter. Keep its output separate from the real run:

```bash
uv run --locked --extra train python -m reasoning_levels.grpo_train \
  --adapter checkpoints/sft \
  --train data/sft/train.jsonl \
  --sft-eval data/sft/eval.jsonl \
  --problems-output data/sft/grpo-smoke-problems.jsonl \
  --problems-manifest data/sft/grpo-smoke-manifest.json \
  --output checkpoints/grpo-smoke \
  --dry-run
```

Then launch the full recipe:

```bash
uv run --locked --extra train python -m reasoning_levels.grpo_train \
  --adapter checkpoints/sft \
  --train data/sft/train.jsonl \
  --sft-eval data/sft/eval.jsonl \
  --output checkpoints/grpo \
  --max-steps 200 \
  --max-new-tokens 65536 \
  --seed 42
```

The command writes the sampled prompts and their manifest under `data/sft/`.
It saves checkpoints every 50 steps and writes the final adapter and
`grpo_run.json` under `checkpoints/grpo/`. The 65,536-token ceiling is shared
across effort levels. Lowering it can reduce generation cost, but changes
the experiment and how the length reward treats truncated completions.

The optional [`launch_grpo_after_sft`](reasoning_levels/launch_grpo_after_sft.py)
wrapper is for runs managed by `tmux`. It waits for a successful SFT exit
status, validates the SFT artifacts, and runs the one-step check before the
full run. It may install `mergekit` if it is missing, but does not merge
the output weights. The commands above do not require `tmux`.

## 7. Try the published model

Use the [merged Preview](https://huggingface.co/valendra/sherry-35b-a3b-0.1-preview)
to serve the model without training or merging an adapter. Follow the
[vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
in a separate inference environment, then start the
[OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/):

```bash
vllm serve valendra/sherry-35b-a3b-0.1-preview \
  --served-model-name sherry --host 127.0.0.1 --dtype bfloat16 \
  --max-model-len 4096
```

From another shell, send an OpenAI-compatible request:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sherry",
    "messages": [{"role": "user", "content": "What is 17 times 23?"}],
    "temperature": 0,
    "max_tokens": 2048,
    "chat_template_kwargs": {"reasoning_effort": "medium"}
  }'
```

Use `low`, `medium`, or `high`; there is no `xhigh` level. The published
SFT adapter does not expose this argument in its chat template. To run that
adapter with PEFT, follow the separate
[SFT model card](https://huggingface.co/valendra/sherry-35b-a3b-0.1-sft-preview#usage),
which shows how to insert the effort token explicitly.

## 8. Benchmark Sherry against the base model

The original evaluation ran Qwen3.6-35B-A3B and Sherry at `high` effort
with BF16 vLLM, matching prompts and completion budgets, and one greedy
generation per problem.

| Benchmark | Metric | Qwen3.6-35B-A3B | Sherry (`high`) |
| --- | --- | ---: | ---: |
| MATH-500 | Math-Verify | 88.3% | 97.1% |
| MATH-500 | Exact match | 25.2% | 56.2% |
| GPQA-Diamond | Exact match | 57.1% | 78.3% |
| Basic Python (40 tasks) | Pass@1 | 75.0% | 97.5% |
| AIME 2025 (preliminary) | Exact match | 63.3% | 50.0% |

On GPQA-Diamond, mean completion length fell from 20,023 to 7,534 tokens
(about 62% fewer). The MATH-500 Math-Verify percentages cover scoreable
samples, not all 500 examples. AIME 2025 is under review; the results are
not a definitive regression. These are controlled results for this preview,
not leaderboard comparisons.

See the [benchmark guide](reasoning_levels/benchmark/README.md) for `uv`
setup, dataset revisions, and step-by-step run commands. The archived
[Markdown report](reasoning_levels/benchmark/reports/2026-09-18-sherry-high-vs-base.md)
and [JSON summary](reasoning_levels/benchmark/reports/2026-09-18-sherry-high-vs-base.json)
contain sample counts, paired intervals, and token statistics.

## Testing and optional utilities

Run the basic tests without a GPU or network access after `uv sync --locked`:

```bash
uv run --locked --offline python -m pytest -q \
  reasoning_levels/tests/test_boxed.py \
  reasoning_levels/tests/test_chat_template.py \
  reasoning_levels/tests/test_generate.py \
  reasoning_levels/tests/test_grpo_reward.py \
  reasoning_levels/tests/test_sft_data.py
```

For the full suite, use
`uv run --locked --extra train python -m pytest reasoning_levels/tests`
on the training host. Some tests import the training stack, and the
NuminaMath sampling integration tests access the Hugging Face dataset.
None of these tests replaces a GPU training or serving run.

[`upload_dataset.py`](reasoning_levels/upload_dataset.py) is optional. It
creates or updates a remote dataset and does not force private visibility.
Use it only if you intend to publish your generated data. Authenticate with
`uv run --locked hf auth login` and explicitly pass a dataset ID in a namespace
you control:

```bash
uv run --locked python -m reasoning_levels.upload_dataset \
  data/generated/reasoning_effort_math.jsonl \
  YOUR_HF_NAMESPACE/sherry-reasoning-traces
```

Keep API keys, local data, and model artifacts out of Git. The repository
ignores `.env` files, `data/`, `checkpoints/`, and the virtual environment.

## License

The code and documentation in this repository are released under the
[MIT License](LICENSE). The Hugging Face models, adapters, and datasets
are separate artifacts governed by their own licenses. See the linked
model and dataset cards for terms and attribution, and
[third-party notices](THIRD_PARTY_NOTICES.md) for the adapted benchmark code.
