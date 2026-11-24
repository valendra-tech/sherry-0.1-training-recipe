# Sherry 35B-A3B 0.1 GRPO Preview vs Base - Public Benchmark Report

> **Historical results (September 18, 2026).** The AIME 2025 comparison is
> preliminary. The
> [Sherry model card](https://huggingface.co/valendra/sherry-35b-a3b-0.1-preview)
> notes that answer extraction, completion truncation, and shared-problem
> accounting are under review. The archived numbers below are unchanged.

All numbers in this section are measured by this harness; the base model was re-run in this harness under the same protocol.

## AIME 2025

| condition | n | accuracy | 95% CI | math_verify | mean tokens | median tokens | p90 tokens | mean reasoning | truncated |
|---|---|---|---|---|---|---|---|---|---|
| base | 30 | 0.633 | 0.455 to 0.781 | 1.000 (n=19) | 26737.4 | 28110.0 | 32768.0 | 26218.5 | 0.333 |
| sherry-high | 30 | 0.500 | 0.332 to 0.668 | 0.517 (n=29) | 10560.3 | 8593.0 | 18285.0 | 9816.8 | 0.067 |

- Paired delta vs base on 30 problems: -0.133 (95% CI -0.333 to +0.067).
- Paired delta vs base on 19 problems (math_verify): -0.368 (95% CI -0.579 to -0.158).

- Published base reference: not available. The model card publishes AIME 26 (92.7), not AIME 2025; different year set.

## MATH-500

| condition | n | accuracy | 95% CI | math_verify | mean tokens | median tokens | p90 tokens | mean reasoning | truncated |
|---|---|---|---|---|---|---|---|---|---|
| base | 500 | 0.252 | 0.216 to 0.292 | 0.883 (n=230) | 5884.9 | 2237.0 | 19241.0 | 5621.9 | 0.028 |
| sherry-high | 500 | 0.562 | 0.518 to 0.605 | 0.971 (n=347) | 3545.3 | 2785.0 | 7841.0 | 3151.4 | 0.004 |

- Paired delta vs base on 500 problems: +0.310 (95% CI +0.262 to +0.358).
- Paired delta vs base on 191 problems (math_verify): +0.094 (95% CI +0.047 to +0.147).

- Published base reference: not available. No MATH-500 value published in the model card.

## GPQA-Diamond

| condition | n | accuracy | 95% CI | math_verify | mean tokens | median tokens | p90 tokens | mean reasoning | truncated |
|---|---|---|---|---|---|---|---|---|---|
| base | 198 | 0.571 | 0.501 to 0.638 | n/a | 20022.6 | 20972.0 | 32768.0 | 19621.7 | 0.278 |
| sherry-high | 198 | 0.783 | 0.720 to 0.835 | n/a | 7534.2 | 6203.0 | 12222.0 | 6470.6 | 0.035 |

- Paired delta vs base on 198 problems: +0.212 (95% CI +0.141 to +0.283).

- Published base reference: 86.0. Comparable benchmark, different protocol. Model-card GPQA (86.0); published sampling protocol differs from this greedy harness.

## Code (basic suite)

| condition | n | accuracy | 95% CI | math_verify | mean tokens | median tokens | p90 tokens | mean reasoning | truncated |
|---|---|---|---|---|---|---|---|---|---|
| base | 40 | 0.750 | 0.598 to 0.858 | n/a | 6264.2 | 2898.0 | 16384.0 | 6170.9 | 0.225 |
| sherry-high | 40 | 0.975 | 0.871 to 0.996 | n/a | 3153.5 | 2054.0 | 5978.0 | 3036.7 | 0.025 |

- Paired delta vs base on 40 problems: +0.225 (95% CI +0.100 to +0.350).

- Published base reference: not available. No published value; this 40-task suite is authored in-repo.

## Base Reference (Published Model Card)

The base model was also re-run in this harness under the same protocol; the published numbers below are context only.

| task | published base value | comparable | note |
|---|---|---|---|
| AIME 2025 | n/a | no | The model card publishes AIME 26 (92.7), not AIME 2025; different year set. |
| MATH-500 | n/a | no | No MATH-500 value published in the model card. |
| GPQA-Diamond | 86.0 | yes | Model-card GPQA (86.0); published sampling protocol differs from this greedy harness. |
| Code (basic suite) | n/a | no | No published value; this 40-task suite is authored in-repo. |

## Caveats

- Preview release: the Sherry model card states independent quality evaluation
  was incomplete.
- Training data derives from NuminaMath, so AIME 2025 and MATH-500 may be
  contaminated; treat the measured scores as an upper bound.
- GPQA-Diamond comes from the public mirror `hendrydong/gpqa_diamond_mc`
  (198 rows) because the official `Idavidrein/gpqa` repository is gated.
- Greedy single-sample decoding with one shared token budget; small tasks have
  wide Wilson confidence intervals.
- The base model was re-run in this harness under the same protocol, so the
  paired deltas above are controlled comparisons. The published model-card
  table is context only, because its sampling protocol differs (temperature
  1.0, top_p 0.95, and an 81,920-token budget for hard math).
- AIME 2025 has no published base value; the model card reports AIME 26.
- `reasoning_tokens_estimate` comes from re-tokenizing the reasoning text and
  is an approximation, not the server's exact reasoning token count.
- `exact_match` uses a heuristic extractor for unboxed answers and can
  under-count prose answers; `math_verify` is the more robust math metric.
