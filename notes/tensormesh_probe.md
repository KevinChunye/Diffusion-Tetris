# Tensormesh serverless probe (Phase 0)

Date: 2026-10-02. Code: `llm/probe_tensormesh.py`, client `llm/tensormesh_client.py`.
Raw data: `runs/explore/probe/` (`models.csv`, `thinking.csv`, `format.csv`, `cache.csv`,
`cache_16k.csv`, `concurrency.csv`, every call in `probe_calls.jsonl`). Total probe spend: **≈ $0.40**.

## TL;DR
1. **All 9 catalog models respond**, stream, and accept `response_format` (`json_object` and strict
   `json_schema`). Exact ids are org-prefixed (table below); `/v1/models` exists and reports `max_model_len`.
2. **Every model has a working prefix cache, but the price card and `usage` don't always say so.**
   - `usage.prompt_tokens_details.cached_tokens` is reported by 8/9 models. **GLM-5.2-NVFP4 always
     reports 0**, yet a repeated 16k-token prefix drops its TTFT from 10.5 s to 0.8 s (13×). For GLM
     you have to infer hits from TTFT.
   - 5 models bill cached tokens at $0 (gpt-oss-20b/120b, gemma-4, MiniMax, Kimi). A warm 16k call costs
     **0.3–0.4 %** of the cold one. The other 4 (DeepSeek-V4-Flash, Qwen3.5, Qwen3.8, GLM) list no cached
     price, so they bill a hit at the full input price: you get the latency win and no cost win.
3. **Prefill is cheap for small models and expensive for big ones.** Cold TTFT for ~16k tokens ranges
   from 1.0 s (gpt-oss-20b) to 10.5 s (GLM-5.2). Warm TTFT is about 0.45–0.8 s for every model. That
   warm figure is mostly the **network/proxy floor (~0.2–0.6 s, median ≈ 0.45 s for a tiny prompt)**.
   At ~3.5k tokens the cache effect is lost in that noise for everything except GLM (2.5×) and Kimi (1.6×).
4. **Concurrency:** no 429s and no retries up to 32 concurrent streams. Latency stays flat to 16 and
   is ~2.5× higher at 32. We use ≤ 16 concurrent episodes.

## Response fields
- With streaming plus `stream_options={"include_usage": true}`, the last chunk carries `usage` with
  `prompt_tokens_details = {cached_tokens, created_cache_tokens}`. vLLM sends its first SSE chunk (the
  role delta) only after prefill, so **time-to-first-chunk ≈ queue + prefill + network**. We log it as
  `ttft_s`.
- Non-streaming responses also carry `kv_transfer_params.cached_token_stats =
  {num_vllm_cached_tokens, num_lmcache_cached_tokens, ...}`. This is the vLLM + **LMCache** stack: a
  16-token-block GPU prefix cache plus 256-token LMCache chunks. Only gpt-oss-120b, MiniMax and Kimi
  fill it in (e.g. vLLM 15488 / LMCache 15360). gemma and gpt-oss-20b return zeros, and the rest omit
  it. So we can't use it to study cache tiers across models.
- Cache granularity, read off `cached_tokens`: 16-token blocks for most models, 64 for DeepSeek-V4-Flash
  (15360 = 240·64), and **448 for Qwen3.8-27B** (15680 = 35·448; hybrid linear-attention alignment). On
  Qwen3.8, up to ~450 tokens of a shared prefix get recomputed on every hit.

## Models, context limits, price card (USD / 1M tokens)

| id | max_model_len | input | output | cached (listed) | cached_tokens reported? |
|:--|--:|--:|--:|--:|:--|
| openai/gpt-oss-20b | 131072 | 0.07 | 0.28 | 0 | yes |
| openai/gpt-oss-120b | 131072 | 0.15 | 0.60 | 0 | yes (+ tier split) |
| google/gemma-4-31B-it | 32768 | 0.14 | 0.56 | 0 | yes |
| MiniMaxAI/MiniMax-M2.5 | 196608 | 0.30 | 1.20 | 0 | yes (+ tier split) |
| moonshotai/Kimi-K2.7-Code | 262144 | 0.96 | 4.00 | 0 | yes (+ tier split) |
| deepseek-ai/DeepSeek-V4-Flash | 1048576 | 0.14 | 0.28 | — | yes (64-tok blocks) |
| Qwen/Qwen3.8-27B-FP8 | 262144 | 0.32 | 3.20 | — | yes (448-tok blocks) |
| Qwen/Qwen3.5-397B-A17B-FP8 | 262144 | 0.60 | 3.60 | — | yes |
| lukealonso/GLM-5.2-NVFP4 | 250000 | 1.40 | 4.40 | — | **no (always 0)** |

## Reasoning control (`thinking.csv`, prompt "17*23?")
All models answered correctly under every variant. The settings that minimise reasoning tokens are
saved in `configs/model_settings.yaml`:
- gpt-oss-20b/120b: `reasoning_effort=low` (~20 tokens vs ~80–95 by default).
- gemma-4: no thinking by default (4 tokens). **Sending `reasoning_effort=low` switches thinking ON**
  (230 tokens), so we never send it to gemma.
- Qwen3.5 / Qwen3.8 / GLM-5.2: think by default (32–330 tokens). `chat_template_kwargs.enable_thinking=false`
  brings them down to 3–4 tokens.
- MiniMax-M2.5: always reasons (~65–95 tokens); none of the switches turn it off.
- Kimi-K2.7-Code: `enable_thinking=false` makes it put its reasoning in `content` instead of
  dropping it, so we keep separated reasoning at `reasoning_effort=low`.
- DeepSeek-V4-Flash: answers directly (2 tokens) by default.

## response_format (`format.csv`)
Every model returns HTTP 200 and valid JSON with `json_object` and with strict `json_schema`. With no
`response_format`, gemma and GLM wrap the JSON in ```json fences, so the action parser strips fences.

## Prefix-cache test (`cache.csv`, `cache_16k.csv`)
Each trial sends a unique nonce, then a shared prefix, then three suffixes in order: cold (stream),
warm (stream), warm (non-stream, for the tier split). Values are medians over 3 trials at ~3.5k
tokens and 2 trials at ~16k tokens.

| model | cold TTFT 3.5k | warm TTFT 3.5k | cold TTFT 16k | warm TTFT 16k | speedup 16k | warm $ / cold $ |
|:--|--:|--:|--:|--:|--:|--:|
| gpt-oss-20b | 0.23 | 0.45 | 1.03 | 0.45 | 2.3× | 0.004 |
| MiniMax-M2.5 | 0.40 | 0.41 | 1.62 | 0.52 | 3.1× | 0.004 |
| gpt-oss-120b | 0.49 | 0.53 | 1.86 | 0.62 | 3.0× | 0.004 |
| Qwen3.5-397B | 0.51 | 0.38 | 2.00 | 0.55 | 3.6× | **1.00** |
| Qwen3.8-27B | 0.53 | 0.49 | 2.33 | 0.75 | 3.1× | **1.00** |
| gemma-4-31B | 0.68 | 0.53 | 3.33 | 0.57 | 5.8× | 0.003 |
| Kimi-K2.7-Code | 0.79 | 0.50 | 3.48 | 0.59 | 5.9× | 0.004 |
| DeepSeek-V4-Flash | 0.67 | 0.61 | 4.00 | 0.73 | 5.5× | **1.00** |
| GLM-5.2-NVFP4 | 2.13 | 0.84 | 10.52 | 0.79 | 13.3× | **1.00** (hits unreported) |

Rough cold prefill throughput, (16k tokens) / (cold TTFT − 0.45 s floor): gpt-oss-20b ≈ 27k tok/s,
gpt-oss-120b ≈ 11k, MiniMax ≈ 13k, Qwen3.5 ≈ 11k, Qwen3.8 ≈ 9k, gemma ≈ 6k, Kimi ≈ 5k,
DeepSeek-V4-Flash ≈ 4k, GLM-5.2 ≈ 1.6k.

## Implications for the experiments
- **On gpt-oss-20b the cache shows up in cost first and latency second.** Cached input is free, but
  prefill only beats the network floor once prompts pass about 8–10k tokens. A history policy that
  stays under ~5k tokens per call will show a cost effect and little TTFT effect.
- **The price card is a confound.** On DeepSeek, Qwen and GLM, cache-hostile memory costs the same as
  cache-friendly memory. The cache still matters for latency there, and matters most on GLM.
- **TTFT needs paired designs and medians.** Per-call network jitter (±0.2 s) is larger than any
  prefill difference under ~5k tokens on fast models. Arms run concurrently (interleaved in time)
  rather than one after another.
- The concurrency ceiling (~16) bounds parallel episodes per pilot.
