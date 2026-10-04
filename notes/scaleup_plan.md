# Scale-up plan (iterations 9–12), written before any data

Date: 2026-10-04. Budget: the whole research program is capped at $20 estimated spend
(`TOTAL_BUDGET_USD` in `llm/run_pilot.py`). $5.88 was spent in iterations 0–8, so these four experiments
share about $14. Each experiment targets at least 5,000 API calls. Pilots (iterations 1–8) were run on
3–4 seeds or blocks; these runs replace them as the paper's main evidence, and pilot numbers are kept
only as history.

Run order, so that our own traffic never contaminates a cache or latency measurement:
1. E3 alone (models in parallel lanes, 3 at a time);
2. E1 and E2 together (game runs; quality and cost are unaffected by load, latency is secondary);
3. E4 alone (the two models in parallel, each model's blocks sequential).

| | Question | Design | Calls | Est. cost | Config |
|:--|:--|:--|--:|--:|:--|
| E1 | Do price or size predict decision quality? | 9 configurations (8 models) × 10 new seeds (5000–5009) × up to 100 pieces, stateless, annotated, legal-placement JSON schema | ≈5,900 | $2.9 | `configs/scaleup/ladder.yaml` |
| E2 | Does the price card decide which memory policy is cheap? | 4 models (cached input $0: gpt-oss-20b, gpt-oss-120b; no cached price: DeepSeek-V4-Flash, Qwen3.8-27B) × 3 policies (no history, append, last 8 turns) × 9 seeds (5100–5108) | ≈5,500 | $3.2 | `configs/scaleup/memory.yaml` |
| E3 | How many concurrent agent contexts does each deployment keep cached? | all 9 models; loads N ∈ {1, 2, 4, 8, 16, 32, 64} concurrent 6.6k-token agents; 30 s idle; repetitions scaled inversely to price (1–5) | 5,014 | $4.6 | `configs/scaleup/capacity.yaml` |
| E4 | Confirm interference and the defer-long rule on held-out data | `corpus_heldout`, gemma-4-31B and DeepSeek-V4-Flash, 13 blocks each, same 4 conditions, hypotheses and margins as `notes/interference_novelty_matrix.md` | ≈5,100 | $2.2 | `configs/interference/confirm.yaml` |

## Hypotheses and analyses (fixed now)

- **E1.**
  - Report Spearman rank correlations between (a) list price per 100 decisions and score relative to
    the beam-search bot, and (b) total and (c) active parameters and score.
  - Use a bootstrap over seeds.
  - The claim "price and size do not predict decision quality" is made only if the 95% CIs for (a)
    and (b) both include 0 or are negative.
- **E2.**
  - Per model: the cost ratio of append to no history, and the per-move regret difference between
    append and no history, paired by seed.
  - Expected: a ratio near 1 for models that bill cached input at $0, and above 5 for models without a
    cached price; append regret above no-history regret.
- **E3.**
  - Per model: the reported hit rate after 30 s against N, and the cold-prefill latency at N = 1.
  - Capacity is the largest N with hit rate ≥ 0.9.
  - Two analyses, stated as associations, not causes:
    - capacity against KV-cache bytes per token computed from each model's published config;
    - cold-prefill latency against active parameters from the model cards.
  - GLM-5.2 reports no hits, so its hits are inferred from latency (probe faster than 0.35 × its median
    cold warm-up at N = 1) and labeled as inferred.
- **E4.** H1, H2 and B2 exactly as preregistered for the pilot, with the same margins (including the
  +0.15 regret margin), on the untouched held-out corpus.

## Fixed choices and known limits

- MiniMax-M2.5 is excluded from the game experiments (E1, E2). It cannot disable reasoning, and in
  iteration 6 smoke tests it exceeded 2,048 reasoning tokens on every move. It is included in E3.
- E3 uses 6.6k-token contexts (pilot: 15k) so that 64 concurrent agents fit the budget.
- The provider's other traffic is unobserved. Each experiment records start and end times.
- Any deviation from this plan is recorded in the paper's appendix.

## Addendum (2026-10-04, 06:45 UTC, before any analysis of E3 part 2, E1, E2, E4 or E5 data)

- **Cost under the provider's stated policy.** The pricing page and a May 2026 blog state that cached
  input is free on every serverless model, although four model cards show no cached row (see
  `notes/model_architectures.md`). E2 therefore reports cost two ways: the stated policy (cached input
  $0 everywhere; primary) and a full-price upper bound (cached input billed at the input price where no
  cached price is listed). The E2 expectation "above 5 for models without a cached price" applies only
  to the upper bound.
- **E3 contended rounds.** Part 1 was stopped because rounds with N ≥ 32 of different models overlapped
  and drew gateway-wide 429s. Part 2 never runs two N ≥ 32 rounds at once. In the analysis, any round
  that was in flight while two N ≥ 32 rounds overlapped is excluded from both parts, and the count is
  reported. Hit rate counts an agent as warm only if its warm-up and probe both succeeded; refusals are
  reported separately.
- **E5 (decision model).** Intelif (open, Qwen3-4B + LoRA + linear scorer, Jev-compatible request
  format) is self-hosted on the container's 4-core CPU (no GPU). Each legal placement is one option,
  described by its simulated outcome. Precision: bf16 weights with fp32 arithmetic. On three positions
  it chose the same move as bf16 with confidences within 0.003, and was 3× faster. Dynamic int8 was
  7× faster but changed the first move and flattened the distribution, so it is not used. Seeds are
  played in a fixed order (1000–1002, then 5000–5009, the E1 seeds) and no new game starts after
  11:45 UTC. Analyses: score relative to the beam-search bot, per-move oracle regret, CPU latency per
  decision against input tokens, and whether the top option's probability tracks regret.

## Addendum 2 (2026-10-04, 06:50 UTC, compute schedule; no results analyzed)

The container has 4 CPU cores and no GPU. The E1/E2 oracle needs about 1.6 CPU-seconds per decision
(≈12,000 decisions, ≈5 core-hours) and Intelif needs ≈20 s × 3 cores per decision. To finish every run by
13:15 UTC:
- E1 and E2 run their API phase with `--skip_oracle`; the identical oracle is added afterwards with
  `python -m llm.add_oracle` (same settings, read from each run's config.yaml).
- E5 is reduced to seeds 1000–1002, the seeds on which all nine iteration-6 language-model configurations
  were played (paired comparison) and which the same-game GIFs show. The Intelif process is stopped as
  soon as its third game ends; any partly played fourth game is discarded. E1's ten new seeds are not
  played by Intelif.
- The oracle then runs on all four cores. E5's CPU latency is measured while E1/E2/E4 API clients run
  alongside (light CPU use); this is reported.

## Addendum 3 (2026-10-04, 06:55 UTC, before E1/E2/E4 data)

- Budget under the guard's full-price accounting is tight. E2's Qwen3.8-27B window8 arm (est. $0.7) is
  dropped; E2's cap becomes $3.6. Qwen3.8 stays for append vs no history.
- E3 part 2 may stop at its $4.1 cap before Kimi/Qwen3.5/GLM/MiniMax finish; any rounds left are run as
  part 3 (same design, alone, before E1/E2).
- E4's number of blocks per model is set just before it runs: the largest number ≤ 13 whose estimated
  cost (pilot: ≈$0.085 per block per model, full price) fits the remaining program budget.

## Deviation log (filled in as runs finished)

- **E2 (07:26 UTC).** The run reached its $3.6 cap (full-price accounting; Qwen3.8 append alone was $2.0) after
  4,168 calls. 26 of 99 games were stopped by the guard: seeds 5107–5108 never started for most arms and a few
  seed-5106/5107 games stopped part-way. These games are excluded; paired contrasts use, per model, the seeds on
  which every policy finished (6–7 seeds). No E2 top-up was run, so that E4 keeps its budget.
- **E1/E2 oracle.** Run after the API phase with `llm.add_oracle` (E2 with one worker while Intelif played).
- **Score metric (07:31 UTC, after seeing E2's seed 5106; before analyzing E1 or E5).** On seed 5106 the
  beam-search reference bot itself scored only 10 points, so per-seed ratios explode (an agent scoring 51 gets
  5.1). Score relative to the bot is therefore pooled over seeds, (Σ score − Σ random) / (Σ beam − Σ random), with
  a bootstrap over seeds; the mean of per-seed ratios is kept in the summaries for reference.
- **E5 latency (07:35 UTC).** The in-game CPU latency of Intelif is inflated by other work on the same 4 cores
  (E1/E2 clients, the E2 oracle). After all runs, Intelif is re-timed alone on 30 positions sampled uniformly
  (seed 0) from its own games, same precision and 3 threads; this quiet-machine latency is the one reported
  as its serving latency, and the in-game latency is reported alongside.
- **E5 interruption (07:13 UTC).** The Intelif process was killed by the container's out-of-memory killer while
  we rendered a GIF next to it (Intelif holds ≈9.3 GB). Game 1 (seed 1000) had finished and is kept
  (`runs/explore/scaleup/intelif`); game 2 had reached piece 22, whose 22 calls remain in that directory's
  calls.jsonl but are not analyzed. Seeds 1001–1002 were restarted from scratch at 07:54 UTC in
  `runs/explore/scaleup/intelif_b` with the same settings; nothing memory-heavy runs beside it.
