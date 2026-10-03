# Do long-context requests delay short Tetris decisions on a hosted endpoint? (iteration 8)

Status: **pilot** (dev corpus, 4 blocks per model, ≤ $1 estimated). The overlap audit, go/no-go
decision and preregistered hypotheses are in
[`interference_novelty_matrix.md`](interference_novelty_matrix.md). The config is
[`configs/interference/pilot_v2.yaml`](../configs/interference/pilot_v2.yaml), with one documented
deviation from [`pilot.yaml`](../configs/interference/pilot.yaml), described below.

Three kinds of evidence appear below and are never pooled.

| Label | What it is | Where |
|:--|:--|:--|
| **LIVE** | Real requests to Tensormesh serverless; costs are price-card estimates, not invoices | `runs/explore/interference/{calibration,calibration_gemma,live_pilot,live_pilot_v2}` |
| **SIMULATED** | `llm/endpoint_sim.py`, a chunked-prefill engine with *assumed* replica counts and step budgets. It validates the pipeline and shows the two regimes; it is not a model of Tensormesh | `runs/explore/interference/{sim_pilot,sim_calibrated}` |
| **MOCK** | A fake streaming client in `tests/test_interference.py`; it checks accounting only | `pytest` |

## Question and contribution

On a black-box serverless endpoint, does a tenant's own stream of long-context requests (about 19k
fresh prompt tokens each) raise the time it takes the same tenant's short decision requests (about
1k tokens) to return a valid action? If so, can a client-side admission policy recover it without
losing useful decisions or decision quality?

The audit found that the scheduling ideas themselves are not new. Server-side, they include chunked
prefill, priority scheduling, disaggregation, and short-first or length-aware queues. Client-side,
DRR and admit/defer have been evaluated in simulation only, and that simulation models latency as
linear in output tokens with no prefill. What this study adds is the measurement on a real endpoint:
- prefill-dominated requests
- frozen agent decisions with an oracle quality score
- open-loop arrival replay
- cache warmth removed by construction
- time-to-valid-action (TTVA) instead of TTFT

## Design

- **Corpus** (`llm/mixed_workload.py`, frozen and hash-verified). 160 short and 40 long decision
  requests drawn from greedy, beam and logged-LLM Tetris trajectories (dev seeds 3000–3003, plus
  historical seeds 1000–1002).
  - Short: stateless raw-track prompt.
  - Long: the same kind of decision preceded by 48 prior turns of the same trajectory (append memory).
  - Each entry stores the legal ids and an oracle Q-table (`q_beam`).
  - Calibration (LIVE) measured about 2.45 bytes/token: shorts are about 1.0k tokens and longs
    18–21k tokens.
  - A separate held-out corpus (seeds 4000–4003) is built and untouched, reserved for confirmation.
- **Traces.** Seeded Poisson arrivals over 40 s, with shorts at λ = 1.0/s and longs at
  λ = 0.15/s, plus a `GET /models` path probe every 2 s. Block *k* uses trace seed 800 + *k*, and
  the same trace is used for every model and condition.
- **Conditions** (`llm/admission.py`). The same policy objects drive the live replayer and the
  simulator.
  - `short_only`: the block's trace with the long requests removed, so shorts keep the same
    arrival times.
  - `fifo`: one FIFO queue with at most 8 requests in flight.
  - `fifo_prio`: `fifo` plus a vLLM-style `priority` field (short 0, long 10). The endpoint
    accepts it; whether it is honored is unknown.
  - `defer_long` (candidate): shorts go first. At most 1 long is in flight, and a long is held
    while any short is waiting or in flight, unless it has waited at least 6 s (the starvation
    bound).
- **Pairing and order.** Every condition in a block replays the identical trace and identical
  frozen requests. Condition order is shuffled per (block, model) with a recorded seed, models are
  interleaved by block, and replays are separated by an 8 s cooldown.
- **Replay** (`llm/trace_replay.py`).
  - Open loop: arrivals are released at their scheduled offsets regardless of completions, so there
    is no coordinated omission.
  - No retries; failures, timeouts and backpressure rejections are recorded outcomes.
  - Every request gets a fresh UUID namespace at position 0 of the system prompt, so no condition
    can reuse another's KV cache. Reported `cached_tokens` is checked against this.
  - Spend is reserved per request from the price card and a byte-length upper bound on prompt
    tokens. The check fails closed when price or usage is missing, and an interrupted run charges
    in-flight requests at their reservation.
- **Metrics** (`llm/interference_analysis.py`).
  - **TTVA** runs from the *scheduled* arrival to the first streamed content prefix that parses as
    a complete JSON object with a legal `action_id`. Requests that never produce one count as +∞,
    so failures can only worsen the tails.
  - Short latency is split into three parts:
    - client queueing (admit − arrival)
    - endpoint time to the first SSE chunk (first chunk − admit)
    - time from first chunk to a valid action
  - **Useful decision:** TTVA ≤ 2.5 s. This is a configured service objective, not a game rule.
  - **Quality:** legal rate and oracle regret.
  - Also reported: path-probe RTT and price-card cost.
- **Statistics.** A two-level paired bootstrap (2000 replicates): resample time blocks, then
  requests within each block, using the same resampled requests for both conditions. Per-block
  estimates are listed so the sign can be checked block by block.
- **Preregistered decisions.**
  - H1 is supported if the CI of the relative change in short p95 TTVA (`fifo` vs `short_only`)
    lies above 0 and the shift exceeds the path-probe shift.
  - H2 is supported if the CI of (`defer_long` vs `fifo`) lies at or below −20%. The guards are
    noninferiority on useful decisions/s (CI lower bound above −5% relative), legal rate (above
    −2 pp) and regret (CI upper bound below +0.15).
  - B2 is descriptive.

## Protocol deviation (before any data from the affected model)

Attempt 1 (`live_pilot`, see its `ABORTED.md`) was stopped after 1.3 replays. Most gpt-oss-20b
short decisions on the raw-track corpus used all 512 completion tokens reasoning and produced no
content, so short requests were decode-bound and their TTVA was undefined in every condition.
gpt-oss-20b was replaced by gemma-4-31B-it with thinking disabled. In calibration it answers in 10
tokens and reports `cached_tokens`. The trace, conditions, hypotheses, margins and analysis are
unchanged. The aborted attempt cost $0.028 (estimated) and is not used for any hypothesis test.

## Results (LIVE, `runs/explore/interference/live_pilot_v2`)

The run took place on 2026-10-03, 01:00–01:31 UTC, against the dev corpus: 4 blocks × 4 conditions ×
2 models = 32 replays, 1570 chat calls and 640 path probes. There were 0 HTTP errors, 0 timeouts and 0
backpressure rejections. Every trace item has exactly one row. The generated tables are in
[`report.md`](../runs/explore/interference/live_pilot_v2/report.md), with `summary.csv`, `paired.csv`,
`overlap.csv` and `exploratory.csv` alongside it. Figure:
[`short_ttva_cdf.png`](../runs/explore/interference/live_pilot_v2/short_ttva_cdf.png).

**Preregistered verdicts.** All CIs are from the two-level paired bootstrap with 2000 replicates.

| | gemma-4-31B-it | DeepSeek-V4-Flash |
|:--|:--|:--|
| **H1**: long requests raise short p95 TTVA (`fifo` vs `short_only`) | **Supported.** 0.92 s → 15.4 s, **+1577% [+726%, +1866%]**; positive in all 4 blocks (+439% to +1865%). The path probe p95 moved only +0.21 s | **Supported.** 3.07 s → +∞, **+∞ [+772%, +∞]**: more than 5% of short decisions never became valid under mixed load. Valid-only p95 (exploratory): +1080% [+281%, +1350%] |
| **H2**: `defer_long` cuts short p95 by ≥ 20% vs `fifo`, with useful decisions/s and quality noninferior | **Inconclusive.** The latency criterion is met (**−69% [−74%, −41%]**, 15.4 s → 4.7 s) and useful decisions/s rose **+63% [+18%, +258%]**. Legal rate is identical (100%). The regret difference is −0.04 [−0.25, **+0.151**], so its upper bound misses the preregistered +0.15 margin by 0.001 and quality noninferiority is *not established*. The margin is not changed after the fact | **Undefined** for the preregistered statistic, because short p95 is +∞ in both arms. Exploratory: valid-only p95 −74% [−77%, −31%], p50 −5.0 s [−21.9, −0.7], useful decisions/s +96% [+42%, +278%]. Regret is too noisy to judge: +0.59 [−2.2, +2.7] |
| **B2**: `priority` hints (short 0, long 10) | No detectable effect: −1% [−11%, +7%] | p95 undefined; p50 +2.7 s [−0.8, +5.4]. No evidence of benefit |

**Where the delay is (API-observable decomposition, pooled over blocks, short requests).**

| | gemma `short_only` | gemma `fifo` | gemma `defer_long` | DeepSeek `short_only` | DeepSeek `fifo` | DeepSeek `defer_long` |
|:--|--:|--:|--:|--:|--:|--:|
| client queue p95 (admit − arrival) | 0.00 s | 10.4 s | 0.07 s | 0.00 s | 29.0 s | 2.8 s |
| endpoint first chunk p50 / p95 (after admission) | 0.50 / 0.65 s | 0.94 / 5.6 s | 0.71 / 3.9 s | 0.57 / 1.8 s | 2.6 / 10.5 s | 2.1 / 5.3 s |
| TTVA p50 / p95 | 0.69 / 0.92 s | 3.5 / 15.4 s | 1.2 / 4.7 s | 0.72 / 3.1 s | 9.2 / +∞ s | 4.3 / +∞ s |
| long TTVA p50 / p95 | — | 11.1 / 18.4 s | 9.5 / 22.6 s | — | 18.6 / 41.2 s | 15.6 / 36.4 s |
| long max client queue | — | 10.5 s | 20.3 s | — | 30.4 s | 35.1 s |
| useful decisions/s (TTVA ≤ 2.5 s) | 1.08 | 0.42 | 0.68 | 0.96 | 0.18 | 0.34 |
| path probe p95 | 0.40 s | 0.61 s | 0.41 s | 0.42 s | 0.57 s | 0.41 s |
| est. cost (4 blocks) | $0.022 | $0.107 | $0.107 | $0.020 | $0.098 | $0.098 |

**Endpoint time to first chunk of a short request, by how many long requests were admitted but had not
yet streamed their first chunk when the short was sent.** This is observational, from `overlap.csv`;
values are p50 / p95.

| longs still prefilling | gemma `fifo` | gemma `defer_long` | DeepSeek `fifo` | DeepSeek `defer_long` |
|:--|--:|--:|--:|--:|
| 0 | 0.54 / 1.2 s (n=86) | 0.53 / 0.71 s (n=84) | 1.2 / 5.2 s (n=88) | 0.77 / 3.1 s (n=94) |
| 1 | 3.0 / 4.6 s (n=60) | 3.0 / 4.0 s (n=89) | 5.5 / 8.7 s (n=59) | 3.9 / 5.9 s (n=79) |
| 2+ | 4.9 / 9.9 s (n=27) | — | 8.4 / 17.1 s (n=26) | — |

### What this shows, and what it does not

1. **Interference is real and large at the endpoint, not only in our client queue.** Under FIFO, part
   of the short-request tail is client head-of-line blocking: longs fill the cap of 8 because the
   endpoint takes 10–40 s to serve them under this load. However, the endpoint's own time to first
   chunk also rises sharply, and only when a long request is still waiting for its first chunk:
   - gemma: 0.5 s → 3.0 s with one such long, and 4.9 s with two.
   - Over the same period the path probe (`GET /models`) moved about 0.2 s.

   Calling the mechanism prefill contention, or queueing behind long prefills inside the provider, is
   a hypothesis *consistent* with this. It is not a measurement: batching, chunk size, replica count
   and routing are not observable through the API.
2. **The candidate policy recovers most of the client-side part and some of the endpoint-side part.**
   - **What it removes.** `defer_long` keeps at most one long in flight and starts a long only when
     no short is pending, which removes the client queue (gemma short p95 queue 10.4 s → 0.07 s) and
     the "2+ longs prefilling" state.
   - **What it cannot remove.** It cannot stop a short that *arrives* while a long is prefilling
     from waiting behind it: the "1 long prefilling" row is unchanged at about 3 s. Hence the
     residual p95 of 4.7 s against 0.92 s alone.
   - **Long requests.** Serializing longs did not hurt their median (gemma 11.1 → 9.5 s), but it
     lengthened their tail and their client wait (p95 18.4 → 22.6 s, max client queue 10.5 → 20.3 s).
     The 6 s starvation bound limits deferral behind *shorts*. Waiting behind the in-flight long
     (L = 1) is not bounded by it, and it is a real cost of this policy.
   - **Cost.** Identical: the same requests and prompt tokens (758.9k vs 758.9k for gemma).
3. **Priority hints did nothing observable.** The endpoint accepts `priority`, but `fifo_prio`
   matched `fifo` within noise on both models. Either it is ignored or it does not act on the
   bottleneck. From the outside these cannot be distinguished.
4. **The quality guard is the weak point of this pilot, not the latency effect.**
   - Admission control does not change prompts. Each condition does, however, send a different
     random namespace, so prompts are not byte-identical, and both models are nondeterministic at
     temperature 0.
   - Same-task answer agreement with `fifo` is 88–90% for gemma and only 45–49% for DeepSeek, whose
     regret estimates are therefore too noisy to test a 0.15 margin with 4 blocks.
   - For gemma, the regret CI is centered on −0.04 but its upper end lands at +0.151.
   - A confirmatory run needs more blocks, or a margin and power analysis set from this pilot's
     variance, before the quality claim can be made.

**Exploratory (not preregistered).** DeepSeek-V4-Flash sometimes answers with the wrong key,
`{"action": N}`. The lenient game parser accepts it; the strict TTVA definition does not. It happened
for 2/173 shorts alone versus 10–12/173 in each mixed condition (Fisher exact p = 0.011–0.035,
unadjusted, 3 comparisons). That is a hint that output format depends on co-running load. It could
also be prompt-namespace sensitivity or chance, and it needs its own preregistered test. gemma
produced no invalid replies in any condition.

**Cache warmth.** gemma reported `cached_tokens` on every request and all values were 0, so it was
cold as designed. DeepSeek does not report cache usage, so for DeepSeek coldness rests on the
per-request namespace by construction and is not verified. Historical evidence that DeepSeek prefix
caching keys on the leading tokens comes from iterations 1–5.

## Simulated evidence (SIMULATED; not Tensormesh)

`sim_pilot` (assumed profiles) and `sim_calibrated` (calibrated token counts and unloaded prefill
rates, *assumed* replica counts and step budgets) reproduce the two regimes the live data
distinguishes between:
- a serialized-prefill endpoint: H1 holds and `defer_long` cuts short p95 by about 60–70%;
- a wide endpoint with spare prefill capacity: no interference, and `defer_long` only delays longs,
  costing useful decisions.

Both live endpoints behaved like the first regime at this load. The simulator ignores `priority`, so
its `fifo_prio` rows equal `fifo` by construction. Simulated numbers are not evidence about
Tensormesh and are not pooled with live data.

## Spend (LIVE, price-card estimates)

| Item | Est. cost |
|:--|--:|
| calibration (DeepSeek, gpt-oss-20b) | $0.0084 |
| attempt 1, aborted (gpt-oss-20b; includes conservative in-flight reservations) | $0.0282 |
| calibration (gemma) | $0.0060 |
| live pilot v2 | $0.6596 |
| **total, iteration 8** | **$0.7022** (≤ $1 cap) |

## Limitations

- **Pilot scale.** 4 blocks per model in one 31-minute window. Other tenants' load is unobserved;
  randomized condition order protects the paired contrasts against slow drift, but not against
  bursts.
- **One load point.** The tenant's own offered prefill (about 2.9k tokens/s of long context) is
  close to the unloaded single-request prefill rate (about 4k tokens/s). Effects at lighter long
  rates, other caps, or L = 2 are untested.
- **Dev corpus only.** The held-out corpus is untouched.
- **Prompt namespaces.** They differ per request (needed to remove cache warmth), so quality
  comparisons carry prompt-perturbation noise.
- **TTVA uses the final attempt.** There were no retries by design, so this is the same as end to
  end.

## Next steps

1. Run a confirmatory held-out run (`corpus_heldout`) with the same protocol.
   - Use ≥ 8 blocks.
   - Preregister the regret margin from this pilot's paired variance.
   - Spread blocks over time of day.
2. Measure a dose-response over λ_long (0.05, 0.1, 0.15) to locate the onset of endpoint-side
   interference.
3. Attack the residual "short arrives while a long is prefilling" delay, which client admission
   cannot remove.
   - Bound the long's own wait behind the in-flight long (L = 2, or an explicit long-queue deadline),
     and compare.
   - Shrink the long context at the application level (window or summary memory from iterations
     1–5) and compare it with admission control at equal decision quality.
4. Run the DeepSeek wrong-key effect as its own preregistered test, with fixed namespaces repeated
   under both load conditions to separate load from prompt sensitivity.
