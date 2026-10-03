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

RESULTS_PLACEHOLDER
